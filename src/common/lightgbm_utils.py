# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This classes provide help to integrate lightgbm
"""
import lightgbm
import numpy as np
import logging
from typing import List
import threading
import time

class LightGBMCallbackHandler():
    """ This class handles LightGBM callbacks for recording metrics. """
    def __init__(self, metrics_logger):
        """
        Args:
            metrics_logger (common.metrics.MetricsLogger): class to log metrics using MLFlow
        """
        self.metrics = {}
        self.metrics_logger = metrics_logger
        self.logger = logging.getLogger(__name__)
    
    def callback(self, env: lightgbm.callback.CallbackEnv) -> None:
        """Callback method to collect metrics produced by LightGBM.

        See https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/callback.html
        """
        # let's record in the object for future use
        self.metrics[env.iteration] = env.evaluation_result_list

        # loop on all the evaluation results tuples
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            # log each as a distinct metric
            self.metrics_logger.log_metric(
                key=f"{data_name}.{eval_name}",
                value=result,
                step=env.iteration # provide iteration as step in mlflow
            )


class LightGBMDistributedCallbackHandler():
    """ This class handles LightGBM callbacks for recording metrics. """
    def __init__(self, metrics_logger, mpi_comm, world_size=1, world_rank=0):
        """Constructor

        Args:
            metrics_logger (common.metrics.MetricsLogger): class to log metrics using MLFlow
            mpi_comm (MPI.COMM_WORLD): communicator
            world_size (int): mpi world size
            world_rank (int): mpi world rank of this node
        """
        self.recording_thread = DistributedMetricCollectionThread(metrics_logger, mpi_comm, world_size=world_size, world_rank=world_rank)
        self.recording_thread.start()
        self.logger = logging.getLogger(__name__)

    def finalize(self):
        """Asks internal thread to finalize"""
        # do one last report
        self.recording_thread.aggregate_and_report_loop()

        # set status to kill and join
        self.recording_thread.killed = True
        self.recording_thread.join()

    def callback(self, env: lightgbm.callback.CallbackEnv) -> None:
        """Callback method to collect metrics produced by LightGBM.

        See https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/callback.html
        """
        # let's record in the object for future use
        self.recording_thread.send_distributed_metric(env)
        self.logger.info("End of callback")


class DistributedMetricCollectionThread(threading.Thread):
    """ This class handles MPI communication of LightGBM callback metrics.
    NOTE: We needed to put this in a thread because having callback()
    do the recv/send directly was interacting with LightGBM's own MPI communication somehow.
    """
    COMM_TAG_METRIC = 209834 # "random tag"

    def __init__(self, metrics_logger, mpi_comm, world_size=1, world_rank=0):
        """Constructor

        Args:
            metrics_logger (common.metrics.MetricsLogger): class to log metrics using MLFlow
            mpi_comm (MPI.COMM_WORLD): communicator
            world_size (int): mpi world size
            world_rank (int): mpi world rank of this node
        """
        threading.Thread.__init__(self)
        self.killed = False # flag, set to True to kill from the inside

        self.logger = logging.getLogger(__name__)
        self.metrics_logger = metrics_logger

        # internal sync storage
        self.distributed_metrics = {}
        self.record_lock = threading.Lock()
        self.send_queue = []
        self.send_lock = threading.Lock()

        # MPI communication
        self.mpi_comm = mpi_comm
        self.world_size = world_size
        self.world_rank = world_rank


    #####################
    ### RUN FUNCTIONS ###
    #####################

    def run_head(self):
        """Run function for node 0 only"""
        while not(self.killed):
            time.sleep(1)

            # collect everything from other nodes into internal record
            for i in range(1, self.world_size):
                self.logger.info(f"Probing metric from node {i}")

                if self.mpi_comm.iprobe(source=i, tag=DistributedMetricCollectionThread.COMM_TAG_METRIC): # non-blocking
                    self.logger.info(f"Collecting metric from node {i}")
                    remote_node_metrics = self.mpi_comm.recv(source=i, tag=DistributedMetricCollectionThread.COMM_TAG_METRIC) # blocking
                else:
                    self.logger.info(f"NO metric from node {i}")
                    continue

                self.record_distributed_metric(i, remote_node_metrics)

            # record node_0's own metrics in internal storage
            with self.send_lock:
                while self.send_queue:
                    entry = self.send_queue.pop()
                    self.record_distributed_metric(0, entry)

            # then aggregate whatever is in the internal record
            self.aggregate_and_report_loop()

    def run_worker(self):
        """Run function for all other nodes"""
        while not(self.killed):
            time.sleep(1)
            # all other nodes send to node_0
            with self.send_lock:
                while self.send_queue:
                    entry = self.send_queue.pop()
                    self.logger.info(f"Reporting metric back to node 0: {entry}")
                    self.mpi_comm.isend(entry, 0, tag=DistributedMetricCollectionThread.COMM_TAG_METRIC) # non-blocking

    def run(self):
        """Main function of the thread"""
        if self.world_rank == 0:
            self.run_head()
        else:
            self.run_worker()

    ###################
    ### SEND / RECV ###
    ###################

    def send_distributed_metric(self, env: lightgbm.callback.CallbackEnv):
        """Stores a metric report in the internal queue
        to be sent by thread using MPI"""

        if self.world_rank == 0: # node_0 also record as mlflow
            # loop on all the evaluation results tuples
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                # log each as a distinct metric
                self.metrics_logger.log_metric(
                    key=f"node_0/{data_name}.{eval_name}",
                    value=result,
                    step=env.iteration # provide iteration as step in mlflow
                )

        self.logger.info(f"Queueing metric to send to node_0: iteration={env.iteration}")
        with self.send_lock:
            self.send_queue.append(env)

    def record_distributed_metric(self, node, env: lightgbm.callback.CallbackEnv):
        """Records a metric report internally to node 0"""
        self.logger.info(f"Recorded metric from node {node}: {env}")
        with self.record_lock:
            iteration = env.iteration
            if iteration not in self.distributed_metrics:
                self.distributed_metrics[iteration] = {}
            self.distributed_metrics[iteration][node] = env


    ##################
    ### PROCESSING ###
    ##################

    def aggregate_and_report_task(self, key: str, iteration: int, eval_name: str, results: List[float]):
        # TODO: devise aggregation method per eval_name
        self.metrics_logger.log_metric(
            key=key,
            value=np.mean(results),
            step=iteration # provide iteration as step in mlflow
        )

    def aggregate_and_report_loop(self):
        aggregation_tasks = {}

        with self.record_lock:
            for iteration in list(self.distributed_metrics.keys()):
                if len(self.distributed_metrics[iteration]) < self.world_size:
                    continue

                # loop on all the evaluation results tuples
                for node_id, node_metrics in self.distributed_metrics[iteration].items():
                    for data_name, eval_name, result, _ in node_metrics.evaluation_result_list:
                        key = f"{data_name}.{eval_name}"
                        if key not in aggregation_tasks:
                            # record name of metric for aggregation method
                            aggregation_tasks[key] = (iteration, eval_name, [])

                        # add value in the list
                        aggregation_tasks[key][2].append(result)

                # once done, remove the data from the "queue"
                del self.distributed_metrics[iteration]

            for key, (iteration, eval_name, results) in aggregation_tasks.items():
                self.aggregate_and_report_task(key, iteration, eval_name, results)
