# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This classes provide help to integrate lightgbm
"""
import lightgbm
import logging
from typing import List

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
    COMM_TAG_METRIC = 209834 # "random id"

    """ This class handles LightGBM callbacks for recording metrics. """
    def __init__(self, metrics_logger, mpi_comm, world_size=1, world_rank=0):
        """
        Args:
            metrics_logger (common.metrics.MetricsLogger): class to log metrics using MLFlow
        """
        self.metrics_logger = metrics_logger
        self.node_metrics = {}
        self.distributed_metrics = {}
        self.world_size = world_size
        self.world_rank = world_rank
        self.mpi_comm = mpi_comm
        self.logger = logging.getLogger(__name__)

    def report_distributed_metric(self, env: lightgbm.callback.CallbackEnv):
        """Sends metrics to node 0"""
        self.mpi_comm.send(env, 0, tag=LightGBMDistributedCallbackHandler.COMM_TAG_METRIC) # blocking

    def collect_distributed_metrics(self, iteration: int):
        """Collect metrics from all nodes other than 0"""
        for i in range(1, self.world_size):
            remote_node_metrics = self.mpi_comm.recv(source=i, tag=LightGBMDistributedCallbackHandler.COMM_TAG_METRIC) # blocking
            if remote_node_metrics.iteration != iteration:
                self.logger.warning(f"Remove node {i} sent metric for iteration {remote_node_metrics.iteration} while node 0 is at iteration {iteration}")
            self.store_metric(i, remote_node_metrics)

    def store_distributed_metric(self, node: int, env: lightgbm.callback.CallbackEnv):
        """Stores a metric in the internal storage
        for processing during aggregate_and_report_loop()"""
        iteration = env.iteration
        if iteration not in self.distributed_metrics:
            self.distributed_metrics[iteration] = {}
        self.distributed_metrics[iteration][node] = env

    def aggregate_and_report_task(self, key: str, iteration: int, eval_name: str, results: List[float]):
        # TODO: devise aggregation method per eval_name
        self.metrics_logger.log_metric(
            key=key,
            value=sum(results),
            step=iteration # provide iteration as step in mlflow
        )

    def aggregate_and_report_loop(self):
        aggregation_tasks = {}

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


    def callback(self, env: lightgbm.callback.CallbackEnv) -> None:
        """Callback method to collect metrics produced by LightGBM.

        See https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/callback.html
        """
        # let's record in the object for future use
        self.node_metrics[env.iteration] = env.evaluation_result_list

        # node 0 gets to report its metrics
        if self.world_rank == 0:
            # loop on all the evaluation results tuples
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                # log each as a distinct metric
                self.metrics_logger.log_metric(
                    key=f"node_0/{data_name}.{eval_name}",
                    value=result,
                    step=env.iteration # provide iteration as step in mlflow
                )

            # store own's metrics in the record
            self.store_distributed_metric(self.world_rank, env)

            # plus collects everybody else's
            self.collect_distributed_metrics(env.iteration)

            # and report aggregates
            self.aggregate_and_report_loop()
        else:
            # all the other just report back to node 0
            self.report_distributed_metric(env)
