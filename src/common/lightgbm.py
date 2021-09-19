# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This classes provide help to integrate lightgbm
"""
import lightgbm
import logging
from mpi4py import MPI

class LightGBMCallbackHandler():
    """ This class handles LightGBM callbacks for recording metrics. """
    def __init__(self, metrics_logger):
        """
        Args:
            metrics_logger (common.metrics.MetricsLogger)
            node_index (int) : if mpi, provide index of the node
        """
        self.metrics = []
        self.metrics_logger = metrics_logger
        self.logger = logging.getLogger(__name__)
    
    def callback(self, env: lightgbm.callback.CallbackEnv) -> None:
        """
        See https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/callback.html
        """
        # let's record in the object for future use
        self.metrics.append(env.evaluation_result_list)

        # loop on all the evaluation results tuples
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            # log each as a distinct metric
            self.metrics_logger.log_metric(
                key=f"{data_name}.{eval_name}",
                value=result,
                step=env.iteration # provide iteration as step in mlflow
            )

    def mpi_catch_last_metric(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        all_metrics = {}

        if rank == 0:
            for i in range(size):
                received_metric_msg = comm.recv(source=i)
                all_metrics[i] = received_metric_msg
        else:
             comm.send(self.metrics[-1], dest=0)
        
        print(all_metrics)

