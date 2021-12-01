# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This classes provide help to integrate lightgbm
"""
import lightgbm
import logging

class LightGBMCallbackHandler():
    """ This class handles LightGBM callbacks for recording metrics. """
    def __init__(self, metrics_logger, metrics_prefix=None, metrics_suffix=None):
        """
        Args:
            metrics_logger (common.metrics.MetricsLogger): class to log metrics using MLFlow
        """
        self.metrics = {}
        self.metrics_logger = metrics_logger
        self.metrics_prefix = metrics_prefix
        self.metrics_suffix = metrics_suffix
        self.logger = logging.getLogger(__name__)
    
    def _format_metric_key(self, data_name, eval_name):
        """Builds a metric key with prefix and suffix"""
        key = f"{data_name}.{eval_name}"

        if self.metrics_prefix:
            key = self.metrics_prefix + key
        if self.metrics_suffix:
            key = key + self.metrics_suffix

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
                key=self._format_metric_key(data_name, eval_name),
                value=result,
                step=env.iteration # provide iteration as step in mlflow
            )
