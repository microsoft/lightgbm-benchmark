# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
These classes provide some tools to automate wall time compute and logging.
"""
import os
import time
import re
from functools import wraps
import mlflow
import platform
import psutil
import json
import traceback
import logging


class MetricType():
    # a metric your script generates once (per node), example: training time
    ONETIME_METRIC = 1

    # a metric generated multiple times, once per "step" or iteration, example: rmse
    ITERATION_METRIC = 2

    # a perf metric generated at regular intervals
    PERF_INTERVAL_METRIC = 2


class MetricsLogger():
    """
    Class for handling metrics logging in MLFlow.
    """
    _initialized = False

    def __init__(self, session_name=None, metrics_prefix=None):
        self._metrics_prefix = metrics_prefix
        self._session_name = session_name
        self._logger = logging.getLogger(__name__)
    
    def open(self):
        """Opens the MLFlow session."""
        if not MetricsLogger._initialized:
            self._logger.info(f"Initializing MLFLOW [session='{self._session_name}', metrics_prefix={self._metrics_prefix}]")
            mlflow.start_run()
            MetricsLogger._initialized = True

    def close(self):
        """Close the MLFlow session."""
        if MetricsLogger._initialized:
            self._logger.info(f"Finalizing MLFLOW [session='{self._session_name}']")
            mlflow.end_run()
            MetricsLogger._initialized = False
        else:
            self._logger.warning(f"Call to finalize MLFLOW [session='{self._session_name}'] that was never initialized.")

    def _remove_non_allowed_chars(self, name_string):
        """ Removes chars not allowed for metric keys in mlflow """
        return re.sub(r'[^a-zA-Z0-9_\-\.\ \/]', '', name_string)

    def log_metric(self, key, value, step=None, type=MetricType.ONETIME_METRIC):
        """Logs a metric key/value pair.

        Args:
            key (str): metric key
            value (str): metric value
            step (int): which step to log this metric? (see mlflow.log_metric())
            type (int): type of the metric
        """
        if self._metrics_prefix:
            key = self._metrics_prefix + key

        key = self._remove_non_allowed_chars(key)

        self._logger.debug(f"mlflow[session={self._session_name}].log_metric({key},{value})")
        # NOTE: there's a limit to the name of a metric
        if len(key) > 50:
            key = key[:50]

        if type == MetricType.PERF_INTERVAL_METRIC:
            pass # for now, do not process those
        else:
            try:
                mlflow.log_metric(key, value, step=step)
            except mlflow.exceptions.MlflowException:
                self._logger.critical(f"Could not log metric using MLFLOW due to exception:\n{traceback.format_exc()}")

    def log_figure(self, figure, artifact_file):
        """Logs a figure using mlflow
        
        Args:
            figure (Union[matplotlib.figure.Figure, plotly.graph_objects.Figure]): figure to log
            artifact_file (str): name of file to record
        """
        try:
            mlflow.log_figure(figure, artifact_file)
        except mlflow.exceptions.MlflowException:
            self._logger.critical(f"Could not log figure using MLFLOW due to exception:\n{traceback.format_exc()}")

    def set_properties(self, **kwargs):
        """Set properties/tags for the session.
        
        Args:
            kwargs (dict): any keyword argument will be passed as tags to MLFLow
        """
        self._logger.debug(f"mlflow[session={self._session_name}].set_tags({kwargs})")
        mlflow.set_tags(kwargs)

    def set_platform_properties(self):
        """ Capture platform sysinfo and record as properties. """
        self.set_properties(
            machine=platform.machine(),
            processor=platform.processor(),
            architecture="-".join(platform.architecture()),
            platform=platform.platform(),
            system=platform.system(),
            system_version=platform.version(),
            cpu_count=os.cpu_count(),
            cpu_frequency=round(psutil.cpu_freq().current),
            system_memory=round((psutil.virtual_memory().total) / (1024*1024*1024))
        )

    def set_properties_from_json(self, json_string):
        """ Set properties/tags for the session from a json_string.
        
        Args:
            json_string (str): a string parsable as json, contains a dict.
        """
        try:
            json_dict = json.loads(json_string)
        except:
            raise ValueError(f"During parsing of JSON properties '{json_string}', an exception occured: {traceback.format_exc()}")

        if not isinstance(json_dict, dict):
            raise ValueError(f"Provided JSON properties should be a dict, instead it was {str(type(json_dict))}: {json_string}")
        
        properties_dict = dict(
            [
                (k, str(v)) # transform whatever as a string
                for k,v in json_dict.items()
            ]
        )
        self.set_properties(**properties_dict)

    def log_parameters(self, **kwargs):
        """ Logs parameters to MLFlow.
        
        Args:
            kwargs (dict): any keyword arguments will be passed as parameters to MLFlow
        """
        self._logger.debug(f"mlflow[session={self._session_name}].log_params({kwargs})")
        # NOTE: to avoid mlflow exception when value length is too long (ex: label_gain)
        for key,value in kwargs.items():
            if isinstance(value, str) and len(value) > 255:
                self._logger.warning(f"parameter {key} (str) could not be logged, value length {len(value)} > 255")
            else:
                mlflow.log_param(key,value)

    def log_time_block(self, metric_name, step=None):
        """ [Proxy] Use in a `with` statement to measure execution time of a code block.
        Uses LogTimeBlock.
        
        Example
        -------
        ```python
        with LogTimeBlock("my_perf_metric_name"):
            print("(((sleeping for 1 second)))")
            time.sleep(1)
        ```
        """
        # see class below with proper __enter__ and __exit__
        return LogTimeBlock(metric_name, step=step, metrics_logger=self)

    def log_inferencing_latencies(self, time_per_batch, batch_length=1, factor_to_usecs=1000000.0):
        """Logs prediction latencies (for inferencing) with lots of fancy metrics and plots.

        Args:
            time_per_batch_list (List[float]): time per inferencing batch
            batch_lengths (Union[List[int],int]): length of each batch (List or constant)
            factor_to_usecs (float): factor to apply to time_per_batch to convert to microseconds
        """
        if isinstance(batch_length, list):
            sum_batch_lengths = sum(batch_length)
        else:
            sum_batch_lengths = batch_length*len(time_per_batch)

        # log metadata
        self.log_metric("prediction_batches", len(time_per_batch))
        self.log_metric("prediction_queries", sum_batch_lengths)

        if len(time_per_batch) > 0:
            self.log_metric("prediction_latency_avg", (sum(time_per_batch) * factor_to_usecs)/sum_batch_lengths) # usecs

        # if there's more than 1 batch, compute percentiles
        if len(time_per_batch) > 1:
            import numpy as np
            import matplotlib.pyplot as plt

            # latency per batch
            batch_run_times = np.array(time_per_batch) * factor_to_usecs
            self.log_metric("batch_latency_p50_usecs", np.percentile(batch_run_times, 50))
            self.log_metric("batch_latency_p75_usecs", np.percentile(batch_run_times, 75))
            self.log_metric("batch_latency_p90_usecs", np.percentile(batch_run_times, 90))
            self.log_metric("batch_latency_p95_usecs", np.percentile(batch_run_times, 95))
            self.log_metric("batch_latency_p99_usecs", np.percentile(batch_run_times, 99))

            # show the distribution prediction latencies
            fig, ax = plt.subplots(1)
            ax.hist(batch_run_times, bins=100)
            ax.set_title("Latency-per-batch histogram (log scale)")
            plt.xlabel("usecs")
            plt.ylabel("occurence")
            plt.yscale('log')

            # record in mlflow
            self.log_figure(fig, "batch_latency_log_histogram.png")

            # latency per query
            if isinstance(batch_length, list):
                prediction_latencies = np.array(time_per_batch) * factor_to_usecs / np.array(batch_length)
            else:
                prediction_latencies = np.array(time_per_batch) * factor_to_usecs / batch_length

            self.log_metric("prediction_latency_p50_usecs", np.percentile(prediction_latencies, 50))
            self.log_metric("prediction_latency_p75_usecs", np.percentile(prediction_latencies, 75))
            self.log_metric("prediction_latency_p90_usecs", np.percentile(prediction_latencies, 90))
            self.log_metric("prediction_latency_p95_usecs", np.percentile(prediction_latencies, 95))
            self.log_metric("prediction_latency_p99_usecs", np.percentile(prediction_latencies, 99))

            # show the distribution prediction latencies
            fig, ax = plt.subplots(1)
            ax.hist(prediction_latencies, bins=100)
            ax.set_title("Latency-per-prediction histogram (log scale)")
            plt.xlabel("usecs")
            plt.ylabel("occurence")
            plt.yscale('log')

            # record in mlflow
            self.log_figure(fig, "prediction_latency_log_histogram.png")


########################
### CODE BLOCK TIMER ###
########################

class LogTimeBlock(object):
    """ This class should be used to time a code block.
    The time diff is computed from __enter__ to __exit__.

    Example
    -------
    ```python
    with LogTimeBlock("my_perf_metric_name"):
        print("(((sleeping for 1 second)))")
        time.sleep(1)
    ```
    """

    def __init__(self, name, **kwargs):
        """
        Constructs the LogTimeBlock.

        Args:
        name (str): key for the time difference (for storing as metric)
        kwargs (dict): any keyword will be added  as properties to metrics for logging (work in progress)
        """
        # kwargs
        self.tags = kwargs.get('tags', None)
        self.step = kwargs.get('step', None)
        self.metrics_logger = kwargs.get('metrics_logger', None)

        # internal variables
        self.name = name
        self.start_time = None
        self._logger = logging.getLogger(__name__)

    def __enter__(self):
        """ Starts the timer, gets triggered at beginning of code block """
        self.start_time = time.time() # starts "timer"

    def __exit__(self, exc_type, value, traceback):
        """ Stops the timer and stores accordingly
        gets triggered at beginning of code block.
        
        Note:
            arguments are by design for with statements.
        """
        run_time = time.time() - self.start_time # stops "timer"

        self._logger.info(f"--- time elapsed: {self.name} = {run_time:2f} s" + (f" [tags: {self.tags}]" if self.tags else ""))
        if self.metrics_logger:
            self.metrics_logger.log_metric(self.name, run_time, step=self.step)
        else:
            MetricsLogger().log_metric(self.name, run_time, step=self.step)
