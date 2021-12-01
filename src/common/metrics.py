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

class MetricsLogger():
    """
    Class for handling metrics logging in MLFlow. This class is a singleton.
    """
    _initialized = False
    _instance = None
    _session_name = None
    _metrics_prefix = None
    _logger = logging.getLogger(__name__)

    def __new__(cls, session_name=None, metrics_prefix=None):
        """ Create a new instance of the Singleton """
        if not cls._initialized:
            # if this is the first time we're initializing
            cls._instance = super(MetricsLogger, cls).__new__(cls)
            cls._metrics_prefix = metrics_prefix
            if not cls._session_name:
                # if no previously recorded session name
                cls._session_name = session_name
            elif session_name:
                # if new session name specified, overwrite
                cls._session_name = session_name
            cls._logger.info(f"Initializing MLFLOW [session='{cls._session_name}', metrics_prefix={cls._metrics_prefix}]")
            mlflow.start_run()
            cls._initialized = True
        else:
            # if this is not the first time, and things are already initialized
            if not cls._metrics_prefix:
                cls._logger.warning(f"New creation of MetricsLogger() with a new prefix {metrics_prefix}")
                cls._metrics_prefix = metrics_prefix
            pass

        return cls._instance

    @classmethod
    def close(cls):
        """Close the MLFlow session."""
        if cls._initialized:
            cls._logger.info(f"Finalizing MLFLOW [session='{cls._session_name}']")
            mlflow.end_run()
            cls._initialized = False
        else:
            cls._logger.warning(f"Call to finalize MLFLOW [session='{cls._session_name}'] that was never initialized.")

    def _remove_non_allowed_chars(self, name_string):
        """ Removes chars not allowed for metric keys in mlflow """
        return re.sub(r'[^a-zA-Z0-9_\-\.\ \/]', '', name_string)

    def log_metric(self, key, value, step=None):
        """Logs a metric key/value pair.
        
        Args:
            key (str): metric key
            value (str): metric value
            step (int): which step to log this metric? (see mlflow.log_metric())
        """
        if self._metrics_prefix:
            key = self._metrics_prefix + key

        key = self._remove_non_allowed_chars(key)

        self._logger.debug(f"mlflow[session={self._session_name}].log_metric({key},{value})")
        # NOTE: there's a limit to the name of a metric
        if len(key) > 50:
            key = key[:50]
        mlflow.log_metric(key, value, step=step)

    def log_figure(self, figure, artifact_file):
        """Logs a figure using mlflow
        
        Args:
            figure (Union[matplotlib.figure.Figure, plotly.graph_objects.Figure]): figure to log
            artifact_file (str): name of file to record
        """
        mlflow.log_figure(figure, artifact_file)

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
        return LogTimeBlock(metric_name, step=step)



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
        MetricsLogger().log_metric(self.name, run_time, step=self.step)


####################
### METHOD TIMER ###
####################

def log_time_function(func):
    """ decorator to log wall time of a function/method """
    @wraps(func)
    def perf_wrapper(*args, **kwargs):
        log_name = "{}.time".format(func.__qualname__)
        start_time = time.time()
        output = func(*args, **kwargs)
        run_time = time.time() - start_time

        logging.getLogger(__name__).info("--- time elapsed: {} = {:2f} s".format(log_name, run_time))
        MetricsLogger().log_metric(log_name, run_time)

        return output
    return perf_wrapper
