# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
These classes provide some tools to automate wall time compute and logging
"""
import os
import time
from functools import wraps
import mlflow


class MetricsLogger():
    """
    Class for handling metrics logging in a singleton

    Example:
    --------
    >> from common.metrics import MetricsLogger
    >>
    >> metrics_logger = MetricsLogger()
    >> metrics_logger.log_metrics("rmse", 0.456)
    """
    _initialized = False
    _instance = None
    _session_name = None

    @classmethod
    def _initialize_azureml_mlflow_client(cls):
        try:
            # if any of that fails, fall back to normal
            from azureml.core.run import Run

            azureml_run = Run.get_context()
            if "_OfflineRun" not in str(azureml_run):
                # if we're running this script REMOTELY, get aml and compute args from run context
                ws = azureml_run.experiment.workspace
                mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
                mlflow.start_run()
                cls._initialized = True
            else:
                # if we're running this script LOCALLY, add your own +aml=X +compute=X arguments
                return
        except:
            print(f"Failed at AzureML initialization for some reason.")

    def __new__(cls, session_name=None):
        """ Create a new instance of the Singleton if necessary """
        if cls._instance is None:
            # if this is the first time we're initializing
            cls._instance = super(MetricsLogger, cls).__new__(cls)
            if not cls._session_name:
                # if no previously recorded session name
                cls._session_name = session_name
            elif session_name:
                # if new session name specified, overwrite
                cls._session_name = session_name
            print(f"Initializing MLFLOW [session='{cls._session_name}']")
            cls._init_azureml_mlflow_client()
        else:
            # if this is not the first time
            pass

        return cls._instance

    def close(self):
        if self._initialized:
            print(f"Finalizing MLFLOW [session='{self._session_name}']")
            mlflow.end_run()

    def log_metric(self, key, value):
        print(f"mlflow[session={self._session_name}].log_metric({key},{value})")
        # NOTE: there's a limit to the name of a metric
        if len(key) > 50:
            key = key[:50]
        mlflow.log_metric(key, value)

    def set_properties(self, **kwargs):
        """ Set properties/tags for the session """
        print(f"mlflow[session={self._session_name}].set_tags({kwargs})")
        mlflow.set_tags(kwargs)

    def log_parameters(self, **kwargs):
        """ Set parameters for the session """
        print(f"mlflow[session={self._session_name}].log_params({kwargs})")
        mlflow.log_params(kwargs)

    def log_time_block(self, metric_name):
        """ [Proxy] Records time of execution for block of code """
        # see class below with proper __enter__ and __exit__
        return LogTimeBlock(metric_name)



########################
### CODE BLOCK TIMER ###
########################

class LogTimeBlock(object):
    """ This class should be used to time a code block.
    The time diff is computed from __enter__ to __exit__
    and can be:
    - printed out (see kwargs verbose)
    - logged as metric in a run (see kwargs run)
    - added to a dictionary (see kwargs profile)

    Example
    -------
    >>> with LogTimeBlock("my_perf_metric_name"):
            print("(((sleeping for 1 second)))")
            time.sleep(1)
    --- time elapsted my_perf_metric_name : 1.0 s
    { 'my_perf_metric_name': 1.0 }
    """

    def __init__(self, name, **kwargs):
        """
        Constructs the LogTimeBlock.

        Arguments
        ---------
        name: {str}
            key for the time difference (for storing as metric)

        Keyword Arguments
        -----------------
        tags: {dict}
            add properties to metrics for logging as log_row()
        """
        # kwargs
        self.tags = kwargs.get('tags', None)

        # internal variables
        self.name = name
        self.start_time = None

    def __enter__(self):
        """ Starts the timer, gets triggered at beginning of code block """
        self.start_time = time.time() # starts "timer"

    def __exit__(self, exc_type, value, traceback):
        """ Stops the timer and stores accordingly
        gets triggered at beginning of code block.
        Note: arguments are by design for with statements. """
        run_time = time.time() - self.start_time # stops "timer"

        print(f"--- time elapsed: {self.name} = {run_time:2f} s" + (f" [tags: {self.tags}]" if self.tags else ""))
        MetricsLogger().log_metric(self.name, run_time)


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

        print("--- time elapsed: {} = {:2f} s".format(log_name, run_time))
        MetricsLogger().log_metric(log_name, run_time)

        return output
    return perf_wrapper