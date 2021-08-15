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

    def __new__(cls):
        """ Create a new instance of the Singleton if necessary """
        if cls._instance is None:
            # if this is the first time we're initializing
            cls._instance = super(MetricsLogger, cls).__new__(cls)
            print("Initializing MLFLOW")
            mlflow.start_run()
        else:
            # if this is not the first time
            pass

        return cls._instance

    def log_metric(self, key, value):
        print(f"mlflow.log_metric({key},{value})")
        # NOTE: there's a limit to the name of a metric
        if len(key) > 45:
            key = key[:45]
        mlflow.log_metric(key, value)


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
    >>> job_profile = {}
    >>> with LogTimeBlock("my_perf_metric_name", methods=['print']):
            print("(((sleeping for 1 second)))")
            time.sleep(1)
    --- time elapsted my_perf_metric_name : 1.0 s
    >>> job_profile
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
        print: {bool}
            prints out time with print()
        tags: {dict}
            add properties to metrics for logging as log_row()
        """
        # kwargs
        self.methods = kwargs.get('methods', ['print'])
        self.tags = kwargs.get('tags', None)

        # internal variables
        self.name = name
        self.start_time = None
        self.metrics_logger = MetricsLogger()

    def __enter__(self):
        """ Starts the timer, gets triggered at beginning of code block """
        self.start_time = time.time() # starts "timer"

    def __exit__(self, exc_type, value, traceback):
        """ Stops the timer and stores accordingly
        gets triggered at beginning of code block.
        Note: arguments are by design for with statements. """
        run_time = time.time() - self.start_time # stops "timer"

        for method in self.methods:
            if method == "print":
                # just prints nicely
                print(f"--- time elapsed: {self.name} = {run_time:2f} s" + (f" [tags: {self.tags}]" if self.tags else ""))
                self.metrics_logger.log_metric(self.name, run_time)
            else:
                # Place holder for mlflow
                raise NotImplementedError("Nothing else exists at this point")



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
        self.metrics_logger.log_metric(log_name, run_time)

        return output
    return perf_wrapper