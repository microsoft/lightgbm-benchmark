# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script contains a class to structure and standardize all scripts
in the lightgbm-benchmark repository. This class factors duplicate code to
achieve usual routines of every script: logging init, MLFlow init,
system properties logging, etc.
"""
import os
import sys
import argparse
import logging
import traceback
from distutils.util import strtobool

from .metrics import MetricsLogger
from .perf import PerformanceMetricsCollector

class RunnableScript():
    """
    This class factors duplicate code to achieve usual routines
    of every script in the lightgbm-benchmark repo: logging init, MLFlow init,
    system properties logging, etc.
    """
    def __init__(self, task, framework, framework_version, metrics_prefix=None):
        """ Generic initialization for all script classes.

        Args:
            task (str): name of task in the pipeline/benchmark (ex: train, score)
            framework (str): name of ML framework
            framework_version (str): a version of this framework
            metrics_prefix (str): any prefix to add to this scripts metrics
        """
        self.task = task
        self.framework = framework
        self.framework_version = framework_version
        self.metrics_prefix = metrics_prefix

        self.logger = logging.getLogger(f"{framework}.{task}")

        # initializes reporting of metrics
        self.metrics_logger = MetricsLogger(
            f"{framework}.{task}",
            metrics_prefix=metrics_prefix
        )

        self.perf_report_collector = None

    @classmethod
    def get_arg_parser(cls, parser=None):
        """Adds component/module arguments to a given argument parser.

        Args:
            parser (argparse.ArgumentParser): an argument parser instance

        Returns:
            ArgumentParser: the argument parser instance

        Notes:
            if parser is None, creates a new parser instance
        """
        # add arguments that are specific to the module
        if parser is None:
            parser = argparse.ArgumentParser()

        # add generic arguments here
        group_general = parser.add_argument_group("General parameters")
        group_general.add_argument(
            "--verbose",
            required=False,
            default=False,
            type=strtobool,  # use this for bool args, do not use action_store=True
            help="set True to show DEBUG logs",
        )
        group_general.add_argument(
            "--custom_properties",
            required=False,
            default=None,
            type=str,
            help="provide custom properties as json dict",
        )
        group_general.add_argument(
            "--enable_perf_metrics",
            required=False,
            default=True,
            type=strtobool,
            help="disable/enable performance metrics (default: enabled)",
        )

        return parser

    def initialize_run(self, args):
        """Initialize the component run, opens/setups what needs to be"""
        self.logger.info("Initializing script run...")

        # record properties of the run
        self.metrics_logger.set_properties(
            task = self.task,
            framework = self.framework,
            framework_version = self.framework_version
        )

        # if provided some custom_properties by the outside orchestrator
        if args.custom_properties:
            self.metrics_logger.set_properties_from_json(args.custom_properties)

        # add properties about environment of this script
        self.metrics_logger.set_platform_properties()

        # enable perf reporting
        if args.enable_perf_metrics:
            self.perf_report_collector = PerformanceMetricsCollector()


    def run(self, args, logger, metrics_logger, unknown_args):
        """The run function of your script. You are required to override this method
        with your own implementation.

        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.logger): a logger initialized for this script
            metrics_logger (common.metrics.MetricLogger): to report metrics for this script, already initialized for MLFlow
            unknown_args (list[str]): list of arguments not recognized during argparse
        """
        raise NotImplementedError(f"run() method from class {self.__class__.__name__} hasn't actually been implemented.")

    def finalize_run(self, args):
        """Finalize the run, close what needs to be"""
        self.logger.info("Finalizing script run...")

        if self.perf_report_collector:
            # plot?
            pass

        # close mlflow
        self.metrics_logger.close()


    @classmethod
    def main(cls, cli_args=None):
        """ Component main function, it is not recommended to override this method.
        It parses arguments and executes run() with the right arguments.

        Args:
            cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
        """
        # initialize root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # construct arg parser
        parser = cls.get_arg_parser()
    
        # if argument parsing fails, or if unknown arguments, will except
        args, unknown_args = parser.parse_known_args(cli_args)
        logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

        # create script instance, initialize mlflow
        script_instance = cls()
        script_instance.initialize_run(args)

        # run the actual thing
        script_instance.run(args, script_instance.logger, script_instance.metrics_logger, unknown_args)

        # close mlflow
        script_instance.finalize_run(args)

class SingleNodeScript(RunnableScript):
    pass
