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

from .metrics import MetricsLogger, MLFlowMetricsLogger, AzureMLRunMetricsLogger
from .perf import PerformanceMetricsCollector, PerfReportPlotter

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

        # default metrics logger is just stdout print
        self.metrics_logger = MetricsLogger(
            f"{framework}.{task}",
            metrics_prefix=self.metrics_prefix
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
        group_general = parser.add_argument_group(f"General parameters [{__name__}:{cls.__name__}]")
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
            "--disable_perf_metrics",
            required=False,
            default=False,
            type=strtobool,
            help="disable performance metrics (default: False)",
        )
        group_general.add_argument(
            "--metrics_driver",
            required=False,
            default="mlflow",
            choices=['mlflow', 'azureml'],
            type=str,
            help="which class to use to report metrics mlflow or azureml",
        )

        return parser

    def initialize_run(self, args):
        """Initialize the component run, opens/setups what needs to be"""
        self.logger.info("Initializing script run...")

        # initializes reporting of metrics
        if args.metrics_driver == 'mlflow':
            self.metrics_logger = MLFlowMetricsLogger(
                f"{self.framework}.{self.task}",
                metrics_prefix=self.metrics_prefix
            )
        elif args.metrics_driver == 'azureml':
            self.metrics_logger = AzureMLRunMetricsLogger(
                f"{self.framework}.{self.task}",
                metrics_prefix=self.metrics_prefix
            )
        else:
            # use default metrics_logger (stdout print)
            pass

        # open mlflow
        self.metrics_logger.open()

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
        if not args.disable_perf_metrics:
            self.perf_report_collector = PerformanceMetricsCollector()
            self.perf_report_collector.start()


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
            self.perf_report_collector.finalize()
            plotter = PerfReportPlotter(self.metrics_logger)
            plotter.add_perf_reports(self.perf_report_collector.perf_reports, node=0)
            plotter.report_nodes_perf()

            # write perf record as artifact
            self.metrics_logger.log_artifact(plotter.save_to())

        # close mlflow
        self.metrics_logger.close()


    ####################
    ### MAIN METHODS ###
    ####################

    @classmethod
    def initialize_root_logger(cls):
        # initialize root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    @classmethod
    def parse_class_arguments(cls, cli_args=None):
        logger = logging.getLogger()

        # show the command used to run
        if cli_args:
            logger.info(f"Running main() with specific cli args: {cli_args}")
        else:
            logger.info(f"Running main() with sys.argv={sys.argv}")

        # construct arg parser
        parser = cls.get_arg_parser()
    
        # if argument parsing fails, or if unknown arguments, will except
        args, unknown_args = parser.parse_known_args(cli_args)
        logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

        return args, unknown_args

    def _main_run_hook(self, args, unknown_args):
        """Run function called from main()"""
        self.run(args, self.logger, self.metrics_logger, unknown_args=unknown_args)

    @classmethod
    def main(cls, cli_args=None):
        """ Component main function, it is not recommended to override this method.
        It parses arguments and executes run() with the right arguments.

        Args:
            cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
        """
        cls.initialize_root_logger()
        args, unknown_args = cls.parse_class_arguments(cli_args)

        # create script instance, initialize mlflow
        script_instance = cls()
        script_instance.initialize_run(args)

        # catch run function exceptions to properly finalize run (kill/join threads)
        try:
            # run the class run method (passthrough)
            script_instance._main_run_hook(args, unknown_args)
        except BaseException as e:
            logging.critical(f"Exception occured during run():\n{traceback.format_exc()}")
            script_instance.finalize_run(args)
            raise e

        # close mlflow
        script_instance.finalize_run(args)

        # return for unit tests
        return script_instance

class SingleNodeScript(RunnableScript):
    pass
