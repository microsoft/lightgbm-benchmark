# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import argparse
import logging
import traceback
from distutils.util import strtobool

from .metrics import MetricsLogger

class RunnableScript():
    def __init__(self, task, framework, framework_version):
        self.task = task
        self.framework = framework
        self.framework_version = framework_version

        self.logger = logging.getLogger(f"{framework}.{task}")

        # initializes reporting of metrics
        self.metrics_logger = MetricsLogger(f"{framework}.{task}")

        self.metrics_logger.set_properties(
            task = self.task,
            framework = self.framework,
            framework_version = self.framework_version
        )

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
            parser = argparse.ArgumentParser("TODO")

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

        return parser

    def run(self, args, unknown_args=[]):
        """Run script with arguments (the core of the component)

        Args:
            args (argparse.namespace): command line arguments provided to script
            unknown_args (list[str]): list of arguments not known
        """
        raise NotImplementedError(f"--type {args.type} is not implemented.")

    @classmethod
    def main(cls, cli_args=None):
        """ Component main function, parses arguments and executes run() function.

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

        # create script instance
        script_instance = cls()

        # get Metrics logger for benchmark metrics
        # below: initialize reporting of metrics with a custom session name
        script_instance.metrics_logger = MetricsLogger(script_instance.task)

        # construct arg parser
        parser = script_instance.get_arg_parser()
    
        # if argument parsing fails, or if unknown arguments, will except
        args, unknown_args = parser.parse_known_args(cli_args)

        logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

        # if provided some custom_properties by the outside orchestrator
        if args.custom_properties:
            script_instance.metrics_logger.set_properties_from_json(args.custom_properties)

        # add properties about environment of this script
        script_instance.metrics_logger.set_platform_properties()

        # run the actual thing
        script_instance.run(args, unknown_args)

        # close mlflow
        script_instance.metrics_logger.close()
