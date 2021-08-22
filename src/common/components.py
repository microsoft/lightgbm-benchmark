# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from .metrics import MetricsLogger

class RunnableScript():
    def __init__(self, task, framework, framework_version):
        # initializes reporting of metrics
        self.metrics_logger = MetricsLogger(f"{framework}.{task}")

        self.metrics_logger.set_properties(
            task = task,
            framework = framework,
            framework_version = framework_version
        )

    def get_arg_parser(self, parser=None):
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
            parser = argparse.ArgumentParser(f"{framework}.{task}")

        # add generic arguments here

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
        script_instance = cls()

        # construct arg parser
        parser = script_instance.get_arg_parser()
    
        # if argument parsing fails, or if unknown arguments, will except
        args, unknown_args = parser.parse_known_args(cli_args)

        # run the actual thing
        script_instance.run(args, unknown_args)

        # close mlflow
        script_instance.metrics_logger.close()

class InferencingScript(RunnableScript):
    def __init__(self, framework, framework_version):
        super().__init__(
            task = "score",
            framework = framework,
            framework_version = framework_version
        )
        