# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/CLI inferencing script
"""
import os
import sys
import argparse
import logging
from distutils.util import strtobool
import lightgbm
from lightgbm import Booster, Dataset
from subprocess import PIPE
from subprocess import run as subprocess_run
from subprocess import TimeoutExpired

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript
from common.io import input_file_path


class LightGBMCLIInferencingScript(RunnableScript):
    def __init__(self):
        super().__init__(
            task="score",
            framework="lightgbm",
            framework_version=lightgbm.__version__
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
        # add generic arguments
        parser = RunnableScript.get_arg_parser(parser)

        group_i = parser.add_argument_group("Input Data")
        group_i.add_argument("--lightgbm_exec_path",
            required=False, type=str, default="lightgbm", help="Path to lightgbm.exe (file path)")
        group_i.add_argument("--data",
            required=True, type=input_file_path, help="Inferencing data location (file path)")
        group_i.add_argument("--model",
            required=False, type=input_file_path, help="Exported model location")
        group_i.add_argument("--output",
            required=False, default=None, type=str, help="Inferencing output location (file path)")

        group_params = parser.add_argument_group("Scoring parameters")
        group_params.add_argument("--num_threads",
            required=False, default=1, type=int, help="number of threads")
        group_params.add_argument("--predict_disable_shape_check",
            required=False, default=False, type=strtobool, help="See LightGBM documentation")

        return parser


    def run(self, args, logger, metrics_logger, unknown_args):
        """Run script with arguments (the core of the component)

        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.getLogger() for this script)
            metrics_logger (common.metrics.MetricLogger)
            unknown_args (list[str]): list of arguments not recognized during argparse
        """
        # record relevant parameters
        metrics_logger.log_parameters(
            num_threads=args.num_threads
        )

        if args.output:
            # make sure the output argument exists
            os.makedirs(args.output, exist_ok=True)
            
            # and create your own file inside the output
            args.output = os.path.join(args.output, "predictions.txt")

        # assemble a command for lightgbm cli
        lightgbm_cli_command = [
            args.lightgbm_exec,
            "task=prediction",
            f"data={args.data}",
            f"input_model={args.model}",
            "verbosity=2",
            f"num_threads={args.num_threads}",
            f"predict_disable_shape_check={bool(args.predict_disable_shape_check)}"
        ]

        if args.output:
            lightgbm_cli_command.append(f"output_result={args.output}")


        logger.info(f"Running .predict()")
        with metrics_logger.log_time_block(metric_name="time_inferencing"):
            lightgbm_cli_call = subprocess_run(
                lightgbm_cli_command,
                stdout=PIPE,
                stderr=PIPE,
                universal_newlines=True,
                check=False, # will not raise an exception if subprocess fails (so we capture with .returncode)
                timeout=None
            )
            logger.info(f"LightGBM stdout: {lightgbm_cli_call.stdout}")
            logger.info(f"LightGBM stderr: {lightgbm_cli_call.stderr}")
            logger.info(f"LightGBM return code: {lightgbm_cli_call.returncode}")


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return LightGBMCLIInferencingScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    LightGBMCLIInferencingScript.main(cli_args)

if __name__ == "__main__":
    main()
