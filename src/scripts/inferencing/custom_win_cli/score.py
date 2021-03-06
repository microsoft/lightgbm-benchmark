# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Custom Binaries inferencing script
"""
import os
import sys
import argparse
import logging
from distutils.util import strtobool
from subprocess import PIPE
from subprocess import run as subprocess_run
from subprocess import TimeoutExpired

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript
from common.io import input_file_path

# STEP 1 : provide the name of your binary executable
# (copy it inside the static_binaries subfolder)
BINARY_FILE_NAME = "lightgbm.exe" # <<< rename to fit your binary
BINARIES_FOLDER = os.path.join(os.path.dirname(__file__), "static_binaries")
BINARY_FILE_PATH = os.path.join(os.path.dirname(__file__), "static_binaries", BINARY_FILE_NAME)

class CustomCLIInferencingScript(RunnableScript):
    def __init__(self):
        # STEP 2 : feel free to update those to reflect your custom binary framework/version
        super().__init__(
            task="score",
            framework="custom_bin",
            framework_version=BINARY_FILE_NAME
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

        # STEP 3 : below are the arguments that will be passed
        # by the inferencing benchmark pipeline, if you want to add more arguments
        # you will have to modify the pipeline itself
        # alternatively, you can hardcode values in the custom_cli_command list below (see STEP 4)
        group_i = parser.add_argument_group(f"Input Data [{__name__}:{cls.__name__}]")
        group_i.add_argument("--data",
            required=True, type=input_file_path, help="Inferencing data location (file path)")
        group_i.add_argument("--model",
            required=True, type=input_file_path, help="Exported model location")
        group_i.add_argument("--output",
            required=False, default=None, type=str, help="Inferencing output location (file path)")

        group_params = parser.add_argument_group(f"Scoring parameters [{__name__}:{cls.__name__}]")
        group_params.add_argument("--num_threads",
            required=False, default=1, type=int, help="number of threads")

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

        # STEP 4: write the command for your custom cli as a list
        # the example below corresponds to commands for lightgbm_cli.exe
        # see https://lightgbm.readthedocs.io/en/latest/Parameters.html
        custom_cli_command = [
            BINARY_FILE_PATH,
            "task=prediction",
            f"model={args.model}",
            f"data={args.data}",
            "verbosity=2",
            f"num_threads={args.num_threads}",
            #f"predict_disable_shape_check=True"
        ]

        # STEP 5 : if you need to add an output
        # the example below corresponds to commands for lightgbm_cli.exe
        if args.output:
            custom_cli_command.append(f"output_result ={args.output}")

        logger.info(f"Running custom command: {custom_cli_command}")
        with metrics_logger.log_time_block(metric_name="time_inferencing"):
            custom_cli_call = subprocess_run(
                custom_cli_command,
                stdout=PIPE,
                stderr=PIPE,
                universal_newlines=True,
                check=False, # will not raise an exception if subprocess fails (so we capture with .returncode)
                timeout=None
            )
        logger.info(f"RETURN CODE: {custom_cli_call.returncode}")
        logger.info(f"STDOUT: {custom_cli_call.stdout}")
        logger.info(f"STDERR: {custom_cli_call.stderr}")

        # OPTIONAL: apply any post processing on logs here (ex: extract metrics)


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return CustomCLIInferencingScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    CustomCLIInferencingScript.main(cli_args)

if __name__ == "__main__":
    main()
