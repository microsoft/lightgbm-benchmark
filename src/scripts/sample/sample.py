# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Sample benchmark script (lightgbm inferencing)
"""
import os
import sys
import argparse
import logging
import lightgbm
import numpy
from distutils.util import strtobool

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.io import input_file_path
from common.components import RunnableScript


class SampleScript(RunnableScript):
    def __init__(self):
        super().__init__(
            task="sample_task",
            framework="sample_framework",
            framework_version="0.0.1"
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

        # add arguments that are specific to the script
        group_i = parser.add_argument_group("I/O Arguments")
        group_i.add_argument(
            "--data",
            required=True,
            type=input_file_path,  # use this helper type for a directory containing a single file
            help="Some input location (directory containing a unique file)",
        )
        group_i.add_argument(
            "--model",
            required=True,
            type=input_file_path,  # use this helper type for a directory containing a single file
            help="Some input location (directory containing a unique file)",
        )
        group_i.add_argument(
            "--output",
            required=True,
            default=None,
            type=str,
            help="Some output location (directory)",
        )

        return parser


    def run(self, args, logger, metrics_logger, unknown_args):
        """Run script with arguments (the core of the component)

        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.getLogger() for this script)
            metrics_logger (common.metrics.MetricLogger)
            unknown_args (list[str]): list of arguments not recognized during argparse
        """
        # make sure the output argument exists
        os.makedirs(args.output, exist_ok=True)

        # and create your own file inside the output
        args.output = os.path.join(args.output, "predictions.txt")

        # CUSTOM CODE STARTS HERE
        # below this line is user code
        logger.info(f"Loading model from {args.model}")
        booster = lightgbm.Booster(model_file=args.model)

        # to log executing time of a code block, use log_time_block()
        logger.info(f"Loading data for inferencing")
        with metrics_logger.log_time_block(metric_name="time_data_loading"):
            inference_data = lightgbm.Dataset(args.data, free_raw_data=False).construct()
            inference_raw_data = inference_data.get_data()

        # optional: add data shape as property
        metrics_logger.set_properties(
            inference_data_length=inference_data.num_data(),
            inference_data_width=inference_data.num_feature(),
        )

        # to log executing time of a code block, use log_time_block()
        logger.info(f"Running .predict()")
        with metrics_logger.log_time_block(metric_name="time_inferencing"):
            booster.predict(data=inference_raw_data)

        # CUSTOM CODE ENDS HERE


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return SampleScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    SampleScript.main(cli_args)

if __name__ == "__main__":
    main()
