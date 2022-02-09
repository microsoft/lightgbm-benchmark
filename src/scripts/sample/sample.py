# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script is a tutorial sample script to explain how all the benchmark
scripts are structured and standardized using the `RunnableScript` helper class.

We've numbered the steps you need to modify and adapt this sample script
to your own needs.

Follow each STEP below, and their associated TODO.
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
    """
    STEP 1 : Package your script as a class.

    This class inherits from RunnableScript, that factors
    duplicate code to achieve usual routines
    of every script in the lightgbm-benchmark repo.
    
    It has a standard main() function (see below)
    that you should not need to modify except for edge cases.

    See `src/common/components.py` for details on that class.

    TODO: name your class specifically for this script
    """
    def __init__(self):
        """
        STEP 3 : Define your benchmark "task" in the constructor.
        
        This Specific constructor for this SampleScript class. It has no arguments,
        as it will be called from the helper `main()` method.

        In your custom script class, you need to call the super constructor with the parameters below.

        TODO: pick your task (score, train, generate, compile, ...)
        TODO: name your framework and version
        """
        super().__init__(
            task="sample_task", # str
            framework="sample_framework", # str
            framework_version="0.0.1" # str
        )

    @classmethod
    def get_arg_parser(cls, parser=None):
        """
        STEP 4 : Define your arguments
        
        This method will be called by the main() function
        to add your script custom arguments to argparse,
        on top of standard arguments of the benchmark.

        TODO: align this section with your requirements.
        
        Args:
            parser (argparse.ArgumentParser): an existing argument parser instance

        Returns:
            ArgumentParser: the argument parser instance
        """
        # IMPORTANT: call this to add generic benchmark arguments
        parser = RunnableScript.get_arg_parser(parser)

        # add arguments that are specific to your script
        # here's a couple examples
        group_i = parser.add_argument_group(f"I/O Arguments [{__name__}:{cls.__name__}]")
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

        # make sure to return parser
        return parser


    def run(self, args, logger, metrics_logger, unknown_args):
        """
        STEP 5 : Define your run function.
        
        This is the core function of your script.
        You are required to override this method with your own implementation.

        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.logger): a logger initialized for this script
            metrics_logger (common.metrics.MetricLogger): to report metrics for this script, already initialized for MLFlow
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
    """
    STEP 2: main function block

    The section below (get_arg_parser(), main() and main block) should go unchanged,
    except for the name of your class.

    Those are standard functions we enforce in order to get some unit tests
    on the module (arguments parsing mainly).

    To ensure compatibility with shrike unit tests
    
    TODO: just replace SampleScript to the name of your class
    """
    return SampleScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    SampleScript.main(cli_args)

if __name__ == "__main__":
    # The main function is defined in src/common/components.py
    # and is standard to all scripts.
    main()
