# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python inferencing script
"""
import os
import sys
import argparse
import logging
import time
import numpy as np
from distutils.util import strtobool
import lightgbm

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if COMMON_ROOT not in sys.path:
    logging.info(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript
from common.io import input_file_path


class LightGBMPythonInferecingScript(RunnableScript):
    def __init__(self):
        super().__init__(
            task = "score",
            framework = "lightgbm",
            framework_version = "PYTHON_API."+str(lightgbm.__version__)
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
        group_i.add_argument("--data",
            required=True, type=input_file_path, help="Inferencing data location (file path)")
        group_i.add_argument("--model",
            required=False, type=input_file_path, help="Exported model location (file path)")
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

        # register logger for lightgbm logs
        lightgbm.register_logger(logger)

        # make sure the output argument exists
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            args.output = os.path.join(args.output, "predictions.txt")

        logger.info(f"Loading model from {args.model}")
        booster = lightgbm.Booster(model_file=args.model)

        logger.info(f"Loading data for inferencing")
        with metrics_logger.log_time_block("time_data_loading"):
            # NOTE: this is bad, but allows for libsvm format (not just numpy)
            inference_data = lightgbm.Dataset(args.data, free_raw_data=False).construct()
            inference_raw_data = inference_data.get_data()

        # capture data shape as property
        metrics_logger.set_properties(
            inference_data_length = inference_data.num_data(),
            inference_data_width = inference_data.num_feature()
        )

        logger.info(f"Running .predict()")
        batch_start_time = time.monotonic()
        predictions_array = booster.predict(
            data=inference_raw_data,
            num_threads=args.num_threads,
            predict_disable_shape_check=bool(args.predict_disable_shape_check)
        )
        prediction_time = (time.monotonic() - batch_start_time)
        metrics_logger.log_metric("time_inferencing", prediction_time)

        # use helper to log latency with the right metric names
        metrics_logger.log_inferencing_latencies(
            [prediction_time], # only one big batch
            batch_length=inference_data.num_data(),
            factor_to_usecs=1000000.0 # values are in seconds
        )

        if args.output:
            np.savetxt(
                args.output,
                predictions_array,
                fmt='%f',
                delimiter=',',
                newline='\n',
                header='',
                footer='',
                comments='# ',
                encoding=None
            )


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return LightGBMPythonInferecingScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    LightGBMPythonInferecingScript.main(cli_args)

if __name__ == "__main__":
    main()
