# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python inferencing script
"""
import os
import sys
import argparse
import logging
from distutils.util import strtobool
import lightgbm
import numpy

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.metrics import MetricsLogger
from common.io import input_file_path


def get_arg_parser(parser=None):
    """Adds component/module arguments to a given argument parser.

    Args:
        parser (argparse.ArgumentParser): an argument parser instance

    Returns:
        ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the script
    if parser is None:
        parser = argparse.ArgumentParser(__doc__)

    group_i = parser.add_argument_group("Input Data")
    group_i.add_argument("--data",
        required=True, type=input_file_path, help="Inferencing data location (file path)")
    group_i.add_argument("--model",
        required=False, type=input_file_path, help="Exported model location (file path)")
    group_i.add_argument("--output",
        required=False, default=None, type=str, help="Inferencing output location (file path)")
    
    group_params = parser.add_argument_group("Scoring parameters")
    group_params.add_argument("--predict_disable_shape_check",
        required=False, default=False, type=strtobool, help="See LightGBM documentation")

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


def run(args, unknown_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    # get logger for general outputs
    logger = logging.getLogger()

    # get Metrics logger for benchmark metrics
    # below: initialize reporting of metrics with a custom session name
    metrics_logger = MetricsLogger("lightgbm_python.score")

    # add common properties to the session
    metrics_logger.set_properties(
        task="score", framework="lightgbm_python", framework_version=lightgbm.__version__
    )

    # if provided some custom_properties by the outside orchestrator
    if args.custom_properties:
        metrics_logger.set_properties_from_json(args.custom_properties)

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
    with metrics_logger.log_time_block("time_inferencing"):
        booster.predict(data=inference_raw_data, predict_disable_shape_check=bool(args.predict_disable_shape_check))

    # Important: close logging session before exiting
    metrics_logger.close()


def main(cli_args=None):
    """Component main function, parses arguments and executes run() function.

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

    # construct arg parser and parse arguments
    parser = get_arg_parser()
    args, unknown_args = parser.parse_known_args(cli_args)

    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # run the actual thing
    run(args, unknown_args)


if __name__ == "__main__":
    main()
