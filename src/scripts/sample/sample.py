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

    # recommended: use groups
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
        default=str,
        type=strtobool,  # use this for bool args, do not use action_store=True
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
    metrics_logger = MetricsLogger("sample_framework.sample_task")

    # add common properties to the session
    metrics_logger.set_properties(
        task="sample_task", framework="sample_framework", framework_version="0.0.1"
    )

    # if provided some custom_properties by the outside orchestrator
    metrics_logger.set_properties_from_json(args.custom_properties)

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
    with metrics_logger.log_time_block("data_loading"):
        inference_data = lightgbm.Dataset(args.data, free_raw_data=False).construct()
        inference_raw_data = inference_data.get_data()

    # optional: add data shape as property
    metrics_logger.set_properties(
        inference_data_length=inference_data.num_data(),
        inference_data_width=inference_data.num_feature(),
    )

    # to log executing time of a code block, use log_time_block()
    logger.info(f"Running .predict()")
    with metrics_logger.log_time_block("inferencing"):
        booster.predict(data=inference_raw_data)

    # CUSTOM CODE ENDS HERE

    # Important: close logging session before exiting
    metrics_logger.close()


def main(cli_args=None):
    """Component main function, parses arguments and executes run() function.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # construct arg parser and parse arguments
    parser = get_arg_parser()
    args, unknown_args = parser.parse_known_args(cli_args)

    # run the actual thing
    run(args, unknown_args)


if __name__ == "__main__":
    main()
