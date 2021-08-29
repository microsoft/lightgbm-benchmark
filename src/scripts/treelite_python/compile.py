# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
TreeLite/Python inferencing script
"""
import os
import sys
import argparse
import logging
import numpy
from distutils.util import strtobool
import pandas as pd
import treelite, treelite_runtime

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
    
    group_treelite = parser.add_argument_group("Treelite parameters")
    group_treelite.add_argument("--model_format",
        required=False, default="lightgbm", type=str, help="format of the input --model")
    group_treelite.add_argument("--toolchain",
        required=False, default="gcc", type=str, help="toolchain for compiling model")

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
    metrics_logger = MetricsLogger("treelite.score")

    # add some properties to the session
    metrics_logger.set_properties(
        framework = 'treelite_python',
        task = 'score',
        lightgbm_version = treelite.__version__
    )

    # if provided some custom_properties by the outside orchestrator
    if args.custom_properties:
        metrics_logger.set_properties_from_json(args.custom_properties)

    # add properties about environment of this script
    metrics_logger.set_platform_properties()

    if args.output:
        # make sure the output argument exists
        os.makedirs(args.output, exist_ok=True)
        
        # and create your own file inside the output
        args.output = os.path.join(args.output, "predictions.txt")



    logger.info(f"Converting model to Treelite")
    with metrics_logger.log_time_block("treelite_model_conversion"):
        model = treelite.Model.load(
            args.model,
            model_format=args.model_format
        )
        model.export_lib(
            toolchain=args.toolchain,
            libpath='./mymodel.so',
            verbose=True,
            params={'parallel_comp':16}
        )

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
