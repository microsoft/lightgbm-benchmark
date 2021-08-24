# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python inferencing script
"""
import os
import sys
import argparse
import lightgbm
import numpy
from distutils.util import strtobool

# let's add the right PYTHONPATH for common module
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# before doing local import
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
    # add arguments that are specific to the module
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
    
    return parser


def run(args, other_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    # create sub dir and output file
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        args.output = os.path.join(args.output, "predictions.txt")

    # initializes reporting of metrics
    metrics_logger = MetricsLogger("lightgbm_python.score")

    # add some properties to the session
    metrics_logger.set_properties(
        framework = 'lightgbm_python',
        task = 'score',
        lightgbm_version = lightgbm.__version__
    )

    print(f"Loading model from {args.model}")
    booster = lightgbm.Booster(model_file=args.model)

    print(f"Loading data for inferencing")
    with metrics_logger.log_time_block("data_loading"):
        # NOTE: this is bad, but allows for libsvm format (not just numpy)
        inference_data = lightgbm.Dataset(args.data, free_raw_data=False).construct()
        inference_raw_data = inference_data.get_data()

    # capture data shape as property
    metrics_logger.set_properties(
        inference_data_length = inference_data.num_data(),
        inference_data_width = inference_data.num_feature()
    )

    # capture data shape as property
    metrics_logger.set_properties(
        inference_data_length = raw_data.shape[0],
        inference_data_width = raw_data.shape[1]
    )

    print(f"Running .predict()")
    with metrics_logger.log_time_block("inferencing"):
        booster.predict(data=inference_raw_data, predict_disable_shape_check=bool(args.predict_disable_shape_check))

    # optional: close logging session
    metrics_logger.close()

    # optional: close logging session
    metrics_logger.close()


def main(cli_args=None):
    """ Component main function, parses arguments and executes run() function.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # construct arg parser
    parser = get_arg_parser()
    args, unknown_args = parser.parse_known_args(cli_args)

    # run the actual thing
    run(args, unknown_args)


if __name__ == "__main__":
    main()    
