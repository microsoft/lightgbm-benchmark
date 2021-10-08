# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python dataset saving to binary
"""
import os
import sys
import argparse
import logging
import traceback
import json
from distutils.util import strtobool
import lightgbm

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if COMMON_ROOT not in sys.path:
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.metrics import MetricsLogger
from common.io import input_file_path, get_all_files


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
    group_i.add_argument("--train",
        required=True, type=input_file_path, help="Training data location (file path or dir path with unique file)")
    group_i.add_argument("--test",
        required=True, type=str, help="Testing data location (file or dir path with multiple files)")
    group_i.add_argument("--header", required=False, default=False, type=strtobool)
    group_i.add_argument("--label_column", required=False, default="0", type=str)
    group_i.add_argument("--group_column", required=False, default=None, type=str)

    group_o = parser.add_argument_group("Outputs")
    group_o.add_argument("--output_train",
        required=False, type=str, help="export binary train data (folder)")
    group_o.add_argument("--output_test",
        required=False, type=str, help="export binary test data (folder)")
    
    # learner params
    group_lgbm = parser.add_argument_group("LightGBM Dataset parameters")
    group_lgbm.add_argument("--max_bin", required=True, type=int)
    group_lgbm.add_argument("--custom_params", required=False, type=str, default=None)

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


def load_lgbm_params_from_cli(args):
    """Gets the right LightGBM parameters from argparse + mpi config
    Args:
        args (argparse.Namespace)
    
    Returns:
        lgbm_params (dict)
    """
    # copy all parameters from argparse
    cli_params = dict(vars(args))

    # removing arguments that are purely CLI
    for key in ['verbose', 'custom_properties', 'output_train', 'output_test', 'test', 'train', 'custom_params']:
        del cli_params[key]

    # doing some fixes and hardcoded values
    lgbm_params = cli_params
    lgbm_params['verbose'] = 2
    lgbm_params['header'] = bool(args.header) # strtobool returns 0 or 1, lightgbm needs actual bool

    # process custom params
    if args.custom_params:
        custom_params = json.loads(args.custom_params)
        lgbm_params.update(custom_params)

    return lgbm_params


def run(args, unknown_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    logger = logging.getLogger()

    # figure out the lgbm params from cli args
    lgbm_params = load_lgbm_params_from_cli(args)

    # get Metrics logger for benchmark metrics
    # below: initialize reporting of metrics with a custom session name
    metrics_logger = MetricsLogger(
        "lightgbm_python.data2bin"
    )

    # make sure the output argument exists
    os.makedirs(args.output_train, exist_ok=True)
    args.output_train = os.path.join(args.output_train, "train.bin")
    os.makedirs(args.output_test, exist_ok=True)

    # add common properties to the session
    metrics_logger.set_properties(
        task="data2bin", framework="lightgbm", framework_version=lightgbm.__version__
    )

    # if provided some custom_properties by the outside orchestrator
    if args.custom_properties:
        metrics_logger.set_properties_from_json(args.custom_properties)

    # add properties about environment of this script
    metrics_logger.set_platform_properties()

    # log lgbm parameters
    logger.info(f"LGBM Params: {lgbm_params}")
    metrics_logger.log_parameters(**lgbm_params)

    # register logger for lightgbm logs
    lightgbm.register_logger(logger)

    logger.info(f"Loading data for training")
    with metrics_logger.log_time_block("time_data_loading.train"):
        # construct dataset
        train_data = lightgbm.Dataset(args.train, params=lgbm_params).construct()

        # capture data shape in metrics
        metrics_logger.log_metric(key="train_data.length", value=train_data.num_data())
        metrics_logger.log_metric(key="train_data.width", value=train_data.num_feature())

    with metrics_logger.log_time_block("time_data_saving.train"):
        # construct dataset
        train_data.save_binary(args.output_train)

    with metrics_logger.log_time_block("time_data_loading.test"):
        # construct dataset
        test_data_paths = get_all_files(args.test)
        val_datasets = [
            train_data.create_valid(test_data_path).construct() for test_data_path in test_data_paths
        ]

        # capture data shape in metrics
        for index, valid_data in enumerate(val_datasets):
            metrics_logger.log_metric(key=f"test_data.valid_{index}.length", value=valid_data.num_data())
            metrics_logger.log_metric(key=f"test_data.valid_{index}.width", value=valid_data.num_feature())

    with metrics_logger.log_time_block("time_data_saving.test"):
        # construct dataset
        for index, valid_data in enumerate(val_datasets):
            valid_data.save_binary(os.path.join(args.output_test, f"test_{index}.bin"))

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
