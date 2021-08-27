# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python training script
"""
import os
import sys
import argparse
import logging
from distutils.util import strtobool
import lightgbm

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
    group_i.add_argument("--train",
        required=True, type=input_file_path, help="Training data location (file path)")
    group_i.add_argument("--test",
        required=True, type=input_file_path, help="Testing data location (file path)")
    group_i.add_argument("--header", required=False, default=False, type=strtobool)
    group_i.add_argument("--label_column", required=False, default="0", type=str)
    group_i.add_argument("--query_column", required=False, default=None, type=str)

    group_o = parser.add_argument_group("Outputs")
    group_o.add_argument("--export_model",
        required=False, type=str, help="Export the model in this location (file path)")
    
    # learner params
    group_lgbm = parser.add_argument_group("LightGBM learning parameters")
    group_lgbm.add_argument("--objective", required=True, type=str)
    group_lgbm.add_argument("--metric", required=True, type=str)
    group_lgbm.add_argument("--boosting_type", required=True, type=str)
    group_lgbm.add_argument("--tree_learner", required=True, type=str)
    group_lgbm.add_argument("--num_trees", required=True, type=int)
    group_lgbm.add_argument("--num_leaves", required=True, type=int)
    group_lgbm.add_argument("--min_data_in_leaf", required=True, type=int)
    group_lgbm.add_argument("--learning_rate", required=True, type=float)
    group_lgbm.add_argument("--max_bin", required=True, type=int)
    group_lgbm.add_argument("--feature_fraction", required=True, type=float)

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
        task="train", framework="lightgbm", framework_version=lightgbm.__version__
    )

    # if provided some custom_properties by the outside orchestrator
    if args.custom_properties:
        metrics_logger.set_properties_from_json(args.custom_properties)

    # make sure the output argument exists
    if args.export_model:
        os.makedirs(args.export_model, exist_ok=True)
        args.export_model = os.path.join(args.export_model, "model.txt")

    # add lgbm params to the session
    lgbm_params = vars(args)
    lgbm_params['feature_pre_filter'] = False
    lgbm_params['verbose'] = 2

    metrics_logger.log_parameters(**lgbm_params)

    logger.info(f"Loading data for training")
    with metrics_logger.log_time_block("time_data_loading"):
        train_data = lightgbm.Dataset(args.train, params=lgbm_params).construct()
        val_data = train_data.create_valid(args.test).construct()

    # capture data shape as property
    metrics_logger.set_properties(
        train_data_length = train_data.num_data(),
        train_data_width = train_data.num_feature(),
        test_data_length = val_data.num_data(),
        test_data_width = val_data.num_feature()
    )

    logger.info(f"Training LightGBM with parameters: {lgbm_params}")
    with metrics_logger.log_time_block("time_training"):
        booster = lightgbm.train(
            lgbm_params,
            train_data,
            valid_sets = val_data,
            callbacks=[]
        )

    if args.export_model:
        logger.info(f"Writing model in {args.export_model}")
        booster.save_model(args.export_model)

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
