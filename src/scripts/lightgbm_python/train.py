# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python training script
"""
import os
import sys
import argparse
from distutils.util import strtobool

from lightgbm import train, Dataset

# let's add the right PYTHONPATH for common module
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# before doing local import
from common.metrics import LogTimeBlock


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
    group_i.add_argument("--train",
        required=True, default="*", type=str, help="Training data location (file path)")
    group_i.add_argument("--test",
        required=True, default="*", type=str, help="Testing data location (file path)")
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

    return parser


def run(args, other_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    # create sub dir
    os.makedirs(os.path.dirname(args.export_model), exist_ok=True)

    lgbm_params = vars(args)
    print(f"Training LightGBM with parameters: {lgbm_params}")

    with LogTimeBlock("lightgbm_data_loading", methods=['print'], tags={'framework':'lightgbm_python'}):
        train_data = Dataset(args.train, params=lgbm_params).construct()
        val_data = train_data.create_valid(args.test)

    with LogTimeBlock("lightgbm_training", methods=['print'], tags=lgbm_params):
        booster = train(
            lgbm_params,
            train_data,
            valid_sets = val_data,
            callbacks=[]
        )

    if args.export_model:
        print(f"Writing model in {args.export_model}")
        booster.save_model(args.export_model)


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
