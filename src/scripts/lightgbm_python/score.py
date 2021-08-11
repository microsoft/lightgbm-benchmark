# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python inferencing script
"""
import os
import sys
import argparse
from lightgbm import Booster, Dataset

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
    group_i.add_argument("--data",
        required=True, default="*", type=str, help="Inferencing data location (directory or file)")
    group_i.add_argument("--model",
        required=False, type=str, help="Exported model location")
    
    return parser


def run(args, other_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    print(f"Loading model from {args.model}")
    booster = Booster(model_file=args.model)

    print(f"Running .predict() on data in {args.data}")
    with LogTimeBlock("lightgbm_inferencing_data_loading", methods=['print']):
        # NOTE: this is bad
        data = Dataset(args.data, free_raw_data=False).construct()
        raw_data = data.get_data()

    with LogTimeBlock("lightgbm_inferencing", methods=['print']):
        booster.predict(raw_data)


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
