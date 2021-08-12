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
        required=True, type=str, help="Inferencing data location (file path)")
    group_i.add_argument("--model",
        required=False, type=str, help="Exported model location (file path)")
    group_i.add_argument("--output",
        required=False, default=None, type=str, help="Inferencing output location (file path)")
    
    return parser


def run(args, other_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    # create sub dir
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading model from {args.model}")
    booster = lightgbm.Booster(model_file=args.model)

    metric_tags = {'framework':'lightgbm_python','task':'score','lightgbm_version':lightgbm.__version__}

    print(f"Loading data for inferencing")
    with LogTimeBlock("data_loading", methods=['print'], tags=metric_tags):
        raw_data = numpy.loadtxt(args.data, delimiter=",")

    print(f"Running .predict()")
    with LogTimeBlock("inferencing", methods=['print'], tags=metric_tags):
        booster.predict(data=raw_data)


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
