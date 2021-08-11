# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Generate synthetic data for LightGBM training/inferencing
"""
import os
import sys
import argparse
from distutils.util import strtobool

import numpy
from lightgbm import train, Dataset
from sklearn.datasets import make_classification, make_regression

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

    group_params = parser.add_argument_group("Synthesis params")
    group_params.add_argument("--type", required=True, type=str, choices=['classification', 'regression'])
    group_params.add_argument("--n_samples", required=True, type=int)
    group_params.add_argument("--n_features", required=True, type=int)
    group_params.add_argument("--n_informative", required=True, type=int)
    group_params.add_argument("--n_redundant", required=False, type=int)
    group_params.add_argument("--random_state", required=True, type=int)

    group_o = parser.add_argument_group("Outputs")
    group_o.add_argument("--output", required=True, default="*", type=str, help="Output data location")

    return parser


def run(args, other_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    # create sub dir
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # record a metric    
    with LogTimeBlock("data_generation", methods=['print']):
        if args.type == "classification":
            X, y = make_classification(
                n_samples=args.n_samples,
                n_features=args.n_features,
                n_informative=args.n_informative,
                n_redundant=args.n_redundant,
                random_state=args.random_state
            )
        elif args.type == "regression":
            X, y = make_regression(
                n_samples=args.n_samples,
                n_features=args.n_features,
                n_informative=args.n_informative,
                random_state=args.random_state
            )
        else:
             raise NotImplementedError(f"--type {args.type} is not implemented.")

        # target as one column
        y = numpy.reshape(y, (y.shape[0], 1))

    # some debugging
    print(f"X shape: {X.shape}")
    print(f"Y shape: {y.shape}")
    output_data = numpy.hstack((y, X)) # keep target as column 0
    print(f"Output data shape: {output_data.shape}")

    # save as CSV
    numpy.savetxt(args.output, output_data, delimiter=",", newline="\n", fmt='%1.3f')



def main(cli_args=None):
    """ Component main function, parses arguments and executes run() function.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # construct arg parser
    parser = get_arg_parser()
 
    # if argument parsing fails, or if unknown arguments, will except
    args, unknown_args = parser.parse_known_args(cli_args)

    # run the actual thing
    run(args, unknown_args)


if __name__ == "__main__":
    main()    
