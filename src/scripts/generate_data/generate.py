# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Generate synthetic data for LightGBM training/inferencing
"""
import os
import sys
import argparse

import numpy
from lightgbm import train, Dataset
import sklearn
from sklearn.datasets import make_classification, make_regression

# let's add the right PYTHONPATH for common module
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# before doing local import
from common.metrics import MetricsLogger


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
    group_params.add_argument("--train_samples", required=True, type=int)
    group_params.add_argument("--test_samples", required=True, type=int)
    group_params.add_argument("--inferencing_samples", required=True, type=int)
    group_params.add_argument("--n_features", required=True, type=int)
    group_params.add_argument("--n_informative", required=True, type=int)
    group_params.add_argument("--n_redundant", required=False, type=int)
    group_params.add_argument("--random_state", required=True, type=int)

    group_o = parser.add_argument_group("Outputs")
    group_o.add_argument("--output_train", required=True, type=str, help="Output data location (directory)")
    group_o.add_argument("--output_test", required=True, type=str, help="Output data location (directory)")
    group_o.add_argument("--output_inference", required=True, type=str, help="Output data location (directory)")

    return parser


def run(args, other_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    # create sub dir
    os.makedirs(args.output_train, exist_ok=True)
    os.makedirs(args.output_test, exist_ok=True)
    os.makedirs(args.output_inference, exist_ok=True)

    # initializes reporting of metrics
    metrics_logger = MetricsLogger("lightgbm_python.score")

    # add some properties to the session
    metrics_logger.set_properties(
        task = 'generate',
        sklean_version = sklearn.__version__
    )
    metrics_logger.log_parameters(
        train_samples = args.train_samples,
        test_samples = args.test_samples,
        inferencing_samples = args.inferencing_samples,
        n_features = args.n_features,
        n_informative = args.n_informative,
        n_redundant = args.n_redundant,
        random_state = args.random_state
    )

    # record a metric    
    print(f"Generating data in memory.")
    with metrics_logger.log_time_block("data_generation"):
        total_samples = args.train_samples + args.test_samples + args.inferencing_samples
        if args.type == "classification":
            X, y = make_classification(
                n_samples=total_samples,
                n_features=args.n_features,
                n_informative=args.n_informative,
                n_redundant=args.n_redundant,
                random_state=args.random_state
            )
        elif args.type == "regression":
            X, y = make_regression(
                n_samples=total_samples,
                n_features=args.n_features,
                n_informative=args.n_informative,
                random_state=args.random_state
            )
        else:
             raise NotImplementedError(f"--type {args.type} is not implemented.")

        # target as one column
        y = numpy.reshape(y, (y.shape[0], 1))

        train_X = X[0:args.train_samples]
        train_y = y[0:args.train_samples]
        train_data = numpy.hstack((train_y, train_X)) # keep target as column 0
        print(f"Train data shape: {train_data.shape}")

        test_X = X[args.train_samples:args.train_samples+args.test_samples]
        test_y = y[args.train_samples:args.train_samples+args.test_samples]
        test_data = numpy.hstack((test_X, test_y)) # keep target as column 0
        print(f"Test data shape: {test_data.shape}")
        
        inference_data = X[args.train_samples+args.test_samples:]
        print(f"Inference data shape: {inference_data.shape}")

    # save as CSV
    print(f"Saving data...")
    with metrics_logger.log_time_block("data_saving"):
        numpy.savetxt(os.path.join(args.output_train, "train.txt"), train_data, delimiter=",", newline="\n", fmt='%1.3f')
        numpy.savetxt(os.path.join(args.output_test, "test.txt"), test_data, delimiter=",", newline="\n", fmt='%1.3f')
        numpy.savetxt(os.path.join(args.output_inference, "inference.txt"), inference_data, delimiter=",", newline="\n", fmt='%1.3f')

    # optional: close logging session
    metrics_logger.close()

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
