# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Generate synthetic data for LightGBM training/inferencing
"""
import os
import sys
import argparse
import logging
from distutils.util import strtoboo
import numpy
import sklearn
from sklearn.datasets import make_classification, make_regression

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
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
    # add arguments that are specific to the script
    if parser is None:
        parser = argparse.ArgumentParser(__doc__)

    group_params = parser.add_argument_group("Synthesis params")
    group_params.add_argument(
        "--type", required=True, type=str, choices=["classification", "regression"]
    )
    group_params.add_argument("--train_samples", required=True, type=int)
    group_params.add_argument("--test_samples", required=True, type=int)
    group_params.add_argument("--inferencing_samples", required=True, type=int)
    group_params.add_argument("--n_features", required=True, type=int)
    group_params.add_argument("--n_informative", required=True, type=int)
    group_params.add_argument("--n_redundant", required=False, type=int)
    group_params.add_argument("--random_state", required=True, type=int)

    group_o = parser.add_argument_group("Outputs")
    group_o.add_argument(
        "--output_train",
        required=True,
        type=str,
        help="Output data location (directory)",
    )
    group_o.add_argument(
        "--output_test",
        required=True,
        type=str,
        help="Output data location (directory)",
    )
    group_o.add_argument(
        "--output_inference",
        required=True,
        type=str,
        help="Output data location (directory)",
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
        task="generate", framework="sklearn", framework_version="sklearn.__version__"
    )

    # if provided some custom_properties by the outside orchestrator
    if args.custom_properties:
        metrics_logger.set_properties_from_json(args.custom_properties)

    # make sure the output arguments exists
    os.makedirs(args.output_train, exist_ok=True)
    os.makedirs(args.output_test, exist_ok=True)
    os.makedirs(args.output_inference, exist_ok=True)

    metrics_logger.log_parameters(
        type=args.type,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        inferencing_samples=args.inferencing_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        n_redundant=args.n_redundant,
        random_state=args.random_state,
    )

    # record a metric
    logger.info(f"Generating data in memory.")
    with metrics_logger.log_time_block("data_generation"):
        total_samples = (
            args.train_samples + args.test_samples + args.inferencing_samples
        )
        if args.type == "classification":
            X, y = make_classification(
                n_samples=total_samples,
                n_features=args.n_features,
                n_informative=args.n_informative,
                n_redundant=args.n_redundant,
                random_state=args.random_state,
            )
        elif args.type == "regression":
            X, y = make_regression(
                n_samples=total_samples,
                n_features=args.n_features,
                n_informative=args.n_informative,
                random_state=args.random_state,
            )
        else:
            raise NotImplementedError(f"--type {args.type} is not implemented.")

        # target as one column
        y = numpy.reshape(y, (y.shape[0], 1))

        train_X = X[0 : args.train_samples]
        train_y = y[0 : args.train_samples]
        train_data = numpy.hstack((train_y, train_X))  # keep target as column 0
        logger.info(f"Train data shape: {train_data.shape}")

        test_X = X[args.train_samples : args.train_samples + args.test_samples]
        test_y = y[args.train_samples : args.train_samples + args.test_samples]
        test_data = numpy.hstack((test_X, test_y))  # keep target as column 0
        logger.info(f"Test data shape: {test_data.shape}")

        inference_data = X[args.train_samples + args.test_samples :]
        logger.info(f"Inference data shape: {inference_data.shape}")

    # save as CSV
    logger.info(f"Saving data...")
    with metrics_logger.log_time_block("data_saving"):
        numpy.savetxt(
            os.path.join(args.output_train, "train.txt"),
            train_data,
            delimiter=",",
            newline="\n",
            fmt="%1.3f",
        )
        numpy.savetxt(
            os.path.join(args.output_test, "test.txt"),
            test_data,
            delimiter=",",
            newline="\n",
            fmt="%1.3f",
        )
        numpy.savetxt(
            os.path.join(args.output_inference, "inference.txt"),
            inference_data,
            delimiter=",",
            newline="\n",
            fmt="%1.3f",
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
