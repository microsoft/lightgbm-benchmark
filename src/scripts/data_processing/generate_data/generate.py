# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Generate synthetic data for LightGBM training/inferencing
"""
import os
import sys
import argparse
import logging
from distutils.util import strtobool
import numpy
import sklearn
from sklearn.datasets import make_classification, make_regression

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript


class GenerateSyntheticDataScript(RunnableScript):
    def __init__(self):
        super().__init__(
            task="generate",
            framework="sklearn",
            framework_version="sklearn.__version__"
        )

    @classmethod
    def get_arg_parser(cls, parser=None):
        """Adds component/module arguments to a given argument parser.

        Args:
            parser (argparse.ArgumentParser): an argument parser instance

        Returns:
            ArgumentParser: the argument parser instance

        Notes:
            if parser is None, creates a new parser instance
        """
        # add generic arguments
        parser = RunnableScript.get_arg_parser(parser)

        # add arguments that are specific to the script
        group_params = parser.add_argument_group("Synthesis params")
        group_params.add_argument(
            "--type", required=True, type=str, choices=["classification", "regression", "lambdarank"]
        )
        group_params.add_argument("--train_samples", required=True, type=int)
        group_params.add_argument("--test_samples", required=True, type=int)
        group_params.add_argument("--inferencing_samples", required=True, type=int)
        group_params.add_argument("--n_features", required=True, type=int)
        group_params.add_argument("--n_informative", required=True, type=int)
        group_params.add_argument("--n_redundant", required=False, type=int)
        group_params.add_argument("--random_state", required=False, default=None, type=int)
        group_params.add_argument("--docs_per_query", required=False, default=20, type=int)
        group_params.add_argument("--delimiter", required=False, default=',', type=str)

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
        group_o.add_argument(
            "--output_header",
            required=True,
            type=str,
            help="Output header location (directory)",
        )

        return parser


    def run(self, args, logger, metrics_logger, unknown_args):
        """Run script with arguments (the core of the component)

        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.getLogger() for this script)
            metrics_logger (common.metrics.MetricLogger)
            unknown_args (list[str]): list of arguments not recognized during argparse
        """
        # make sure the output arguments exists
        os.makedirs(args.output_train, exist_ok=True)
        os.makedirs(args.output_test, exist_ok=True)
        os.makedirs(args.output_inference, exist_ok=True)
        os.makedirs(args.output_header, exist_ok=True)

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
        with metrics_logger.log_time_block("time_data_generation"):
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
            elif args.type == "lambdarank":
                X, y = make_regression(
                    n_samples=total_samples,
                    n_features=args.n_features,
                    n_informative=args.n_informative,
                    random_state=args.random_state,
                )
                # add query column
                query_col = [[i // args.docs_per_query] for i in range(total_samples)]
                X = numpy.hstack((query_col, X))
                # create 30 ranking labels
                y = ((y - min(y))/(max(y)-min(y))*30).astype(int)

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
            test_data = numpy.hstack((test_y, test_X))  # keep target as column 0
            logger.info(f"Test data shape: {test_data.shape}")

            inference_data = X[args.train_samples + args.test_samples :]
            logger.info(f"Inference data shape: {inference_data.shape}")

        # create a header 
        header = [f'Column_{i}' for i in range(train_data.shape[1])]
        if args.delimiter == 'tab':
            args.delimiter = "\t"
        # save as CSV
        logger.info(f"Saving data...")
        with metrics_logger.log_time_block("time_data_saving"):
            numpy.savetxt(
                os.path.join(args.output_train, "train.txt"),
                train_data,
                delimiter=args.delimiter,
                newline="\n",
                fmt="%1.3f",
            )
            numpy.savetxt(
                os.path.join(args.output_test, "test.txt"),
                test_data,
                delimiter=args.delimiter,
                newline="\n",
                fmt="%1.3f",
            )
            numpy.savetxt(
                os.path.join(args.output_inference, "inference.txt"),
                inference_data,
                delimiter=args.delimiter,
                newline="\n",
                fmt="%1.3f",
            )
            with open(os.path.join(args.output_header, "header.txt"), 'w') as hf:
                hf.writelines(args.delimiter.join(header))


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return GenerateSyntheticDataScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    GenerateSyntheticDataScript.main(cli_args)

if __name__ == "__main__":
    main()
