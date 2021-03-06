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
from common.data import RegressionDataGenerator, RankingDataGenerator, ClassificationDataGenerator

class GenerateSyntheticDataScript(RunnableScript):
    def __init__(self):
        super().__init__(
            task="generate",
            framework="sklearn",
            framework_version="sklearn.__version__"
        )

        # this will collect all the "tasks" (file to generate)
        # for multi-tasking if available (work in progress)
        self.generation_tasks = []

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
        group_params = parser.add_argument_group(f"Data generation params [{__name__}:{cls.__name__}]")
        group_params.add_argument(
            "--type", required=True, type=str, choices=["classification", "regression", "lambdarank"]
        )
        group_params.add_argument("--train_samples", required=True, type=int)
        group_params.add_argument("--train_partitions", required=False, type=int, default=1)
        group_params.add_argument("--test_samples", required=True, type=int)
        group_params.add_argument("--test_partitions", required=False, type=int, default=1)
        group_params.add_argument("--inferencing_samples", required=True, type=int)
        group_params.add_argument("--inferencing_partitions", required=False, type=int, default=1)
        group_params.add_argument("--n_features", required=True, type=int)
        group_params.add_argument("--n_informative", required=True, type=int)
        group_params.add_argument("--n_redundant", required=False, type=int)
        group_params.add_argument("--random_state", required=False, default=None, type=int)
        group_params.add_argument("--docs_per_query", required=False, default=20, type=int)
        group_params.add_argument("--n_label_classes", required=False, default=10, type=int)
    
        group_params = parser.add_argument_group("Format params")
        group_params.add_argument(
            "--delimiter", required=False, type=str, choices=['tab', 'comma', 'space'], default='comma'
        )
        group_params.add_argument(
            "--generate_header", required=False, type=strtobool, default=False, help="generate header on top of output files (independent from external_header)"
        )

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
            "--external_header",
            required=False,
            type=str,
            help="Output the header in an external location (directory)",
        )


        return parser
   

    def generate_tasks(self, args):
        """Create generation tasks based on arguments"""
        train_partition_size = args.train_samples // args.train_partitions
        test_partition_size = args.test_samples // args.test_partitions
        inferencing_partition_size = args.inferencing_samples // args.inferencing_partitions

        batch_size = min(
            10000,
            train_partition_size,
            test_partition_size,
            inferencing_partition_size,
        )

        self.logger.info(f"Using batch size {batch_size}")

        if args.type =="regression":
            generator = RegressionDataGenerator(
                    batch_size=batch_size,
                    n_features=args.n_features,
                    n_informative=args.n_informative,
                    bias=0.0,
                    noise=0.0,
                    seed=args.random_state,
                )
        elif args.type =="lambdarank":
            generator = RankingDataGenerator(
                    docs_per_query=args.docs_per_query,
                    n_label_classes=args.n_label_classes,
                    batch_size=batch_size,
                    n_features=args.n_features,
                    n_informative=args.n_informative,
                    bias=0.0,
                    noise=0.0,
                    seed=args.random_state,
                )
        elif args.type =="classification":
            generator = ClassificationDataGenerator(
                    n_label_classes=args.n_label_classes,
                    batch_size=batch_size,
                    n_features=args.n_features,
                    n_informative=args.n_informative,
                    bias=0.0,
                    noise=0.0,
                    seed=args.random_state,
                )


        # add train partitions to list of tasks
        for i in range(args.train_partitions):
            self.generation_tasks.append(
                (os.path.join(args.output_train, f"train_{i}.txt"), generator, train_partition_size//batch_size)
            )

        # add test partitions to list of tasks
        for i in range(args.test_partitions):
            self.generation_tasks.append(
                (os.path.join(args.output_test, f"test_{i}.txt"), generator, test_partition_size//batch_size)
            )

        # add inferencing partitions to list of tasks
        for i in range(args.inferencing_partitions):
            self.generation_tasks.append(
                (os.path.join(args.output_inference, f"inference_{i}.txt"), generator, inferencing_partition_size//batch_size)
            )

    def execute_tasks(self, args):
        # show some outputs first
        for output_file_path, generator, batches in self.generation_tasks:
            self.logger.info(f"Will generate output {output_file_path} with {batches} batches")

        # generate a fake header
        if args.generate_header or args.external_header:
            if args.type == "lambdarank":
                synthetic_header = (args.delimiter.join([
                            str(i) for i in range(args.n_features+2) # label + queryid + features
                ]))
            else:
                synthetic_header = (args.delimiter.join([
                            str(i) for i in range(args.n_features+1) # label + features
                ]))
            

        partition_count=0
        # generate each data outputs
        for output_file_path, generator, batches in self.generation_tasks:
            self.logger.info(f"Opening file {output_file_path} for writing...")

            # create/erase file
            with open(output_file_path, "w") as output_file:
                if args.generate_header:
                    # generate synthetic header
                    output_file.write(synthetic_header)
                    output_file.write("\n")
                else:
                    output_file.write("")

            # iterate and append
            for i in range(batches):
                with self.metrics_logger.log_time_block("time_data_generation_batch"):
                    X,y = generator.generate(partition_count)
                    y = numpy.reshape(y, (y.shape[0], 1))
                    data = numpy.hstack((y, X))  # keep target as column 0

                with self.metrics_logger.log_time_block("time_data_saving_batch"):
                    with open(output_file_path, "a") as output_file:
                        numpy.savetxt(
                            output_file,
                            data,
                            delimiter=args.delimiter,
                            newline="\n",
                            fmt="%1.3f",
                        )

                self.logger.info(f"Wrote batch {i+1}/{batches}")

                del X
                del y

                partition_count+=1

            self.logger.info(f"Finished generating file {output_file_path}.")
        
        # create an external header 
        if args.external_header:
            self.logger.info(f"Creating an external header file for the generated data")
            os.makedirs(args.external_header, exist_ok=True)
            with open(os.path.join(args.external_header, "header.txt"), 'w') as hf:
                hf.writelines(synthetic_header)


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
        os.makedirs(args.external_header, exist_ok=True)


        # transform delimiter
        if args.delimiter == "comma":
            args.delimiter = ","
        elif args.delimiter == "tab":
            args.delimiter = "\t"
        elif args.delimiter == "space":
            args.delimiter = " "

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
        logger.info(f"Generating data.")
        self.generate_tasks(args)
        self.execute_tasks(args)

def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return GenerateSyntheticDataScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    GenerateSyntheticDataScript.main(cli_args)

if __name__ == "__main__":
    main()
