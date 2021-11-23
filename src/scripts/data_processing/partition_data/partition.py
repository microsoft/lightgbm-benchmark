"""
Partitions input data (text/lines) into chunks for parallel processing.

NOTE: current script assumes all records are independent.
"""
import os
import sys
import argparse
import logging
from distutils.util import strtobool
import ray
from pyarrow import csv

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript


class PartitionDataScript(RunnableScript):
    def __init__(self):
        super().__init__(
            task = "partition",
            framework = "python",
            framework_version = "n/a"
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
        group = parser.add_argument_group('Partitioning arguments')
        group.add_argument("--input", dest="input", type=str, required=True, help="file/directory to split")
        group.add_argument("--output", dest="output", type=str, help="location to store partitioned files", required=True)
        group.add_argument("--mode", type=str, choices=['partition','chunk'], required=True, help="Partitioning mode")
        group.add_argument("--number", type=int, required=True, help="If roundrobin number of partition, if chunk number of records per partition")
        group.add_argument("--header", type=strtobool, required=False, default=False, help="Should we preserve firstline into each partition?")

        return parser

    def run(self, args, logger, metrics_logger, unknown_args):
        """Run script with arguments (the core of the component)

        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.getLogger() for this script)
            metrics_logger (common.metrics.MetricLogger)
            unknown_args (list[str]): list of arguments not recognized during argparse
        """
        # Create output folder
        os.makedirs(args.output, exist_ok=True)

        # Retrieve all input files
        if os.path.isfile(args.input):
            logger.info("Input is one unique file")
            file_names = [os.path.basename(args.input)]
            input_files = [args.input]
        else:
            logger.info("Input is a directory, listing all of them for processing")
            file_names = os.listdir(args.input)
            input_files = [os.path.join(args.input, file) for file in file_names]
            logger.info("Found {} files in {}".format(len(input_files), args.input))

        logger.info(f"Running partitioning...")
        with metrics_logger.log_time_block("time_partitioning"):
            if args.mode == "partition":
                if args.header:
                    csv_data = ray.data.read_csv(input_files)
                    csv_data.repartition(args.number).write_csv(args.output)
                else:
                    csv_data = ray.data.read_text(input_files)
                    csv_data.repartition(args.number).write_csv(args.output)
            elif args.mode == "chunk":
                raise NotImplementedError()
            else:
                raise NotImplementedError(f"Mode --mode {args.mode} is not implemented yet.")


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return PartitionDataScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    PartitionDataScript.main(cli_args)

if __name__ == "__main__":
    main()
