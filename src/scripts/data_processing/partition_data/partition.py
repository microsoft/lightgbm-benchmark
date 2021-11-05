"""
Partitions input data (text/lines) into chunks for parallel processing.

NOTE: current script assumes all records are independent.
"""
import os
import sys
import argparse
import logging
from distutils.util import strtobool

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.io import PartitioningEngine
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
        group.add_argument("--mode", type=str, choices=PartitioningEngine.PARTITION_MODES, required=True, help="Partitioning mode")
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

        # create instance of partitioner
        partition_engine = PartitioningEngine(
            mode = args.mode,
            number = args.number,
            header = args.header,
            logger=logger
        )

        # simply run
        logger.info(f"Running partitioning...")
        with metrics_logger.log_time_block("time_partitioning"):
            partition_engine.run(args.input, args.output)


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return PartitionDataScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    PartitionDataScript.main(cli_args)

if __name__ == "__main__":
    main()
