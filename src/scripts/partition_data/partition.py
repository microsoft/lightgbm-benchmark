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
from common.metrics import MetricsLogger
from common.io import input_file_path
from common.io import PartitioningEngine


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

    # add arguments that are specific to the module
    group = parser.add_argument_group('Partitioning arguments')
    group.add_argument("--input", dest="input", type=str, required=True, help="file/directory to split")
    group.add_argument("--output", dest="output", type=str, help="location to store partitioned files", required=True)
    group.add_argument("--mode", type=str, choices=PartitioningEngine.PARTITION_MODES, required=True, help="Partitioning mode")
    group.add_argument("--number", type=int, required=True, help="If roundrobin number of partition, if chunk number of records per partition")
    group.add_argument("--header", type=strtobool, required=False, default=False, help="Should we preserve firstline into each partition?")

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
    metrics_logger = MetricsLogger("partition_data")

    # add common properties to the session
    metrics_logger.set_properties(
        task="partition", framework="python", framework_version="n/a"
    )

    # if provided some custom_properties by the outside orchestrator
    if args.custom_properties:
        metrics_logger.set_properties_from_json(args.custom_properties)

    # add properties about environment of this script
    metrics_logger.set_platform_properties()

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
