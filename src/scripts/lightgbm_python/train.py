# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python training script
"""
import os
import sys
import argparse
import logging
from distutils.util import strtobool
import lightgbm
from mpi4py import MPI
from collection import namedtuple

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.metrics import MetricsLogger
from common.io import input_file_path
from common.lightgbm import LightGBMCallbackHandler

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

    group_i = parser.add_argument_group("Input Data")
    group_i.add_argument("--train",
        required=True, type=str, help="Training data location (file or dir path)")
    group_i.add_argument("--test",
        required=True, type=input_file_path, help="Testing data location (file path)")
    group_i.add_argument("--header", required=False, default=False, type=strtobool)
    group_i.add_argument("--label_column", required=False, default="0", type=str)
    group_i.add_argument("--group_column", required=False, default=None, type=str)

    group_o = parser.add_argument_group("Outputs")
    group_o.add_argument("--export_model",
        required=False, type=str, help="Export the model in this location (file path)")
    
    # learner params
    group_lgbm = parser.add_argument_group("LightGBM learning parameters")
    group_lgbm.add_argument("--objective", required=True, type=str)
    group_lgbm.add_argument("--metric", required=True, type=str)
    group_lgbm.add_argument("--boosting_type", required=True, type=str)
    group_lgbm.add_argument("--tree_learner", required=True, type=str)
    group_lgbm.add_argument("--num_trees", required=True, type=int)
    group_lgbm.add_argument("--num_leaves", required=True, type=int)
    group_lgbm.add_argument("--min_data_in_leaf", required=True, type=int)
    group_lgbm.add_argument("--learning_rate", required=True, type=float)
    group_lgbm.add_argument("--max_bin", required=True, type=int)
    group_lgbm.add_argument("--feature_fraction", required=True, type=float)

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


def detect_mpi_config():
    """ Detects if we're running in MPI.
    Args:
        None

    Returns:
        mpi_config (namedtuple)
    """
    # check if we're running multi or single node
    mpi_config_tuple = namedtuple("mpi_config", ['world_size', 'world_rank', 'mpi_available', 'main_node'])

    comm = MPI.COMM_WORLD
    mpi_config = mpi_config_tuple(
        comm.Get_size(), # world_size
        comm.Get_rank(), # world_rank
        (comm.Get_size() > 1), # mpi_available
        (comm.Get_rank() == 0), # main_node
    )

    return mpi_config


def load_lgbm_params_from_cli(args, mpi_config):
    """Gets the right LightGBM parameters from argparse + mpi config
    Args:
        args (argparse.Namespace)
        mpi_config (namedtuple): as returned from detect_mpi_config()
    
    Returns:
        lgbm_params (dict)
    """
    lgbm_params = vars(args)
    lgbm_params['feature_pre_filter'] = False
    lgbm_params['verbose'] = 2
    lgbm_params['header'] = bool(args.header) # strtobool returns 0 or 1, lightgbm needs actual bool

    if mpi_config.mpi_available:
        # add mpi parameters
        lgbm_params['num_machines'] = mpi_config.world_size
        lgbm_params['machines'] = ":"

    return lgbm_params


def get_train_files(path):
    """ Scans input path and returns a list of files. """
    # if input path is already a file, return as list
    if os.path.isfile(path):
        logging.getLogger().info(f"Found INPUT file {path}")
        return [path]
    
    # if input path is a directory, list all files and return
    if os.path.isdir(path):
        all_files = [ os.path.join(path, entry) for entry in os.listdir(path) ]
        if not all_files:
            raise Exception(f"Could not find any file in specified input directory {path}")
        return all_files

    logging.getLogger(__name__).critical(f"Provided INPUT path {path} is neither a directory or a file???")
    return path


def assign_train_data(args, mpi_config):
    """ Identifies which training file to load on this node.
    Checks for consistency between number of files and mpi config.

    Args:
        args (argparse.Namespace)
        mpi_config (namedtuple): as returned from detect_mpi_config()
    
    Returns:
        str: path to the data file for this node
    """
    train_file_paths = get_train_files(args.train)

    if mpi_config.mpi_available:    
        if args.tree_learner == "data" or args.tree_learner == "voting":
            if len(train_file_paths) == mpi_config.world_size:
                train_data = train_file_paths[mpi_config.world_rank]
            else:
                raise Exception(f"To use MPI with tree_learner={args.tree_learner} and node count {mpi_config.world_rank}, you need to partition the input data into {mpi_config.world_rank} files (currently found {len(train_file_paths)})")
        elif args.tree_learner == "parallel":
            if len(train_file_paths) == 1:
                train_data = train_file_paths[0]
            else:
                raise Exception(f"To use MPI with tree_learner=parallel you need to provide only 1 input file, but {len(train_file_paths)} were found")
        elif args.tree_learner == "serial":
            if len(train_file_paths) == 1:
                train_data = train_file_paths[0]
            else:
                raise Exception(f"To use single node training, you need to provide only 1 input file, but {len(train_file_paths)} were found")
        else:
            NotImplementedError(f"tree_learner mode {args.tree_learner} does not exist or is not implemented.")

    else:
        if args.tree_learner != "serial":
            logging.getLogger().warning(f"Using tree_learner={args.tree_learner} on single node does not make sense, switching back to tree_learner=serial")
            args.tree_learner = "serial"

        # check input data
        if len(train_file_paths) == 1:
            train_data = train_file_paths[0]
        else:
            raise Exception(f"To use single node training, you need to provide only 1 input file, but {len(train_file_paths)} were found")
    return train_data


def run(args, unknown_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    logger = logging.getLogger()

    # detect if we're running in mpi
    mpi_config = detect_mpi_config()

    # figure out the lgbm params from cli args + mpi config
    lgbm_params = load_lgbm_params_from_cli(args, mpi_config)

    # get Metrics logger for benchmark metrics
    # below: initialize reporting of metrics with a custom session name
    metrics_logger = MetricsLogger(
        "lightgbm_python.train",
        #metrics_prefix=f"node_{mpi_config.world_rank}" if mpi_config.mpi_available else None
    )
    callbacks_handler = LightGBMCallbackHandler(
        metrics_logger = metrics_logger,
        metrics_prefix = f"node_{mpi_config.world_rank}" if mpi_config.mpi_available else None
    )

    # make sure the output argument exists
    if args.export_model and mpi_config.main_node:
        os.makedirs(args.export_model, exist_ok=True)
        args.export_model = os.path.join(args.export_model, "model.txt")

    # log params only once by doing it only on main node (node 0)
    if mpi_config.main_node:
        # add common properties to the session
        metrics_logger.set_properties(
            task="train", framework="lightgbm", framework_version=lightgbm.__version__
        )

        # if provided some custom_properties by the outside orchestrator
        if args.custom_properties:
            metrics_logger.set_properties_from_json(args.custom_properties)

        # add properties about environment of this script
        metrics_logger.set_platform_properties()

        # log lgbm parameters
        #metrics_logger.log_parameters(**lgbm_params)

    # register logger for lightgbm logs
    lightgbm.register_logger(logger)

    logger.info(f"Loading data for training")
    with metrics_logger.log_time_block("time_data_loading"):
        # obtain the path to the train data for this node
        train_data_path = assign_train_data(args, mpi_config)
        train_data = lightgbm.Dataset(train_data_path, params=lgbm_params).construct()
        val_data = train_data.create_valid(args.test).construct()

    # capture data shape in metrics
    metrics_logger.log_metric(key="train_data.length", value=train_data.num_data())
    metrics_logger.log_metric(key="train_data.width", value=train_data.num_feature())
    metrics_logger.log_metric(key="test_data.length", value=val_data.num_data())
    metrics_logger.log_metric(key="test_data.width", value=val_data.num_feature())

    logger.info(f"Training LightGBM with parameters: {lgbm_params}")
    with metrics_logger.log_time_block("time_training"):
        booster = lightgbm.train(
            lgbm_params,
            train_data,
            valid_sets = val_data,
            callbacks=[callbacks_handler.callback]
        )

    if args.export_model and mpi_config.main_node:
        logger.info(f"Writing model in {args.export_model}")
        booster.save_model(args.export_model)

    # clean exit from mpi
    if MPI.Is_initialized():
        logger.info("MPI was initialized, calling MPI.finalize()")
        MPI.Finalize()

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
