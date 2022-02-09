# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python training script
"""
import os
import sys
import argparse
import logging
import traceback
import json
from distutils.util import strtobool
import lightgbm
from collections import namedtuple

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript
from common.io import get_all_files
from common.lightgbm_utils import LightGBMCallbackHandler
from common.distributed import MultiNodeScript


class LightGBMPythonMultiNodeTrainingScript(MultiNodeScript):
    def __init__(self):
        super().__init__(
            task = "train",
            framework = "lightgbm",
            framework_version = lightgbm.__version__
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
        parser = MultiNodeScript.get_arg_parser(parser)

        group_i = parser.add_argument_group(f"Input Data [{__name__}:{cls.__name__}]")
        group_i.add_argument("--train",
            required=True, type=str, help="Training data location (file or dir path)")
        group_i.add_argument("--test",
            required=True, type=str, help="Testing data location (file path)")
        group_i.add_argument("--construct",
            required=False, default=True, type=strtobool, help="use lazy initialization during data loading phase")
        group_i.add_argument("--header", required=False, default=False, type=strtobool)
        group_i.add_argument("--label_column", required=False, default="0", type=str)
        group_i.add_argument("--group_column", required=False, default=None, type=str)

        group_o = parser.add_argument_group(f"Outputs [{__name__}:{cls.__name__}]")
        group_o.add_argument("--export_model",
            required=False, type=str, help="Export the model in this location (file path)")
        
        # learner params
        group_lgbm = parser.add_argument_group(f"LightGBM learning parameters [{__name__}:{cls.__name__}]")
        group_lgbm.add_argument("--objective", required=True, type=str)
        group_lgbm.add_argument("--metric", required=True, type=str)
        group_lgbm.add_argument("--boosting_type", required=True, type=str)
        group_lgbm.add_argument("--tree_learner", required=True, type=str)
        group_lgbm.add_argument("--label_gain", required=False, type=str, default=None)
        group_lgbm.add_argument("--num_trees", required=True, type=int)
        group_lgbm.add_argument("--num_leaves", required=True, type=int)
        group_lgbm.add_argument("--min_data_in_leaf", required=True, type=int)
        group_lgbm.add_argument("--learning_rate", required=True, type=float)
        group_lgbm.add_argument("--max_bin", required=True, type=int)
        group_lgbm.add_argument("--feature_fraction", required=True, type=float)
        group_lgbm.add_argument("--device_type", required=False, type=str, default="cpu")
        group_lgbm.add_argument("--custom_params", required=False, type=str, default=None)

        return parser


    def load_lgbm_params_from_cli(self, args, multinode_config):
        """Gets the right LightGBM parameters from argparse + mpi config
        Args:
            args (argparse.Namespace)
            multinode_config (namedtuple): as returned from MultiNodeDriver (see common/distributed.py)
        
        Returns:
            lgbm_params (dict)
        """
        # copy all parameters from argparse
        cli_params = dict(vars(args))

        # removing arguments from argparse that are not actual lightgbm parameters
        non_lgbm_params = [
            'verbose',
            'custom_properties',
            'export_model',
            'test',
            'train',
            'custom_params',
            'construct',
            'disable_perf_metrics',
            'multinode_driver',
            'multinode_machines',
            'multinode_listen_port',
        ]
        for key in non_lgbm_params:
            del cli_params[key]

        # doing some fixes and hardcoded values
        lgbm_params = cli_params
        lgbm_params['verbose'] = 2
        lgbm_params['header'] = bool(args.header) # strtobool returns 0 or 1, lightgbm needs actual bool

        # configure multinode
        if multinode_config.multinode_available:
            if isinstance(multinode_config.machines, list):
                # use ip addresses
                lgbm_params['num_machines'] = len(multinode_config.machines)
                lgbm_params['machines'] = ",".join([f"{ip}:{multinode_config.local_listen_port}" for ip in multinode_config.machines])
                lgbm_params['local_listen_port'] = multinode_config.local_listen_port
            else:
                # use mpi
                lgbm_params['num_machines'] = multinode_config.world_size
                lgbm_params['machines'] = ":"

        # process custom params
        if args.custom_params:
            custom_params = json.loads(args.custom_params)
            lgbm_params.update(custom_params)

        return lgbm_params


    def assign_train_data(self, args, multinode_config):
        """ Identifies which training file to load on this node.
        Checks for consistency between number of files and mpi config.
        Args:
            args (argparse.Namespace)
            multinode_config (namedtuple): as returned from detect_multinode_config()
        
        Returns:
            str: path to the data file for this node
        """
        train_file_paths = get_all_files(args.train)

        if multinode_config.multinode_available:    
            # depending on mode, we'll require different number of training files
            if args.tree_learner == "data" or args.tree_learner == "voting":
                if len(train_file_paths) == multinode_config.world_size:
                    train_data = train_file_paths[multinode_config.world_rank]
                else:
                    raise Exception(f"To use MPI with tree_learner={args.tree_learner} and node count {multinode_config.world_rank}, you need to partition the input data into {multinode_config.world_rank} files (currently found {len(train_file_paths)})")
            elif args.tree_learner == "feature":
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
            # if not using mpi, let's just use serial mode with one unique input file
            if args.tree_learner != "serial":
                logging.getLogger().warning(f"Using tree_learner={args.tree_learner} on single node does not make sense, switching back to tree_learner=serial")
                args.tree_learner = "serial"

            if len(train_file_paths) == 1:
                train_data = train_file_paths[0]
            else:
                raise Exception(f"To use single node training, you need to provide only 1 input file, but {len(train_file_paths)} were found")
        return train_data


    def run(self, args, logger, metrics_logger, unknown_args):
        """Run script with arguments (the core of the component)

        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.getLogger() for this script)
            metrics_logger (common.metrics.MetricLogger)
            unknown_args (list[str]): list of arguments not recognized during argparse
        """
        # get mpi config as a namedtuple
        multinode_config = self.get_multinode_config()

        # figure out the lgbm params from cli args + mpi config
        lgbm_params = self.load_lgbm_params_from_cli(args, multinode_config)

        # create a handler for the metrics callbacks
        callbacks_handler = LightGBMCallbackHandler(
            metrics_logger=metrics_logger,
            metrics_prefix=f"node_{multinode_config.world_rank}/"
        )

        # make sure the output argument exists
        if args.export_model and multinode_config.main_node:
            os.makedirs(args.export_model, exist_ok=True)
            args.export_model = os.path.join(args.export_model, "model.txt")

        # log params only once by doing it only on main node (node 0)
        if multinode_config.main_node:
            # log lgbm parameters
            logger.info(f"LightGBM parameters: {lgbm_params}")
            metrics_logger.log_parameters(**lgbm_params)

        # register logger for lightgbm logs
        lightgbm.register_logger(logger)

        logger.info(f"Loading data for training")
        with metrics_logger.log_time_block("time_data_loading", step=multinode_config.world_rank):
            # obtain the path to the train data for this node
            train_data_path = self.assign_train_data(args, multinode_config)
            test_data_paths = get_all_files(args.test)

            logger.info(f"Running with 1 train file and {len(test_data_paths)} test files.")

            # construct datasets
            if args.construct:
                train_data = lightgbm.Dataset(train_data_path, params=lgbm_params).construct()
                val_datasets = [
                    train_data.create_valid(test_data_path).construct() for test_data_path in test_data_paths
                ]
                # capture data shape in metrics
                metrics_logger.log_metric(key="train_data.length", value=train_data.num_data(), step=multinode_config.world_rank)
                metrics_logger.log_metric(key="train_data.width", value=train_data.num_feature(), step=multinode_config.world_rank)
            else:
                train_data = lightgbm.Dataset(train_data_path, params=lgbm_params)
                val_datasets = [
                    train_data.create_valid(test_data_path) for test_data_path in test_data_paths
                ]
                # can't count rows if dataset is not constructed 
                # mlflow can only log float. 
                # metrics_logger.log_metric(key="train_data.length", value="n/a")
                # metrics_logger.log_metric(key="train_data.width", value="n/a")

        logger.info(f"Training LightGBM with parameters: {lgbm_params}")
        with metrics_logger.log_time_block("time_training", step=multinode_config.world_rank):
            booster = lightgbm.train(
                lgbm_params,
                train_data,
                valid_sets = val_datasets,
                callbacks=[callbacks_handler.callback]
            )

        if args.export_model and multinode_config.main_node:
            logger.info(f"Writing model in {args.export_model}")
            booster.save_model(args.export_model)


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return LightGBMPythonMultiNodeTrainingScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    LightGBMPythonMultiNodeTrainingScript.main(cli_args)

if __name__ == "__main__":
    main()
