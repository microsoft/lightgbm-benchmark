# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python dataset saving to binary
"""
import os
import sys
import argparse
import logging
import traceback
import json
from distutils.util import strtobool
import lightgbm

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if COMMON_ROOT not in sys.path:
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript
from common.io import input_file_path, get_all_files


class LightGBMData2BinScript(RunnableScript):
    def __init__(self):
        super().__init__(
            task="data2bin",
            framework="lightgbm_python",
            framework_version=lightgbm.__version__
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

        group_i = parser.add_argument_group(f"Input Data [{__name__}:{cls.__name__}]")
        group_i.add_argument("--train",
            required=True, type=input_file_path, help="Training data location (file path or dir path with unique file)")
        group_i.add_argument("--test",
            required=True, type=str, help="Testing data location (file or dir path with multiple files)")
        group_i.add_argument("--header", required=False, default=False, type=strtobool)
        group_i.add_argument("--label_column", required=False, default="0", type=str)
        group_i.add_argument("--group_column", required=False, default=None, type=str)

        group_o = parser.add_argument_group(f"Outputs [{__name__}:{cls.__name__}]")
        group_o.add_argument("--output_train",
            required=False, type=str, help="export binary train data (folder)")
        group_o.add_argument("--output_test",
            required=False, type=str, help="export binary test data (folder)")
        
        # learner params
        group_lgbm = parser.add_argument_group(f"LightGBM Dataset parameters [{__name__}:{cls.__name__}]")
        group_lgbm.add_argument("--max_bin", required=True, type=int)
        group_lgbm.add_argument("--custom_params", required=False, type=str, default=None)

        return parser


    def load_lgbm_params_from_cli(self, args):
        """Gets the right LightGBM parameters from argparse + mpi config
        Args:
            args (argparse.Namespace)
        
        Returns:
            lgbm_params (dict)
        """
        # copy all parameters from argparse
        cli_params = dict(vars(args))

        # removing arguments that are purely CLI
        for key in ['verbose', 'custom_properties', 'output_train', 'output_test', 'test', 'train', 'custom_params']:
            del cli_params[key]

        # doing some fixes and hardcoded values
        lgbm_params = cli_params
        lgbm_params['verbose'] = 2
        lgbm_params['header'] = bool(args.header) # strtobool returns 0 or 1, lightgbm needs actual bool

        # process custom params
        if args.custom_params:
            custom_params = json.loads(args.custom_params)
            lgbm_params.update(custom_params)

        return lgbm_params


    def run(self, args, logger, metrics_logger, unknown_args):
        """Run script with arguments (the core of the component)

        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.getLogger() for this script)
            metrics_logger (common.metrics.MetricLogger)
            unknown_args (list[str]): list of arguments not recognized during argparse
        """
        # figure out the lgbm params from cli args
        lgbm_params = self.load_lgbm_params_from_cli(args)

        # make sure the output argument exists
        os.makedirs(args.output_train, exist_ok=True)
        args.output_train = os.path.join(args.output_train, "train.bin")
        os.makedirs(args.output_test, exist_ok=True)

        # log lgbm parameters
        logger.info(f"LGBM Params: {lgbm_params}")
        metrics_logger.log_parameters(**lgbm_params)

        # register logger for lightgbm logs
        lightgbm.register_logger(logger)

        logger.info(f"Loading data for training")
        with metrics_logger.log_time_block("time_data_loading.train"):
            # construct dataset
            train_data = lightgbm.Dataset(args.train, params=lgbm_params).construct()

            # capture data shape in metrics
            metrics_logger.log_metric(key="train_data.length", value=train_data.num_data())
            metrics_logger.log_metric(key="train_data.width", value=train_data.num_feature())

        with metrics_logger.log_time_block("time_data_saving.train"):
            # construct dataset
            train_data.save_binary(args.output_train)

        with metrics_logger.log_time_block("time_data_loading.test"):
            # construct dataset
            test_data_paths = get_all_files(args.test)
            val_datasets = [
                train_data.create_valid(test_data_path).construct() for test_data_path in test_data_paths
            ]

            # capture data shape in metrics
            for index, valid_data in enumerate(val_datasets):
                metrics_logger.log_metric(key=f"test_data.valid_{index}.length", value=valid_data.num_data())
                metrics_logger.log_metric(key=f"test_data.valid_{index}.width", value=valid_data.num_feature())

        with metrics_logger.log_time_block("time_data_saving.test"):
            # construct dataset
            for index, valid_data in enumerate(val_datasets):
                valid_data.save_binary(os.path.join(args.output_test, f"test_{index}.bin"))


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return LightGBMData2BinScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    LightGBMData2BinScript.main(cli_args)

if __name__ == "__main__":
    main()
