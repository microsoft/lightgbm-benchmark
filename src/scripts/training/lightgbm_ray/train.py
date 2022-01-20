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
from collections import namedtuple
import glob

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript
from common.io import get_all_files
from common.ray import RayScript
import lightgbm_ray # RayDMatrix, RayParams, train
from common.lightgbm_utils import LightGBMCallbackHandler

class LightGBMOnRayTrainingScript(RayScript):
    def __init__(self):
        super().__init__(
            task = "train",
            framework = "lightgbm_ray",
            framework_version = lightgbm_ray.__version__
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
        parser = RayScript.get_arg_parser(parser)

        group_i = parser.add_argument_group("Input Data")
        group_i.add_argument("--train",
            required=True, type=str, help="Training data location (file or dir path)")
        group_i.add_argument("--train_data_format",
            required=False, type=str, choices=['CSV', 'PARQUET', 'PETAFORM'], default='CSV', help="type of input train data (CSV, PARQUET, PETAFORM), default CSV")
        group_i.add_argument("--test",
            required=True, type=str, help="Testing data location (file path)")
        group_i.add_argument("--test_data_format",
            required=False, type=str, choices=['CSV', 'PARQUET', 'PETAFORM'], default=None, help="type of input test data (CSV, PARQUET, PETAFORM), default using same as train")
        #group_i.add_argument("--header", required=False, default=False, type=strtobool)
        group_i.add_argument("--label_column", required=True, type=str)
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
        group_lgbm.add_argument("--label_gain", required=False, type=str, default=None)
        group_lgbm.add_argument("--num_iterations", required=True, type=int)
        group_lgbm.add_argument("--num_leaves", required=True, type=int)
        group_lgbm.add_argument("--min_data_in_leaf", required=True, type=int)
        group_lgbm.add_argument("--learning_rate", required=True, type=float)
        group_lgbm.add_argument("--max_bin", required=True, type=int)
        group_lgbm.add_argument("--feature_fraction", required=True, type=float)
        group_lgbm.add_argument("--device_type", required=False, type=str, default="cpu")
        group_lgbm.add_argument("--custom_params", required=False, type=str, default=None)

        group_lgbm = parser.add_argument_group("LightGBM/Ray runsettings")
        group_lgbm.add_argument("--ray_actors", required=False, default=None, type=int, help="number of actors (default: count available nodes, or 1)")
        group_lgbm.add_argument("--ray_data_distributed", required=False, default=True, type=strtobool, help="is data pre-partitioned (True) or should ray distribute it (False)")


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
            'ray_head_port',
            'ray_on_aml',
            'train_data_format',
            'ray_data_distributed'
        ]
        for key in non_lgbm_params:
            if key in cli_params:
                del cli_params[key]

        # doing some fixes and hardcoded values
        lgbm_params = cli_params
        lgbm_params['verbose'] = 2

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

        # create a handler for the metrics callbacks
        callbacks_handler = LightGBMCallbackHandler(
            metrics_logger=metrics_logger
        )

        # make sure the output argument exists
        if args.export_model:
            os.makedirs(args.export_model, exist_ok=True)
            args.export_model = os.path.join(args.export_model, "model.txt")

        # log lgbm parameters
        logger.info(f"LightGBM parameters: {lgbm_params}")
        metrics_logger.log_parameters(**lgbm_params)

        # register logger for lightgbm logs
        #lightgbm.register_logger(logger)

        logger.info(f"Loading data for training")
        train_paths = list(sorted(glob.glob(os.path.join(args.train, "*"))))
        logger.info(f"Found {len(train_paths)} training files")
        train_data_format = getattr(lightgbm_ray.RayFileType, args.train_data_format)
        train_set = lightgbm_ray.RayDMatrix(
            train_paths,
            label=args.label_column,  # Will select this column as the label
            #columns=[str(i) for i in range(4001)],
            filetype=train_data_format,
            distributed=args.ray_data_distributed,

        )

        logger.info(f"Loading data for validation")
        validation_paths = list(sorted(glob.glob(os.path.join(args.test, "*"))))
        logger.info(f"Found {len(validation_paths)} validation files")
        
        # NOTE: it seems we need to have as many test sets as actors
        required_validation_sets = args.ray_actors or self.available_nodes
        if len(validation_paths) == 1 and required_validation_sets > 1:
            logger.info(f"Creating artificial {required_validation_sets} test sets")
            validation_paths = [ validation_paths[0] for _ in range(required_validation_sets) ]

        # load the validation data into RayDMatrix
        val_data_format = getattr(lightgbm_ray.RayFileType, args.test_data_format or args.train_data_format)
        val_set = lightgbm_ray.RayDMatrix(
            validation_paths,
            label=args.label_column,  # Will select this column as the label
            #columns=[str(i) for i in range(4001)],
            filetype=val_data_format,
            distributed=True
        )

        logger.info(f"Training LightGBM with parameters: {lgbm_params}")
        evals_result = {}
        additional_results = {}
        booster = lightgbm_ray.train(
            lgbm_params,
            train_set,
            num_boost_round=lgbm_params['num_iterations'],
            evals_result=evals_result,
            additional_results=additional_results,
            valid_sets=[ val_set ],
            valid_names=[ "test" ],
            verbose_eval=True,
            ray_params=lightgbm_ray.RayParams(
                num_actors=args.ray_actors or self.available_nodes, # number of VMs
                #cpus_per_actor=2
            ),
            #callbacks=[callbacks_handler.callback]
        )
        print(f"evals_results={evals_result}")
        print(f"additional_results={additional_results}")

        if args.export_model:
            logger.info(f"Writing model in {args.export_model}")
            booster.booster_.save_model(args.export_model)


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return LightGBMOnRayTrainingScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    LightGBMOnRayTrainingScript.main(cli_args)

if __name__ == "__main__":
    main()
