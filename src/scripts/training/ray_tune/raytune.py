# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM sweep script using ray tune.
"""

from functools import partial
from genericpath import exists
import json
import os
import sys
import lightgbm
from distutils.util import strtobool

import ray
from ray import tune
from ray.tune.integration.lightgbm import TuneReportCheckpointCallback
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from flaml import BlendSearch
from ray.tune.schedulers import FIFOScheduler
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.suggest.basic_variant import BasicVariantGenerator

# set tune environment variable for forward compatibility
# https://docs.ray.io/en/latest/tune/user-guide.html#tune-env-vars
os.environ["TUNE_RESULT_DELIM"] = '/'

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))


# useful imports from common
from common.ray import RayScript
from common.raytune_param import RayTuneParameterParser

def process_raytune_parameters(args):
    """Parses config and spots sweepable paraneters

    Args:
        args: parsed config parameters

    Returns:
        tunable_params (dict)
        fixed_params (dict)
    """
    cli_params = dict(vars(args))

    # removing arguments from argparse that are not actual lightgbm parameters
    non_lgbm_params = [
        'custom_properties',
        'custom_params',
        'test',
        'train',
        'construct',
        'disable_perf_metrics',
        'ray_head_addr',
        'ray_head_port',
        'ray_redis_password',
        'ray_on_aml',
        'search_alg',
        'scheduler',
        'mode',
        'num_samples',
        'time_budget',
        'max_concurrent_trials',
        'cpus_per_trial',
        'output_path'
    ]
    for key in non_lgbm_params:
        del cli_params[key]

    # doing some fixes and hardcoded values
    lgbm_params = cli_params
    lgbm_params['verbose'] = -1
    # strtobool returns 0 or 1, lightgbm needs actual bool
    lgbm_params['header'] = bool(args.header)

    # process custom params
    if args.custom_params:
        custom_params = json.loads(args.custom_params)
        lgbm_params.update(custom_params)

    # the class below automates parsing of sweepable parameters
    raytune_param_parser = RayTuneParameterParser(
        tunable_parameters=[
            "num_iterations",
            "num_leaves",
            "min_data_in_leaf",
            "learning_rate",
            "max_bin",
            "feature_fraction"
        ],
    )

    # parser gonna parse
    raytune_param_parser.parse_from_dict(lgbm_params)

    # update get_tunable_params and get_fixed_params
    tunable_params = raytune_param_parser.get_tunable_params()
    fixed_params = raytune_param_parser.get_fixed_params()

    # return dictionaries to fed as params into our pipeline
    return tunable_params, fixed_params


class LightGBMRayTuneScript(RayScript):

    SEARCH_ALG_MAP = {"BasicVariantGenerator": BasicVariantGenerator,
                      "AxSearch": AxSearch,
                      "BayesOptSearch": BayesOptSearch,
                      "BlendSearch": BlendSearch}

    SCHEDULER_MAP = {"FIFOScheduler": FIFOScheduler,
                     'ASHAScheduler': ASHAScheduler,
                     'HyperBandScheduler': HyperBandScheduler,
                     'MedianStoppingRule': MedianStoppingRule}

    def __init__(self):
        super().__init__(
            task="sweep",
            framework="ray_tune",
            framework_version=ray.__version__
        )

    @classmethod
    def get_arg_parser(cls, parser=None):
        """Adds component/module arguments to a given argument parser.

        Args:
            parser (argparse.ArgumentParser): an existing argument parser instance

        Returns:
            ArgumentParser: the argument parser instance
        """
        # IMPORTANT: call this to add generic benchmark arguments
        parser = RayScript.get_arg_parser(parser)

        group_i = parser.add_argument_group("Input Data")
        group_i.add_argument("--train",
                             required=True, type=str, help="Training data location (file or dir path)")
        group_i.add_argument("--test",
                             required=True, type=str, help="Testing data location (file path)")
        group_i.add_argument("--construct",
                             required=False, default=True, type=strtobool, help="use lazy initialization during data loading phase")
        group_i.add_argument("--header", required=False,
                             default=False, type=strtobool)
        group_i.add_argument(
            "--label_column", required=False, default="0", type=str)
        group_i.add_argument(
            "--group_column", required=False, default=None, type=str)

        group_o = parser.add_argument_group("Outputs")
        group_o.add_argument("--output_path",
                             required=False, type=str, help="Export the best result as a csv file in this dir path.")

        # learner params
        group_lgbm = parser.add_argument_group("LightGBM learning parameters")
        group_lgbm.add_argument("--objective", required=True, type=str)
        group_lgbm.add_argument("--metric", required=True, type=str)
        group_lgbm.add_argument("--boosting_type", required=True, type=str)
        group_lgbm.add_argument("--tree_learner", required=True, type=str)
        group_lgbm.add_argument(
            "--label_gain", required=False, type=str, default=None)
        group_lgbm.add_argument(
            "--num_iterations", required=False, type=str, default="100")
        group_lgbm.add_argument(
            "--num_leaves", required=False, type=str, default="31")
        group_lgbm.add_argument("--min_data_in_leaf",
                                required=False, type=str, default="20")
        group_lgbm.add_argument(
            "--learning_rate", required=False, type=str, default="0.1")
        group_lgbm.add_argument(
            "--max_bin", required=False, type=str, default="255")
        group_lgbm.add_argument("--feature_fraction",
                                required=False, type=str, default="1.0")
        group_lgbm.add_argument(
            "--device_type", required=False, type=str, default="cpu")
        group_lgbm.add_argument(
            "--custom_params", required=False, type=str, default=None)

        # ray tune parameters
        group_raytune = parser.add_argument_group("Ray Tune config parameters")
        group_raytune.add_argument(
            "--mode", required=True, type=str, choices=['min', 'max'])
        group_raytune.add_argument("--search_alg", required=False, type=str, default="BasicVariantGenerator",
                                   choices=["BasicVariantGenerator", "AxSearch", "BayesOptSearch", "BlendSearch"])
        group_raytune.add_argument("--scheduler", required=False, default="FIFOScheduler", type=str, choices=[
                                   "FIFOScheduler", 'ASHAScheduler', 'HyperBandScheduler', 'MedianStoppingRule'])
        group_raytune.add_argument(
            "--num_samples", required=False, default=-1, type=int)
        group_raytune.add_argument(
            "--time_budget", required=False, default=1800, type=int)
        group_raytune.add_argument(
            "--cpus_per_trial", required=False, default=1, type=int)
        group_raytune.add_argument(
            "--max_concurrent_trials", required=False, default=0, type=int)

        return parser

    # get the search algorithm
    # TODO: add arguments of scheduler for advanced usage.
    def get_search_alg(self, args):
        if args.search_alg not in LightGBMRayTuneScript.SEARCH_ALG_MAP:
            raise NotImplementedError(
                f'The specified search algo {args.search_alg} is not supported yet.')
        search_alg_func = LightGBMRayTuneScript.SEARCH_ALG_MAP[args.search_alg]

        if args.search_alg == "BasicVariantGenerator":
            search_alg = search_alg_func(
                max_concurrent=args.max_concurrent_trials)
        else:
            search_alg = search_alg_func()
            search_alg = ConcurrencyLimiter(
                search_alg, max_concurrent=args.max_concurrent_trials)

        return search_alg

    # get scheduler
    # TODO: add arguments of scheduler for advanced usage.
    def get_scheduler(self, args):
        if args.scheduler not in LightGBMRayTuneScript.SCHEDULER_MAP:
            raise Exception(
                f'The specified scheduler {args.scheduler} is not supported yet.')
        else:
            scheduler_func = LightGBMRayTuneScript.SCHEDULER_MAP[args.scheduler]
        return scheduler_func()

    def run(self, args, logger, metrics_logger, unknown_args):
        """
        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.logger): a logger initialized for this script
            metrics_logger (common.metrics.MetricLogger): to report metrics for this script, already initialized for MLFlow
            unknown_args (list[str]): list of arguments not recognized during argparse
        """
        # record relevant parameters
        # check if num_threads are correct with metrics_logger.
        metrics_logger.log_parameters(
            num_threads=1
        )

        # parse args parameters, fixed parameters go to lgbm_params.
        tunable_params, fixed_params = process_raytune_parameters(args)

        logger.info(f'The fixed_params: {fixed_params}')
        logger.info(f'The tunable_params: {tunable_params}')

        # make sure the export_model argument exists
        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
            args.output_path = os.path.join(
                args.output_path, "best_result.csv")

        # I need a local copy of the function to make it available to ray tune training func.
        # TODO: find a better way for this. 
        def input_file_path(path):
            """ Argparse type to resolve input path as single file from directory.
            Given input path can be either a file, or a directory.
            If it's a directory, this returns the path to the unique file it contains.

            Args:
                path (str): either file or directory path
            
            Returns:
                str: path to file, or to unique file in directory
            """
            if os.path.isfile(path):
                logger.info(f"Found INPUT file {path}")
                return path
            if os.path.isdir(path):
                all_files = os.listdir(path)
                if not all_files:
                    raise Exception(f"Could not find any file in specified input directory {path}")
                if len(all_files) > 1:
                    raise Exception(f"Found multiple files in input file path {path}, use input_directory_path type instead.")
                logger.info(f"Found INPUT directory {path}, selecting unique file {all_files[0]}")
                return os.path.join(path, all_files[0])
            
            logger.critical(f"Provided INPUT path {path} is neither a directory or a file???")
            return path

        # define training function
        def training_function(config, train_data_path, test_data_path):

            # get the tuned parameters from config
            fixed_params.update(config)
            # remove mode
            del fixed_params['mode']
            # set the num_threads to the number of cores.
            fixed_params['num_threads'] = os.cpu_count()
            logger.info(f'The updated config parameters {fixed_params}')

            # logging the train/test_data_path
            logger.info(f'The train_data_path is {train_data_path}')
            logger.info(f'The test_data_path is {test_data_path}')

            # retrive file from the directory path
            train_file = input_file_path(train_data_path)
            test_file = input_file_path(test_data_path)

            train_data_exist = exists(train_data_path)
            if train_data_exist == False:
                raise Exception(
                    f"The specified path {train_data_path} does not exist.")

            train_data = lightgbm.Dataset(train_file, params=fixed_params)
            test_data = lightgbm.Dataset(test_file, params=fixed_params)

            gbm = lightgbm.train(
                config,
                train_data,
                valid_sets=[test_data],
                valid_names=[valid_name],
                callbacks=[
                    TuneReportCheckpointCallback(
                        {report_metric: report_metric}
                    ),
                ],
            )

        # add must-have non-lgbm-specific arguments in the basic config.
        config = {"metric": args.metric,
                  "mode": args.mode}
        # update config with tunable parameters
        config.update(tunable_params)

        # logging the configs
        logger.info(f'The tune configs are: {config}')

        valid_name = 'eval'
        report_metric = valid_name + '-' + args.metric

        # instantiate and setup the search agl
        search_alg = self.get_search_alg(args)

        scheduler = self.get_scheduler(args)
        analysis = tune.run(
            partial(training_function,
                    train_data_path=args.train,
                    test_data_path=args.test,
                    ),
            metric=report_metric,
            mode=args.mode,
            config=config,
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=args.num_samples,
            time_budget_s=args.time_budget,
            resources_per_trial={"cpu": args.cpus_per_trial, "gpu": 0},
            raise_on_failed_trial=False
        )

        logger.info(f'Best hyperparameters found were: {analysis.best_config}')
        # save the best results to csv file.
        if args.output_path:
            logger.info(f"Writing best result in {args.output_path}")
            best_df = analysis.best_result_df
            best_df.to_csv(args.output_path, index=False)


def get_arg_parser(parser=None):
    """
    To ensure compatibility with shrike unit tests
    """
    return LightGBMRayTuneScript.get_arg_parser(parser)


def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    LightGBMRayTuneScript.main(cli_args)


if __name__ == "__main__":
    main()
