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
from distutils.util import strtobool

import ray
from ray import tune
from lightgbm_ray.tune import TuneReportCheckpointCallback
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from flaml import BlendSearch
from ray.tune.schedulers import FIFOScheduler
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.suggest.basic_variant import BasicVariantGenerator

import lightgbm_ray

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

# TODO: refactor this in a common function shared.

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
        'search_alg',
        'scheduler',
        'mode',
        'num_samples',
        'time_budget',
        'max_concurrent_trials',
        'cpus_per_actor',
        'output_path',
        'log_path',
        'metrics_driver'
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


class LightGBMRayTuneDistributedScript(RayScript):

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
        group_i.add_argument("--train_data_format",
                             required=False, type=str, choices=['CSV', 'TSV', 'PARQUET', 'PETAFORM'], default='CSV', help="type of input train data (CSV, PARQUET, PETAFORM), default CSV")
        group_i.add_argument("--test",
                             required=True, type=str, help="Testing data location (file path)")
        group_i.add_argument("--test_data_format",
                             required=False, type=str, choices=['CSV', 'TSV', 'PARQUET', 'PETAFORM'], default='CSV', help="type of input test data (CSV, PARQUET, PETAFORM), default using same as train")
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
        group_o.add_argument("--log_path", required=True,
                             type=str, help="the path for ray tune log files.")

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
            "--cpus_per_actor", required=False, default=1, type=int)
        group_raytune.add_argument(
            "--max_concurrent_trials", required=False, default=0, type=int)

        group_lgbm = parser.add_argument_group(
            f"LightGBM/Ray runsettings [{__name__}:{cls.__name__}]")
        group_lgbm.add_argument("--lightgbm_ray_actors", required=False, default=None,
                                type=int, help="number of actors (default: count available nodes, or 1)")
        group_lgbm.add_argument("--ray_data_distributed", required=False, default=True, type=strtobool,
                                help="is data pre-partitioned (True) or should ray distribute it (False)")

        return parser

    # get the search algorithm
    # TODO: add arguments of scheduler for advanced usage.
    def get_search_alg(self, args):
        if args.search_alg not in LightGBMRayTuneDistributedScript.SEARCH_ALG_MAP:
            raise NotImplementedError(
                f'The specified search algo {args.search_alg} is not supported yet.')
        search_alg_func = LightGBMRayTuneDistributedScript.SEARCH_ALG_MAP[args.search_alg]

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
        if args.scheduler not in LightGBMRayTuneDistributedScript.SCHEDULER_MAP:
            raise Exception(
                f'The specified scheduler {args.scheduler} is not supported yet.')
        else:
            scheduler_func = LightGBMRayTuneDistributedScript.SCHEDULER_MAP[args.scheduler]
        return scheduler_func()

    def run(self, args, logger, metrics_logger, unknown_args):
        """
        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.logger): a logger initialized for this script
            metrics_logger (common.metrics.MetricLogger): to report metrics for this script, already initialized for MLFlow
            unknown_args (list[str]): list of arguments not recognized during argparse
        """
        # parse args parameters, fixed parameters go to lgbm_params.
        tunable_params, fixed_params = process_raytune_parameters(args)

        # log tunable metrics to show in the component
        metrics_logger.log_parameters(
            **tunable_params
        )

        logger.info(f'The fixed_params: {fixed_params}')
        logger.info(f'The tunable_params: {tunable_params}')

        # make sure the export_model argument exists
        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
            args.output_path = os.path.join(
                args.output_path, "best_result.csv")

        # I need a local copy of the function to make it available to ray tune training func.
        # TODO: find a better way for this.
        def get_all_files(path, fail_on_unknown_type=False):
            """ Scans some input path and returns a list of files.

            Args:
                path (str): either a file, or directory path
                fail_on_unknown_type (bool): fails if path is neither a file or a dir?
            Returns:
                List[str]: list of paths contained in path
            """
            # check the existence of the path
            if exists(path) == False:
                raise Exception(f"The specified path {path} does not exist.")

            # if input path is already a file, return as list
            if os.path.isfile(path):
                print(f"Found INPUT file {path}")
                return [path]

            # if input path is a directory, list all files and return
            if os.path.isdir(path):
                print(f"Found INPUT directory {path}")
                all_files = [os.path.join(path, entry)
                             for entry in os.listdir(path)]
                print(f"Found INPUT files {all_files}")
                if not all_files:
                    raise Exception(
                        f"Could not find any file in specified input directory {path}")
                return all_files

            if fail_on_unknown_type:
                raise FileNotFoundError(
                    f"Provided INPUT path {path} is neither a directory or a file???")
            else:
                print(
                    f"Provided INPUT path {path} is neither a directory or a file???")

            return path

        # define training function
        def training_function(config, train_data_path, test_data_path):

            # get the tuned parameters from config
            fixed_params.update(config)
            # remove mode
            del fixed_params['mode']
            print(f'The updated config parameters {fixed_params}')

            ### DATA LOADING (train) ###

            print(f"Loading data for training")
            train_paths = get_all_files(train_data_path)

            # detect sharding strategy exceptions
            if len(train_paths) != num_actors:
                # NOTE: when this happens, data is read entirely on head node
                # then ray distributes it to cluster nodes (distributed=False)
                print(
                    f"Found {len(train_paths)} training files != num_actors={num_actors} => forcing distributed=False")
                args.ray_data_distributed = False
            else:
                # NOTE: in this situation, each node will read one shard of data which means 1 file
                # https://github.com/ray-project/xgboost_ray/blob/master/xgboost_ray/matrix.py#L605
                print(
                    f"Found {len(train_paths)} training files == num_actors={num_actors} => using distributed from args.ray_data_distributed={args.ray_data_distributed}")

            if args.train_data_format == 'TSV':
                train_data = lightgbm_ray.RayDMatrix(
                    train_paths,
                    label=args.label_column,  # Will select this column as the label
                    filetype=lightgbm_ray.RayFileType.CSV,
                    distributed=args.ray_data_distributed,
                    sharding=lightgbm_ray.RayShardingMode.INTERLEAVED,
                    sep='\t'
                )
            else:
                train_data_format = getattr(
                    lightgbm_ray.RayFileType, args.train_data_format)
                train_data = lightgbm_ray.RayDMatrix(
                    train_paths,
                    label=args.label_column,  # Will select this column as the label
                    filetype=train_data_format,
                    distributed=args.ray_data_distributed,
                    sharding=lightgbm_ray.RayShardingMode.INTERLEAVED
                )

            # ### DATA LOADING (validation) ###

            print(f"Loading data for validation")
            # validation_paths = list(sorted(glob.glob(os.path.join(test_data_path, "*"))))
            validation_paths = get_all_files(test_data_path)
            print(f"Found {len(validation_paths)} validation files")

            # NOTE: it seems we need to have as many test sets as actors
            # args.lightgbm_ray_actors or self.available_nodes
            required_validation_sets = num_actors
            if len(validation_paths) == 1 and required_validation_sets > 1:
                print(
                    f"Creating artificial {required_validation_sets} test sets")
                validation_paths = [validation_paths[0]
                                    for _ in range(required_validation_sets)]

            # load the validation data into RayDMatrix
            if args.test_data_format == 'TSV':
                test_data = lightgbm_ray.RayDMatrix(
                    validation_paths,
                    label=args.label_column,  # Will select this column as the label
                    filetype=lightgbm_ray.RayFileType.CSV,
                    distributed=True,
                    sep='\t'
                )
            else:
                val_data_format = getattr(
                    lightgbm_ray.RayFileType, args.test_data_format)
                test_data = lightgbm_ray.RayDMatrix(
                    validation_paths,
                    label=args.label_column,  # Will select this column as the label
                    filetype=val_data_format,
                    distributed=True,
                )

            ### TRAINING ###

            print(f"Training LightGBM with parameters: {fixed_params}")
            evals_result = {}
            additional_results = {}

            lightgbm_ray.train(
                fixed_params,
                train_data,
                # this is required, num_iterations in lgbm_params will be discarded anyway
                num_boost_round=fixed_params['num_iterations'],
                evals_result=evals_result,
                additional_results=additional_results,
                valid_sets=[test_data],
                valid_names=[valid_name],
                verbose_eval=True,
                ray_params=ray_params,
                callbacks=[
                    TuneReportCheckpointCallback(
                        {report_metric: report_metric}
                    ),
                ],
            )
        # TODO: log best trial metrics with time.

        # add must-have non-lgbm-specific arguments in the basic config.
        config = {"metric": args.metric,
                  "mode": args.mode}
        # update config with tunable parameters
        config.update(tunable_params)

        # logging the configs
        logger.info(f'The tune configs are: {config}')

        valid_name = 'valid_0'
        # TODO: use metric name like "node_0/valid_0.metric" to align with other components.
        report_metric = valid_name + '-' + args.metric

        # instantiate and setup the search agl
        search_alg = self.get_search_alg(args)

        scheduler = self.get_scheduler(args)

        # distributed training setup
        num_actors = args.lightgbm_ray_actors or self.available_nodes
        num_cpus_per_actor = args.cpus_per_actor

        ray_params = lightgbm_ray.RayParams(num_actors=num_actors,
                                            cpus_per_actor=num_cpus_per_actor
                                            )
        # understand the resource placement stragety.
        from ray.tune import PlacementGroupFactory
        tune_resources = ray_params.get_tune_resources()
        if isinstance(tune_resources, PlacementGroupFactory):
            tune_resources._strategy = 'SPREAD'
            print("bundles=", tune_resources._bundles, "strategy=", tune_resources._strategy)
        else:
            print("Not PlacementGroupFactory.")

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
            resources_per_trial=tune_resources,
            raise_on_failed_trial=False,
            local_dir ='./outputs',
            sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
            )
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
    return LightGBMRayTuneDistributedScript.get_arg_parser(parser)


def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    LightGBMRayTuneDistributedScript.main(cli_args)


if __name__ == "__main__":
    main()
