# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license. 

"""
TreeLite/Python inferencing script
"""
import os
import sys
import re
import json
import argparse
import logging
from distutils.util import strtobool
from jinja2 import Template
import mlflow
import pandas as pd
import numpy as np

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if COMMON_ROOT not in sys.path:
    logging.debug(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.metrics import MetricsLogger
from common.io import input_file_path
from shrike.pipeline.aml_connect import add_cli_args, azureml_connect_cli


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

    group_exp = parser.add_argument_group(f"MLFlow Experiment [{__name__}:{cls.__name__}]")
    group_exp.add_argument("--experiment-id", dest="experiment_id",
        required=True, type=str)
    group_exp.add_argument("--benchmark-id", dest="benchmark_id",
        required=True, type=str)
    group_aml = parser.add_argument("--mlflow-target", dest="mlflow_target", required=False, type=str, choices=['azureml', 'local'], default='local')

    group_data = parser.add_argument_group(f"Data operations [{__name__}:{cls.__name__}]")
    group_data.add_argument("--data-load", dest="data_load",
        required=False, default=None, type=str, help="path to file export benchmark data"
    )
    group_data.add_argument("--data-save", dest="data_save",
        required=False, default=None, type=str, help="path to file read benchmark data"
    )

    group_analysis = parser.add_argument_group(f"Analysis parameters [{__name__}:{cls.__name__}]")
    group_analysis.add_argument(
        "--template",
        required=True,
        type=str,
        choices=['inferencing'],
        help="which analysis template to use"
    )
    group_analysis.add_argument(
        "--output",
        required=False,
        default=None,
        type=str,
        help="Path to write report in markdown"
    )
    group_analysis.add_argument(
        "--verbose",
        required=False,
        default=False,
        type=strtobool,  # use this for bool args, do not use action_store=True
        help="set True to show DEBUG logs",
    )

    group_aml = parser.add_argument_group(f"AzureML Connect (if using --mlflow-target azureml) [{__name__}:{cls.__name__}]")
    add_cli_args(group_aml)

    return parser


class AnalysisEngine():
    """
    Class to run the analysis of multiple AzureML runs
    and generate a benchmark report
    """
    def __init__(self):
        """ Constructor """
        # list to store lines of data obtained from AzureML runs
        self.benchmark_data = None
        self.variants = None
        self.models = None
        self.datasets = None

        # location of the jinja templates to generate reports
        self.templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))

        self.logger = logging.getLogger(__name__)

    def load_benchmark_data(self, file_path):
        """ Loads the previously saved benchmark data (and skip fetching) """
        # reset internal data list
        self.benchmark_data = []

        # read line by line json
        self.benchmark_data = pd.read_json(file_path)

    def save_benchmark_data(self, file_path):
        """ Saves the fetched benchmark data into a file """
        # create output directory
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # write data line by line in json
        self.benchmark_data.to_json(file_path)

    def fetch_benchmark_data(self, experiment_id, filter_string):
        """ Gets the data from fetching AzureML runs with a given set of filters """
        self.logger.info("Fetching Experiment")
        mlflow.set_experiment(experiment_id)

        self.logger.info("Fetching Benchmark Runs")
        # NOTE: returns a pandas dataframe
        self.benchmark_data = mlflow.search_runs(
            filter_string=filter_string
        )

        # extract all model information if present
        if 'tags.benchmark_model' in self.benchmark_data.columns:
            models = self.benchmark_data[['tags.benchmark_model']].drop_duplicates()
            model_info = models['tags.benchmark_model'].str.extract(r"model-([a-zA-Z0-9]+)-([a-zA-Z0-9]+)-([0-9]+)cols-([0-9]+)trees-([0-9]+)leaves")
            model_info.columns = ['model_origin', 'model_task', 'model_columns', 'model_trees', 'model_leaves']
            models = models.join(model_info)
            self.benchmark_data = pd.merge(self.benchmark_data, models, how='left', on='tags.benchmark_model')
            #print("*** MODELS ***")
            #print(models.to_markdown())

        # extract all dataset information if present
        if 'tags.benchmark_dataset' in self.benchmark_data.columns:
            datasets = self.benchmark_data[['tags.benchmark_dataset']].drop_duplicates()
            dataset_info = datasets['tags.benchmark_dataset'].str.extract(r"data-([a-zA-Z0-9]+)-([a-zA-Z0-9]+)-([0-9]+)cols-([0-9]+)samples-([a-zA-Z0-9]+)")
            dataset_info.columns = ['dataset_origin', 'dataset_task', 'dataset_columns', 'dataset_samples', 'dataset_benchmark_task']
            datasets = datasets.join(dataset_info)
            self.benchmark_data = pd.merge(self.benchmark_data, datasets, how='left', on='tags.benchmark_dataset')
            #print("*** DATASETS ***")
            #print(datasets.to_markdown())

        return self.benchmark_data


    def report_inferencing(self, output_path):
        """ Uses fetched or load data to produce a reporting for inferencing tasks. """
        # create variant readable id
        self.benchmark_data['variant_id'] = self.benchmark_data['tags.framework'] + "#" + self.benchmark_data['tags.variant_index']

        # extract variants
        variants = self.benchmark_data[
            [
                # select variant specific columns/tags
                'variant_id',
                'tags.variant_index',
                'tags.framework',
                'tags.framework_version',
                'tags.framework_build',
                'tags.cpu_count',
                'params.num_threads',
                'tags.machine',
                'tags.system'
            ]
        ].drop_duplicates().set_index('variant_id').sort_values(by='tags.variant_index')

        # get a list of variant_id ordered by tags.variant_index
        variant_indices = (
            self.benchmark_data[['tags.variant_index', 'variant_id']]
            .drop_duplicates()
            .set_index('tags.variant_index')
            .to_dict()
        )['variant_id']
        variant_indices_sorted_keys = sorted(list(variant_indices.keys()))
        variant_indices_sorted = [ variant_indices[k] for k in variant_indices_sorted_keys ]

        variants.columns = ['index', 'framework', 'version', 'build', 'cpu count', 'num threads', 'machine', 'system']
        #variants = variants.transpose()

        # reduce time_inferencing to predict time per request, in micro seconds
        self.benchmark_data['avg_predict_time_usecs'] = self.benchmark_data['metrics.time_inferencing'].astype(float) / self.benchmark_data['dataset_samples'].astype(int) * 1000000

        # create a readable name for each task configuration
        self.benchmark_data['inferencing task config'] = (
            self.benchmark_data['model_trees'] + " trees<br/>"
            + self.benchmark_data['model_leaves'] + " leaves<br/>"
            + self.benchmark_data['model_columns'] + " cols"
        )

        # pivot metrics table
        metrics = self.benchmark_data.pivot(
            index=['inferencing task config'],
            columns=['variant_id'],
            values=['avg_predict_time_usecs']
        )
        # rename columns to have only variant_id
        metrics.columns = [ col[1] for col in metrics.columns ]
        metrics = metrics[variant_indices_sorted] # order columns by increasing tags.variant_index

        percentile_metrics_reports = []

        for variant_id in variant_indices_sorted:
            percentile_metrics_values = (
                self.benchmark_data.loc[self.benchmark_data['variant_id'] == variant_id][[
                    'inferencing task config',
                    'variant_id',
                    'metrics.batch_time_inferencing_p50_usecs',
                    'metrics.batch_time_inferencing_p90_usecs',
                    'metrics.batch_time_inferencing_p99_usecs'
                ]]
            ).dropna()
            
            if len(percentile_metrics_values) == 0:
                continue

            percentile_metrics = (
                percentile_metrics_values.pivot(
                    index=['inferencing task config'],
                    columns=['variant_id'],
                    values=['metrics.batch_time_inferencing_p50_usecs', 'metrics.batch_time_inferencing_p90_usecs', 'metrics.batch_time_inferencing_p99_usecs']
                )
            )
            percentile_metrics.columns = [ col[0].lstrip("metrics.batch_time_inferencing_") for col in percentile_metrics.columns ]

            percentile_metrics_reports.append(
                {
                    'variant_id' : variant_id,
                    'report' : percentile_metrics.to_markdown()
                }
            )

        # load the jinja template from local files
        with open(os.path.join(self.templates_dir, "inferencing.md"), "r") as i_file:
            template_str = i_file.read()

        # use jinja Template
        template_obj = Template(template_str)

        # render the template
        rendered_report = template_obj.render(
            variants_table=variants.to_markdown(),
            metrics_table=metrics.to_markdown(),
            percentile_metrics_reports=percentile_metrics_reports
        )

        # save or print
        if output_path:
            with open(output_path, "w") as o_file:
                o_file.write(rendered_report)
        else:
            print(rendered_report)


def run(args, unknown_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    # get logger for general outputs
    logger = logging.getLogger()

    analysis_engine = AnalysisEngine()

    if args.template == 'inferencing':
        if args.data_load:
            analysis_engine.load_benchmark_data(args.data_load)
        else:
            if args.mlflow_target == 'azureml':
                # use helper to connect to AzureML
                print("Connecting to AzureML...")
                ws = azureml_connect_cli(args)
                mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
            elif args.mlflow_target == 'local':
                pass # nothing to do here
            else:
                raise NotImplementedError(f"--mlflow-target {args.mlflow_target} is not implemented (yet)")

            # querying runs for specific filters
            analysis_engine.fetch_benchmark_data(
                experiment_id=args.experiment_id,
                filter_string=f"tags.task = 'score' and tags.benchmark_name = '{args.benchmark_id}'"
            )
        
        if args.data_save:
            analysis_engine.save_benchmark_data(args.data_save)
        
        analysis_engine.report_inferencing(args.output)

    else:
        raise NotImplementedError(f"Analysis template {args.template} does not exist (yet?)")


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
