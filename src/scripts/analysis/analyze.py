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

    group_exp = parser.add_argument_group("MLFlow Experiment")
    group_exp.add_argument("--experiment-id", dest="experiment_id",
        required=True, type=str)
    group_exp.add_argument("--benchmark-id", dest="benchmark_id",
        required=True, type=str)
    group_aml = parser.add_argument("--mlflow-target", dest="mlflow_target", required=False, type=str, choices=['azureml', 'local'], default='local')

    group_data = parser.add_argument_group("Data operations")
    group_data.add_argument("--data-load", dest="data_load",
        required=False, default=None, type=str, help="path to file export benchmark data"
    )
    group_data.add_argument("--data-save", dest="data_save",
        required=False, default=None, type=str, help="path to file read benchmark data"
    )

    group_analysis = parser.add_argument_group("Analysis parameters")
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

    group_aml = parser.add_argument_group("AzureML Connect (if using --mlflow-target azureml)")
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
        with open(file_path, "r") as i_file:
            for line in i_file:
                self.benchmark_data.append(json.loads(line))

    def save_benchmark_data(self, file_path):
        """ Saves the fetched benchmark data into a file """
        # create output directory
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # write data line by line in json
        with open(file_path, "w") as o_file:
            for entry in self.benchmark_data:
                o_file.write(json.dumps(entry))
                o_file.write("\n")

    def fetch_benchmark_data(self, experiment_id, filter_string):
        """ Gets the data from fetching AzureML runs with a given set of filters """
        self.logger.info("Fetching Experiment")
        mlflow.set_experiment(experiment_id)

        self.logger.info("Fetching Benchmark Runs")
        # NOTE: returns a pandas dataframe
        self.benchmark_data = mlflow.search_runs(
            filter_string=filter_string
        )

        # extract variants
        self.variants = self.benchmark_data[
            [
                # select variant specific columns/tags
                'tags.variant_index',
                'tags.framework',
                'tags.framework_version',
                'tags.framework_build',
                'tags.framework_build_os'
            ]
        ].drop_duplicates().set_index('tags.variant_index') # reduce to unique rows

        print("*** VARIANTS ***")
        print(self.variants.to_markdown())

        # extract all model information
        if 'tags.benchmark_model' in self.benchmark_data.columns:
            models = self.benchmark_data[['tags.benchmark_model']].drop_duplicates()
            model_info = models['tags.benchmark_model'].str.extract(r"model-([a-zA-Z0-9]+)-([a-zA-Z0-9]+)-([0-9]+)cols-([0-9]+)trees-([0-9]+)leaves")
            model_info.columns = ['model_origin', 'model_task', 'model_columns', 'model_trees', 'model_leaves']
            self.models = models.join(model_info)
            print("*** MODELS ***")
            print(self.models.to_markdown())

        # extract all dataset information
        if 'tags.benchmark_dataset' in self.benchmark_data.columns:
            datasets = self.benchmark_data[['tags.benchmark_dataset']].drop_duplicates()
            dataset_info = datasets['tags.benchmark_dataset'].str.extract(r"data-([a-zA-Z0-9]+)-([a-zA-Z0-9]+)-([0-9]+)cols-([0-9]+)samples-([a-zA-Z0-9]+)")
            dataset_info.columns = ['dataset_origin', 'dataset_task', 'dataset_columns', 'dataset_samples', 'dataset_benchmark_task']
            self.datasets = datasets.join(dataset_info)
            print("*** DATASETS ***")
            print(self.datasets.to_markdown())

        return self.benchmark_data


    def report_inferencing(self, output_path):
        """ Uses fetched or load data to produce a reporting for inferencing tasks. """
        # set of all tasks on which frameworks are evaluated
        #tasks = self.benchmark_data.join(
        #    self.datasets,
        #    on=['tags.benchmark_dataset']
        #)

        metrics = self.benchmark_data.pivot(
            index=['tags.benchmark_model', 'tags.benchmark_dataset'],
            columns=['tags.variant_index'],
            values=['metrics.time_inferencing'],
            #aggfunc=np.sum
        )
        print(metrics.to_markdown())
        raise NotImplementedError()


        # gets data we want to report on from the fetched data
        for entry in self.benchmark_data:
            # key to store metrics for this variant
            variant_key = (entry['framework'],entry['framework_version'],entry['framework_build'])
            variants.add(variant_key)
            
            # if not already, create a dict for metrics
            if variant_key not in metrics:
                metrics[variant_key] = {}

            # key to store metrics for this task
            task_key = (entry['model_trees'], entry['model_leaves'], entry['dataset_columns'])
            tasks.add(task_key)

            # store the metric at the right location in metrics dict
            metrics[variant_key][task_key] = entry['metrics'].get('time_inferencing', None) / entry['dataset_samples'] * 1000000 # mu secs per query
        
        # load the jinja template from local files
        with open(os.path.join(self.templates_dir, "inferencing.md"), "r") as i_file:
            template_str = i_file.read()

        # use jinja Template
        template_obj = Template(template_str)

        # prepare data for the template rendering
        variants_list = sorted(list(variants)) # sorting variants in increasing order (tuple)
        tasks_list = sorted(list(tasks)) # sorting tasks in increasing order (tuple)

        # create a table of metrics
        metrics_list = [
            [
                metrics[v][c] for v in variants_list
            ] for c in tasks_list
        ]

        # render the template
        rendered_report = template_obj.render(
            variants=variants_list,
            tasks=sorted(list(tasks)),
            metrics=metrics_list
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
