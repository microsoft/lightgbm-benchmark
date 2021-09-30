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

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.metrics import MetricsLogger
from common.io import input_file_path
from shrike.pipeline.aml_connect import add_cli_args, azureml_connect_cli
from azureml.core import Experiment
from azureml.pipeline.core import PipelineRun


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

    group_aml = parser.add_argument_group("AzureML Connect")
    add_cli_args(group_aml)

    group_exp = parser.add_argument_group("AzureML Experiment")
    group_exp.add_argument("--experiment-id", dest="experiment_id",
        required=True, type=str)
    group_exp.add_argument("--benchmark-id", dest="benchmark_id",
        required=True, type=str)

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

    return parser


class AnalysisEngine():
    """
    Class to run the analysis of multiple AzureML runs
    and generate a benchmark report
    """
    def __init__(self):
        """ Constructor """
        # list to store lines of data obtained from AzureML runs
        self.benchmark_data = []

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

    def fetch_benchmark_data(self, azureml_ws, experiment_id, **filters):
        """ Gets the data from fetching AzureML runs with a given set of filters """
        self.logger.info("Fetching Experiment")
        experiment = Experiment(workspace=azureml_ws, name=experiment_id)

        self.logger.info("Fetching Benchmark Runs")
        benchmark_runs = experiment.get_runs(tags=filters, include_children=True)

        # iterate through runs to get all the data we need for the analysis
        self.logger.info("Iterating through runs")
        self.benchmark_data = [] # reset internal list

        for run in benchmark_runs:
            benchmark_data_entry = {}
            run_tags = dict(run.tags)

            # get the model name from a tag
            benchmark_data_entry['model_name'] = run_tags.get('benchmark_model', None)
            
            # parse the model name to get number of trees and leaves
            if benchmark_data_entry['model_name']:
                model_pattern = r"synthetic-regression-([0-9]+)cols-model-([0-9]+)trees-([0-9]+)leaves"
                model_matched = re.match(model_pattern, benchmark_data_entry['model_name'])
                if model_matched:
                    benchmark_data_entry['model_trees'] = int(model_matched.group(2))
                    benchmark_data_entry['model_leaves'] = int(model_matched.group(3))

            # get the dataset name from a tag
            benchmark_data_entry['dataset_name'] = run_tags.get('benchmark_dataset', None)

            # parse the dataset name for numbers of columns
            if benchmark_data_entry['dataset_name']:
                dataset_pattern = r"synthetic-regression-([0-9]+)cols-inference"
                dataset_matched = re.match(dataset_pattern, benchmark_data_entry['dataset_name'])
                if dataset_matched:
                    benchmark_data_entry['dataset_columns'] = int(dataset_matched.group(1))

            # get framework (variant) info
            benchmark_data_entry['framework'] = run_tags['framework']
            benchmark_data_entry['framework_version'] = run_tags.get('framework_version', None)
            benchmark_data_entry['framework_build'] = run_tags.get('framework_build', None)

            # get all existing metrics in module
            run_metrics = run.get_metrics()
            for metric in run_metrics.keys():
                benchmark_data_entry[metric] = run_metrics[metric]

            benchmark_data_entry['system_cpu_count'] = run_tags['cpu_count']
            benchmark_data_entry['system_os'] = run_tags['system']
            benchmark_data_entry['system_machine'] = run_tags['machine']

            self.benchmark_data.append(benchmark_data_entry)

        return self.benchmark_data


    def report_inferencing(self, output_path):
        # TODO: replace by pandas data operations
        variants = set()
        configs = set()
        metrics = {}

        for entry in self.benchmark_data:
            variant_key = (entry['framework'],entry['framework_version'],entry['framework_build'])
            variants.add(variant_key)
            if variant_key not in metrics:
                metrics[variant_key] = {}

            config_key = (entry['model_trees'], entry['model_leaves'], entry['dataset_columns'])
            configs.add(config_key)

            metrics[variant_key][config_key] = entry['time_inferencing']
        
        with open(os.path.join(self.templates_dir, "inferencing.md"), "r") as i_file:
            template_str = i_file.read()
        
        template_obj = Template(template_str)

        variants_list = sorted(list(variants))
        configs_list = sorted(list(configs))
        metrics_list = [
            [
                metrics[v][c] for v in variants_list
            ] for c in configs_list
        ]

        rendered_report = template_obj.render(
            variants=variants_list,
            configs=sorted(list(configs)),
            metrics=metrics_list
        )

        if output_path:
            with open(output_path, "w") as o_file:
                o_file.write(rendered_report)



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
            # use helper to connect to AzureML
            print("Connecting to AzureML...")
            ws = azureml_connect_cli(args)

            # querying runs for specific filters
            analysis_engine.fetch_benchmark_data(
                azureml_ws=ws,
                experiment_id=args.experiment_id,
                benchmark_name=args.benchmark_id,
                task='score'
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
