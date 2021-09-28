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
    def __init__(self, ws):
        self.ws = ws
        self.benchmark_data = []

    def get_benchmark_data(self, experiment_id, **filters):
        print("*** Fetching Experiment")
        experiment = Experiment(workspace=self.ws, name=experiment_id)

        print("*** Fetching Benchmark Runs")
        benchmark_runs = experiment.get_runs(tags=filters, include_children=True)

        # iterate through runs to get all the data we need for the analysis
        print("*** Iterating through runs")

        self.benchmark_data = [] # list of dict for pandas
        for run in benchmark_runs:
            benchmark_data_entry = {}
            run_tags = dict(run.tags)

            benchmark_data_entry['model_name'] = run_tags.get('benchmark_model', None)
            if benchmark_data_entry['model_name']:
                model_pattern = r"synthetic-regr-([0-9]+)cols-model-([0-9]+)trees"
                model_matched = re.match(model_pattern, benchmark_data_entry['model_name'])
                if model_matched:
                    benchmark_data_entry['model_trees'] = int(model_matched.group(2))

            benchmark_data_entry['dataset_name'] = run_tags.get('benchmark_dataset', None)
            if benchmark_data_entry['dataset_name']:
                dataset_pattern = r"synthetic-regr-([0-9]+)cols-inference-([0-9]+)rows"
                dataset_matched = re.match(dataset_pattern, benchmark_data_entry['dataset_name'])
                if dataset_matched:
                    benchmark_data_entry['dataset_columns'] = int(dataset_matched.group(1))
                    benchmark_data_entry['dataset_rows'] = int(dataset_matched.group(2))

            benchmark_data_entry['framework'] = run_tags['framework']
            benchmark_data_entry['framework_version'] = run_tags.get('framework_version', None)
            benchmark_data_entry['framework_build'] = run_tags.get('framework_build', None)

            run_metrics = run.get_metrics()
            for metric in run_metrics.keys():
                benchmark_data_entry[metric] = run_metrics[metric]

            benchmark_data_entry['system_cpu_count'] = run_tags['cpu_count']
            benchmark_data_entry['system_os'] = run_tags['system']
            benchmark_data_entry['system_machine'] = run_tags['machine']

            print(benchmark_data_entry)
            self.benchmark_data.append(benchmark_data_entry)

        return self.benchmark_data


    def analyze_inferencing(self, experiment_id, benchmark_id, output_path):
        # querying runs for specific filters
        benchmark_data = self.get_benchmark_data(
            experiment_id=experiment_id,
            benchmark_name=benchmark_id,
            task='score'
        )



def run(args, unknown_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    # get logger for general outputs
    logger = logging.getLogger()

    # use helper to connect to AzureML
    print("Connecting to AzureML...")
    ws = azureml_connect_cli(args)

    analysis_engine = AnalysisEngine(ws)

    if args.template == 'inferencing':
        analysis_engine.analyze_inferencing(args.experiment_id, args.benchmark_id, args.output)
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
