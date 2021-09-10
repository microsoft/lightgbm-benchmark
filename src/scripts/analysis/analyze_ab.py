import os

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license. 

"""
TreeLite/Python inferencing script
"""
import os
import sys
import argparse
import logging
import numpy
from distutils.util import strtobool
import pandas as pd
import treelite, treelite_runtime

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
    group_exp.add_argument("--aml-experiment-name", dest="aml_experiment_name",
        required=True, type=str)
    group_exp.add_argument("--aml-run-id", dest="aml_run_id",
        required=True, type=str)

    group_general = parser.add_argument_group("Analysis parameters")
    group_general.add_argument(
        "--output",
        required=False,
        default=None,
        type=str,
        help="Path to write report in markdown"
    )
    group_general.add_argument(
        "--verbose",
        required=False,
        default=False,
        type=strtobool,  # use this for bool args, do not use action_store=True
        help="set True to show DEBUG logs",
    )

    return parser


def analyze_ab_pipeline_run(args, experiment, pipeline_run):
    """Generated a markdown report on this experiment.
    NOTE: will wait for pipeline to finish.
    
    Args:
        args (argparse.Namespace): cli arguments
        experiment (azure.ml.core.Experiment) : AzureML Experiment object
        pipeline_run (azureml.core.run.PipelineRun) : the pipeline run object from our canary (once completed)
    """
    # report header
    benchmark_report = [
        "# LightGBM master vs custom report",
        "",
        f"- Pipeline name: {pipeline_run.name}",
        #f"- Pipeline Run Url: {pipeline_run.get_portal_url()}",
        #f"- Run Id: {pipeline_run.id}",
        f"- Tags: `{str(pipeline_run.get_tags())}`",
        f"- Properties: `{str(pipeline_run.get_properties())}`",
        f"- Status: {pipeline_run.get_status()}",
    ]

    # report the pipeline metrics
    benchmark_report.append("")
    benchmark_report.append("## Pipeline Level Metrics")
    benchmark_report.append("")

    # custom function to report metric tables
    def _write_metrics_report(report, metrics_dict):
        flat_metrics_report = [
            "| Metric | &nbsp; | Value(s) |",
            "| :-- | :-- | :-- |"
        ]
        for key,value in metrics_dict.items():
            #print(f"metric {key}={value}")
            if isinstance(value, dict):
                for metric_key, metric_value in value.items():
                    flat_metrics_report.append(f"| {key} | key={metric_key} | {metric_value} |") # TODO
            elif isinstance(value, list):
                for metric_index, metric_value in enumerate(value):
                    flat_metrics_report.append(f"| {key} | step={metric_index+1} | {metric_value} |") # TODO
            else:
                flat_metrics_report.append(f"| {key} | `{str(value)}` |")

        if not metrics_dict:
            report.append("(no pipeline level metrics available)")
        else:
            report.extend(flat_metrics_report)

    # using it once at pipeline level
    _write_metrics_report(benchmark_report, pipeline_run.get_metrics())

    # adding table of pipeline steps
    benchmark_report.append("")
    benchmark_report.append("## Pipeline Steps")
    benchmark_report.append("")


    metrics_pivot_table = {}
    steps_names = []
    for step in pipeline_run.get_steps():
        steps_names.append(step.name)
        #metrics_pivot_table[step.name] = step.get_metrics()
        metrics_reported = step.get_metrics()
        for key in metrics_reported:
            if key not in metrics_pivot_table:
                metrics_pivot_table[key] = {}
            metrics_pivot_table[key][step.name] = metrics_reported[key]

    print(metrics_pivot_table)

    # this is a simple table with direct links to the Metrics panels
    benchmark_report.append("| " + " | ".join(["Metric"] + steps_names) + " |")
    benchmark_report.append("| " + " | ".join([":--"] * (len(steps_names)+1)) + " |")

    for key in metrics_pivot_table:
        benchmark_report.append(
            "| " +
            " | ".join(
                [ key ] +
                [
                    "{:2f}".format((metrics_pivot_table[key][name]))
                    for name in steps_names
                ]
            ) +
            " |"
        )

        #benchmark_report.append(f"| [{step.name}]({step.get_portal_url()}) | {step.get_status()} | [{len(metrics_reported)} metrics available]({step.get_portal_url()}#metrics): {', '.join(list(metrics_reported.keys()))} |")

    # wrap it up
    benchmark_report.append("")
    markdown_report = "\n".join(benchmark_report)
    print(markdown_report)

    # save if provided with a path
    if args.output:
        with open(args.output, "w") as o_file:
            o_file.write(markdown_report)


def run(args, unknown_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        unknown_args (list[str]): list of arguments not known
    """
    # get logger for general outputs
    logger = logging.getLogger()

    # use helper to connect to AzureML
    ws = azureml_connect_cli(args)

    exp = Experiment(ws, args.aml_experiment_name)

    pipeline_run = PipelineRun(
        experiment=exp,
        run_id=args.aml_run_id,
    )

    analyze_ab_pipeline_run(args, exp, pipeline_run)

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
