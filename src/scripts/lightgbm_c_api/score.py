# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/CLI inferencing script
"""
import os
import sys
import argparse
import logging
import re
import lightgbm
import numpy as np
from distutils.util import strtobool
from subprocess import PIPE
from subprocess import run as subprocess_run
from subprocess import TimeoutExpired

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if COMMON_ROOT not in sys.path:
    logging.info(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript
from common.io import input_file_path


class LightGBMCAPIInferecingScript(RunnableScript):
    def __init__(self):
        super().__init__(
            task = "score",
            framework = "lightgbm",
            framework_version = "C_API."+str(lightgbm.__version__)
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

        group_i = parser.add_argument_group("Input Data")
        group_i.add_argument("--lightgbm_lib_path",
            required=False, type=str, default=None, help="Path to lightgbm library (file path)")
        group_i.add_argument("--binaries_path",
            required=False, type=str, default=os.environ.get("LIGHTGBM_BENCHMARK_BINARIES_PATH", None), help="Path to lightgbm_predict (file path)")
        group_i.add_argument("--data",
            required=True, type=input_file_path, help="Inferencing data location (file path)")
        group_i.add_argument("--model",
            required=False, type=input_file_path, help="Exported model location (file path)")
        group_i.add_argument("--output",
            required=False, default=None, type=str, help="Inferencing output location (file path)")

        group_params = parser.add_argument_group("Scoring parameters")
        group_params.add_argument("--num_threads",
            required=False, default=1, type=int, help="number of threads")
        group_params.add_argument("--predict_disable_shape_check",
            required=False, default=False, type=strtobool, help="See LightGBM documentation")

        return parser


    def run(self, args, logger, metrics_logger, unknown_args):
        """Run script with arguments (the core of the component)

        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.getLogger() for this script)
            metrics_logger (common.metrics.MetricLogger)
            unknown_args (list[str]): list of arguments not recognized during argparse
        """
        # record relevant parameters
        metrics_logger.log_parameters(
            num_threads=args.num_threads
        )

        if args.output:
            # make sure the output argument exists
            os.makedirs(args.output, exist_ok=True)
            
            # and create your own file inside the output
            args.output = os.path.join(args.output, "predictions.txt")

        lightgbm_predict_path = os.path.join(os.path.abspath(args.binaries_path), "lightgbm_predict")

        # assemble a command for lightgbm cli
        lightgbm_predict_command = [
            lightgbm_predict_path,
            f"{args.model}",
            f"{args.data}",
            "verbosity=2",
            "num_threads=1",
            f"predict_disable_shape_check={bool(args.predict_disable_shape_check)}"
        ]

        if args.output:
            lightgbm_predict_command.append(f"output_result={args.output}")

        # create custom environment variables for the exec
        custom_env = os.environ.copy()
        if args.lightgbm_lib_path:
            logger.info(f"Adding to PATH: {args.lightgbm_lib_path}")
            custom_env["PATH"] = os.path.abspath(args.lightgbm_lib_path) + ":" + custom_env["PATH"]

        logger.info("Running command {}".format(" ".join(lightgbm_predict_command)))
        lightgbm_predict_call = subprocess_run(
            lightgbm_predict_command,
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
            check=False, # will not raise an exception if subprocess fails (so we capture with .returncode)
            timeout=None,
            env=custom_env
        )
        logger.info(f"stdout: {lightgbm_predict_call.stdout}")
        logger.info(f"stderr: {lightgbm_predict_call.stderr}")
        logger.info(f"return code: {lightgbm_predict_call.returncode}")

        if lightgbm_predict_call.returncode != 0:
            raise Exception("Return code != 0, see stderr above.")

        # now parsing executable logs for prediction per query time in ms
        time_inferencing_per_query = []
        for line in lightgbm_predict_call.stdout.split("\n"):
            if line.startswith("ROW"):
                row_pattern = r"ROW line=([0-9\.]+) label=([0-9\.e\-]+) null_elem=([0-9\.]+) prediction=([0-9\.e\-]+) time_usecs=([0-9\.e\-]+)"
                row_matched = re.match(row_pattern, line.strip())
                if row_matched:
                    time_inferencing_per_query.append(float(row_matched.group(5)))
                else:
                    logger.warning(f"log row {line} does not match expected pattern {row_pattern}")
            elif line.startswith("METRIC"):
                row_pattern = r"METRIC ([a-zA-Z0-9_]+)=([a-zA-Z0-9\.e\-]+)"
                row_matched = re.match(row_pattern, line.strip())
                if row_matched:
                    metrics_logger.log_metric(row_matched.group(1), float(row_matched.group(2)))
                else:
                    logger.warning(f"log metric {line} does not match expected pattern {row_pattern}")
            elif line.startswith("PROPERTY"):
                row_pattern = r"PROPERTY ([a-zA-Z0-9_]+)=([a-zA-Z0-9\.e\-]+)"
                row_matched = re.match(row_pattern, line.strip())
                if row_matched:
                    metrics_logger.set_properties(**{row_matched.group(1): row_matched.group(2)})
                else:
                    logger.warning(f"log metric {line} does not match expected pattern {row_pattern}")


        if len(time_inferencing_per_query) > 1:
            batch_run_times = np.array(time_inferencing_per_query)
            metrics_logger.log_metric("batch_time_inferencing_p50_usecs", np.percentile(batch_run_times, 50))
            metrics_logger.log_metric("batch_time_inferencing_p90_usecs", np.percentile(batch_run_times, 90))
            metrics_logger.log_metric("batch_time_inferencing_p99_usecs", np.percentile(batch_run_times, 99))
            metrics_logger.log_metric("time_inferencing", np.sum(batch_run_times))


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return LightGBMCAPIInferecingScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    LightGBMCAPIInferecingScript.main(cli_args)

if __name__ == "__main__":
    main()

