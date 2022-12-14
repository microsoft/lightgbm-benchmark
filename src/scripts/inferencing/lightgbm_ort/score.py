# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python inferencing script
"""
import os
import sys
import logging
import timeit
import numpy as np
from distutils.util import strtobool
import lightgbm

import onnxruntime as ort
from onnxmltools.convert import convert_lightgbm
from onnxconverter_common.data_types import FloatTensorType

COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if COMMON_ROOT not in sys.path:
    logging.info(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript
from common.io import input_file_path


class LightGBMONNXRTInferecingScript(RunnableScript):
    def __init__(self):
        framework = "onnx"
        if "--run_parallel" in sys.argv and strtobool(sys.argv[sys.argv.index("--run_parallel") + 1]):
            framework += "_parallel"
        if "--run_batch" in sys.argv and strtobool(sys.argv[sys.argv.index("--run_batch") + 1]):
            framework += "_batch"
        if "--num_threads" in sys.argv:
            framework += f"_threads_{sys.argv[sys.argv.index('--num_threads') + 1]}"

        super().__init__(
            task="score",
            framework=framework,
            framework_version="ONNXRT." + str(ort.__version__),
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
        group_i.add_argument(
            "--data",
            required=True,
            type=input_file_path,
            help="Inferencing data location (file path)",
        )
        group_i.add_argument(
            "--data_format",
            type=str,
            choices=["CSV", "PARQUET", "PETASTORM"],
            default="CSV",
        )
        group_i.add_argument(
            "--model",
            required=False,
            type=input_file_path,
            help="Exported model location (file path)",
        )
        group_i.add_argument(
            "--output",
            required=False,
            default=None,
            type=str,
            help="Inferencing output location (file path)",
        )

        group_params = parser.add_argument_group(
            f"Scoring parameters [{__name__}:{cls.__name__}]"
        )
        group_params.add_argument(
            "--num_threads",
            required=False,
            default=0,
            type=int,
            help="number of threads",
        )
        group_params.add_argument(
            "--run_parallel",
            required=False,
            default="False",
            type=strtobool,
            help="allows intra sample parallelism",
        )
        group_params.add_argument(
            "--run_batch",
            required=False,
            default="False",
            type=strtobool,
            help="runs inference in a single batch",
        )
        group_params.add_argument(
            "--predict_disable_shape_check",
            required=False,
            default="False",
            type=strtobool,
            help="See LightGBM documentation",
        )

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
        metrics_logger.log_parameters(num_threads=args.num_threads)

        # register logger for lightgbm logs
        lightgbm.register_logger(logger)

        # make sure the output argument exists
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            args.output = os.path.join(args.output, "predictions.txt")

        logger.info(f"Loading model from {args.model}")
        # BUG: https://github.com/onnx/onnxmltools/issues/338
        with open(args.model, "r") as mf:
            model_str = mf.read()
            model_str = model_str.replace(
                "objective=lambdarank", "objective=regression"
            )
        booster = lightgbm.Booster(model_str=model_str)

        logger.info(f"Loading data for inferencing")
        assert args.data_format == "CSV"
        with metrics_logger.log_time_block("time_data_loading"):
            # NOTE: this is bad, but allows for libsvm format (not just numpy)
            # inference_data = lightgbm.Dataset(
            #     args.data, free_raw_data=False
            # ).construct()
            # inference_raw_data = inference_data.get_data()
            # if type(inference_raw_data) == str:
            inference_raw_data = np.loadtxt(
                args.data, delimiter=","
            ).astype(np.float32)[:, : booster.num_feature()]

        logger.info(f"Converting model to ONNX")
        onnx_input_types = [
            (
                "input",
                FloatTensorType([None, inference_raw_data.shape[1]]),
            )
        ]
        onnx_ml_model = convert_lightgbm(booster, initial_types=onnx_input_types)

        logger.info(f"Creating inference session")
        sess_options = ort.SessionOptions()

        if args.num_threads > 0:
            logger.info(f"Setting number of threads to {args.num_threads}")
            sess_options.intra_op_num_threads = args.num_threads
            sess_options.inter_op_num_threads = args.num_threads

        if args.run_parallel:
            logger.info(f"Creating multithreaded inference session")

        sess_options.execution_mode = (
            ort.ExecutionMode.ORT_PARALLEL
            if args.run_parallel
            else ort.ExecutionMode.ORT_SEQUENTIAL
        )
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sessionml = ort.InferenceSession(
            onnx_ml_model.SerializeToString(), sess_options
        )

        # capture data shape as property
        metrics_logger.set_properties(
            inference_data_length=inference_raw_data.shape[0],
            inference_data_width=inference_raw_data.shape[1],
        )

        logger.info(f"Running .predict()")

        # Warmup and compute results
        for _ in range(100):
            sessionml.run(
                [sessionml.get_outputs()[0].name],
                {sessionml.get_inputs()[0].name: inference_raw_data[0:1]},
            )

        predictions_array = sessionml.run(
            [sessionml.get_outputs()[0].name],
            {sessionml.get_inputs()[0].name: inference_raw_data},
        )[0]

        time_inferencing_per_query = []
        timeit_loops = 10

        if args.run_batch:
            batch_length = len(inference_raw_data)
            prediction_time = timeit.timeit(
                lambda: sessionml.run(
                    [sessionml.get_outputs()[0].name],
                    {sessionml.get_inputs()[0].name: inference_raw_data},
                ),
                number=timeit_loops,
            )
            prediction_time /= timeit_loops
            metrics_logger.log_metric("time_inferencing", prediction_time)
            time_inferencing_per_query = [prediction_time]
        else:
            batch_length = 1
            for i in range(len(inference_raw_data)):
                prediction_time = timeit.timeit(
                    lambda: sessionml.run(
                        [sessionml.get_outputs()[0].name],
                        {sessionml.get_inputs()[0].name: inference_raw_data[i : i + 1]},
                    ),
                    number=timeit_loops,
                )
                prediction_time /= timeit_loops
                time_inferencing_per_query.append(prediction_time)
            metrics_logger.log_metric(
                "time_inferencing", sum(time_inferencing_per_query)
            )

        logger.info(f"Batch size: {batch_length}")

        # use helper to log latency with the right metric names
        metrics_logger.log_inferencing_latencies(
            time_inferencing_per_query,  # only one big batch
            batch_length=batch_length,
            factor_to_usecs=1000000.0,  # values are in seconds
        )

        if args.output:
            np.savetxt(
                args.output,
                predictions_array,
                fmt="%f",
                delimiter=",",
                newline="\n",
                header="",
                footer="",
                comments="# ",
                encoding=None,
            )


def get_arg_parser(parser=None):
    """To ensure compatibility with shrike unit tests"""
    return LightGBMONNXRTInferecingScript.get_arg_parser(parser)


def main(cli_args=None):
    """To ensure compatibility with shrike unit tests"""
    LightGBMONNXRTInferecingScript.main(cli_args)


if __name__ == "__main__":
    main()
