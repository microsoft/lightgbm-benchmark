# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
TreeLite/Python inferencing script
"""
import os
import sys
import argparse
import logging
import time
import numpy as np
from distutils.util import strtobool
import treelite, treelite_runtime

# Add the right path to PYTHONPATH
# so that you can import from common.*
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if COMMON_ROOT not in sys.path:
    logging.info(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript
from common.io import input_file_path, CustomLightGBMDataBatchIterator


class TreeLightInferencingScript(RunnableScript):
    def __init__(self):
        super().__init__(
            task = "score",
            framework = 'treelite_python',
            framework_version = "PYTHON_API."+str(treelite.__version__)
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
        group_i.add_argument("--data",
            required=True, type=input_file_path, help="Inferencing data location (file path)")
        group_i.add_argument("--so_path",
            required=False, default = "./mymodel.so" , help="full path to model so")
        group_i.add_argument("--output",
            required=False, default=None, type=str, help="Inferencing output location (file path)")
        
        group_params = parser.add_argument_group("Scoring parameters")
        group_params.add_argument("--num_threads",
            required=False, default=1, type=int, help="number of threads")
        group_params.add_argument("--batch_size",
            required=False, default=0, type=int, help="size of batches for predict call")

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
            num_threads=args.num_threads,
            batch_size=args.batch_size,
        )

        # make sure the output argument exists
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            
            # and create your own file inside the output
            args.output = os.path.join(args.output, "predictions.txt")

        logger.info(f"Loading model from {args.so_path}")
        predictor = treelite_runtime.Predictor(
            args.so_path,
            verbose=True,
            nthread=args.num_threads
        )

        # accumulate predictions and latencies
        predictions = []
        time_inferencing_per_batch = []
        batch_lengths = []

        # loop through batches
        for batch in CustomLightGBMDataBatchIterator(args.data, batch_size=args.batch_size, file_format="csv").iter():
            if len(batch) == 0:
                break
            batch_lengths.append(len(batch))

            # transform into dense matrix for treelite
            batch_data = np.array(batch)
            batch_dmat = treelite_runtime.DMatrix(batch_data)

            # run prediction on batch
            batch_start_time = time.monotonic()
            predictions.extend(predictor.predict(batch_dmat))
            time_inferencing_per_batch.append((time.monotonic() - batch_start_time)) # usecs

        # log overall time
        metrics_logger.log_metric("time_inferencing", sum(time_inferencing_per_batch))

        # use helper to log latency with the right metric names
        metrics_logger.log_inferencing_latencies(
            time_inferencing_per_batch,
            batch_length=batch_lengths,
            factor_to_usecs=1000000.0 # values are in seconds
        )

        if args.output:
            np.savetxt(
                args.output,
                predictions,
                fmt='%f',
                delimiter=',',
                newline='\n',
                header='',
                footer='',
                comments='# ',
                encoding=None
            )


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return TreeLightInferencingScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    TreeLightInferencingScript.main(cli_args)

if __name__ == "__main__":
    main()
