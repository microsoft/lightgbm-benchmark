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
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# useful imports from common
from common.components import RunnableScript
from common.io import input_file_path


class TreeLightInferencingScript(RunnableScript):
    def __init__(self):
        super().__init__(
            task = 'score',
            framework = 'treelite_python',
            framework_version = treelite.__version__
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
        group_i.add_argument("--data",
            required=True, type=input_file_path, help="Inferencing data location (file path)")
        group_i.add_argument("--so_path",
            required=False, default = "./mymodel.so" , help="full path to model so")
        group_i.add_argument("--output",
            required=False, default=None, type=str, help="Inferencing output location (file path)")
        
        group_params = parser.add_argument_group(f"Scoring parameters [{__name__}:{cls.__name__}]")
        group_params.add_argument("--num_threads",
            required=False, default=1, type=int, help="number of threads")

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


        logger.info(f"Loading data for inferencing")
        with metrics_logger.log_time_block("time_data_loading"):
            my_data = pd.read_csv(args.data).to_numpy()
            
            predictor = treelite_runtime.Predictor(
                args.so_path,
                verbose=True,
                nthread=args.num_threads
            )
            dmat = treelite_runtime.DMatrix(my_data)

        logger.info(f"Running .predict()")
        with metrics_logger.log_time_block("time_inferencing"):
            predictor.predict(dmat)


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return TreeLightInferencingScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    TreeLightInferencingScript.main(cli_args)

if __name__ == "__main__":
    main()
