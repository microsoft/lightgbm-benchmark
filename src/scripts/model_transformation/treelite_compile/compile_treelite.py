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
from common.components import RunnableScript
from common.io import input_file_path


class TreeLightCompileScript(RunnableScript):
    def __init__(self):
        super().__init__(
            task = 'compile',
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

        group_i = parser.add_argument_group("Input Data")
        group_i.add_argument("--model",
            required=False, type=input_file_path, help="Exported model location (file path)")
        
        group_treelite = parser.add_argument_group("Treelite parameters")
        group_treelite.add_argument("--model_format",
            required=False, default="lightgbm", type=str, help="format of the input --model")
        group_treelite.add_argument("--so_path",
            required=False, default="./mymodel.so", type=str, help="full path to the saved model")
        group_treelite.add_argument("--toolchain",
            required=False, default="gcc", type=str, help="toolchain for compiling model")

        return parser


    def run(self, args, logger, metrics_logger, unknown_args):
        """Run script with arguments (the core of the component)

        Args:
            args (argparse.namespace): command line arguments provided to script
            logger (logging.getLogger() for this script)
            metrics_logger (common.metrics.MetricLogger)
            unknown_args (list[str]): list of arguments not recognized during argparse
        """
        logger.info(f"Converting model to Treelite")
        with metrics_logger.log_time_block("model_compilation"):
            model = treelite.Model.load(
                args.model,
                model_format=args.model_format
            )
            model.export_lib(
                toolchain=args.toolchain,
                libpath=args.so_path,
                verbose=True,
                params={'parallel_comp':16}
            )


def get_arg_parser(parser=None):
    """ To ensure compatibility with shrike unit tests """
    return TreeLightCompileScript.get_arg_parser(parser)

def main(cli_args=None):
    """ To ensure compatibility with shrike unit tests """
    TreeLightCompileScript.main(cli_args)

if __name__ == "__main__":
    main()
