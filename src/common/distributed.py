# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python training script
"""
import os
import sys
import argparse
import logging
import traceback
import json
from .components import RunnableScript
from collections import namedtuple

# doing our own initialization of MPI to have fine-grain control
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

def detect_mpi_config():
    """ Detects if we're running in MPI.
    Args:
        None

    Returns:
        mpi_config (namedtuple)
    """
    # check if we're running multi or single node
    mpi_config_tuple = namedtuple("mpi_config", ['world_size', 'world_rank', 'mpi_available', 'main_node'])

    try:
        comm = MPI.COMM_WORLD
        mpi_config = mpi_config_tuple(
            comm.Get_size(), # world_size
            comm.Get_rank(), # world_rank
            (comm.Get_size() > 1), # mpi_available
            (comm.Get_rank() == 0), # main_node
        )
        logging.getLogger().info(f"MPI detection results: {mpi_config}")
    except:
        mpi_config = mpi_config_tuple(
            1, # world_size
            0, # world_rank
            False, # mpi_available
            True, # main_node
        )
        logging.getLogger().critical(f"MPI detection failed, switching to single node: {mpi_config}, see traceback below:\n{traceback.format_exc()}")

    return mpi_config

class MultiNodeScript(RunnableScript):
    def __init__(self, task, framework, framework_version, metrics_prefix=None, mpi_init_mode=MPI.THREAD_MULTIPLE):
        """ Generic initialization for all script classes.

        Args:
            task (str): name of task in the pipeline/benchmark (ex: train, score)
            framework (str): name of ML framework
            framework_version (str): a version of this framework
            metrics_prefix (str): any prefix to add to this scripts metrics
            mpi_init_mode (int): mode to initialize MPI
        """
        # just use the regular init
        super().__init__(
            task = task,
            framework = framework,
            framework_version = framework_version,
            metrics_prefix = metrics_prefix
        )

        # but also add mpi_config
        self._mpi_config = None
        self._mpi_init_mode = mpi_init_mode

    def mpi_config(self):
        """Getter method"""
        return self._mpi_config

    def initialize_run(self, args):
        """Initialize the component run, opens/setups what needs to be"""
        self.logger.info("Initializing multi node component script...")

        if MPI.Is_initialized():
            self.logger.info("MPI is already initialized.")
        else:
            self.logger.info("Initializing MPI.")
            MPI.Init_thread(required=self._mpi_init_mode)

        self._mpi_config = detect_mpi_config()

        # open mlflow
        self.metrics_logger.open()
        
        if self._mpi_config.main_node:
            # record properties only from the main node
            self.metrics_logger.set_properties(
                task = self.task,
                framework = self.framework,
                framework_version = self.framework_version
            )

            # if provided some custom_properties by the outside orchestrator
            if args.custom_properties:
                self.metrics_logger.set_properties_from_json(args.custom_properties)

            # add properties about environment of this script
            self.metrics_logger.set_platform_properties()

    def finalize_run(self, args):
        """Finalize the run, close what needs to be"""
        self.logger.info("Finalizing multi node component script...")
        # clean exit from mpi
        if MPI.Is_initialized():
            self.logger.info("MPI was initialized, calling MPI.finalize()")
            MPI.Finalize()
        
        # then use the super finalization method
        super().finalize_run(args)
