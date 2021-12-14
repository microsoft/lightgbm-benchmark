# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python training script
"""
import logging
import traceback
from .components import RunnableScript
from dataclasses import dataclass
from omegaconf import MISSING

from .perf import PerformanceMetricsCollector, PerfReportPlotter

@dataclass
class mpi_config_class:
    world_size: int = MISSING
    world_rank: int = MISSING
    mpi_available: bool = MISSING
    main_node: bool = MISSING

class MPIHandler():
    """Handling MPI initialization in a separate class
    so we can patch/mock it during unit testing of MultiNodeScript"""
    def __init__(self, mpi_init_mode=None):
        """Constructor"""
        self._mpi_module = None
        self._comm = None
        self._mpi_config = None
        self._mpi_init_mode = mpi_init_mode
        self.logger = logging.getLogger(__name__)

    @classmethod
    def _mpi_import(cls):
        import mpi4py
        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
        from mpi4py import MPI

        return MPI

    def initialize(self):
        # doing our own initialization of MPI to have fine-grain control
        self._mpi_module = self._mpi_import()
        self.comm = self._mpi_module.COMM_WORLD

        if self._mpi_init_mode is None:
            self._mpi_init_mode = self._mpi_module.THREAD_MULTIPLE

        try:
            self._mpi_module.Init_thread(required=self._mpi_init_mode)
        except self._mpi_module.Exception:
            self.logger.warning(f"Exception occured during MPI initialization:\n{traceback.format_exc()}")

        self._mpi_config = self.detect_mpi_config()

    def finalize(self):
        if self._mpi_module.Is_initialized():
            self.logger.info("MPI was initialized, calling MPI.finalize()")
            self._mpi_module.Finalize()

    def mpi_config(self):
        return self._mpi_config

    def detect_mpi_config(self):
        """ Detects if we're running in MPI.
        Args:
            None

        Returns:
            mpi_config (namedtuple)
        """
        try:
            mpi_config = mpi_config_class(
                self.comm.Get_size(), # world_size
                self.comm.Get_rank(), # world_rank
                (self.comm.Get_size() > 1), # mpi_available
                (self.comm.Get_rank() == 0), # main_node
            )
            logging.getLogger().info(f"MPI detection results: {mpi_config}")
        except:
            mpi_config = mpi_config_class(
                1, # world_size
                0, # world_rank
                False, # mpi_available
                True, # main_node
            )
            logging.getLogger().critical(f"MPI detection failed, switching to single node: {mpi_config}, see traceback below:\n{traceback.format_exc()}")

        return mpi_config


class MultiNodeScript(RunnableScript):
    def __init__(self, task, framework, framework_version, metrics_prefix=None, mpi_init_mode=None):
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

        self._mpi_handler = MPIHandler(mpi_init_mode=mpi_init_mode)
        self._mpi_config = None

    def mpi_config(self):
        """Getter"""
        return self._mpi_config

    def initialize_run(self, args):
        """Initialize the component run, opens/setups what needs to be"""
        self.logger.info("Initializing multi node component script...")

        self.logger.info("Initializing MPI.")
        self._mpi_handler.initialize()
        self._mpi_config = self._mpi_handler.mpi_config()

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

        # enable perf reporting
        if not args.disable_perf_metrics:
            self.perf_report_collector = PerformanceMetricsCollector()
            self.perf_report_collector.start()


    def finalize_run(self, args):
        """Finalize the run, close what needs to be"""
        self.logger.info("Finalizing multi node component script...")

        # clean exit from mpi
        self.logger.info("Finalizing MPI.")
        self._mpi_handler.finalize()

        if self.perf_report_collector:
            self.perf_report_collector.finalize()
            plotter = PerfReportPlotter(self.metrics_logger)
            plotter.add_perf_reports(self.perf_report_collector.perf_reports, node=self._mpi_config.world_rank)
            plotter.report_nodes_perf()

        # close mlflow
        self.metrics_logger.close()
