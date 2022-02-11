# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python training script
"""
import os
import logging
import traceback
from .components import RunnableScript
from dataclasses import dataclass
from typing import Optional

from .metrics import MLFlowMetricsLogger, AzureMLRunMetricsLogger
from .perf import PerformanceMetricsCollector, PerfReportPlotter

@dataclass
class multinode_config_class:
    world_size: int = 1
    world_rank: int = 0
    multinode_available: bool = False
    main_node: bool = True
    machines: Optional[str] = None
    local_listen_port: Optional[int] = None


class MultiNodeDriver:
    """Handling multinode initialization"""
    def __init__(self, **kwargs):
        """Constructor"""
        self.logger = logging.getLogger(__name__)
        self._multinode_config = None
        self._kwargs = kwargs

    def initialize(self):
        """Initialize the driver"""
        self.logger.info(f"{self.__class__.__name__}.initialize(): pass.")

    def get_multinode_config(self):
        """Get internal multinode config"""
        return self._multinode_config

    def finalize(self):
        """Finalize/close resources used by the driver"""
        self.logger.info(f"{self.__class__.__name__}.finalize(): pass.")


class MultiNodeSocketDriver(MultiNodeDriver):
    """Handling multinode initialization for socket"""
    def initialize(self, **kwargs):
        """Initialize the driver"""
        self.logger.info(f"{self.__class__.__name__}.initialize(): discovering nodes from within the job.")
        if 'AZ_BATCH_NODE_LIST' in os.environ: # if within AzureML
            self.logger.info("Discovering multinode socket config from inside AzureML.")
            machines = os.environ.get('AZ_BATCH_NODE_LIST').split(";")
            local_ip = os.environ.get('AZ_BATCHAI_NODE_IP')

            world_size = len(machines)
            world_rank = machines.index(local_ip)

            self._multinode_config = multinode_config_class(
                world_size,
                world_rank,
                world_size > 1, # multinode_available
                (world_rank == 0), # main_node
                machines = machines,
                local_listen_port = 12345
            )
        else:
            self.logger.warning("MultiNodeSocketDriver could not discover socker config.")
            # we can't detect anything not given to us,
            # let's consider this single node
            self._multinode_config = multinode_config_class(
                1, # world_size
                0, # world_rank
                False, # multinode_available
                True, # main_node
            )
        
        self.logger.info(f"Socket discovery obtained config: {self._multinode_config}")


class MultiNodeMPIDriver(MultiNodeDriver):
    """Handling MPI initialization in a separate class
    so we can patch/mock it during unit testing of MultiNodeScript"""
    def __init__(self, mpi_init_mode=None, **kwargs):
        """Constructor"""
        super().__init__(**kwargs)
        self._mpi_module = None
        self._comm = None
        self._mpi_init_mode = mpi_init_mode

    @classmethod
    def _mpi_import(cls):
        # doing our own initialization of MPI to have fine-grain control
        import mpi4py
        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
        from mpi4py import MPI

        return MPI

    def initialize(self, **kwargs):
        """Initialize the driver"""
        self._mpi_module = self._mpi_import()

        if self._mpi_init_mode is None:
            # do not init mpi, but use openmpi env vars to detect mpi config
            self.logger.info(f"no MPI init, using environment variables instead")
            world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
            world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))

            self._multinode_config = multinode_config_class(
                world_size, # world_size
                world_rank, # world_rank
                (world_size > 1), # mpi_available
                (world_rank == 0), # main_node
            )
            self._comm = None
        else:
            # init mpi and use comm to detect mpi config
            self.logger.info(f"Running MPI.Init_thread(required={self._mpi_init_mode})")
            try:
                self._mpi_module.Init_thread(required=self._mpi_init_mode)
            except self._mpi_module.Exception:
                self.logger.warning(f"Exception occured during MPI initialization:\n{traceback.format_exc()}")

            self._comm = self._mpi_module.COMM_WORLD
            try:
                self._multinode_config = multinode_config_class(
                    self._comm.Get_size(), # world_size
                    self._comm.Get_rank(), # world_rank
                    (self._comm.Get_size() > 1), # mpi_available
                    (self._comm.Get_rank() == 0), # main_node
                )
                self.logger.info(f"MPI detection results: {self._multinode_config}")
            except:
                self._multinode_config = multinode_config_class(
                    1, # world_size
                    0, # world_rank
                    False, # mpi_available
                    True, # main_node
                )
                self.logger.critical(f"MPI detection failed, switching to single node: {self._multinode_config}, see traceback below:\n{traceback.format_exc()}")

    def get_multinode_config(self):
        """Get internal multinode config"""
        return self._multinode_config

    def finalize(self):
        """Finalize/close resources used by the driver"""
        if self._mpi_module.Is_initialized() and not self._mpi_module.Is_finalized():
            self.logger.info("MPI was initialized, calling MPI.finalize()")
            self._mpi_module.Finalize()
        else:
            self.logger.warning(f"MPIHandler.finalize() was called, but MPI.Is_initialized={self._mpi_module.Is_initialized()} and MPI.Is_finalized={self._mpi_module.Is_finalized()}")


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

        # keep those for initialization
        self._mpi_init_mode = mpi_init_mode

    @classmethod
    def get_arg_parser(cls, parser=None):
        """Adds multinode arguments to a given argument parser.

        Args:
            parser (argparse.ArgumentParser): an argument parser instance

        Returns:
            ArgumentParser: the argument parser instance

        Notes:
            if parser is None, creates a new parser instance
        """
        # add generic arguments
        parser = RunnableScript.get_arg_parser(parser)

        group_runtime = parser.add_argument_group(f"MultiNode runtime parameters [{__name__}:{cls.__name__}]")
        group_runtime.add_argument("--multinode_driver", type=str, choices=['mpi', 'socket'], default='socket', required=False)
        group_runtime.add_argument("--multinode_machines", type=str, default='auto', required=False, help="list of machines, use only when running locally, default will use 'auto' to discover")
        group_runtime.add_argument("--multinode_listen_port", type=str, default=12345, required=False, help="used for socket only, default 12345")

        return parser

    def initialize_run(self, args):
        """Initialize the component run, opens/setups what needs to be"""
        self.logger.info("Initializing multi node component script...")

        # initializes reporting of metrics
        if args.metrics_driver == 'mlflow':
            self.metrics_logger = MLFlowMetricsLogger(
                f"{self.framework}.{self.task}",
                metrics_prefix=self.metrics_prefix
            )
        elif args.metrics_driver == 'azureml':
            self.metrics_logger = AzureMLRunMetricsLogger(
                f"{self.framework}.{self.task}",
                metrics_prefix=self.metrics_prefix
            )
        else:
            # use default metrics_logger (stdout print)
            pass

        if args.multinode_driver == 'socket':
            self.multinode_driver = MultiNodeSocketDriver(machines=args.multinode_machines, listen_port=args.multinode_listen_port)
        elif args.multinode_driver == 'mpi':
            self.multinode_driver = MultiNodeMPIDriver(mpi_init_mode=self._mpi_init_mode)
        else:
            raise NotImplementedError(f"multinode_driver={args.multinode_driver} is not implemented, use 'socket' or 'mpi'")

        # initialize driver
        self.multinode_driver.initialize()
        self.multinode_config = self.multinode_driver.get_multinode_config()

        # open mlflow
        self.metrics_logger.open()
        
        if self.multinode_config.main_node:
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

    def get_multinode_config(self):
        """Get internal multinode config"""
        return self.multinode_driver._multinode_config

    def finalize_run(self, args):
        """Finalize the run, close what needs to be"""
        self.logger.info("Finalizing multi node component script...")

        # clean exit from driver
        self.multinode_driver.finalize()

        if self.perf_report_collector:
            self.perf_report_collector.finalize()
            plotter = PerfReportPlotter(self.metrics_logger)
            plotter.add_perf_reports(self.perf_report_collector.perf_reports, node=self.multinode_config.world_rank)
            plotter.report_nodes_perf()

            # write perf record as artifact
            self.metrics_logger.log_artifact(plotter.save_to())

        # close mlflow
        self.metrics_logger.close()
