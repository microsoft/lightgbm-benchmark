# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
LightGBM/Python training script
"""
import os
import logging
import traceback
import subprocess
import uuid
import time
from .components import RunnableScript
from dataclasses import dataclass
from typing import Optional
from distutils.util import strtobool

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


class MultiNodeClusterSyncSetupScript(RunnableScript):
    # not even sure we need tags, but let's do it
    COMM_TAG_CLUSTER_SETUP = 42
    COMM_TAG_SETUP_FINISHED = 43
    COMM_TAG_CLUSTER_SHUTDOWN = 44

    def __init__(self, task, framework, framework_version, metrics_prefix=None, mpi_init_mode=3):
        """ Generic initialization for all script classes.

        Args:
            task (str): name of task in the pipeline/benchmark (ex: train, score)
            framework (str): name of ML framework
            framework_version (str): a version of this framework
            metrics_prefix (str): any prefix to add to this scripts metrics
            mpi_init_mode (int): mode to initialize MPI (default: THREAD)
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
        self._setup_config = {}


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

        # add generic arguments here
        group_args = parser.add_argument_group(f"Cluster auto setup arguments [{__name__}:{cls.__name__}]")
        group_args.add_argument(
            "--cluster_auto_setup",
            required=False,
            default=False,
            type=strtobool,
            help="Runs the components setup/teardown methods and sync using MPI.",
        )

        return parser

    def run_cli_command(self, cli_command, timeout=60):
        """Runs subprocess for a cli setup command"""
        self.logger.info(f"Launching cli with command: {cli_command}")
        cli_command_call = subprocess.run(
            cli_command,
            # stdout=PIPE,
            # stderr=PIPE,
            universal_newlines=True,
            check=False,  # will not raise an exception if subprocess fails (so we capture with .returncode)
            timeout=timeout,  # TODO: more than a minute would be weird?
            # env=custom_env
        )
        self.logger.info(f"return code: {cli_command_call.returncode}")

        if cli_command_call.returncode != 0:
            raise RuntimeError("Cli command returned code != 0")

        return cli_command_call.returncode


    #####################
    ### SETUP METHODS ###
    #####################

    def setup_config_add_key(self, key, value):
        self._setup_config[key] = value

    def setup_config_get_key(self, key, default_value=None):
        return self._setup_config.get(key, default_value)

    # For specific setups, override methods below

    def setup_local(self, args):
        """Setup method if custom_sync_setup=False"""
        self.logger.info(f"{self.__class__.__name__}.setup_local() called.")

    def setup_head_node(self):
        """Setup to run only on head node"""
        self.logger.info(f"{self.__class__.__name__}.setup_head_node() called to set up HEAD node.")
        self.setup_config_add_key("_session_id", str(uuid.uuid4()))

    def setup_cluster_node(self):
        """Setup to run only on non-head cluster nodes"""
        self.logger.info(f"{self.__class__.__name__}.setup_cluster_node() called to set up cluster node.")

    def head_node_teardown(self):
        """Un-setup a cluster node"""
        self.logger.info(f"{self.__class__.__name__}.head_node_teardown() called to teardown a HEAD node.")

    def cluster_node_teardown(self):
        """Un-setup a cluster node"""
        self.logger.info(f"{self.__class__.__name__}.cluster_node_teardown() called to teardown a cluster node.")


    ################
    ### MPI COMM ###
    ################

    def broadcast_config_from_head_to_cluster_nodes(self):
        """[HEAD only] Sends the cluster setup params to each non-head node"""
        self.logger.info(f"Sending cluster setup from head node to cluster nodes: {self._setup_config}")
        for i in range(1, self.multinode_config.world_size):
            self.multinode_driver._comm.send(
                self._setup_config, i, tag=MultiNodeClusterSyncSetupScript.COMM_TAG_CLUSTER_SETUP
            )

    def listen_cluster_setup_from_head_node(self):
        """[NODE only] Waits for head node to send cluster setup params"""
        self._setup_config = self.multinode_driver._comm.recv(
            source=0, tag=MultiNodeClusterSyncSetupScript.COMM_TAG_CLUSTER_SETUP
        )
        self.logger.info(f"Obtained cluster setup from head node: {self._setup_config}")


    def wait_on_nodes_setup_ready(self):
        """[HEAD only] Waits for each node to report completion of their setup"""
        self.logger.info("Checking setup status from each node...")

        # loop on each node in the world and wait for status
        for i in range(1, self.multinode_config.world_size):
            status = self.multinode_driver._comm.recv(source=i, tag=MultiNodeClusterSyncSetupScript.COMM_TAG_SETUP_FINISHED)
            self.logger.info(f"Node #{i}: {status}")

            if status != "OK":
                raise RuntimeError(f"Node #{i} failed to setup.")

    def report_node_setup_complete(self):
        """[NODE only] Report to head that this node setup is complete"""
        self.logger.info("Reporting status OK to head node.")
        self.multinode_driver._comm.send("OK", 0, tag=MultiNodeClusterSyncSetupScript.COMM_TAG_SETUP_FINISHED)

    def broadcast_shutdown_signal(self):
        """[HEAD only] Sends message to shutdown to all nodes"""
        for i in range(1, self.multinode_config.world_size):
            self.logger.info(f"Broadcasting shutdown message to node #{i}")
            self.multinode_driver._comm.send("SHUTDOWN", i, tag=MultiNodeClusterSyncSetupScript.COMM_TAG_CLUSTER_SHUTDOWN)

    def non_block_wait_for_shutdown(self):
        """[NODE only] Checks if head node has sent shutdown message"""
        return self.multinode_driver._comm.iprobe(source=0, tag=MultiNodeClusterSyncSetupScript.COMM_TAG_CLUSTER_SHUTDOWN)


    ######################
    ### SCRIPT METHODS ###
    ######################

    def initialize_run(self, args):
        """Initialize the component run, opens/setups what needs to be"""
        self.logger.info(f"Initializing {self.__class__.__name__} run...")

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

        # initialize mpi comm
        self.multinode_driver = MultiNodeMPIDriver(mpi_init_mode=self._mpi_init_mode)
        self.multinode_driver.initialize()
        self.multinode_config = self.multinode_driver.get_multinode_config()

        if args.cluster_auto_setup:
            # initialize setup accross nodes
            if self.multinode_config.main_node:
                # run setup on head node
                self.setup_head_node()

                # send cluster config to all other nodes
                self.broadcast_config_from_head_to_cluster_nodes()

                # then wait for all nodes to finish setup
                self.wait_on_nodes_setup_ready()
            else:
                # get cluster setup from head node using mpi
                self.listen_cluster_setup_from_head_node()

                # run setup on cluster node
                self.setup_cluster_node()

                # then report that setup is complete
                self.report_node_setup_complete()
        else:
            # run custom method for setup
            self.setup_local(args)

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


    def finalize_run(self, args):
        """Finalize the run, close what needs to be"""
        self.logger.info(f"Finalizing {self.__class__.__name__} run...")

        if args.cluster_auto_setup:
            # properly teardown all nodes
            if self.multinode_config.main_node:
                # run teardown on head node
                self.head_node_teardown()

                # send signal to teardown to each node
                self.broadcast_shutdown_signal()
            else:
                # wait for teardown signal from head node
                while True:
                    time.sleep(10)
                    self.logger.info(f"Waiting for teardown signal from HEAD node...")

                    if self.non_block_wait_for_shutdown():
                        break

                # run teardown on cluster
                self.cluster_node_teardown()

        if self.perf_report_collector:
            self.perf_report_collector.finalize()
            plotter = PerfReportPlotter(self.metrics_logger)
            plotter.add_perf_reports(self.perf_report_collector.perf_reports, node=self.multinode_config.world_rank)
            plotter.report_nodes_perf()

        # close mlflow
        self.metrics_logger.close()

        # clean exit from driver
        self.multinode_driver.finalize()
