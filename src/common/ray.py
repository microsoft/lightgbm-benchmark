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

from .perf import PerformanceMetricsCollector, PerfReportPlotter
from distutils.util import strtobool

import subprocess
import ray

class RayScript(RunnableScript):
    def __init__(self, task, framework, framework_version, metrics_prefix=None):
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

        # ray init settings
        self.head_address = None
        self.head_port = 6379
        self.redis_password = None

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
        group_general = parser.add_argument_group("Ray parameters")
        group_general.add_argument(
            "--ray_head",
            required=False,
            default=None,
            type=str,
            help="address of ray cluster (if running this script locally)",
        )
        group_general.add_argument(
            "--ray_head_port",
            required=False,
            default=6379,
            type=int,
            help="port of ray cluster (if running this script locally)",
        )
        group_general.add_argument(
            "--ray_redis_password",
            required=False,
            default=None,
            type=str,
            help="redis password of ray cluster (if running this script locally)",
        )
        group_general.add_argument(
            "--ray_on_aml",
            required=False,
            default=False,
            type=strtobool,
            help="if running this script within an AzureML run (head/port will be discovered)",
        )

        return parser

    def initialize_run(self, args):
        """Initialize the component run, opens/setups what needs to be"""
        self.logger.info("Initializing Ray component script...")

        # open mlflow
        self.metrics_logger.open()

        if args.ray_on_aml:
            # if running on AzureML, get context of cluster from env variables
            self.head_address = os.environ.get("AZ_BATCHAI_JOB_MASTER_NODE_IP")
            self.redis_password = os.environ.get("AZUREML_RUN_TOKEN_RAND", "12345")
            self_is_head = (os.environ.get("OMPI_COMM_WORLD_NODE_RANK", "0") == "0")
            available_nodes = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))

            if self_is_head:
                self.setup_head_node()
            else:
                self.setup_cluster_node()
                # go to sleep
                self.cluster_node_sleep()
                # and never return...

            ray_init_ret_val = ray.init(address="auto", _redis_password=self.redis_password)
            self.logger.info(f"Ray init returned: {ray_init_ret_val}")
            self.logger.info("Ray resources: {}".format(ray.available_resources()))

            for i in range(60):
                self.logger.info(f"Waiting for ray cluster to reach available nodes size... [{len(ray.nodes())}/{available_nodes}]")
                if (len(ray.nodes()) < available_nodes):
                    break
                time.sleep(1)
            else:
                raise Exception("Could not reach maximum number of nodes before 60 seconds.")

        else:
            if args.ray_head:
                # initialize ray for remote ray cluster
                ray.init(
                    redis_addr=f"{args.ray_head}:{args.ray_head_port}",
                    _redis_password=args.ray_redis_password
                )
            else:
                # initialize ray locally
                ray.init()

    def finalize_run(self, args):
        """Finalize the run, close what needs to be"""
        self.logger.info("Finalizing Ray component script...")

        # clean ray exit
        ray.shutdown()

        # close mlflow
        self.metrics_logger.close()

    #####################
    ### SETUP METHODS ###
    #####################

    def run_ray_cli(self, ray_cli_command, timeout=60):
        """Runs subprocess for ray setup command"""
        self.logger.info(f"Launching ray cli with command: {ray_cli_command}")
        ray_cli_command_call = subprocess.run(
            ray_cli_command,
            #stdout=PIPE,
            #stderr=PIPE,
            universal_newlines=True,
            check=False, # will not raise an exception if subprocess fails (so we capture with .returncode)
            timeout=timeout, # TODO: more than a minute would be weird?
            #env=custom_env
        )
        self.logger.info(f"return code: {ray_cli_command_call.returncode}")

        if ray_cli_command_call.returncode != 0:
            raise RuntimeError("Ray cli command returned code != 0")

        return ray_cli_command_call.returncode

    def setup_head_node(self):
        """Setup to run only on head node"""
        self.logger.info("Setting up Ray for HEAD node.")

        # run ray cli
        ray_setup_command = [
            "ray",
            "start",
            "--head",
            f"--port={self.head_port}",
            f"--redis-password={self.redis_password}"
        ]
        self.run_ray_cli(ray_setup_command)

    def setup_cluster_node(self):
        """Setup to run only on non-head cluster nodes"""
        self.logger.info("Setting up Ray for CLUSTER node.")
        
        # run ray cli
        ray_setup_command = [
            "ray",
            "start",
            f"--address={self.head_address}:{self.head_port}",
            f"--redis-password={self.redis_password}",
            "--block" # should remain in subprocess forever
        ]
        self.run_ray_cli(ray_setup_command, timeout=None)

        # then report that setup is complete
        self.report_node_setup_complete()

    def cluster_node_sleep(self):
        # then go into sleep and show some useful logs
        self.logger.info("Getting into sleep now...")
        while(True):
            time.sleep(10)

            # TODO: figure out the exit strategy here

        self.logger.warning("Received shutdown signal, shutting down node.")

        # not sure we need this, but doing it anyway
        self.run_ray_cli(["ray", "stop", "--force", "-v"])

        # exit, do not return!
        sys.exit(0)
