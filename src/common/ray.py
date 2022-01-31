# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Helper code to run Ray distributed scripts [EXPERIMENTAL]
"""
import os
import logging
import traceback
from .components import RunnableScript
from dataclasses import dataclass

from distutils.util import strtobool

import subprocess
import ray
import time

from .perf import PerformanceMetricsCollector, PerfReportPlotter

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
        self.self_is_head = True
        self.head_address = None
        self.head_port = 6379
        self.redis_password = None
        self.available_nodes = 1

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

        if args.ray_on_aml:
            # if running on AzureML, get context of cluster from env variables
            self.head_address = os.environ.get("AZ_BATCHAI_JOB_MASTER_NODE_IP")
            self.redis_password = os.environ.get("AZUREML_RUN_TOKEN_RAND", "12345")
            self.self_is_head = (os.environ.get("OMPI_COMM_WORLD_RANK", "0") == "0")
            self.available_nodes = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))

            if self.self_is_head: # if we're on the first node of this job
                if self.available_nodes > 1: # and if number of nodes if more than one
                    # then initialize head node to listen to cluster nodes
                    self.logger.info(f"Available nodes = {self.available_nodes}, initializing ray for HEAD node.")
                    self.setup_head_node()

                    # then run ray init
                    ray_init_ret_val = ray.init(address="auto", _redis_password=self.redis_password)
                    self.logger.info(f"Ray init returned: {ray_init_ret_val}")
                    self.logger.info("Ray resources: {}".format(ray.available_resources()))

                    # and wait for cluster nodes to be initialized as well
                    for i in range(60):
                        self.logger.info(f"Waiting for ray cluster to reach available nodes size... [{len(ray.nodes())}/{self.available_nodes}]")
                        if (len(ray.nodes()) >= self.available_nodes):
                            break
                        time.sleep(1)
                    else:
                        raise Exception("Could not reach maximum number of nodes before 60 seconds.")
                else:
                    # if just one node, nothing to do here
                    self.logger.info(f"Available nodes = {self.available_nodes}, running ray.init() as for a single node...")
                    ray.init()

            else:
                self.setup_cluster_node()

        else:
            # considering this one as head
            self.self_is_head = True

            # if not running this script in AzureML...
            if args.ray_head:
                # initialize ray for remote ray cluster
                ray.init(
                    redis_addr=f"{args.ray_head}:{args.ray_head_port}",
                    _redis_password=args.ray_redis_password
                )
            else:
                # initialize ray locally
                ray.init()

        # open mlflow
        self.metrics_logger.open()

        if self.self_is_head:
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
            self.perf_report_collector = PerformanceMetricsCollector()
            self.perf_report_collector.start()


    def finalize_run(self, args):
        """Finalize the run, close what needs to be"""
        self.logger.info(f"Finalizing Ray component script...")

        # clean ray exit on HEAD node only
        if self.self_is_head:
            self.logger.info(f"At finalization, number of nodes is [nodes={len(ray.nodes())}]")
            ray.shutdown()

        if self.perf_report_collector:
            self.perf_report_collector.finalize()
            plotter = PerfReportPlotter(self.metrics_logger)
            plotter.add_perf_reports(self.perf_report_collector.perf_reports, node=0)
            plotter.report_nodes_perf()

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
            #"--block" # should remain in subprocess forever
        ]
        self.run_ray_cli(ray_setup_command, timeout=None)


    ############################
    ### SPECIFIC MAIN METHOD ###
    ############################

    @classmethod
    def main(cls, cli_args=None):
        """ Component main function, it is not recommended to override this method.
        It parses arguments and executes run() with the right arguments.

        Args:
            cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
        """
        # initialize root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # construct arg parser
        parser = cls.get_arg_parser()

        # if argument parsing fails, or if unknown arguments, will except
        args, unknown_args = parser.parse_known_args(cli_args)
        logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

        # create script instance, initialize mlflow
        script_instance = cls()
        script_instance.initialize_run(args)

        # catch run function exceptions to properly finalize run (kill/join threads)
        try:
            # run the actual run method ONLY ON HEAD
            if script_instance.self_is_head:
                script_instance.run(args, script_instance.logger, script_instance.metrics_logger, unknown_args)
            else:
                script_instance.logger.warning("This is not HEAD node, exiting script now")
        except BaseException as e:
            logging.critical(f"Exception occured during run():\n{traceback.format_exc()}")
            script_instance.finalize_run(args)
            raise e

        # close mlflow
        script_instance.finalize_run(args)

        # return for unit tests
        return script_instance
