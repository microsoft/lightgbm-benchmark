# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Helper code to run Ray distributed scripts [EXPERIMENTAL]
"""
import os
import logging
import traceback
import uuid
import time

import ray

from .distributed import MultiNodeClusterSyncSetupScript


class RayScript(MultiNodeClusterSyncSetupScript):
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
        parser = MultiNodeClusterSyncSetupScript.get_arg_parser(parser)

        # add generic arguments here
        group_args = parser.add_argument_group(f"Ray setup arguments [{__name__}:{cls.__name__}]")
        group_args.add_argument(
            "--ray_head_addr",
            required=False,
            default=None,
            type=str,
            help="address of ray cluster (if running this script locally)",
        )
        group_args.add_argument(
            "--ray_head_port",
            required=False,
            default=6379,
            type=int,
            help="port of ray cluster (if running this script locally)",
        )
        group_args.add_argument(
            "--ray_redis_password",
            required=False,
            default=None,
            type=str,
            help="redis password of ray cluster (if running this script locally)",
        )

        return parser

    #####################
    ### SETUP METHODS ###
    #####################

    def setup_local(self, args):
        """Setup method if custom_sync_setup=False"""
        self.logger.info(f"{self.__class__.__name__}.setup_local() called.")

        # create setup config
        self.head_address = args.ray_head_addr
        self.head_port = args.ray_head_port
        self.redis_password = args.ray_redis_password


    def setup_head_node(self):
        """Setup to run only on head node"""
        super().setup_head_node()

        # create setup config
        self.head_address = os.environ.get('AZ_BATCHAI_NODE_IP')
        self.head_port = 6379
        self.redis_password = str(uuid.uuid4())

        # record what's needed to setup cluster nodes
        self.setup_config_add_key("head_address", self.head_address)
        self.setup_config_add_key("head_port", self.head_port)
        self.setup_config_add_key("redis_password", self.redis_password)

        # run ray cli
        ray_setup_command = [
            "ray",
            "start",
            "--head",
            f"--port={self.head_port}",
            f"--redis-password={self.redis_password}"
        ]
        self.run_cli_command(ray_setup_command)

    def setup_cluster_node(self):
        """Setup to run only on non-head cluster nodes"""
        super().setup_head_node()
        
        self.head_address = self.setup_config_get_key("head_address")
        self.head_port = self.setup_config_get_key("head_port")
        self.redis_password = self.setup_config_get_key("redis_password")

        # run ray cli
        ray_setup_command = [
            "ray",
            "start",
            f"--address={self.head_address}:{self.head_port}",
            f"--redis-password={self.redis_password}",
        ]
        self.run_cli_command(ray_setup_command)

    def head_node_teardown(self):
        """Un-setup a cluster node"""
        super().head_node_teardown()

        ray.shutdown()

        self.run_cli_command(["ray", "stop", "--force", "-v"])

    def cluster_node_teardown(self):
        """Un-setup a cluster node"""
        super().cluster_node_teardown()
        self.run_cli_command(["ray", "stop", "--force", "-v"])


    ############################
    ### SPECIFIC MAIN METHOD ###
    ############################

    def _main_run_hook(self, args, unknown_args):
        """Run function called from main()"""
        # call run() only on main node
        if self.multinode_config.main_node:
            # initialize ray lib
            if self.head_address is None:
                ray.init()
            else:
                ray.init(
                    address="{self.head_address}:{self.head_port}",
                    _redis_password=self.redis_password
                )

            # making absolutely sure all nodes are there...
            for i in range(60):
                self.logger.info(f"Waiting for ray cluster to reach available nodes size... [{len(ray.nodes())}/{self.multinode_config.world_size}]")
                if (len(ray.nodes()) >= self.multinode_config.world_size):
                    break
                time.sleep(1)
            else:
                raise Exception("Could not reach maximum number of nodes before 60 seconds.")

            self.run(args, self.logger, self.metrics_logger, unknown_args=unknown_args)
        else:
            self.logger.info("Running on cluster node, run() is skipped.")
            return
