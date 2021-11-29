"""
Replacement-code for Shrike
"""
# pylint: disable=no-member
# NOTE: because it raises 'dict' has no 'outputs' member in dsl.pipeline construction
import os
import sys
import json
import logging
import argparse

# config management
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from shrike.pipeline.pipeline_config import default_config_dict
from shrike.pipeline.aml_connect import azureml_connect

# when running this script directly, needed to import common
from .paths import COMPONENTS_ROOT, CONFIG_PATH

def pipeline_cli_main(pipeline_config_dataclass, pipeline_func, cli_args = None):
    """ (soon to be) Standard main function
    
    Args:
        pipeline_config_dataclass (dataclass): class for hosting the config of pipeline_func
        pipeline_func (function): pipeline building function to call with config object
        cli_args (List): command line arguments (if None, use sys.argv)
    """
    # create an argument parser just to catch --exp-conf
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("--exp-conf", required=False, default=None)
    # all remaining arguments will be passed to hydra
    args, unknown_args = arg_parser.parse_known_args()

    # resolve config_dir and config_name from --exp-conf
    # hacky, need a better solution
    if args.exp_conf:
        config_abspath = os.path.abspath(args.exp_conf)
        config_relpath = os.path.relpath(config_abspath, start=CONFIG_PATH) # relative path from config folder to specified config yaml
        config_name = os.path.dirname(config_relpath).replace("\\", "/") + "/" + os.path.basename(config_relpath)
        print(f"Using config_name={config_name} and config_dir={CONFIG_PATH}")
    else:
        config_name = None

    # override argv with only unknown args (TODO: do better)
    sys.argv = (
        [sys.argv[0]]
        + unknown_args
        + [
            "--config-dir", CONFIG_PATH,
            "--config-name", config_name,
        ]
    )

    # create config with pipeline dataclass
    # store it in hydra default
    config_store = ConfigStore.instance()
    config_dict = default_config_dict()
    config_dict[pipeline_config_dataclass.__name__] = pipeline_config_dataclass
    config_store.store(name="default", node=config_dict)

    def _run(pipeline_config):
        """ transient run function to call from hydra main"""
        # Connect to AzureML
        workspace = azureml_connect(
            aml_subscription_id=pipeline_config.aml.subscription_id,
            aml_resource_group=pipeline_config.aml.resource_group,
            aml_workspace_name=pipeline_config.aml.workspace_name,
            aml_auth=pipeline_config.aml.auth,
            aml_tenant=pipeline_config.aml.tenant,
            #aml_force=pipeline_config.aml.force,
        )  # NOTE: this also stores aml workspace in internal global variable

        # run the pipeline function with the given config
        pipeline_instance = pipeline_func(pipeline_config)

        # Submit or Validate ?
        pipeline_instance.validate(workspace=workspace)


    @hydra.main(config_name="default")
    def hydra_main(cfg : DictConfig) -> None:
        #cfg = OmegaConf.merge(config_dict, cfg)
        print(OmegaConf.to_yaml(cfg))

        _run(cfg)

    hydra_main()

