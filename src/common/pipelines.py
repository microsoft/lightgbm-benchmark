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
from omegaconf import MISSING
from typing import Any, Optional
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from shrike.pipeline.pipeline_config import default_config_dict
from shrike.pipeline.aml_connect import azureml_connect

# when running this script directly, needed to import common
from .paths import COMPONENTS_ROOT, CONFIG_PATH


@dataclass
class aml_connection_config:  # pylint: disable=invalid-name
    """AML connection configuration"""
    subscription_id: str = MISSING
    resource_group: str = MISSING
    workspace_name: str = MISSING
    tenant: Optional[str] = None
    auth: str = "interactive"
    force: bool = False

@dataclass
class compute_config:  # pylint: disable=invalid-name
    """AML workspace compute targets and I/O modes"""
    default_compute_target: str = "cpu-cluster"
    linux_cpu: str = MISSING
    linux_gpu: str = MISSING
    windows_cpu: str = MISSING

@dataclass
class run_config:  # pylint: disable=invalid-name
    """Pipeline config for command line parameters"""
    regenerate_outputs: bool = False
    continue_on_failure: bool = False
    submit: bool = False
    validate: bool = True

@dataclass
class experiment_config:  # pylint: disable=invalid-name
    """Pipeline config for command line parameters"""
    name: str = MISSING
    description: Optional[str] = None
    display_name: Optional[str] = None
    tags: Optional[Any] = None


def pipeline_cli_main(pipeline_config_dataclass,
                      pipeline_func,
                      experiment_name=None,
                      experiment_description=None,
                      display_name=None,
                      tags=None,
                      cli_args=None):
    """ (soon to be) Standard main function
    
    Args:
        pipeline_config_dataclass (dataclass): class for hosting the config of pipeline_func
        pipeline_func (function): pipeline building function to call with config object
        cli_args (List): command line arguments (if None, use sys.argv)
    """
    # create an argument parser just to catch --exp-conf
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("--exp-config", dest="exp_conf", required=False, default=None)
    # all remaining arguments will be passed to hydra
    args, unknown_args = arg_parser.parse_known_args(cli_args)

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
    config_dict = {
        "aml": aml_connection_config,
        "compute": compute_config,
        "run": run_config,
        "experiment": experiment_config
    }

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
        if pipeline_config.run.validate:
            pipeline_instance.validate(workspace=workspace)
        
        if pipeline_config.run.submit:
            pipeline_run = pipeline_instance.submit(
                workspace=workspace,
                experiment_name=(experiment_name or pipeline_config.experiment.name),
                description=(experiment_description or pipeline_config.experiment.description),
                display_name=(display_name or pipeline_config.experiment.display_name),
                tags=(tags or pipeline_config.experiment.tags),
                default_compute_target=pipeline_config.compute.default_compute_target,
                regenerate_outputs=pipeline_config.run.regenerate_outputs,
                continue_on_step_failure=pipeline_config.run.continue_on_failure,
            )

    @hydra.main(config_name="default")
    def hydra_main(cfg : DictConfig) -> None:
        cfg = OmegaConf.merge(config_dict, cfg)
        print(OmegaConf.to_yaml(cfg))

        _run(cfg)

    hydra_main()
