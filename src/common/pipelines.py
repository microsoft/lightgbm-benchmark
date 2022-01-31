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

from azureml.core import Workspace
from azureml.pipeline.core import Pipeline
from shrike.pipeline.aml_connect import azureml_connect as shrike_azureml_connect

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
    """AML workspace compute targets"""
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
    """Pipeline config for experiment parameters"""
    name: str = MISSING
    description: Optional[str] = None
    display_name: Optional[str] = None
    tags: Optional[Any] = None


_GLOBAL_CONFIG = None

def parse_pipeline_config(pipeline_config_dataclass: dataclass, cli_args: list=None):
    """Standard helper function to submit a pipeline to AzureML.

    This is a lightweight version of what Shrike does (https://github.com/Azure/shrike).
    
    Args:
        pipeline_config_dataclass (dataclass): class for hosting the config of pipeline_func
        cli_args (List): command line arguments (if None, use sys.argv)
    """
    # create an argument parser just to catch --exp-conf
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("--exp-config", dest="exp_conf", required=False, default=None)
    # all remaining arguments will be passed to hydra
    args, unknown_args = arg_parser.parse_known_args(cli_args)

    # resolve config_dir and config_name from --exp-conf
    if args.exp_conf:
        config_abspath = os.path.abspath(args.exp_conf)
        config_relpath = os.path.relpath(config_abspath, start=CONFIG_PATH) # relative path from config folder to specified config yaml
        config_name = os.path.dirname(config_relpath).replace("\\", "/") + "/" + os.path.basename(config_relpath)
        logging.getLogger(__name__).info(f"Using config_name={config_name} and config_dir={CONFIG_PATH}")
    else:
        config_name = None

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

    # override argv with only hydra args (TODO: do better)
    sys.argv = (
        [sys.argv[0]]
        + unknown_args
        + [
            "--config-dir", CONFIG_PATH,
            "--config-name", config_name,
        ]
    )

    # create a hydra main function to get overrides
    @hydra.main(config_name="default")
    def hydra_main(cfg : DictConfig) -> None:
        global _GLOBAL_CONFIG
        cfg = OmegaConf.merge(config_dict, cfg)
        #logging.getLogger(__name__).info(OmegaConf.to_yaml(cfg))

        _GLOBAL_CONFIG = cfg


    # call the hydra main function
    hydra_main()

    return _GLOBAL_CONFIG.copy()


def azureml_connect(config: DictConfig):
    """Connects to AzureML.
    
    Args:
        config (DictConfig): containing aml_config dataclass

    Returns:
        workspace (azure.ml.core.Workspace)
    """
    return shrike_azureml_connect(
        aml_subscription_id=config.aml.subscription_id,
        aml_resource_group=config.aml.resource_group,
        aml_workspace_name=config.aml.workspace_name,
        aml_auth=config.aml.auth,
        aml_tenant=config.aml.tenant,
        aml_force=config.aml.force
    )

def pipeline_submit(workspace: Workspace,
                    pipeline_config: DictConfig,
                    pipeline_instance: Pipeline,
                    experiment_name: str=None,
                    experiment_description: str=None,
                    display_name: str=None,
                    tags: dict=None):
    """Standard helper function to submit a pipeline to AzureML.

    Args:
        workspace (azure.ml.core.Workspace): AzureML workspace (see azureml_connect())
        pipeline_config (DictConfig): class for hosting the config of pipeline_func
        pipeline_instance (Pipeline): pipeline object
        experiment_name (str): override config.experiment.name at runtime
        experiment_description (str): override config.experiment.description at runtime
        display_name (str): override config.experiment.display_name at runtime
        tags (dict): override config.experiment.tags at runtime

    Returns:
        pipeline (azure.ml.core.PipelineRun)
    """
    if pipeline_config.run.validate:
        pipeline_instance.validate(workspace=workspace)

    experiment_description = (experiment_description or pipeline_config.experiment.description)
    if experiment_description and len(experiment_description) > 5000:
        experiment_description = experiment_description[:5000-50] + "\n<<<TRUNCATED DUE TO SIZE LIMIT>>>"

    if pipeline_config.run.submit:
        pipeline_run = pipeline_instance.submit(
            workspace=workspace,
            experiment_name=(experiment_name or pipeline_config.experiment.name),
            description=experiment_description,
            display_name=(display_name or pipeline_config.experiment.display_name),
            tags=(tags or pipeline_config.experiment.tags),
            default_compute_target=pipeline_config.compute.default_compute_target,
            regenerate_outputs=pipeline_config.run.regenerate_outputs,
            continue_on_step_failure=pipeline_config.run.continue_on_failure,
        )

        logging.info(
            f"""
#################################
#################################
#################################

Follow link below to access your pipeline run directly:
-------------------------------------------------------
{pipeline_run.get_portal_url()}

#################################
#################################
#################################
        """
        )

        return pipeline_run
    else:
        logging.warning("Pipeline was not submitted, to submit it please add +run.submit=true to your command.")
