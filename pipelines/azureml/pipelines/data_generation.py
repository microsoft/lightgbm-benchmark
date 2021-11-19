"""
Runs LightGBM using distributed (mpi) training.

to execute:
> python pipelines/data_generation.py --config-dir ./conf --config-name experiments/data-generation run.submit=True
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
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, MISSING
from typing import Optional, List

# AzureML
from azure.ml.component import Component
from azure.ml.component import dsl
from shrike.pipeline.pipeline_config import default_config_dict
from shrike.pipeline.aml_connect import azureml_connect, current_workspace

# when running this script directly, needed to import common
LIGHTGBM_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SCRIPTS_SOURCES_ROOT = os.path.join(LIGHTGBM_REPO_ROOT, 'src')
COMPONENTS_ROOT = os.path.join(LIGHTGBM_REPO_ROOT, 'src', 'scripts')
CONFIG_PATH = os.path.join(LIGHTGBM_REPO_ROOT, 'pipelines', 'azureml', 'conf')

if SCRIPTS_SOURCES_ROOT not in sys.path:
    logging.info(f"Adding {SCRIPTS_SOURCES_ROOT} to path")
    sys.path.append(str(SCRIPTS_SOURCES_ROOT))

from common.tasks import data_generation_task

### CONFIG DATACLASS ###

@dataclass
class data_generation: # pylint: disable=invalid-name
    """ Config object constructed as a dataclass.

    NOTE: the name of this class will be used as namespace in your config yaml file.
    See conf/reference/evaluate_qas_model.yaml for an example.
    """
    # NOTE: all those values are REQUIRED in your yaml config file
    benchmark_name: str = MISSING

    tasks: List[data_generation_task] = MISSING

    # OUTPUT REGISTRATION
    register_outputs: bool = False
    register_outputs_prefix: str = "synthetic"

### PIPELINE COMPONENTS ###

generate_data_component = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "data_processing", "generate_data", "spec.yaml"))

### DATA GENERATION TASKS ###

@dsl.pipeline(name="generate_all_datasets", # pythonic name
              non_pipeline_parameters=["config"])
def all_inferencing_tasks_pipeline_function(config):
    benchmark_custom_properties = json.dumps({
        'benchmark_name' : config.data_generation.benchmark_name
    })
    full_pipeline_description="\n".join([
        "Generate all datasets for lightgbm benchmark",
        "```yaml""",
        OmegaConf.to_yaml(config.data_generation),
        "```"
    ])

    if len(full_pipeline_description) > 5000:
        full_pipeline_description = full_pipeline_description[:5000-50] + "\n<<<TRUNCATED DUE TO SIZE LIMIT>>>"

    for generation_task in config.data_generation.tasks:
        generate_data_step = generate_data_component(
            learning_task = generation_task.task,
            train_samples = generation_task.train_samples,
            test_samples = generation_task.test_samples,
            inferencing_samples = generation_task.inferencing_samples,
            n_features = generation_task.n_features,
            n_informative = generation_task.n_informative,
            random_state = 5,
            verbose = False,
            custom_properties = benchmark_custom_properties
        )

        if config.data_generation.register_outputs:
            dataset_prefix = "{prefix}-{task}-{cols}cols".format(
                prefix=config.data_generation.register_outputs_prefix,
                task=generation_task.task,
                cols=generation_task.n_features
            )
            
            generate_data_step.outputs.train.register_as(
                name=f"{dataset_prefix}-{generation_task.train_samples}samples-train",
                create_new_version=True
            )  
            generate_data_step.outputs.test.register_as(
                name=f"{dataset_prefix}-{generation_task.test_samples}samples-test",
                create_new_version=True
            )  
            generate_data_step.outputs.inference.register_as(
                name=f"{dataset_prefix}-{generation_task.inferencing_samples}samples-inference",
                create_new_version=True
            )  

def run(pipeline_config):
    pipeline_instance = all_inferencing_tasks_pipeline_function(pipeline_config)

    # Connect to AzureML
    workspace = azureml_connect(
        aml_subscription_id=pipeline_config.aml.subscription_id,
        aml_resource_group=pipeline_config.aml.resource_group,
        aml_workspace_name=pipeline_config.aml.workspace_name,
        aml_auth=pipeline_config.aml.auth,
        aml_tenant=pipeline_config.aml.tenant,
        #aml_force=pipeline_config.aml.force,
    )  # NOTE: this also stores aml workspace in internal global variable

    # Submit or Validate ?
    pipeline_instance.validate(workspace=workspace)


def cli_main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exp-conf", required=True)
    args, hydra_args = arg_parser.parse_known_args()

    #config_dict = default_config()
    # resolve config_dir and config_name
    config_abspath = os.path.abspath(args.exp_conf)
    print(f"CONFIG_PATH={CONFIG_PATH}")
    print(f"config_path={args.exp_conf}")
    print(f"config_abspath={config_abspath}")
    print(f"unknown_args={hydra_args}")

    # hacky, need a better solution
    config_name = os.path.relpath(config_abspath, start=CONFIG_PATH).replace("\\","/").rstrip(".yaml")
    print(f"config_name={config_name}")

    @hydra.main(config_path=CONFIG_PATH, config_name=config_name)
    def hydra_main(cfg : DictConfig) -> None:
        #cfg = OmegaConf.merge(config_dict, cfg)
        print(OmegaConf.to_yaml(cfg))

        run(cfg)


    # override argv with only unknown args
    sys.argv = [sys.argv[0]] + hydra_args
    hydra_main()


# NOTE: main block is necessary only if script is intended to be run from command line
if __name__ == "__main__":
    cli_main()
