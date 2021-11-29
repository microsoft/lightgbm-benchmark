"""
Runs LightGBM using distributed (mpi) training.

to execute:
> python pipelines/data_generation.py --exp-config ./pipelines/azureml/conf/experiments/data-generation.yaml
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
from omegaconf import DictConfig, OmegaConf, MISSING
from typing import Optional, List

# AzureML
from azure.ml.component import Component
from azure.ml.component import dsl

# when running this script directly, needed to import common
LIGHTGBM_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SCRIPTS_SOURCES_ROOT = os.path.join(LIGHTGBM_REPO_ROOT, 'src')

if SCRIPTS_SOURCES_ROOT not in sys.path:
    logging.info(f"Adding {SCRIPTS_SOURCES_ROOT} to path")
    sys.path.append(str(SCRIPTS_SOURCES_ROOT))

from common.tasks import data_generation_task
from common.pipelines import pipeline_cli_main, COMPONENTS_ROOT

### CONFIG DATACLASS ###

# Step 1 : to configure your pipeline, add all your fields inside a
# properly defined dataclass, pipeline_cli_main will figure out how
# to read that config from a given yaml file + hydra override commands

@dataclass
class data_generation_config: # pylint: disable=invalid-name
    """ Config object constructed as a dataclass.

    The name of this class will be used as namespace in your config yaml file.
    """
    # NOTE: all those values are REQUIRED in your yaml config file
    benchmark_name: str = MISSING

    tasks: List[data_generation_task] = MISSING

    # OUTPUT REGISTRATION
    register_outputs: bool = False
    register_outputs_prefix: str = "synthetic"

### PIPELINE COMPONENTS ###

# Step 2 : your pipeline consists in assembling components
# load those components from local yaml specifications
# use COMPONENTS_ROOT as base folder

generate_data_component = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "data_processing", "generate_data", "spec.yaml"))

### DATA GENERATION TASKS ###

# Step 3: your pipeline consists in creating a python function
# decorated with @dsl.pipeline.
# You can create as many subgraphs as you want,
# but `pipeline_cli_main` will need one pipeline function
# taking a single config argument, not a pipeline parameter.

@dsl.pipeline(
    name="generate_all_datasets", # pythonic name
    non_pipeline_parameters=["config"] # required to use config object
)
def data_generation_main_pipeline_function(config):
    """Pipeline's main building function.

    Args:
        config (DictObject): the pipeline configuration object containing pipeline config dataclass
            This is not an actual pipeline parameter
    
    Returns:
        None
    """
    benchmark_custom_properties = json.dumps({
        'benchmark_name' : config.data_generation_config.benchmark_name
    })
    full_pipeline_description="\n".join([
        "Generate all datasets for lightgbm benchmark",
        "```yaml""",
        OmegaConf.to_yaml(config.data_generation_config),
        "```"
    ])

    if len(full_pipeline_description) > 5000:
        full_pipeline_description = full_pipeline_description[:5000-50] + "\n<<<TRUNCATED DUE TO SIZE LIMIT>>>"

    for generation_task in config.data_generation_config.tasks:
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

        if config.data_generation_config.register_outputs:
            dataset_prefix = "{prefix}-{task}-{cols}cols".format(
                prefix=config.data_generation_config.register_outputs_prefix,
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

if __name__ == "__main__":
    # use standard cli main to get arguments from CLI
    pipeline_cli_main(
        # pipeline configuration class
        data_generation_config,

        # main pipeline function
        data_generation_main_pipeline_function
    )
