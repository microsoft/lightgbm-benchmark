"""
Generates synthetic data with multiple parameters.

See config file /conf/experiments/data-generation.yaml

to execute:
> python src/pipelines/azureml/data_generation.py --exp-config conf/experiments/data-generation.yaml
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
from omegaconf import OmegaConf, MISSING
from typing import Optional, List

# AzureML SDK 2.0
from azure.ml import dsl

# when running this script directly, needed to import common
LIGHTGBM_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SCRIPTS_SOURCES_ROOT = os.path.join(LIGHTGBM_REPO_ROOT, 'src')

if SCRIPTS_SOURCES_ROOT not in sys.path:
    logging.info(f"Adding {SCRIPTS_SOURCES_ROOT} to path")
    sys.path.append(str(SCRIPTS_SOURCES_ROOT))

from common.tasks import data_generation_task
from common.pipelines import (
    parse_pipeline_config,
    azureml_connect,
    pipeline_submit,
    COMPONENTS_ROOT
)
from common.aml import format_run_name


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

generate_data_component = dsl.load_component(yaml_file=os.path.join(COMPONENTS_ROOT, "data_processing", "generate_data", "spec.yaml"))

### DATA GENERATION PIPELINE ###

# Step 3: your pipeline consists in creating a python function
# decorated with @dsl.pipeline.
# You can create as many subgraphs as you want,
# but `pipeline_cli_main` will need one pipeline function
# taking a single config argument, not a pipeline parameter.

def data_generation_main_pipeline_function(config):
    """Pipeline's main building function.

    Args:
        config (DictObject): the pipeline configuration object containing pipeline config dataclass
            This is not an actual pipeline parameter
    
    Returns:
        None
    """
    @dsl.pipeline(
        name="generate_all_datasets", # pythonic name
    )
    def _data_generation_main_pipeline_function():
        benchmark_custom_properties = json.dumps({
            'benchmark_name' : config.data_generation_config.benchmark_name
        })

        # for each task provided in the general config
        for generation_task in config.data_generation_config.tasks:

            # run a generation step with the right parameters
            generate_data_step = generate_data_component(
                learning_task = generation_task.task,
                train_samples = generation_task.train_samples,
                train_partitions = generation_task.train_partitions,
                test_samples = generation_task.test_samples,
                test_partitions = generation_task.test_partitions,
                inferencing_samples = generation_task.inferencing_samples,
                inferencing_partitions = generation_task.inferencing_partitions,
                n_features = generation_task.n_features,
                n_informative = generation_task.n_informative,
                n_label_classes = generation_task.n_label_classes,
                docs_per_query = generation_task.docs_per_query,
                delimiter = generation_task.delimiter,
                header = generation_task.header,
                random_state = 5,
                verbose = False,
                custom_properties = benchmark_custom_properties
            )
            # run it on the right compute target
            generate_data_step.runsettings.configure(target=config.compute.linux_cpu)

        # generate a readable run name
        generate_data_step.node_name = format_run_name("generate_{}_train{}test{}inf{}_feat{}".format(
            generation_task.task,
            generation_task.train_samples,
            generation_task.test_samples,
            generation_task.inferencing_samples,
            generation_task.n_features
        ))

        # if config asks to register the outputs automatically...
        if config.data_generation_config.register_outputs:
            # create a prefix for the dataset
            dataset_prefix = "{prefix}-{task}-{cols}cols".format(
                prefix=config.data_generation_config.register_outputs_prefix,
                task=generation_task.task,
                cols=generation_task.n_features
            )
            # run it on the right compute target
            generate_data_step.compute = config.compute.linux_cpu

            # if config asks to register the outputs automatically...
            if config.data_generation_config.register_outputs:
                # create a prefix for the dataset
                dataset_prefix = "{prefix}-{task}-{cols}cols".format(
                    prefix=config.data_generation_config.register_outputs_prefix,
                    task=generation_task.task,
                    cols=generation_task.n_features
                )
                
                # register each output (train, test, inference)
                generate_data_step.outputs.output_train.register_as(
                    name=f"{dataset_prefix}-{generation_task.train_samples}samples-train",
                    create_new_version=True,
                    tags={ # add tags that will show up in AzureML
                        'type':'train',
                        'task':generation_task.task,
                        'origin':'synthetic',
                        'samples':generation_task.train_samples,
                        'features':generation_task.n_features,
                        'informative':generation_task.n_informative
                    }
                )
                generate_data_step.outputs.output_test.register_as(
                    name=f"{dataset_prefix}-{generation_task.test_samples}samples-test",
                    create_new_version=True,
                    tags={ # add tags that will show up in AzureML
                        'type':'test',
                        'task':generation_task.task,
                        'origin':'synthetic',
                        'samples':generation_task.test_samples,
                        'features':generation_task.n_features,
                        'informative':generation_task.n_informative
                    }
                )
                generate_data_step.outputs.output_inference.register_as(
                    name=f"{dataset_prefix}-{generation_task.inferencing_samples}samples-inference",
                    create_new_version=True,
                    tags={ # add tags that will show up in AzureML
                        'type':'inference',
                        'task':generation_task.task,
                        'origin':'synthetic',
                        'samples':generation_task.inferencing_samples,
                        'features':generation_task.n_features,
                        'informative':generation_task.n_informative
                    }
                )  

    return _data_generation_main_pipeline_function()

### MAIN BLOCK ###

# Step 4: implement main block using helper functions

def main():
    # use parse helper function to get arguments from CLI
    config = parse_pipeline_config(data_generation_config)
    
    # you'll need a workspace object to connect
    ml_client = azureml_connect(config)

    # run the pipeline function with the given arguments
    pipeline_instance = data_generation_main_pipeline_function(config)

    # generate a nice markdown description
    experiment_description="\n".join([
        "Generating synthetic datasets (see yaml below).",
        "```yaml",
        "data_generation_config:",
        OmegaConf.to_yaml(config.data_generation_config),
        "```"
    ])

    # validate/submit the pipeline (if run.submit=True)
    pipeline_submit(
        ml_client,
        config,
        pipeline_instance,
        experiment_description=experiment_description
    )

if __name__ == "__main__":
    main()
