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
from dataclasses import dataclass
from omegaconf import MISSING, OmegaConf
from typing import Optional, List
from azure.ml.component import dsl
from shrike.pipeline.pipeline_helper import AMLPipelineHelper
from azure.ml.component.environment import Docker

# when running this script directly, needed to import common
LIGHTGBM_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
COMPONENTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'scripts'))

if LIGHTGBM_BENCHMARK_ROOT not in sys.path:
    print(f"Adding {LIGHTGBM_BENCHMARK_ROOT} to path")
    sys.path.append(str(LIGHTGBM_BENCHMARK_ROOT))

from common.sweep import SweepParameterParser
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

generate_data_component = Component.from_yaml(os.path.join(COMPONENTS_ROOT, "data_processing", "generate_data", "spec.yaml"))

### DATA GENERATION TASKS ###

@dsl.pipeline(name="generate_all_datasets", # pythonic name
              non_pipeline_parameters=["data_generation_config"])
def all_inferencing_tasks_pipeline_function(data_generation_config):
    benchmark_custom_properties = json.dumps({
        'benchmark_name' : data_generation_config.benchmark_name
    })
    full_pipeline_description="\n".join([
        "Generate all datasets for lightgbm benchmark",
        "```yaml""",
        OmegaConf.to_yaml(data_generation_config),
        "```"
    ])

    if len(full_pipeline_description) > 5000:
        full_pipeline_description = full_pipeline_description[:5000-50] + "\n<<<TRUNCATED DUE TO SIZE LIMIT>>>"

    for generation_task in data_generation_config.tasks:
        generate_data_step = generate_data_module(
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

        if data_generation_config.register_outputs:
            dataset_prefix = "{prefix}-{task}-{cols}cols".format(
                prefix=data_generation_config.register_outputs_prefix,
                task=generation_task.task,
                cols=generation_task.n_features
            )
            
            generation_task_subgraph_step.outputs.train.register_as(
                name=f"{dataset_prefix}-{generation_task.train_samples}samples-train",
                create_new_version=True
            )  
            generation_task_subgraph_step.outputs.test.register_as(
                name=f"{dataset_prefix}-{generation_task.test_samples}samples-test",
                create_new_version=True
            )  
            generation_task_subgraph_step.outputs.inference.register_as(
                name=f"{dataset_prefix}-{generation_task.inferencing_samples}samples-inference",
                create_new_version=True
            )  


def main():


# NOTE: main block is necessary only if script is intended to be run from command line
if __name__ == "__main__":
    main()
