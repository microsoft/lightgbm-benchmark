"""
Compares 3 versions of LightGBM Inferencing
A - vanilla
B - custom build
C - treelite

to execute:
> python src/pipelines/azureml/lightgbm_inferencing.py --exp-config  conf/experiments/lightgbm-inferencing.yaml
"""
# pylint: disable=no-member
# NOTE: because it raises 'dict' has no 'outputs' member in dsl.pipeline construction
import os
import sys
import json
import logging

# config management
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf, MISSING
from typing import Optional, List

# AzureML
from azure.ml.component import Component
from azure.ml.component import dsl
from azure.ml.component.environment import Docker

# when running this script directly, needed to import common
LIGHTGBM_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SCRIPTS_SOURCES_ROOT = os.path.join(LIGHTGBM_REPO_ROOT, 'src')

if SCRIPTS_SOURCES_ROOT not in sys.path:
    logging.info(f"Adding {SCRIPTS_SOURCES_ROOT} to path")
    sys.path.append(str(SCRIPTS_SOURCES_ROOT))

from common.tasks import inferencing_task, inferencing_variants
from common.pipelines import (
    parse_pipeline_config,
    azureml_connect,
    pipeline_submit,
    COMPONENTS_ROOT
)
from common.aml import load_dataset_from_data_input_spec, format_run_name

### CONFIG DATACLASS ###

# Step 1 : to configure your pipeline, add all your fields inside a
# properly defined dataclass, pipeline_cli_main will figure out how
# to read that config from a given yaml file + hydra override commands

@dataclass
class lightgbm_inferencing_config: # pylint: disable=invalid-name
    """ Config object constructed as a dataclass.

    NOTE: the name of this class will be used as namespace in your config yaml file.
    See conf/reference/evaluate_qas_model.yaml for an example.
    """
    benchmark_name: str = MISSING
    tasks: List[inferencing_task] = MISSING
    variants: List[inferencing_variants] = MISSING

### PIPELINE COMPONENTS ###

# Step 2 : your pipeline consists in assembling components
# load those components from local yaml specifications
# use COMPONENTS_ROOT as base folder

lightgbm_python_score_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "inferencing", "lightgbm_python", "spec.yaml"))
lightgbm_c_api_score_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "inferencing", "lightgbm_c_api", "spec.yaml"))
lightgbm_ray_score_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "inferencing", "lightgbm_ray", "spec.yaml"))
lightgbm_ort_score_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "inferencing", "lightgbm_ort", "spec.yaml"))
custom_win_cli_score_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "inferencing", "custom_win_cli", "spec.yaml"))
treelite_compile_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "model_transformation", "treelite_compile", "spec.yaml"))
treelite_score_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "inferencing", "treelite_python", "spec.yaml"))

### INFERENCING TASKS ###

# Step 3: your pipeline consists in creating a python function
# decorated with @dsl.pipeline.
# You can create as many subgraphs as you want,
# but `pipeline_cli_main` will need one pipeline function
# taking a single config argument, not a pipeline parameter.

@dsl.pipeline(name=f"lightgbm_inferencing", # pythonic name
                description=f"LightGBM inferencing on user defined dataset/model",
                non_pipeline_parameters=['benchmark_custom_properties', 'config'])
def inferencing_task_pipeline_function(benchmark_custom_properties,
                                       config,
                                       data,
                                       model,
                                       predict_disable_shape_check):
    """This pipeline consists in running multiple inferencing
    frameworks in parallel on a given input data/model pair.

    Args:
        TODO

    Returns:
        dict[str->PipelineOutputData]: a dictionary of your pipeline outputs
            for instance to be consumed by other graphs
    """
    # creating a dict to store pipeline outputs
    pipeline_outputs = {}

    # loop through all inferencing variants
    for variant_index, variant in enumerate(config.lightgbm_inferencing_config.variants):
        # add last minute custom proeprties
        custom_properties = benchmark_custom_properties.copy()
        custom_properties.update({
            # adding build settings (docker)
            'framework_build' : variant.build or "default",
            # adding variant_index to spot which variant is the reference
            'variant_index' : variant_index
        })
        # passing as json string that each module parses to digest as tags/properties
        custom_properties = json.dumps(custom_properties)

        # list of comments to surface on the component itself
        variant_comment = [
            f"variant #{variant_index}"
        ]

        if variant.framework == "treelite_python":
            treelite_compile_step = treelite_compile_module(
                model = model,
                verbose = False,
                custom_properties = custom_properties
            )
            treelite_compile_step.runsettings.configure(target=config.compute.linux_cpu)

            inferencing_step = treelite_score_module(
                data = data,
                compiled_model = treelite_compile_step.outputs.compiled_model,
                verbose = False,
                custom_properties = custom_properties
            )
            inferencing_step.runsettings.configure(target=config.compute.linux_cpu)

        elif variant.framework == "lightgbm_c_api":
            # call module with all the right arguments
            inferencing_step = lightgbm_c_api_score_module(
                data = data,
                model = model,
                predict_disable_shape_check = predict_disable_shape_check,
                verbose = False,
                custom_properties = custom_properties
            )
            inferencing_step.runsettings.configure(target=config.compute.linux_cpu)

        elif variant.framework == "custom_win_cli":
            # call module with all the right arguments
            inferencing_step = custom_win_cli_score_module(
                data = data,
                model = model,
                verbose = False,
                custom_properties = custom_properties.replace("\"","\\\"")
            )
            inferencing_step.runsettings.configure(target=config.compute.windows_cpu)

        elif variant.framework == "lightgbm_python":
            # call module with all the right arguments
            inferencing_step = lightgbm_python_score_module(
                data = data,
                model = model,
                predict_disable_shape_check = predict_disable_shape_check,
                verbose = False,
                custom_properties = custom_properties
            )
            inferencing_step.runsettings.configure(target=config.compute.linux_cpu)

        elif variant.framework == "lightgbm_ray":
            # call module with all the right arguments
            inferencing_step = lightgbm_ray_score_module(
                data = data,
                model = model,
                verbose = False,
                custom_properties = custom_properties
            )
            inferencing_step.runsettings.configure(target=config.compute.linux_cpu)

        elif variant.framework == "lightgbm_ort":
            # call module with all the right arguments
            inferencing_step = lightgbm_ort_score_module(
                data = data,
                model = model,
                verbose = False,
                run_parallel = variant.parallel_exec,
                run_batch = variant.batch_exec,
                n_threads = variant.threads,
                custom_properties = custom_properties
            )
            inferencing_step.runsettings.configure(target=config.compute.linux_cpu)

            if variant.parallel_exec:
                variant_comment.append(f"parallel execution")
            if variant.batch_exec:
                variant_comment.append(f"batch execution")
            variant_comment.append(f"num threads {variant.threads}")

        else:
            raise NotImplementedError(f"framework {variant.framework} not implemented (yet)")

        if variant.build:
            # build path is relative to docker/ subfolder
            custom_docker = Docker(file=os.path.join(LIGHTGBM_REPO_ROOT, variant.build))
            inferencing_step.runsettings.environment.configure(
                docker=custom_docker
            )
            variant_comment.append(f"build {variant.build}")
        else:
            variant_comment.append(f"default build")


        # add some comment to the component
        inferencing_step.comment = " -- ".join(variant_comment)

        # provide step readable display name
        inferencing_step.node_name = format_run_name(f"inferencing_{variant.framework}_{variant_index}")

    # return {key: output}'
    return pipeline_outputs


@dsl.pipeline(
    name="inferencing_all_tasks",
    non_pipeline_parameters=["workspace", "config"] # required to use config object
)
def inferencing_all_tasks(workspace, config):
    """Pipeline's main building function.

    Args:
        workspace (azureml.core.Workspace): the AzureML workspace
            This is not an actual pipeline parameter
        config (DictConfig): the pipeline configuration object containing pipeline config dataclass
            This is not an actual pipeline parameter

    Returns:
        None
    """
    for task_index, inferencing_task in enumerate(config.lightgbm_inferencing_config.tasks):
        data = load_dataset_from_data_input_spec(workspace, inferencing_task.data)
        model = load_dataset_from_data_input_spec(workspace, inferencing_task.model)

        # create custom properties for this task
        benchmark_custom_properties = {
            'benchmark_name' : config.lightgbm_inferencing_config.benchmark_name, 
            'benchmark_dataset' : inferencing_task.data.name,
            'benchmark_model' : inferencing_task.model.name,
        }

        inferencing_task_subgraph_step = inferencing_task_pipeline_function(
            data=data,
            model=model,
            predict_disable_shape_check=inferencing_task.predict_disable_shape_check or False,
            benchmark_custom_properties=benchmark_custom_properties,
            config=config
        )

        # add some relevant comments on the subgraph
        inferencing_task_subgraph_step.comment = " -- ".join([
            f"benchmark name: {config.lightgbm_inferencing_config.benchmark_name}",
            # NOTE: add more here?
        ])

        # provide readable subgraph display name
        inferencing_task_subgraph_step.node_name = f"inferencing_task_{task_index}"


### MAIN BLOCK ###

# Step 4: implement main block

def main():
    # use parse helper function to get arguments from CLI
    config = parse_pipeline_config(lightgbm_inferencing_config)
    
    # you'll need a workspace object to connect
    workspace = azureml_connect(config)

    # run the pipeline function with the given arguments
    pipeline_instance = inferencing_all_tasks(workspace, config)

    experiment_description="\n".join([
        "Inferencing on all specified tasks (see yaml below).",
        "```yaml",
        "lightgbm_inferencing_config:",
        OmegaConf.to_yaml(config.lightgbm_inferencing_config),
        "```"
    ])

    # validate/submit the pipeline (if run.submit=True)
    pipeline_submit(
        workspace,
        config,
        pipeline_instance,
        experiment_description=experiment_description
    )

if __name__ == "__main__":
    main()
