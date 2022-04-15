"""
Runs LightGBM using distributed (mpi) training.

to execute:
> python src/pipelines/azureml/lightgbm_training.py --exp-config conf/experiments/lightgbm_training/cpu.yaml
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
from typing import Optional, Any, List

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

from common.tasks import training_task, training_variant
from common.sweep import SweepParameterParser
from common.aml import load_dataset_from_data_input_spec
from common.aml import apply_sweep_settings
from common.aml import format_run_name
from common.pipelines import (
    parse_pipeline_config,
    azureml_connect,
    pipeline_submit,
    COMPONENTS_ROOT
)

### CONFIG DATACLASS ###

# Step 1 : to configure your pipeline, add all your fields inside a
# properly defined dataclass, pipeline_cli_main will figure out how
# to read that config from a given yaml file + hydra override commands

@dataclass
class lightgbm_training_config: # pragma: no cover
    """ Config object constructed as a dataclass.

    NOTE: the name of this class will be used as namespace in your config yaml file.
    """
    # NOTE: all those values are REQUIRED in your yaml config file
    benchmark_name: str = MISSING

    # INPUT DATASETS
    tasks: List[training_task] = MISSING

    # TRAINING PARAMS
    reference: training_variant = MISSING

    # free changing parameters on top of reference
    variants: Optional[Any] = None

### PIPELINE COMPONENTS ###

# Step 2 : your pipeline consists in assembling components
# load those components from local yaml specifications
# use COMPONENTS_ROOT as base folder

# lightgbm python api with socket (pip install lightgbm)
lightgbm_basic_train_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "training", "lightgbm_python", "spec.yaml"))
lightgbm_basic_train_sweep_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "training", "lightgbm_python", "sweep_spec.yaml"))

# lightgbm ray api
lightgbm_ray_train_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "training", "lightgbm_ray", "spec.yaml"))

# preprocessing/utility modules
partition_data_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "data_processing", "partition_data", "spec.yaml"))
lightgbm_data2bin_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "data_processing", "lightgbm_data2bin", "spec.yaml"))

# load ray tune module.
lightgbm_ray_tune_module = Component.from_yaml(yaml_file=os.path.join(COMPONENTS_ROOT, "training", "ray_tune", "spec.yaml"))

### PIPELINE SPECIFIC CODE ###

def process_sweep_parameters(params_dict, sweep_algorithm):
    """Parses config and spots sweepable paraneters
    
    Args:
        params_dict (dict): configuration object (see get_config_class())
        sweep_algorithm (str): random, grid, bayesian

    Returns:
        tunable_params (dict)
    """
    # the class below automates parsing of sweepable parameters
    sweep_param_parser = SweepParameterParser(
        tunable_parameters=[
            # those are keys and their default values
            "num_iterations",
            "num_leaves",
            "min_data_in_leaf",
            "learning_rate",
            "max_bin",
            "feature_fraction"
        ],
        cli_prefix=None, # this is not argparse
        parameter_sampling=sweep_algorithm
    )

    # provide config as a dictionary to the parser
    sweep_parameters = {
        "num_iterations": params_dict['num_iterations'],
        "num_leaves": params_dict['num_leaves'],
        "min_data_in_leaf": params_dict['min_data_in_leaf'],
        "learning_rate": params_dict['learning_rate'],
        "max_bin": params_dict['max_bin'],
        "feature_fraction": params_dict['feature_fraction'],
    }

    # parser gonna parse
    sweep_param_parser.parse_from_dict(sweep_parameters)

    # and return params as we want them
    tunable_params = sweep_param_parser.get_tunable_params()
    fixed_params = sweep_param_parser.get_fixed_params()

    # return dictionaries to fed as params into our pipeline
    return tunable_params, fixed_params


### TRAINING PIPELINE ###

# Step 3: your pipeline consists in creating a python function
# decorated with @dsl.pipeline.
# You can create as many subgraphs as you want,
# but `pipeline_cli_main` will need one pipeline function
# taking a single config argument, not a pipeline parameter.

# Here you should create an instance of a pipeline function (using your custom config dataclass)
@dsl.pipeline(
    name="lightgbm_training", # pythonic name
    description="LightGBM distributed training (mpi)",
    non_pipeline_parameters=['config', 'benchmark_custom_properties']
)
def lightgbm_training_pipeline_function(config,
                                        benchmark_custom_properties,
                                        train_dataset,
                                        test_dataset):
    """Pipeline function for this graph.

    Args:
        TODO

    Returns:
        dict[str->PipelineOutputData]: a dictionary of your pipeline outputs
            for instance to be consumed by other graphs
    """
    # create list of all variants params
    training_variants_params = [
        config.lightgbm_training_config.reference
    ]

    # if there's any variant specified
    if config.lightgbm_training_config.variants:
        # create distinct training params for each variant
        for variant_index, training_variant_config in enumerate(config.lightgbm_training_config.variants):
            # create a specific dict of params for the variant
            variant_config = OmegaConf.merge(config.lightgbm_training_config.reference, training_variant_config)
            training_variants_params.append(variant_config)

    # for each variant, check if sweep needs to be applied
    for variant_index, variant_params in enumerate(training_variants_params):
        ############
        ### DATA ###
        ############

        # if we're using multinode, add partitioning
        if variant_params.data.auto_partitioning and (variant_params.training.tree_learner == "data" or variant_params.training.tree_learner == "voting"):
            # if using data parallel, train data has to be partitioned first
            if (variant_params.runtime.nodes * variant_params.runtime.processes) > 1:
                partition_data_step = partition_data_module(
                    input_data=train_dataset,
                    mode="roundrobin",
                    number=(variant_params.runtime.nodes * variant_params.runtime.processes),
                    header=variant_params.data.header,
                    verbose=variant_params.training.verbose
                )
                partition_data_step.runsettings.configure(target=config.compute.linux_cpu)
                partitioned_train_data = partition_data_step.outputs.output_data
            else:
                # for other modes, train data has to be one file
                partitioned_train_data = train_dataset
        else:
            # for other modes, train data has to be one file
            partitioned_train_data = train_dataset
        
        # convert into binary files
        if variant_params.data.pre_convert_to_binary:
            convert_data2bin_step = lightgbm_data2bin_module(
                train=partitioned_train_data,
                test=test_dataset,
                header=variant_params.data.header,
                label_column=variant_params.data.label_column,
                group_column=variant_params.data.group_column,
                max_bin=variant_params.training.max_bin,
                custom_params=json.dumps(dict(variant_params.training.custom_params or {})),
                verbose=variant_params.training.verbose
            )
            convert_data2bin_step.runsettings.configure(target=config.compute.linux_cpu)

            prepared_train_data = convert_data2bin_step.outputs.output_train
            prepared_test_data = convert_data2bin_step.outputs.output_test
        else:
            prepared_train_data = partitioned_train_data
            prepared_test_data = test_dataset

        ################
        ### TRAINING ###
        ################

        # copy params into dict for flexibility
        training_params = dict(variant_params.training)

        # add all data-related params
        training_params['header'] = variant_params.data.header
        training_params['label_column'] = variant_params.data.label_column
        training_params['group_column'] = variant_params.data.group_column
        training_params['construct'] = variant_params.data.construct

        # extract and construct "sweepable" params
        if variant_params.sweep:
            tunable_params, fixed_params = process_sweep_parameters(
                variant_params.training,
                variant_params.sweep.algorithm
            )

            # test if we have sweepable parameters in the learning params
            if len(tunable_params) > 0:
                use_sweep = True
                training_params.update(tunable_params)
            else:
                use_sweep = False
        else:
            use_sweep = False

        # create custom properties and serialize to pass as argument
        variant_custom_properties = {
            'variant_index': variant_index,
            'framework': variant_params.framework,
            'framework_build': variant_params.runtime.build,
        }
        variant_custom_properties.update(benchmark_custom_properties)
        training_params['custom_properties'] = json.dumps(variant_custom_properties)

        # serialize custom_params to pass as argument
        training_params['custom_params'] = json.dumps(dict(variant_params.training.custom_params or {}))

        # some debug outputs to expose variant parameters
        print(f"*** lightgbm variant#{variant_index}: {training_params}")

        # figuring out target (cpu/gpu)
        training_target = variant_params.runtime.target
        if not training_target:
            if (variant_params.training.device_type == 'gpu' or variant_params.training.device_type == 'cuda'):
                training_target = config.compute.linux_gpu
            else:
                training_target = config.compute.linux_cpu

        # switch between frameworks (work in progress)
        if variant_params.framework == "lightgbm_python":
            if use_sweep:
                lightgbm_train_module = lightgbm_basic_train_sweep_module
                # make sure there's a metric
                if variant_params.sweep.primary_metric is None:
                    variant_params.sweep.primary_metric=f"node_0/valid_0.{variant_params.training.metric}"
            else:
                lightgbm_train_module = lightgbm_basic_train_module
        elif variant_params.framework == "lightgbm_ray":
            if use_sweep:
                raise NotImplementedError("Sweep on lightgbm_ray component is not implemented, use framework lightgbm_ray_tune instead.")
            lightgbm_train_module = lightgbm_ray_train_module

            # remove arguments that are not in lightgbm_ray component
            if 'multinode_driver' in training_params:
                del training_params['multinode_driver']
            if 'header' in training_params:
                del training_params['header']
            if 'construct' in training_params:
                del training_params['construct']
            if 'custom_properties' in training_params:
                del training_params['custom_properties']
            if 'verbose' in training_params:
                del training_params['verbose']

        elif variant_params.framework == 'lightgbm_ray_tune':
            lightgbm_train_module = lightgbm_ray_tune_module
            use_sweep = False

            # manually add ray tune parameters.
            training_params['mode'] = variant_params.raytune.mode
            training_params['search_alg'] = variant_params.raytune.search_alg
            training_params['scheduler'] = variant_params.raytune.scheduler
            training_params['num_samples'] = variant_params.raytune.num_samples
            training_params['time_budget'] = variant_params.raytune.time_budget
            training_params['concurrent_trials'] = variant_params.raytune.concurrent_trials

            # remove arguments that are not in lightgbm_ray_tune component
            if 'multinode_driver' in training_params:
                del training_params['multinode_driver']
            if 'header' in training_params:
                del training_params['header']
            if 'construct' in training_params:
                del training_params['construct']
            if 'custom_properties' in training_params:
                del training_params['custom_properties']
            if 'verbose' in training_params:
                del training_params['verbose']
        else:
            raise NotImplementedError(f"training framework {variant_params.framework} hasn't been implemented yet.")

        # configure the training module
        lightgbm_train_step = lightgbm_train_module(
            train = prepared_train_data,  # see end of DATA section
            test = prepared_test_data,  # see end of DATA section
            **training_params
        )
        # apply runsettings
        lightgbm_train_step.runsettings.target=training_target
        lightgbm_train_step.runsettings.resource_layout.node_count = variant_params.runtime.nodes
        # This line is never used. It might run into the error saying "process_count_per_node' is not an expected key" 
        # lightgbm_train_step.runsettings.resource_layout.process_count_per_node = variant_params.runtime.processes

        if use_sweep:
            # apply settings from our custom yaml config
            apply_sweep_settings(lightgbm_train_step, variant_params.sweep)

        ###############
        ### RUNTIME ###
        ###############

        # optional: override docker (ex: to test custom builds)
        if 'build' in variant_params.runtime and variant_params.runtime.build:
            custom_docker = Docker(file=os.path.join(LIGHTGBM_REPO_ROOT, variant_params.runtime.build))
            lightgbm_train_step.runsettings.environment.configure(
                docker=custom_docker
            )

        ##############
        ### OUTPUT ###
        ##############

        # add some relevant comments on the component
        lightgbm_train_step.comment = " -- ".join(
            [
                f"variant #{variant_index}",
                # add more
            ]
        )

        # provide step readable display name
        lightgbm_train_step.node_name = format_run_name(f"training_{variant_params.framework}_{variant_index}")

        # optional: save output model
        if variant_params.output and variant_params.output.register_model:
            # "{register_model_prefix}-{task_key}-{num_iterations}trees-{num_leaves}leaves-{register_model_suffix}"
            model_basename = "{num_iterations}trees-{num_leaves}leaves".format(
                num_iterations=variant_params.training.num_iterations,
                num_leaves=variant_params.training.num_leaves
            )
            # prepend task_key if given
            if benchmark_custom_properties.get('benchmark_task_key', None):
                model_basename = benchmark_custom_properties['benchmark_task_key'] + "-" + model_basename
            # prepend prefix if given
            if variant_params.output.register_model_prefix:
                model_basename = variant_params.output.register_model_prefix + "-" + model_basename
            # append suffix if given
            if variant_params.output.register_model_suffix:
                model_basename += "-" + variant_params.output.register_model_suffix

            print(f"*** Will output model at {model_basename}")
            # auto-register output with model basename
            lightgbm_train_step.outputs.model.register_as(
                name=model_basename,
                create_new_version=True
            )

    # return {key: output}'
    return {}

# creating an overall pipeline using pipeline_function for each task given
@dsl.pipeline(
    name="training_all_tasks",
    non_pipeline_parameters=['workspace', 'config']
)
def training_all_tasks(workspace, config):
    # loop on all training tasks
    for training_task in config.lightgbm_training_config.tasks:
        # load the given train dataset
        train_data = load_dataset_from_data_input_spec(workspace, training_task.train)
        test_data = load_dataset_from_data_input_spec(workspace, training_task.test)

        # create custom properties for this task
        # they will be passed on to each job as tags
        benchmark_custom_properties = {
            'benchmark_name' : config.lightgbm_training_config.benchmark_name,
            'benchmark_task_key' : training_task.task_key
        }

        # call pipeline_function as a subgraph here
        training_task_subgraph_step = lightgbm_training_pipeline_function(
            # NOTE: benchmark_custom_properties is not an actual pipeline input, just passed to the python code
            config=config,
            benchmark_custom_properties=benchmark_custom_properties,
            train_dataset=train_data,
            test_dataset=test_data
        )

        # add some relevant comments on the subgraph
        training_task_subgraph_step.comment = " -- ".join([
            "LightGBM training pipeline",
            f"benchmark name: {config.lightgbm_training_config.benchmark_name}",
            f"benchmark task key: {training_task.task_key}"
        ])


### MAIN BLOCK ###

# Step 4: implement main block using helper functions

def main():
    # use parse helper function to get arguments from CLI
    config = parse_pipeline_config(lightgbm_training_config)
    
    # you'll need a workspace object to connect
    workspace = azureml_connect(config)

    # run the pipeline function with the given arguments
    pipeline_instance = training_all_tasks(workspace, config)

    # generate a nice markdown description
    experiment_description="\n".join([
        "Training on all specified tasks (see yaml below).",
        "```yaml""",
        "data_generation_config:",
        OmegaConf.to_yaml(config.lightgbm_training_config),
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
