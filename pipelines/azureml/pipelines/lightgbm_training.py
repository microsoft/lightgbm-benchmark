"""
Runs LightGBM using distributed (mpi) training.

to execute:
> python pipelines/lightgbm_training.py --config-dir ./conf --config-name experiments/lightgbm-training run.submit=True
"""
# pylint: disable=no-member
# NOTE: because it raises 'dict' has no 'outputs' member in dsl.pipeline construction
import os
import sys
import json
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Optional, Any, List
from azure.ml.component import dsl
from shrike.pipeline.pipeline_helper import AMLPipelineHelper
from azure.ml.component.environment import Docker

# when running this script directly, needed to import common
LIGHTGBM_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

if LIGHTGBM_BENCHMARK_ROOT not in sys.path:
    print(f"Adding {LIGHTGBM_BENCHMARK_ROOT} to path")
    sys.path.append(str(LIGHTGBM_BENCHMARK_ROOT))

from common.sweep import SweepParameterParser
from common.tasks import training_task, training_variant

class LightGBMTraining(AMLPipelineHelper):
    """Runnable/reusable pipeline helper class

    This class inherits from AMLPipelineHelper which provides
    helper functions to create reusable production pipelines for SmartCompose.
    """
    @classmethod
    def get_config_class(cls):
        """ Returns the config object (dataclass) for this runnable script.

        Returns:
            dataclass: class for configuring this runnable pipeline.
        """
        @dataclass
        class lightgbm_training: # pylint: disable=invalid-name
            """ Config object constructed as a dataclass.

            NOTE: the name of this class will be used as namespace in your config yaml file.
            See conf/reference/evaluate_qas_model.yaml for an example.
            """
            # NOTE: all those values are REQUIRED in your yaml config file
            benchmark_name: str = MISSING

            # INPUT DATASETS
            tasks: List[training_task] = MISSING

            # TRAINING PARAMS
            reference_training: training_variant = MISSING

            # free changing parameters on top of reference
            variants: Optional[Any] = None

        # return the dataclass itself
        # for helper class to construct config file
        return lightgbm_training


    def process_sweep_parameters(self, params_dict, sweep_algorithm):
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


    def build(self, config):
        """ Builds a pipeline function for this pipeline using AzureML SDK (dsl.pipeline).

        This method should build your graph using the provided config object.
        Your pipeline config will be under config.CONFIGNAME.*
        where CONFIGNAME is the name of the dataclass returned by get_config_class()

        This method returns a constructed pipeline function (decorated with @dsl.pipeline).

        Args:
            config (DictConfig): configuration object (see get_config_class())

        Returns:
            dsl.pipeline: the function to create your pipeline
        """
        # required modules
        lightgbm_train_module = self.module_load("lightgbm_python_train")
        lightgbm_train_sweep_module = self.module_load("lightgbm_python_train_sweep")
        partition_data_module = self.module_load("partition_data")

        pipeline_name = f"lightgbm_training_{config.lightgbm_training.reference_training.objective}"
        pipeline_description = f"LightGBM {config.lightgbm_training.reference_training.objective} distributed training (mpi) on synthetic data"

        # Here you should create an instance of a pipeline function (using your custom config dataclass)
        @dsl.pipeline(name=pipeline_name, # pythonic name
                      description=pipeline_description,
                      default_datastore=config.compute.noncompliant_datastore,
                      non_pipeline_parameters=['benchmark_custom_properties'])
        def lightgbm_training_pipeline_function(benchmark_custom_properties, train_dataset, test_dataset):
            """Pipeline function for this graph.

            Args:
                TODO

            Returns:
                dict[str->PipelineOutputData]: a dictionary of your pipeline outputs
                    for instance to be consumed by other graphs
            """
            # create dict of all params for training module
            reference_training_params = {
                'header' : False,
                'label_column' : "0",

                # training params
                'objective' : config.lightgbm_training.reference_training.objective,
                'metric' : config.lightgbm_training.reference_training.metric,
                'boosting' : config.lightgbm_training.reference_training.boosting,
                'tree_learner' : config.lightgbm_training.reference_training.tree_learner,

                # sweepable params
                'num_iterations' : config.lightgbm_training.reference_training.num_iterations,
                'num_leaves' : config.lightgbm_training.reference_training.num_leaves,
                'min_data_in_leaf' : config.lightgbm_training.reference_training.min_data_in_leaf,
                'learning_rate' : config.lightgbm_training.reference_training.learning_rate,
                'max_bin' : config.lightgbm_training.reference_training.max_bin,
                'feature_fraction' : config.lightgbm_training.reference_training.feature_fraction,

                # generic params
                'verbose' : False,
                'custom_properties' : json.dumps(benchmark_custom_properties),

                # compute params
                'device_type' : config.lightgbm_training.reference_training.device_type,
            }

            # create a dict with all compute/runsettings
            reference_runsettings = {
                'nodes' : config.lightgbm_training.reference_training.nodes,
                'processes' : config.lightgbm_training.reference_training.processes,
                'target' : config.lightgbm_training.reference_training.target,
                'register_model_as' : config.lightgbm_training.reference_training.register_model_as,
                'override_docker' : config.lightgbm_training.reference_training.override_docker,
                'override_os' : config.lightgbm_training.reference_training.override_os,
            }

            # create list of all training params
            training_variants_params = [
                reference_training_params.copy()
            ]
            runsettings_variants_params = [
                reference_runsettings.copy()
            ]

            # create distinct training params for each variant
            for training_variant in config.lightgbm_training.variants:
                # create a specific dict of params for the variant
                variant_params = reference_training_params.copy()
                variant_runsettings = reference_runsettings.copy()

                variant_config = dict(training_variant)

                # we don't copy everything here in order to not mix up params and
                for key in variant_params:
                    if key in variant_config:
                        variant_params[key] = variant_config[key]

                # we don't copy everything here in order to not mix up params and
                for key in variant_runsettings:
                    if key in variant_config:
                        variant_runsettings[key] = variant_config[key]

                # add to training params list
                training_variants_params.append(variant_params)
                runsettings_variants_params.append(variant_runsettings)
            
            # iterate to create runsettings
            for index, variant_training_params in enumerate(training_variants_params):
                # test if we have sweepable parameters in the learning params
                print(f"*** lightgbm training variant#{index}: {variant_training_params}")

            # for each training variant, create a module sequence
            for training_params,runsettings in zip(training_variants_params,runsettings_variants_params):
                # if we're using multinode, add partitioning
                if training_params['tree_learner'] == "data" or training_params['tree_learner'] == "voting":
                    # if using data parallel, train data has to be partitioned first
                    if (runsettings['nodes'] * runsettings['processes']) > 1:
                        partition_data_step = partition_data_module(
                            input_data=train_dataset,
                            mode="roundrobin",
                            number=(runsettings['nodes'] * runsettings['processes'])
                        )
                        self.apply_smart_runsettings(partition_data_step)
                        partitioned_train_data = train_dataset
                    else:
                        # for other modes, train data has to be one file
                        partitioned_train_data = train_dataset
                else:
                    # for other modes, train data has to be one file
                    partitioned_train_data = train_dataset

                lightgbm_train_step = lightgbm_train_module(
                    train = partitioned_train_data,
                    test = test_dataset,
                    **training_params
                )

                self.apply_smart_runsettings(
                    lightgbm_train_step,
                    node_count = runsettings['nodes'],
                    process_count_per_node = runsettings['processes'],
                    gpu = (training_params['device_type'] == 'gpu' or training_params['device_type'] == 'cuda'),
                    target = runsettings['target']
                )

            # optional: save output model
            if 'register_model_as' in runsettings and runsettings['register_model_as']:
                lightgbm_train_step.outputs.model.register_as(
                    name=runsettings['register_model_as'],
                    create_new_version=True
                )

            # optional: override environment (ex: to test custom builds)
            if 'override_docker' in runsettings and runsettings['override_docker']:
                custom_docker = Docker(file=runsettings['override_docker'])
                lightgbm_train_step.runsettings.environment.configure(
                    docker=custom_docker,
                    os=runsettings.get('override_os', 'Linux')
                )

            # return {key: output}'
            return {}

        # finally return the function itself to be built by helper code
        return lightgbm_training_pipeline_function


    def pipeline_instance(self, pipeline_function, config):
        """ Given a pipeline function, creates a runnable instance based on provided config.

        This is used only when calling this as a runnable pipeline using .main() function (see below).
        The goal of this function is to map the config to the pipeline_function inputs and params.

        Args:
            pipeline_function (function): the pipeline function obtained from self.build()
            config (DictConfig): configuration object (see get_config_class())

        Returns:
            azureml.core.Pipeline: the instance constructed with its inputs and params.
        """
        # Here you should create an instance of a pipeline function (using your custom config dataclass)
        @dsl.pipeline(name="training_all_tasks", # pythonic name
                      description="Training on all specified tasks",
                      default_datastore=config.compute.noncompliant_datastore)
        def training_all_tasks():
            for training_task in config.lightgbm_training.tasks:
                train_data = self.dataset_load(
                    name = training_task.train_dataset,
                    version = training_task.train_dataset_version # use latest if None
                )
                test_data = self.dataset_load(
                    name = training_task.test_dataset,
                    version = training_task.test_dataset_version # use latest if None
                )

                # create custom properties for this task
                benchmark_custom_properties = {
                    'benchmark_name' : config.lightgbm_training.benchmark_name,
                    'benchmark_train_dataset' : training_task.train_dataset,
                    'benchmark_test_dataset' : training_task.test_dataset,
                }

                training_task_subgraph_step = pipeline_function(
                    benchmark_custom_properties=benchmark_custom_properties,
                    train_dataset=train_data,
                    test_dataset=test_data
                )

        # return the instance of this general function
        return training_all_tasks()


# NOTE: main block is necessary only if script is intended to be run from command line
if __name__ == "__main__":
    # calling the helper .main() function
    LightGBMTraining.main()
