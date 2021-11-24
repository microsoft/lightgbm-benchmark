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
from omegaconf import MISSING, OmegaConf
from typing import Optional, Any, List
from azure.ml.component import dsl
from shrike.pipeline.pipeline_helper import AMLPipelineHelper
from azure.ml.component.environment import Docker

# when running this script directly, needed to import common
LIGHTGBM_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

if LIGHTGBM_BENCHMARK_ROOT not in sys.path:
    print(f"Adding {LIGHTGBM_BENCHMARK_ROOT}/src/ to path")
    sys.path.append(str(os.path.join(LIGHTGBM_BENCHMARK_ROOT, "src")))

from common.sweep import SweepParameterParser
from common.tasks import training_task, training_variant
from common.aml import dataset_from_dstore_path, apply_sweep_settings


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
            reference: training_variant = MISSING

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
        lightgbm_data2bin_module = self.module_load("lightgbm_python_data2bin")

        pipeline_name = f"lightgbm_training"
        pipeline_description = f"LightGBM distributed training (mpi)"

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
            # create list of all variants params
            training_variants_params = [
                config.lightgbm_training.reference
            ]

            # if there's any variant specified
            if config.lightgbm_training.variants:
                # create distinct training params for each variant
                for variant_index, training_variant_config in enumerate(config.lightgbm_training.variants):
                    # create a specific dict of params for the variant
                    variant_config = OmegaConf.merge(config.lightgbm_training.reference, training_variant_config)
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
                        self.apply_smart_runsettings(partition_data_step)
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
                    self.apply_smart_runsettings(convert_data2bin_step)

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

                # extract and construct "sweepable" params
                if variant_params.sweep:
                    tunable_params, fixed_params = self.process_sweep_parameters(
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
                    'framework': "lightgbm",
                    'framework_build': variant_params.runtime.build,
                }
                variant_custom_properties.update(benchmark_custom_properties)
                training_params['custom_properties'] = json.dumps(variant_custom_properties)

                # serialize custom_params to pass as argument
                training_params['custom_params'] = json.dumps(dict(variant_params.training.custom_params or {}))

                # some debug outputs to expose variant parameters
                print(f"*** lightgbm variant#{variant_index}: {training_params}")

                if use_sweep:
                    # sweep training
                    if variant_params.sweep.primary_metric is None:
                        variant_params.sweep.primary_metric=f"node_0/valid_0.{variant_params.training.metric}"

                    lightgbm_train_step = lightgbm_train_sweep_module(
                        train = prepared_train_data,
                        test = prepared_test_data,
                        **training_params
                    )
                    # apply runsettings
                    self.apply_smart_runsettings(
                        lightgbm_train_step,
                        node_count = variant_params.runtime.nodes,
                        process_count_per_node = variant_params.runtime.processes,
                        gpu = (variant_params.training.device_type == 'gpu' or variant_params.training.device_type == 'cuda'),
                        target = variant_params.runtime.target
                    )
                    # apply settings from our custom yaml config
                    apply_sweep_settings(lightgbm_train_step, variant_params.sweep)

                else:
                    # regular training, no sweep
                    lightgbm_train_step = lightgbm_train_module(
                        train = prepared_train_data,
                        test = prepared_test_data,
                        **training_params
                    )
                    # apply runsettings
                    self.apply_smart_runsettings(
                        lightgbm_train_step,
                        node_count = variant_params.runtime.nodes,
                        process_count_per_node = variant_params.runtime.processes,
                        gpu = (variant_params.training.device_type == 'gpu' or variant_params.training.device_type == 'cuda'),
                        target = variant_params.runtime.target
                    )

                ###############
                ### RUNTIME ###
                ###############

                # # optional: override docker (ex: to test custom builds)
                if 'build' in variant_params.runtime and variant_params.runtime.build:
                    custom_docker = Docker(file=os.path.join(LIGHTGBM_BENCHMARK_ROOT, variant_params.runtime.build))
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
        full_pipeline_description="\n".join([
            "Training on all specified tasks (see yaml below).",
            "```yaml""",
            "lightgbm_training:",
            OmegaConf.to_yaml(config.lightgbm_training),
            "```"
        ])

        if len(full_pipeline_description) > 5000:
            full_pipeline_description = full_pipeline_description[:5000-50] + "\n<<<TRUNCATED DUE TO SIZE LIMIT>>>"

        # creating an overall pipeline using pipeline_function for each task given
        @dsl.pipeline(name="training_all_tasks", # pythonic name
                      description=full_pipeline_description,
                      default_datastore=config.compute.noncompliant_datastore)
        def training_all_tasks():
            # loop on all training tasks
            for training_task in config.lightgbm_training.tasks:
                # load the given train dataset
                if training_task.train_dataset:
                    train_data = self.dataset_load(
                        name = training_task.train_dataset,
                        version = training_task.train_dataset_version # use latest if None
                    )
                elif training_task.train_datastore and training_task.train_datastore_path:
                    train_data = dataset_from_dstore_path(self.workspace(), training_task.train_datastore, training_task.train_datastore_path, validate=training_task.train_datastore_path_validate)
                else:
                    raise ValueError(f"In training_task {training_task}, you need to provide either train_dataset or train_datastore+train_datastore_path")

                # load the given test dataset
                if training_task.test_dataset:
                    test_data = self.dataset_load(
                        name = training_task.test_dataset,
                        version = training_task.test_dataset_version # use latest if None
                    )
                elif training_task.test_datastore and training_task.test_datastore_path:
                    test_data = dataset_from_dstore_path(self.workspace(), training_task.test_datastore, training_task.test_datastore_path, validate=training_task.test_datastore_path_validate)
                else:
                    raise ValueError(f"In training_task {training_task}, you need to provide either test_dataset or test_datastore+test_datastore_path")

                # create custom properties for this task
                # they will be passed on to each job as tags
                benchmark_custom_properties = {
                    'benchmark_name' : config.lightgbm_training.benchmark_name,
                    'benchmark_train_dataset' : training_task.train_dataset,
                    'benchmark_test_dataset' : training_task.test_dataset,
                    'benchmark_task_key' : training_task.task_key
                }

                # call pipeline_function as a subgraph here
                training_task_subgraph_step = pipeline_function(
                    # NOTE: benchmark_custom_properties is not an actual pipeline input, just passed to the python code
                    benchmark_custom_properties=benchmark_custom_properties,
                    train_dataset=train_data,
                    test_dataset=test_data
                )

                # add some relevant comments on the subgraph
                training_task_subgraph_step.comment = " -- ".join([
                    "LightGBM training pipeline",
                    f"benchmark name: {config.lightgbm_training.benchmark_name}",
                    f"benchmark task key: {training_task.task_key}"
                ])

        # return the instance of this general function
        return training_all_tasks()


# NOTE: main block is necessary only if script is intended to be run from command line
if __name__ == "__main__":
    # calling the helper .main() function
    LightGBMTraining.main()
