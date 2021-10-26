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
LIGHTGBM_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

if LIGHTGBM_BENCHMARK_ROOT not in sys.path:
    sys.path.append(str(LIGHTGBM_BENCHMARK_ROOT))

from common.sweep import SweepParameterParser
from common.tasks import training_task, training_variant
from common.aml import dataset_from_dstore_path


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
        lightgbm_data2bin_module = self.module_load("lightgbm_python_data2bin")

        pipeline_name = f"lightgbm_training_{config.lightgbm_training.reference_training.objective}"
        pipeline_description = f"LightGBM {config.lightgbm_training.reference_training.objective} distributed training (mpi)"

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
                'header' : config.lightgbm_training.reference_training.header,
                'label_column' : config.lightgbm_training.reference_training.label_column,
                'group_column' : config.lightgbm_training.reference_training.group_column,
                'construct' : config.lightgbm_training.reference_training.construct,

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
                'label_gain' : config.lightgbm_training.reference_training.label_gain,
                'custom_params' : config.lightgbm_training.reference_training.custom_params,

                # generic params
                'verbose' : config.lightgbm_training.reference_training.verbose,
                'custom_properties' : benchmark_custom_properties,

                # compute params
                'device_type' : config.lightgbm_training.reference_training.device_type
            }

            # create specific dict for sweep parameters
            reference_sweep_params = {
                'sweep_algorithm': config.lightgbm_training.reference_training.sweep_algorithm,
                'sweep_goal': config.lightgbm_training.reference_training.sweep_goal,
                'sweep_max_total_trials': config.lightgbm_training.reference_training.sweep_max_total_trials,
                'sweep_max_concurrent_trials': config.lightgbm_training.reference_training.sweep_max_concurrent_trials,
                'sweep_timeout_minutes': config.lightgbm_training.reference_training.sweep_timeout_minutes
            }

            # create a dict with all compute/runsettings
            reference_runsettings = {
                'nodes' : config.lightgbm_training.reference_training.nodes,
                'processes' : config.lightgbm_training.reference_training.processes,
                'target' : config.lightgbm_training.reference_training.target,
                'auto_partitioning' : config.lightgbm_training.reference_training.auto_partitioning,
                'pre_convert_to_binary' : config.lightgbm_training.reference_training.pre_convert_to_binary,
                'register_model' : config.lightgbm_training.reference_training.register_model,
                'register_model_prefix' : config.lightgbm_training.reference_training.register_model_prefix,
                'register_model_suffix' : config.lightgbm_training.reference_training.register_model_suffix,
                'override_docker' : config.lightgbm_training.reference_training.override_docker,
                'override_os' : config.lightgbm_training.reference_training.override_os,
            }

            # create list of all variants params
            training_variants_params = [
                reference_training_params.copy()
            ]
            sweep_variants_params = [
                reference_sweep_params.copy()
            ]
            runsettings_variants_params = [
                reference_runsettings.copy()
            ]
            variant_comments = [
                " -- ".join([
                    f"variant #0 (reference)",
                    f"device type: {config.lightgbm_training.reference_training.device_type}"
                ])
            ]

            # if there's any variant specified
            if config.lightgbm_training.variants:
                # create distinct training params for each variant
                for variant_index, training_variant in enumerate(config.lightgbm_training.variants):
                    # create a specific dict of params for the variant
                    variant_params = reference_training_params.copy()
                    variant_sweep_params = reference_sweep_params.copy()
                    variant_runsettings = reference_runsettings.copy()

                    variant_config = dict(training_variant)

                    # we don't copy everything here in order to not mix up params and runsettings
                    for key in variant_params:
                        if key in variant_config:
                            variant_params[key] = variant_config[key]

                    # we don't copy everything here in order to not mix up params and runsettings
                    for key in variant_sweep_params:
                        if key in variant_config:
                            variant_sweep_params[key] = variant_config[key]

                    # we don't copy everything here in order to not mix up params and runsettings
                    for key in variant_runsettings:
                        if key in variant_config:
                            variant_runsettings[key] = variant_config[key]
                      
                    # add to training params list
                    training_variants_params.append(variant_params)
                    sweep_variants_params.append(variant_sweep_params)
                    runsettings_variants_params.append(variant_runsettings)

                    # create smart comment for this variant
                    variant_comments.append(" -- ".join([
                        f"variant #{variant_index+1}", # needs +1 since variant 0 is reference
                        "variant diff: {}".format(
                            ", ".join([
                                f"{k}={v}" for k,v in variant_config.items()
                            ])
                        )
                    ]))


            # for each variant, check if sweep needs to be applied
            for index, variant_training_params in enumerate(training_variants_params):
                # extract and construct "sweepable" params
                tunable_params, fixed_params = self.process_sweep_parameters(
                    variant_training_params,
                    sweep_variants_params[index]['sweep_algorithm']
                )

                # test if we have sweepable parameters in the learning params
                if len(tunable_params) > 0:
                    # create runsettings
                    runsettings = runsettings_variants_params[index]

                    runsettings['sweep'] = True

                    # copy constructed sweep params into variant training params
                    for key in tunable_params:
                        variant_training_params[key] = tunable_params[key]

                # some debug outputs to expose variant parameters
                print(f"*** lightgbm training variant#{index}: {variant_training_params}")
                print(f"*** lightgbm sweep variant#{index}: {sweep_variants_params[index]}")
                print(f"*** lightgbm runsettings variant#{index}: {runsettings_variants_params[index]}")


            # for each training variant, create a module sequence
            for variant_index,(training_params,sweep_params,runsettings,variant_comment) in enumerate(zip(training_variants_params,sweep_variants_params,runsettings_variants_params,variant_comments)):
                # if we're using multinode, add partitioning
                if runsettings['auto_partitioning'] and (training_params['tree_learner'] == "data" or training_params['tree_learner'] == "voting"):
                    # if using data parallel, train data has to be partitioned first
                    if (runsettings['nodes'] * runsettings['processes']) > 1:
                        partition_data_step = partition_data_module(
                            input_data=train_dataset,
                            mode="roundrobin",
                            number=(runsettings['nodes'] * runsettings['processes']),
                            header=training_params['header']
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
                if runsettings['pre_convert_to_binary']:
                    convert_data2bin_step = lightgbm_data2bin_module(
                        train=partitioned_train_data,
                        test=test_dataset,
                        max_bin=training_params['max_bin'],
                        custom_params=json.dumps(dict(training_params['custom_params'] or {}))
                    )
                    self.apply_smart_runsettings(convert_data2bin_step)

                    prepared_train_data = convert_data2bin_step.outputs.output_train
                    prepared_test_data = convert_data2bin_step.outputs.output_test
                else:
                    prepared_train_data = partitioned_train_data
                    prepared_test_data = test_dataset

                # NOTE: last minute addition to custom_properties before transforming into json for tagging
                # adding variant_index to spot which variant is the reference
                training_params['custom_properties']['variant_index'] = variant_index
                # adding build settings (docker+os)
                training_params['custom_properties']['framework_build'] = runsettings.get('override_docker') or "n/a"
                training_params['custom_properties']['framework_build_os'] = runsettings.get('override_os') or "n/a"
                # passing as json string that each module parses to digest as tags/properties
                training_params['custom_properties'] = json.dumps(training_params['custom_properties'] or {})
                if training_params['custom_params']:
                    training_params['custom_params'] = json.dumps(dict(training_params['custom_params'] or {}))

                # create instance of training module and apply training params
                if runsettings.get('sweep', None):
                    # apply training parameters (including sweepable params)
                    lightgbm_train_step = lightgbm_train_sweep_module(
                        train = prepared_train_data,
                        test = prepared_test_data,
                        **training_params
                    )
                    # apply runsettings
                    self.apply_smart_runsettings(
                        lightgbm_train_step,
                        node_count = runsettings['nodes'],
                        process_count_per_node = runsettings['processes'],
                        gpu = (training_params['device_type'] == 'gpu' or training_params['device_type'] == 'cuda'),
                        target = runsettings['target'],
                        primary_metric=f"node_0/valid_0.{training_params['metric']}",
                        sweep = True,
                        algorithm = sweep_params['sweep_algorithm'],
                        goal = sweep_params['sweep_goal'],
                        max_total_trials = sweep_params['sweep_max_total_trials'],
                        max_concurrent_trials = sweep_params['sweep_max_concurrent_trials'],
                        timeout_minutes = sweep_params['sweep_timeout_minutes']
                    )
                else:
                    # apply training params
                    lightgbm_train_step = lightgbm_train_module(
                        train = prepared_train_data,
                        test = prepared_test_data,
                        **training_params
                    )
                    # apply runsettings
                    self.apply_smart_runsettings(
                        lightgbm_train_step,
                        node_count = runsettings['nodes'],
                        process_count_per_node = runsettings['processes'],
                        gpu = (training_params['device_type'] == 'gpu' or training_params['device_type'] == 'cuda'),
                        target = runsettings['target']
                    )
                
                # add some relevant comments on the component
                lightgbm_train_step.comment = variant_comment

                # optional: override environment (ex: to test custom builds)
                if 'override_docker' in runsettings and runsettings['override_docker']:
                    custom_docker = Docker(file=os.path.join(config.module_loader.local_steps_folder, "lightgbm_python", runsettings['override_docker']))
                    lightgbm_train_step.runsettings.environment.configure(
                        docker=custom_docker,
                        os=runsettings.get('override_os', 'Linux')
                    )

                # optional: save output model
                if 'register_model' in runsettings and runsettings['register_model']:
                    # "{register_model_prefix}-{task_key}-{num_iterations}trees-{num_leaves}leaves-{register_model_suffix}"
                    model_basename = "{num_iterations}trees-{num_leaves}leaves".format(
                        num_iterations=training_params['num_iterations'],
                        num_leaves=training_params['num_leaves']
                    )
                    # prepend task_key if given
                    if benchmark_custom_properties.get('benchmark_task_key', None):
                        model_basename = benchmark_custom_properties['benchmark_task_key'] + "-" + model_basename
                    # prepend prefix if given
                    if runsettings.get('register_model_prefix', None):
                        model_basename = runsettings['register_model_prefix'] + "-" + model_basename
                    # append suffix if given
                    if runsettings.get('register_model_suffix', None):
                        model_basename += "-" + runsettings.get('register_model_suffix')

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
            OmegaConf.to_yaml(config),
            "```"
        ])

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
