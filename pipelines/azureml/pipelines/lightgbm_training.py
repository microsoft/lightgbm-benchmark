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
from typing import Optional
from azure.ml.component import dsl
from shrike.pipeline.pipeline_helper import AMLPipelineHelper
from azure.ml.component.environment import Docker

# when running this script directly, needed to import common
LIGHTGBM_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

if LIGHTGBM_BENCHMARK_ROOT not in sys.path:
    print(f"Adding {LIGHTGBM_BENCHMARK_ROOT} to path")
    sys.path.append(str(LIGHTGBM_BENCHMARK_ROOT))

from common.sweep import SweepParameterParser

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

            # DATA
            learning_task: str = MISSING
            train_samples: int = MISSING
            test_samples: int = MISSING
            inferencing_samples: int = MISSING
            n_features: int = MISSING
            n_informative: int = MISSING

            # TRAINING
            # fixed training parameters
            objective: str = MISSING
            metric: str = MISSING
            boosting: str = MISSING
            tree_learner: str = MISSING

            # sweepable training parameters
            # NOTE: need to be str so they can be parsed (ex: 'choice(100,200)')
            num_iterations: str = MISSING
            num_leaves: str = MISSING
            min_data_in_leaf: str = MISSING
            learning_rate: str = MISSING
            max_bin: str = MISSING
            feature_fraction: str = MISSING

            # COMPUTE
            device_type: str = "cpu"
            nodes: int = 1
            processes: int = 1
            target: Optional[str] = None
            override_docker: Optional[str] = None
            override_os: Optional[str] = None

            # SWEEP
            # TODO: add all parameters from shrike https://github.com/Azure/shrike/blob/387fadb47d69e46bd7e5ac6f243250dc6044afaa/shrike/pipeline/pipeline_helper.py#L809
            sweep_algorithm: str = "random"
            sweep_goal: str = "minimize"
            sweep_max_total_trials: Optional[int] = None
            sweep_max_concurrent_trials: Optional[int] = None
            sweep_timeout_minutes: Optional[int] = None

        # return the dataclass itself
        # for helper class to construct config file
        return lightgbm_training


    def process_sweep_parameters(self, config):
        """Parses config and spots sweepable paraneters
        
        Args:
            config (DictConfig): configuration object (see get_config_class())

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
            parameter_sampling=config.lightgbm_training.sweep_algorithm
        )

        # provide config as a dictionary to the parser
        sweep_parameters = {
            "num_iterations": config.lightgbm_training.num_iterations,
            "num_leaves": config.lightgbm_training.num_leaves,
            "min_data_in_leaf": config.lightgbm_training.min_data_in_leaf,
            "learning_rate": config.lightgbm_training.learning_rate,
            "max_bin": config.lightgbm_training.max_bin,
            "feature_fraction": config.lightgbm_training.feature_fraction,
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
        tunable_params, fixed_params = self.process_sweep_parameters(config)
        enable_sweep = (len(tunable_params.keys())>0)

        # Data modules
        generate_data_module = self.module_load("generate_synthetic_data")
        partition_data_module = self.module_load("partition_data")

        # Training modules
        if enable_sweep:
            # there is at least one sweep parameter, use sweep instead
            lightgbm_train_module = self.module_load("lightgbm_python_train_sweep")
            
            # merge both params to feed into module
            training_params = fixed_params.copy()
            training_params.update(tunable_params)
        else:
            # there are no sweep parameters, use regular training
            lightgbm_train_module = self.module_load("lightgbm_python_train")
            training_params = fixed_params

        benchmark_custom_properties = json.dumps({
            'benchmark_name' : config.lightgbm_training.benchmark_name
        })
        pipeline_name = f"lightgbm_training_{config.lightgbm_training.learning_task}"
        pipeline_description = f"LightGBM {config.lightgbm_training.learning_task} distributed training (mpi) on synthetic data"

        # Here you should create an instance of a pipeline function (using your custom config dataclass)
        @dsl.pipeline(name=pipeline_name, # pythonic name
                      description=pipeline_description,
                      default_datastore=config.compute.noncompliant_datastore)
        def lightgbm_benchmark_pipeline_function(train_samples=10000,
                                                 test_samples=1000,
                                                 inferencing_samples=10000,
                                                 n_features=4000,
                                                 n_informative=4000):
            """Pipeline function for this graph.

            Args:
                TODO

            Returns:
                dict[str->PipelineOutputData]: a dictionary of your pipeline outputs
                    for instance to be consumed by other graphs
            """
            generate_data_step = generate_data_module(
                learning_task = config.lightgbm_training.learning_task,
                train_samples = train_samples,
                test_samples = test_samples,
                inferencing_samples = inferencing_samples,
                n_features = n_features,
                n_informative = n_informative,
                random_state = 5,
                verbose = False,
                custom_properties = benchmark_custom_properties
            )
            self.apply_smart_runsettings(generate_data_step)
            
            if config.lightgbm_training.tree_learner == "data" or config.lightgbm_training.tree_learner == "voting":
                # if using data parallel, train data has to be partitioned first
                if (config.lightgbm_training.nodes * config.lightgbm_training.processes) > 1:
                    partition_data_step = partition_data_module(
                        input_data=generate_data_step.outputs.output_train,
                        mode="roundrobin",
                        number=(config.lightgbm_training.nodes * config.lightgbm_training.processes)
                    )
                    self.apply_smart_runsettings(partition_data_step)
                    train_data = partition_data_step.outputs.output_data
                else:
                    # for other modes, train data has to be one file
                    train_data = generate_data_step.outputs.output_train
            else:
                # for other modes, train data has to be one file
                train_data = generate_data_step.outputs.output_train

            lightgbm_train_step = lightgbm_train_module(
                train = train_data,
                test = generate_data_step.outputs.output_test,
                header = False,
                label_column = "0",
                #group_column = None,
                objective = config.lightgbm_training.objective,
                metric = config.lightgbm_training.metric,
                boosting = config.lightgbm_training.boosting,
                tree_learner = config.lightgbm_training.tree_learner,

                # sweepable params
                num_iterations = training_params['num_iterations'],
                num_leaves = training_params['num_leaves'],
                min_data_in_leaf = training_params['min_data_in_leaf'],
                learning_rate = training_params['learning_rate'],
                max_bin = training_params['max_bin'],
                feature_fraction = training_params['feature_fraction'],

                # generic params
                verbose = False,
                custom_properties = benchmark_custom_properties,

                # compute params
                device_type = config.lightgbm_training.device_type
            )

            if enable_sweep:
                # apply runsettings specific to sweep
                self.apply_smart_runsettings(
                    lightgbm_train_step,
                    gpu = (config.lightgbm_training.device_type == 'gpu' or config.lightgbm_training.device_type == 'cuda'),
                    target = config.lightgbm_training.target,
                    node_count=config.lightgbm_training.nodes,
                    process_count_per_node=config.lightgbm_training.processes,
                    sweep = True,
                    primary_metric = "valid_0." + config.lightgbm_training.metric,
                    goal = config.lightgbm_training.sweep_goal,
                    algorithm = config.lightgbm_training.sweep_algorithm,
                    max_total_trials = config.lightgbm_training.sweep_max_total_trials,
                    max_concurrent_trials = config.lightgbm_training.sweep_max_concurrent_trials,
                    timeout_minutes = config.lightgbm_training.sweep_timeout_minutes
                )
            else:
                self.apply_smart_runsettings(
                    lightgbm_train_step,
                    node_count = config.lightgbm_training.nodes,
                    process_count_per_node = config.lightgbm_training.processes,
                    gpu = (config.lightgbm_training.device_type == 'gpu' or config.lightgbm_training.device_type == 'cuda'),
                    target = config.lightgbm_training.target
                )

            if config.lightgbm_training.override_docker:
                custom_docker = Docker(file=config.lightgbm_training.override_docker)
                lightgbm_train_step.runsettings.environment.configure(
                    docker=custom_docker,
                    os=config.lightgbm_training.override_os or "Linux"
                )

            # return {key: output}'
            return {}

        # finally return the function itself to be built by helper code
        return lightgbm_benchmark_pipeline_function


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
        # when all inputs are obtained, we call the pipeline function
        experiment_pipeline = pipeline_function(
            train_samples=config.lightgbm_training.train_samples,
            test_samples=config.lightgbm_training.test_samples,
            inferencing_samples=config.lightgbm_training.inferencing_samples,
            n_features=config.lightgbm_training.n_features,
            n_informative=config.lightgbm_training.n_informative
        )

        # and we return that function so that helper can run it.
        return experiment_pipeline


# NOTE: main block is necessary only if script is intended to be run from command line
if __name__ == "__main__":
    # calling the helper .main() function
    LightGBMTraining.main()
