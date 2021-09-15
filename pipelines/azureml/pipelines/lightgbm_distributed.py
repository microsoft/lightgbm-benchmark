"""
Runs LightGBM using distributed (mpi) training.

to execute:
> python pipelines/lightgbm_distributed.py --config-dir ./conf --config-name experiments/benchmark-distributed run.submit=True
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

# when running this script directly, needed to import common
LIGHTGBM_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

if LIGHTGBM_BENCHMARK_ROOT not in sys.path:
    print(f"Adding {LIGHTGBM_BENCHMARK_ROOT} to path")
    sys.path.append(str(LIGHTGBM_BENCHMARK_ROOT))

class LightGBMDistributed(AMLPipelineHelper):
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
        class lightgbm_distributed: # pylint: disable=invalid-name
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
            objective: str = MISSING
            metric: str = MISSING
            boosting: str = MISSING
            tree_learner: str = MISSING
            num_iterations: int = MISSING
            num_leaves: int = MISSING
            min_data_in_leaf: int = MISSING
            learning_rate: float = MISSING
            max_bin: int = MISSING
            feature_fraction: float = MISSING

            # DISTRIBUTED
            nodes: int = MISSING

        # return the dataclass itself
        # for helper class to construct config file
        return lightgbm_distributed


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
        # load the right module depending on config

        # Data modules
        generate_data_module = self.module_load("generate_synthetic_data")
        partition_data_module = self.module_load("partition_data")

        # Training modules
        lightgbm_train_module = self.module_load("lightgbm_python_train")

        benchmark_custom_properties = json.dumps({
            'benchmark_name' : config.lightgbm_distributed.benchmark_name
        })
        pipeline_name = f"lightgbm_distributed_{config.lightgbm_distributed.learning_task}"
        pipeline_description = f"LightGBM {config.lightgbm_distributed.learning_task} distributed training (mpi) on synthetic data"

        # Here you should create an instance of a pipeline function (using your custom config dataclass)
        @dsl.pipeline(name=pipeline_name, # pythonic name
                      description=pipeline_description,
                      default_datastore=config.compute.noncompliant_datastore)
        def lightgbm_benchmark_pipeline_function(train_samples=10000,
                                                 test_samples=1000,
                                                 inferencing_samples=10000,
                                                 n_features=4000,
                                                 n_informative=4000,
                                                 num_iterations=100,
                                                 num_leaves=31):
            """Pipeline function for this graph.

            Args:
                TODO

            Returns:
                dict[str->PipelineOutputData]: a dictionary of your pipeline outputs
                    for instance to be consumed by other graphs
            """
            generate_data_step = generate_data_module(
                learning_task = config.lightgbm_distributed.learning_task,
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
            
            if config.lightgbm_distributed.tree_learner == "data" or config.lightgbm_distributed.tree_learner == "voting":
                # if using data parallel, train data has to be partitioned first
                partition_data_step = partition_data_module(
                    input_data=generate_data_step.outputs.output_train,
                    mode="roundrobin",
                    number=config.lightgbm_distributed.nodes
                )
                self.apply_smart_runsettings(partition_data_step)
                train_data = partition_data_step.outputs.output_data
            else:
                # for other modes, train data has to be one file
                train_data = generate_data_step.outputs.output_train

            lightgbm_train_step = lightgbm_train_module(
                train = train_data,
                test = generate_data_step.outputs.output_test,
                header = False,
                label_column = "0",
                #group_column = None,
                objective = config.lightgbm_distributed.objective,
                metric = config.lightgbm_distributed.metric,
                boosting = config.lightgbm_distributed.boosting,
                tree_learner = config.lightgbm_distributed.tree_learner,
                num_iterations = num_iterations,
                num_leaves = num_leaves,
                min_data_in_leaf = config.lightgbm_distributed.min_data_in_leaf,
                learning_rate = config.lightgbm_distributed.learning_rate,
                max_bin = config.lightgbm_distributed.max_bin,
                feature_fraction = config.lightgbm_distributed.feature_fraction,
                verbose = False,
                custom_properties = benchmark_custom_properties
            )
            self.apply_smart_runsettings(
                lightgbm_train_step,
                node_count = config.lightgbm_distributed.nodes
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
            train_samples=config.lightgbm_distributed.train_samples,
            test_samples=config.lightgbm_distributed.test_samples,
            inferencing_samples=config.lightgbm_distributed.inferencing_samples,
            n_features=config.lightgbm_distributed.n_features,
            n_informative=config.lightgbm_distributed.n_informative,
            num_iterations=config.lightgbm_distributed.num_iterations,
            num_leaves=config.lightgbm_distributed.num_leaves
        )

        # and we return that function so that helper can run it.
        return experiment_pipeline


# NOTE: main block is necessary only if script is intended to be run from command line
if __name__ == "__main__":
    # calling the helper .main() function
    LightGBMDistributed.main()