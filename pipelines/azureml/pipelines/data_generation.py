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
from omegaconf import MISSING
from typing import Optional, List
from azure.ml.component import dsl
from shrike.pipeline.pipeline_helper import AMLPipelineHelper
from azure.ml.component.environment import Docker

# when running this script directly, needed to import common
LIGHTGBM_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

if LIGHTGBM_BENCHMARK_ROOT not in sys.path:
    print(f"Adding {LIGHTGBM_BENCHMARK_ROOT} to path")
    sys.path.append(str(LIGHTGBM_BENCHMARK_ROOT))

from common.sweep import SweepParameterParser
from common.tasks import data_generation_task

class DataGenerationPipeline(AMLPipelineHelper):
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

        # return the dataclass itself
        # for helper class to construct config file
        return data_generation


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
        # Data modules
        generate_data_module = self.module_load("generate_synthetic_data")

        pipeline_name = f"data_generation"
        pipeline_description = f"Data Generation for the LightGBM benchmark"

        # Here you should create an instance of a pipeline function (using your custom config dataclass)
        @dsl.pipeline(name=pipeline_name, # pythonic name
                      description=pipeline_description,
                      default_datastore=config.compute.noncompliant_datastore)
        def data_generation_pipeline_function(task,
                                              train_samples=10000,
                                              test_samples=1000,
                                              inferencing_samples=10000,
                                              n_features=4000,
                                              n_informative=4000,
                                              benchmark_custom_properties=None):
            """Pipeline function for this graph.

            Args:
                TODO

            Returns:
                dict[str->PipelineOutputData]: a dictionary of your pipeline outputs
                    for instance to be consumed by other graphs
            """          
            generate_data_step = generate_data_module(
                learning_task = task,
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

            # return {key: output}'
            return {
                'train' : generate_data_step.outputs.output_train,
                'test' : generate_data_step.outputs.output_test,
                'inference' : generate_data_step.outputs.output_inference
            }

        # finally return the function itself to be built by helper code
        return data_generation_pipeline_function


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
        benchmark_custom_properties = json.dumps({
            'benchmark_name' : config.data_generation.benchmark_name
        })

        # Here you should create an instance of a pipeline function (using your custom config dataclass)
        @dsl.pipeline(name="generate_all_datasets", # pythonic name
                      description="Generate all datasets for lightgbm benchmark",
                      default_datastore=config.compute.noncompliant_datastore)
        def generate_all_tasks():
            for generation_task in config.data_generation.tasks:
                generation_task_subgraph_step = pipeline_function(
                    task=generation_task.task,
                    train_samples=generation_task.train_samples,
                    test_samples=generation_task.test_samples,
                    inferencing_samples=generation_task.inferencing_samples,
                    n_features=generation_task.n_features,
                    n_informative=generation_task.n_informative,
                    benchmark_custom_properties=benchmark_custom_properties
                )

                if config.data_generation.register_outputs:
                    dataset_prefix = "{prefix}-{task}-{cols}cols".format(
                        prefix=config.data_generation.register_outputs_prefix,
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


        # and we return that function so that helper can run it.
        return generate_all_tasks()


# NOTE: main block is necessary only if script is intended to be run from command line
if __name__ == "__main__":
    # calling the helper .main() function
    DataGenerationPipeline.main()
