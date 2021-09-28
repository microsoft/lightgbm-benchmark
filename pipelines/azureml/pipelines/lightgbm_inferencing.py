"""
Compares 3 versions of LightGBM Inferencing
A - vanilla
B - custom build
C - treelite

to execute:
> python pipelines/lightgbm_inferencing.py --config-dir ./conf --config-name experiments/benchmark-inferencing run.submit=True
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

from common.tasks import inferencing_task, inferencing_variants

class LightGBMInferencing(AMLPipelineHelper):
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
        class lightgbm_inferencing: # pylint: disable=invalid-name
            """ Config object constructed as a dataclass.

            NOTE: the name of this class will be used as namespace in your config yaml file.
            See conf/reference/evaluate_qas_model.yaml for an example.
            """
            benchmark_name: str = MISSING
            tasks: List[inferencing_task] = MISSING
            variants: List[inferencing_variants] = MISSING

        # return the dataclass itself
        # for helper class to construct config file
        return lightgbm_inferencing


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
        # Inferencing modules
        lightgbm_score_module = self.module_load("lightgbm_python_score")
        treelite_compile_module = self.module_load("treelite_compile")
        treelite_score_module = self.module_load("treelite_score")

        pipeline_name = f"lightgbm_inferencing"
        pipeline_description = f"LightGBM inferencing on user defined dataset/model"

        # Here you should create an instance of a pipeline function (using your custom config dataclass)
        @dsl.pipeline(name=pipeline_name, # pythonic name
                      description=pipeline_description,
                      default_datastore=config.compute.noncompliant_datastore)
        def lightgbm_inferencing_pipeline_function(data, model, benchmark_custom_properties, predict_disable_shape_check):
            """Pipeline function for this graph.

            Args:
                TODO

            Returns:
                dict[str->PipelineOutputData]: a dictionary of your pipeline outputs
                    for instance to be consumed by other graphs
            """
            pipeline_outputs = {}
            for variant in config.lightgbm_inferencing.variants:
                if variant.framework == "treelite_python":
                    treelite_compile_step = treelite_compile_module(
                        model = model,
                        verbose = False,
                        custom_properties = benchmark_custom_properties
                    )
                    self.apply_smart_runsettings(treelite_compile_step)

                    treelite_score_step = treelite_score_module(
                        data = data,
                        compiled_model = treelite_compile_step.outputs.compiled_model,
                        verbose = False,
                        custom_properties = benchmark_custom_properties
                    )
                    self.apply_smart_runsettings(treelite_score_step)
                    
                    # pipeline_outputs[f"{variant}_predictions"] = treelite_score_step.outputs.predictions

                elif variant.framework == "lightgbm_python":
                    # call module with all the right arguments
                    lightgbm_score_step = lightgbm_score_module(
                        data = data,
                        model = model,
                        predict_disable_shape_check = predict_disable_shape_check,
                        verbose = False,
                        custom_properties = benchmark_custom_properties
                    )
                    self.apply_smart_runsettings(lightgbm_score_step)

                    if variant.build:
                        custom_docker = Docker(file=os.path.join(config.module_loader.local_steps_folder, variant.framework, variant.build))
                        lightgbm_score_step.runsettings.environment.configure(
                            docker=custom_docker,
                            os=variant.os or "Linux" # linux by default
                        )

                    # pipeline_outputs[f"{variant}_predictions"] = lightgbm_score_step.outputs.predictions

            # return {key: output}'
            return pipeline_outputs

        # finally return the function itself to be built by helper code
        return lightgbm_inferencing_pipeline_function


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
        @dsl.pipeline(name="inferencing_all_tasks", # pythonic name
                      description="Inferencing on all specified tasks",
                      default_datastore=config.compute.noncompliant_datastore)
        def inferencing_all_tasks():
            for inferencing_task in config.lightgbm_inferencing.tasks:
                data = self.dataset_load(inferencing_task.dataset)
                model = self.dataset_load(inferencing_task.model)

                # create custom properties for this task
                benchmark_custom_properties = json.dumps({
                    'benchmark_name' : config.lightgbm_inferencing.benchmark_name, 
                    'benchmark_dataset' : inferencing_task.dataset,
                    'benchmark_model' : inferencing_task.model,
                })

                inferencing_task_subgraph_step = pipeline_function(
                    data=data,
                    model=model,
                    benchmark_custom_properties=benchmark_custom_properties,
                    predict_disable_shape_check=inferencing_task.predict_disable_shape_check or False
                )

        # return the instance of this general function
        return inferencing_all_tasks()


# NOTE: main block is necessary only if script is intended to be run from command line
if __name__ == "__main__":
    # calling the helper .main() function
    LightGBMInferencing.main()
