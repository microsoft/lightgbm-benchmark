"""
to execute:
> python src/pipelines/pipeline_score.py
"""
# pylint: disable=no-member
# NOTE: because it raises 'dict' has no 'outputs' member in dsl.pipeline construction
import os
import sys
from dataclasses import MISSING, dataclass
from typing import Optional
from azure.ml.component import dsl
from shrike.pipeline.pipeline_helper import AMLPipelineHelper

# when running this script directly, needed to import common
LIGHTGBM_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

if LIGHTGBM_BENCHMARK_ROOT not in sys.path:
    print(f"Adding {LIGHTGBM_BENCHMARK_ROOT} to path")
    sys.path.append(str(LIGHTGBM_BENCHMARK_ROOT))

class LightGBMBenchmarkPipeline(AMLPipelineHelper):
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
        class lightgbm_benchmark: # pylint: disable=invalid-name
            """ Config object constructed as a dataclass.

            NOTE: the name of this class will be used as namespace in your config yaml file.
            See conf/reference/evaluate_qas_model.yaml for an example.
            """
            # INPUTS
            data: str = ""
            model: str = ""
            predict_disable_shape_check: bool = False
            os: str = "linux" # linux or windows

        # return the dataclass itself
        # for helper class to construct config file
        return lightgbm_benchmark


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
        if config.lightgbm_benchmark.os.lower() == "windows":
            lightgbm_score_module = self.module_load("lightgbm_python_score_win")
        elif config.lightgbm_benchmark.os.lower() == "linux":
            lightgbm_score_module = self.module_load("lightgbm_python_score")
        else:
            raise Exception(f"lightgbm_benchmark.os should be either linux or windows, not '{config.lightgbm_benchmark.os}'")
        treelite_score_module = self.module_load("treelite_python_score")

        # Here you should create an instance of a pipeline function (using your custom config dataclass)
        @dsl.pipeline(name="lightgbm_benchmark", # pythonic name
                      description="LightGBM Benchmark",
                      default_datastore=config.compute.noncompliant_datastore)
        def lightgbm_benchmark_pipeline_function(data, model):
            """Pipeline function for this graph.

            Args:
                TODO

            Returns:
                dict[str->PipelineOutputData]: a dictionary of your pipeline outputs
                    for instance to be consumed by other graphs
            """
            # call module with all the right arguments
            lightgbm_score_step = lightgbm_score_module(
                data = data,
                model = model,
                predict_disable_shape_check = config.lightgbm_benchmark.predict_disable_shape_check
            )
            self.apply_smart_runsettings(lightgbm_score_step)

            treelite_score_step = treelite_score_module(
                data = data,
                model = model
            )
            self.apply_smart_runsettings(treelite_score_step)

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
        data = self.dataset_load(config.lightgbm_benchmark.data)
        model = self.dataset_load(config.lightgbm_benchmark.model)

        # when all inputs are obtained, we call the pipeline function
        experiment_pipeline = pipeline_function(data=data, model=model)

        # and we return that function so that helper can run it.
        return experiment_pipeline


# NOTE: main block is necessary only if script is intended to be run from command line
if __name__ == "__main__":
    # calling the helper .main() function
    LightGBMBenchmarkPipeline.main()
