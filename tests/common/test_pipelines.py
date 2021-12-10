"""Tests src/common/pipelines.py"""
import os
import sys
import pytest
from unittest.mock import call, Mock, patch
import time

from dataclasses import dataclass
from omegaconf import DictConfig

from common.pipelines import (
    aml_connection_config,
    compute_config,
    run_config,
    experiment_config,
    pipeline_submit_main
)

@patch('common.pipelines.azureml_connect')
def test_pipeline_submit_main(aml_connect_mock):
    # need a mock pipeline function (dsl.pipeline)
    pipeline_func_mock = Mock()

    # need a mock pipeline instance (returned by pipeline func)
    pipeline_instance_mock = Mock()
    pipeline_func_mock.return_value = pipeline_instance_mock

    # no need for a workspace, but we can test call to pipeline instance
    aml_connect_mock.return_value = "fake_workspace"

    @dataclass
    class test_config:
        test_param: str = "default_str"


    script_args = [
        "test_pipelines.py",
        "+aml.subscription_id=test_subscription",
        "+aml.resource_group=test_resource_group",
        "+aml.workspace_name=test_workspace_name",
        "+aml.tenant=test_tenant",
        "+experiment.name=test_experiment_name",
        "+compute.linux_cpu=test-cluster",
        "+compute.linux_gpu=test-gpu-cluster",
        "+compute.windows_cpu=test-win-cpu",
        "test_config.test_param=foo",
        "run.submit=True"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        ret_value = pipeline_submit_main(
            # config class
            test_config,

            # dsl.pipeline func
            pipeline_func_mock,

            # test hardcoded overrides
            experiment_description="test_description",
            display_name="test_display_name",
            tags={'foo':'bar'}
        )

    # checking call to pipeline_func
    pipeline_func_mock.assert_called_once()

    # argument should be a DictConfig
    pipeline_call_args = pipeline_func_mock.call_args
    pipeline_config = pipeline_call_args.args[0]
    assert isinstance(pipeline_config, DictConfig)

    # checking config fields (see dataclass above)  
    # aml connect
    assert pipeline_config.aml.subscription_id == "test_subscription"
    assert pipeline_config.aml.resource_group == "test_resource_group"
    assert pipeline_config.aml.workspace_name == "test_workspace_name"
    assert pipeline_config.aml.tenant == "test_tenant"

    # compute
    assert pipeline_config.compute.linux_cpu == "test-cluster"
    assert pipeline_config.compute.linux_gpu == "test-gpu-cluster"
    assert pipeline_config.compute.windows_cpu == "test-win-cpu"

    # custom params
    assert pipeline_config.test_config.test_param == "foo"

    pipeline_instance_mock.validate.assert_called_once()
    pipeline_instance_mock.validate.assert_called_with(
        workspace="fake_workspace" # what's returned by aml_connect mock
    )

    pipeline_instance_mock.submit.assert_called_once()
    pipeline_instance_mock.submit.assert_called_with(
        workspace="fake_workspace", # what's returned by aml_connect mock
        experiment_name="test_experiment_name",
        description="test_description",
        display_name="test_display_name",
        tags={'foo':'bar'},
        default_compute_target="cpu-cluster",
        regenerate_outputs=False, # default
        continue_on_step_failure=False # default
    )
