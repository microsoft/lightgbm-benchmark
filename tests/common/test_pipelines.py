"""Tests src/common/pipelines.py"""
import os
import sys
import pytest
from unittest.mock import call, Mock, patch
import time

from dataclasses import dataclass
from omegaconf import DictConfig

from common.pipelines import (
    parse_pipeline_config,
    azureml_connect,
    pipeline_submit
)


def test_parse_pipeline_config():
    """Creates a config dataclass and tests parsing it from CLI"""
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
        "test_config.test_param=test_str_value",
        "run.submit=True"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        pipeline_config = parse_pipeline_config(test_config)

    # test return value type
    assert isinstance(pipeline_config, DictConfig)

    # test some custom value
    assert pipeline_config.test_config.test_param == "test_str_value"

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

    return pipeline_config

def test_pipeline_submit():
    # need a mock pipeline mock (Pipeline)
    pipeline_instance_mock = Mock()
    workspace_mock = "fake_workspace"

    # reusing config from previous test
    pipeline_config = test_parse_pipeline_config()

    pipeline_run = pipeline_submit(
        workspace_mock,
        pipeline_config,
        pipeline_instance_mock,

        # test hardcoded overrides
        experiment_description="test_description",
        display_name="test_display_name",
        tags={'foo':'bar'}
    )

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
