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
    pipeline_func_mock = Mock()
    aml_connect_mock.return_value = None

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
        "test_config.test_param=foo"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        ret_value = pipeline_submit_main(
            test_config,
            pipeline_func_mock
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

    # experiment
    assert pipeline_config.experiment.name == "test_experiment_name"

    # custom params
    assert pipeline_config.test_config.test_param == "foo"
