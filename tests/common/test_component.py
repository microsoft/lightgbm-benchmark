"""Tests src/common/io.py"""
import os
import pytest
from unittest.mock import call, Mock, patch
import time
import json

from common.components import SingleNodeScript
from common.metrics import MetricsLogger

class FakeSingleNodeScript(SingleNodeScript):
    def __init__(self):
        super().__init__(
            task="unittest",
            framework="pytest",
            framework_version=pytest.__version__
        )

    def run(self, args, logger, metrics_logger, unknown_args):
        # don't do anything
        with metrics_logger.log_time_block("fake_time_block", step=1):
            time.sleep(1)


@patch('mlflow.end_run')
@patch('mlflow.log_metric')
@patch('mlflow.set_tags')
@patch('mlflow.start_run')
def test_single_node_script_metrics(mlflow_start_run_mock, mlflow_set_tags_mock, mlflow_log_metric_mock, mlflow_end_run_mock):
    # just run main
    test_component = FakeSingleNodeScript.main(
        [
            "foo.py",
            "--verbose", "True",
            "--custom_properties", json.dumps({'benchmark_name':'unittest'})
        ]
    )

    ##################
    ### PROPERTIES ###
    ##################

    tags_calls = mlflow_set_tags_mock.call_args_list
    assert len(tags_calls) == 3

    # benchmark common proeprties
    assert (tags_calls[0].args[0] == {"task": "unittest", "framework": "pytest", "framework_version": pytest.__version__}), "first call to set_tags() is supposed to be for benchmark properties"

    # custom properties
    assert (tags_calls[1].args[0] == {"benchmark_name": "unittest"}), "2nd call to set_tags() is for custom properties parsed from json argument"

    # test all platform properties
    platform_property_keys = [
        "machine",
        "processor",
        "system",
        "system_version",
        "cpu_count",
        "architecture",
        "platform",
        "cpu_frequency",
        "system_memory"
    ]
    for key in platform_property_keys:
        assert key in tags_calls[2].args[0], f"platform property {key} is expected in the 3nd call to set_tags()"

    ###############
    ### METRICS ###
    ###############

    # mlflow initialization
    mlflow_start_run_mock.assert_called_once()
    mlflow_end_run_mock.assert_called_once()

    # now let's test all metrics
    metrics_calls = mlflow_log_metric_mock.call_args_list

    # 1 user metric, 11 performance metrics
    assert len(metrics_calls) == 12

    # user metric (time block)
    assert metrics_calls[0].args[0] == "fake_time_block"
    assert isinstance(metrics_calls[0].args[1], float)
    assert "step" in metrics_calls[0].kwargs
    assert metrics_calls[0].kwargs["step"] == 1

    # perf metrics
    perf_metrics_call_args = [
        "max_t_(cpu_pct_per_cpu_avg)",
        "max_t_(cpu_pct_per_cpu_min)",
        "max_t_(cpu_pct_per_cpu_max)",
        "max_t_(mem_percent)",
        "max_t_(disk_usage_percent)",
        "max_t_(disk_io_read_mb)",
        "max_t_(disk_io_write_mb)",
        "max_t_(net_io_lo_sent_mb)",
        "max_t_(net_io_ext_sent_mb)",
        "max_t_(net_io_lo_recv_mb)",
        "max_t_(net_io_ext_recv_mb)",
    ]
    for index, metric_key in enumerate(perf_metrics_call_args):
        assert metrics_calls[index+1].args[0] == MetricsLogger._remove_non_allowed_chars(metric_key)
        assert "step" in metrics_calls[index+1].kwargs
        assert metrics_calls[index+1].kwargs["step"] == 0 # using node id as step


class FailingSingleNodeScript(SingleNodeScript):
    def __init__(self):
        super().__init__(
            task="failure",
            framework="pytest",
            framework_version=pytest.__version__
        )

    def run(self, args, logger, metrics_logger, unknown_args):
        # don't do anything
        with metrics_logger.log_time_block("fake_time_block", step=1):
            time.sleep(1)
            raise Exception("Some fake issue occured during code!")


@patch('mlflow.end_run')
@patch('mlflow.log_metric')
@patch('mlflow.set_tags')
@patch('mlflow.start_run')
def test_single_node_script_metrics(mlflow_start_run_mock, mlflow_set_tags_mock, mlflow_log_metric_mock, mlflow_end_run_mock):
    # just run main
    with pytest.raises(Exception) as e_test:
        test_component = FailingSingleNodeScript.main(
            [
                "foo.py",
                "--verbose", "True",
                "--custom_properties", json.dumps({'benchmark_name':'unittest'})
            ]
        )
