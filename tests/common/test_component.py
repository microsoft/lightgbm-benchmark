"""Tests src/common/io.py"""
import os
import pytest
from unittest.mock import call, Mock, patch
import time
import json

from common.components import SingleNodeScript
from common.metrics import MetricsLogger


def assert_runnable_script_properties(script_instance: SingleNodeScript, benchmark_name: str, mlflow_set_tags_mock: Mock):
    """Tests properties recorded by a SingleNodeScript class"""
    tags_calls = mlflow_set_tags_mock.call_args_list
    assert len(tags_calls) == 3

    # benchmark common proeprties
    assert (tags_calls[0].args[0] == {
        "task": script_instance.task,
        "framework": script_instance.framework,
        "framework_version": script_instance.framework_version
    }), "first call to set_tags() is supposed to be for benchmark properties"

    # custom properties
    assert (tags_calls[1].args[0] == {"benchmark_name": benchmark_name}), "2nd call to set_tags() is for custom properties parsed from json argument"

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


def assert_runnable_script_metrics(script_instance: SingleNodeScript, user_metrics: list, mlflow_log_metric_mock: Mock):
    """Tests metrics recorded by a SingleNodeScript class"""
    # now let's test all metrics
    metrics_calls = mlflow_log_metric_mock.call_args_list

    # N user metric + 18 performance metrics
    assert len(metrics_calls) == (18 + len(user_metrics))

    # user metric testing
    assert isinstance(user_metrics, list)
    for entry in user_metrics:
        assert isinstance(entry, dict)
        if 'key' in entry:
            assert metrics_calls[0].args[0] == entry['key']
        assert isinstance(metrics_calls[0].args[1], float)
        if 'value' in entry:
            assert metrics_calls[0].args[1] == entry['value']
        assert "step" in metrics_calls[0].kwargs
        if 'step' in entry:
            assert metrics_calls[0].kwargs["step"] == entry['step']

    # perf metrics
    perf_metrics_call_args = [
        "max_t_(cpu_pct_per_cpu_avg)",
        "cpu_avg_utilization_pct",
        "cpu_avg_utilization_at100_pct",
        "cpu_avg_utilization_over80_pct",
        "cpu_avg_utilization_over40_pct",
        "cpu_avg_utilization_over20_pct",
        "max_t_(cpu_pct_per_cpu_min)",
        "max_t_(cpu_pct_per_cpu_max)",
        "node_cpu_hours",
        "node_unused_cpu_hours",
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

    # mlflow initialization
    mlflow_start_run_mock.assert_called_once()
    mlflow_end_run_mock.assert_called_once()

    assert_runnable_script_properties(
        test_component,
        "unittest",
        mlflow_set_tags_mock
    )

    assert_runnable_script_metrics(
        test_component,
        [{'key':'fake_time_block', 'step':1}], # user_metrics
        mlflow_log_metric_mock
    )


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

def test_failure_single_node_script_metrics():
    # just run main
    with pytest.raises(Exception) as e_test:
        test_component = FailingSingleNodeScript.main(
            [
                "foo.py",
                "--verbose", "True",
                "--custom_properties", json.dumps({'benchmark_name':'unittest'})
            ]
        )
