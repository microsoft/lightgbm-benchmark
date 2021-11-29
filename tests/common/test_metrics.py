"""Tests src/common/metrics.py"""
import os
import pytest
from unittest.mock import call, Mock, patch
import time
import platform
import psutil

from common.metrics import MetricsLogger

@patch('mlflow.end_run')
@patch('mlflow.start_run')
def test_unique_mlflow_initialization(mlflow_start_run_mock, mlflow_end_run_mock):
    """ Tests MetricsLogger() unique initialization of mlflow"""
    metrics_logger = MetricsLogger()
    metrics_logger_2 = MetricsLogger()
    metrics_logger.close()
    metrics_logger_2.close()

    mlflow_start_run_mock.assert_called_once()
    mlflow_end_run_mock.assert_called_once()


@patch('mlflow.log_metric')
def test_metrics_logger_log_metric(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_metric() """
    metrics_logger = MetricsLogger()
    metrics_logger.close()

    metrics_logger.log_metric("foo", "bar", step=16)
    mlflow_log_metric_mock.assert_called_with(
        "foo", "bar", step=16
    )



@patch('mlflow.log_metric')
def test_metrics_logger_log_metric_with_prefix(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_metric() """
    metrics_logger = MetricsLogger(metrics_prefix="foo/")
    metrics_logger.close()

    metrics_logger.log_metric("foo", "bar", step=16)
    mlflow_log_metric_mock.assert_called_with(
        "foo/foo", "bar", step=16
    )


@patch('mlflow.log_metric')
def test_metrics_logger_log_metric_with_prefix_2sessions(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_metric() """
    metrics_logger = MetricsLogger(metrics_prefix="foo/")
    metrics_logger_2 = MetricsLogger()

    metrics_logger.log_metric("foo", "bar", step=16)
    mlflow_log_metric_mock.assert_called_with(
        "foo/foo", "bar", step=16
    )

    metrics_logger_2.log_metric("foo2", "bar2", step=12)
    mlflow_log_metric_mock.assert_called_with(
        "foo/foo2", "bar2", step=12
    )

    metrics_logger.close()
    metrics_logger_2.close()



@patch('mlflow.log_metric')
def test_metrics_logger_log_metric_too_long(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_metric() """
    metrics_logger = MetricsLogger()
    metrics_logger.close()

    metric_key = "x" * 250
    assert len(metric_key), 250

    short_metric_key = "x" * 50
    assert len(short_metric_key), 50

    metrics_logger.log_metric(
        metric_key, "bar", step=15
    )
    mlflow_log_metric_mock.assert_called_with(
        short_metric_key, "bar", step=15
    )


def test_metrics_logger_log_metric_non_allowed_chars():
    """ Tests MetricsLogger().log_metric() """
    metrics_logger = MetricsLogger()
    metrics_logger.close()

    test_cases = [
        {
            'input':   "a!@$b%^&c_-/d",
            'expected':"abc_-/d"
        },
        {
            'input':   "abcd",
            'expected':"abcd"
        },
        {
            'input':   "node_0/valid_0.ndcg@1",
            'expected':"node_0/valid_0.ndcg1"
        },
    ]

    for test_case in test_cases:
        assert metrics_logger._remove_non_allowed_chars(test_case['input']) == test_case['expected']


@patch('mlflow.set_tags')
def test_metrics_logger_set_properties(mlflow_set_tags_mock):
    """ Tests MetricsLogger().set_properties() """
    metrics_logger = MetricsLogger()
    metrics_logger.close()

    metrics_logger.set_properties(
        key1 = "foo",
        key2 = 0.45
    )
    mlflow_set_tags_mock.assert_called_with(
        { 'key1' : "foo", 'key2' : 0.45 }
    )


@patch('mlflow.set_tags')
def test_metrics_logger_set_platform_properties(mlflow_set_tags_mock):
    """ Tests MetricsLogger().set_properties() """
    metrics_logger = MetricsLogger()
    metrics_logger.close()

    platform_properties = {
        "machine":platform.machine(),
        "processor":platform.processor(),
        "system":platform.system(),
        "system_version":platform.version(),
        "cpu_count":os.cpu_count(),
        "architecture":"-".join(platform.architecture()),
        "platform":platform.platform(),
        "cpu_frequency":round(psutil.cpu_freq().current),
        "system_memory":round((psutil.virtual_memory().total) / (1024*1024*1024))
    }
    metrics_logger.set_platform_properties()

    mlflow_set_tags_mock.assert_called_with(
        platform_properties
    )

@patch('mlflow.set_tags')
def test_metrics_logger_set_properties_from_json(mlflow_set_tags_mock):
    """ Tests MetricsLogger().set_properties_from_json() """
    metrics_logger = MetricsLogger()
    metrics_logger.close()

    metrics_logger.set_properties_from_json(
        "{ \"key1\" : \"foo\", \"key2\" : 0.45 }"
    )
    mlflow_set_tags_mock.assert_called_with(
        { 'key1' : "foo", 'key2' : '0.45' }
    )

    # test failure during json parsing
    with pytest.raises(ValueError) as exc_info:
        metrics_logger.set_properties_from_json(
            "{ 'foo': NOTHING }"
        )
    # making sure it's the right exception
    assert str(exc_info.value).startswith("During parsing of JSON properties")

    # test failure if dict is not provided
    with pytest.raises(ValueError) as exc_info:
        metrics_logger.set_properties_from_json(
            "[\"bla\", \"foo\"]"
        )
    # making sure it's the right exception
    assert str(exc_info.value).startswith("Provided JSON properties should be a dict")

@patch('mlflow.log_param')
def test_metrics_logger_log_parameters(mlflow_log_param_mock):
    """ Tests MetricsLogger().log_parameters() """
    metrics_logger = MetricsLogger()
    metrics_logger.close()

    metrics_logger.log_parameters(
        key1 = "foo",
        key2 = 0.45,
        str_way_too_long = ("*" * 1024)
    )
    mlflow_log_param_mock.assert_has_calls(
        [
            call("key1", "foo"),
            call("key2", 0.45),
        ],
        any_order=True
    )


@patch('mlflow.log_metric')
def test_metrics_logger_log_time_block(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_time_block() """
    metrics_logger = MetricsLogger()
    metrics_logger.close()

    with metrics_logger.log_time_block("foo_metric"):
        time.sleep(0.01)

    mlflow_log_metric_mock.assert_called_once()
