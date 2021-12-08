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
def test_unique_mlflow_parallel_initialization(mlflow_start_run_mock, mlflow_end_run_mock):
    """ Tests MetricsLogger() unique initialization of mlflow"""
    # if open is called twice, we initialize mlflow only once
    metrics_logger = MetricsLogger()
    metrics_logger_2 = MetricsLogger()
    metrics_logger.open()
    metrics_logger_2.open()
    metrics_logger.close()
    metrics_logger_2.close()

    mlflow_start_run_mock.assert_called_once()
    mlflow_end_run_mock.assert_called_once()


@patch('mlflow.end_run')
@patch('mlflow.start_run')
def test_unique_mlflow_sequence_initialization(mlflow_start_run_mock, mlflow_end_run_mock):
    """ Tests MetricsLogger() unique initialization of mlflow"""
    # when open/close multiple times in a sequence, we initialize mlflow each time
    metrics_logger = MetricsLogger()
    metrics_logger.open()
    metrics_logger.close()

    assert mlflow_start_run_mock.call_count == 1
    assert mlflow_end_run_mock.call_count == 1

    metrics_logger_2 = MetricsLogger()
    metrics_logger_2.open()
    metrics_logger_2.close()

    assert mlflow_start_run_mock.call_count == 2
    assert mlflow_end_run_mock.call_count == 2


@patch('mlflow.log_metric')
def test_metrics_logger_log_metric(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_metric() """
    metrics_logger = MetricsLogger()
    metrics_logger.open()
    metrics_logger.log_metric("foo", "bar", step=16)
    metrics_logger.close()

    mlflow_log_metric_mock.assert_called_with(
        "foo", "bar", step=16
    )


@patch('mlflow.log_metric')
def test_metrics_logger_log_metric_with_prefix(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_metric() """
    metrics_logger = MetricsLogger(metrics_prefix="foo/")
    metrics_logger.open()
    metrics_logger.log_metric("foo", "bar", step=16)
    metrics_logger.close()

    mlflow_log_metric_mock.assert_called_with(
        "foo/foo", "bar", step=16
    )


@patch('mlflow.log_metric')
def test_mlflow_metrics_logger_log_metric_with_prefix_2sessions(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_metric() """
    # when initializing multiple times, each logger has its own prefix
    metrics_logger = MetricsLogger(metrics_prefix="foo/")
    metrics_logger.open()

    metrics_logger_2 = MetricsLogger()
    metrics_logger_2.open()

    metrics_logger.log_metric("foo", "bar", step=16)
    mlflow_log_metric_mock.assert_called_with(
        "foo/foo", "bar", step=16
    )

    metrics_logger_2.log_metric("foo2", "bar2", step=12)
    mlflow_log_metric_mock.assert_called_with(
        "foo2", "bar2", step=12
    )

    metrics_logger.close()
    metrics_logger_2.close()


@patch('mlflow.log_metric')
def test_metrics_logger_log_metric_too_long(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_metric() """
    metrics_logger = MetricsLogger()
    metrics_logger.open()

    metric_key = "x" * 250
    assert len(metric_key), 250

    short_metric_key = "x" * 50
    assert len(short_metric_key), 50

    metrics_logger.log_metric(
        metric_key, "bar", step=15
    )

    metrics_logger.close()

    mlflow_log_metric_mock.assert_called_with(
        short_metric_key, "bar", step=15
    )


def test_metrics_logger_log_metric_non_allowed_chars():
    """ Tests MetricsLogger().log_metric() """
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
        assert MetricsLogger._remove_non_allowed_chars(test_case['input']) == test_case['expected']


@patch('mlflow.set_tags')
def test_metrics_logger_set_properties(mlflow_set_tags_mock):
    """ Tests MetricsLogger().set_properties() """
    metrics_logger = MetricsLogger()
    metrics_logger.open()
    metrics_logger.set_properties(
        key1 = "foo",
        key2 = 0.45
    )
    metrics_logger.close()

    mlflow_set_tags_mock.assert_called_with(
        { 'key1' : "foo", 'key2' : 0.45 }
    )


@patch('mlflow.set_tags')
def test_metrics_logger_set_platform_properties(mlflow_set_tags_mock):
    """ Tests MetricsLogger().set_properties() """
    metrics_logger = MetricsLogger()
    metrics_logger.open()

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

    metrics_logger.close()

    mlflow_set_tags_mock.assert_called_with(
        platform_properties
    )

@patch('mlflow.set_tags')
def test_metrics_logger_set_properties_from_json(mlflow_set_tags_mock):
    """ Tests MetricsLogger().set_properties_from_json() """
    metrics_logger = MetricsLogger()
    metrics_logger.open()

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

    metrics_logger.close()

@patch('mlflow.log_param')
def test_metrics_logger_log_parameters(mlflow_log_param_mock):
    """ Tests MetricsLogger().log_parameters() """
    metrics_logger = MetricsLogger()
    metrics_logger.open()

    metrics_logger.log_parameters(
        key1 = "foo",
        key2 = 0.45,
        str_way_too_long = ("*" * 1024)
    )

    metrics_logger.close()

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
    metrics_logger = MetricsLogger(metrics_prefix="foo_time_block/")
    metrics_logger.open()

    with metrics_logger.log_time_block("foo_metric", step=2):
        time.sleep(0.01)

    metrics_logger.close()

    # there should be only one call in this case
    metric_calls = mlflow_log_metric_mock.call_args_list
    assert mlflow_log_metric_mock.call_count == 1
    assert len(metric_calls) == 1

    # test metric key argument
    assert (metric_calls[0].args[0] == "foo_time_block/foo_metric")
    assert (metric_calls[0].kwargs["step"] == 2)


@patch('mlflow.log_figure')
@patch('mlflow.log_metric')
def test_log_inferencing_latencies(mlflow_log_metric_mock, mlflow_log_figure_mock):
    """ Tests MetricsLogger().log_inferencing_larencies() """
    metrics_logger = MetricsLogger()
    metrics_logger.open()

    test_latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 5.0]
    test_batch_sizes = [1, 1, 1, 1, 1, 1, 1, 5]

    metrics_logger.log_inferencing_latencies(test_latencies, batch_length=test_batch_sizes, factor_to_usecs=1000000.0)

    metrics_logger.close()

    #assert mlflow_log_metric_mock.call_args_list == []

    mlflow_log_metric_mock.assert_has_calls(
        [
            call("prediction_batches", 8, step=None), # len(test_latencies)
            call("prediction_queries", 12, step=None), # sum(test_batch_sizes)

            # reference values based on test_latencies above
            call('prediction_latency_avg', 650000.0, step=None),
            call('batch_latency_p50_usecs', 450000.0, step=None),
            call('batch_latency_p75_usecs', 625000.0, step=None),
            call('batch_latency_p90_usecs', 1989999.9999999993, step=None),
            call('batch_latency_p95_usecs', 3494999.9999999977, step=None),
            call('batch_latency_p99_usecs', 4698999.999999999, step=None),
            call('prediction_latency_p50_usecs', 450000.0, step=None),
            call('prediction_latency_p75_usecs', 625000.0, step=None),
            call('prediction_latency_p90_usecs', 790000.0, step=None),
            call('prediction_latency_p95_usecs', 894999.9999999999, step=None),
            call('prediction_latency_p99_usecs', 978999.9999999999, step=None),
        ],
        any_order=True
    )

    # testing logging figures
    figure_calls = mlflow_log_figure_mock.call_args_list
    assert mlflow_log_figure_mock.call_count == 2
    assert len(figure_calls) == 2

    # second argument of each call is file name
    assert (figure_calls[0].args[1] == "batch_latency_log_histogram.png")
    assert (figure_calls[1].args[1] == "prediction_latency_log_histogram.png")
