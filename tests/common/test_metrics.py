"""Tests src/common/metrics.py"""
import os
import pytest
from unittest.mock import Mock, patch
import time
import platform

from common.metrics import MetricsLogger

@patch('mlflow.start_run')
def test_unique_mlflow_initialization(mlflow_start_run_mock):
    """ Tests MetricsLogger() unique initialization of mlflow"""
    metrics_logger = MetricsLogger()
    metrics_logger_2 = MetricsLogger()
    mlflow_start_run_mock.assert_called_once()


@patch('mlflow.log_metric')
def test_metrics_logger_log_metric(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_metric() """
    metrics_logger = MetricsLogger()

    metrics_logger.log_metric("foo", "bar")
    mlflow_log_metric_mock.assert_called_with(
        "foo", "bar"
    )


@patch('mlflow.log_metric')
def test_metrics_logger_log_metric_too_long(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_metric() """
    metrics_logger = MetricsLogger()

    metric_key = "x" * 250
    assert len(metric_key), 250

    short_metric_key = "x" * 50
    assert len(short_metric_key), 50

    metrics_logger.log_metric(
        metric_key, "bar"
    )
    mlflow_log_metric_mock.assert_called_with(
        short_metric_key, "bar"
    )


@patch('mlflow.set_tags')
def test_metrics_logger_set_properties(mlflow_set_tags_mock):
    """ Tests MetricsLogger().set_properties() """
    metrics_logger = MetricsLogger()

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

    platform_properties = {
        "machine":platform.machine(),
        "processor":platform.processor(),
        "system":platform.system(),
        "system_version":platform.version(),
        "cpu_count":os.cpu_count()
    }
    metrics_logger.set_platform_properties()

    mlflow_set_tags_mock.assert_called_with(
        platform_properties
    )


@patch('mlflow.log_params')
def test_metrics_logger_set_properties(mlflow_log_params_mock):
    """ Tests MetricsLogger().log_parameters() """
    metrics_logger = MetricsLogger()

    metrics_logger.log_parameters(
        key1 = "foo",
        key2 = 0.45
    )
    mlflow_log_params_mock.assert_called_with(
        { 'key1' : "foo", 'key2' : 0.45 }
    )


@patch('mlflow.log_metric')
def test_metrics_logger_log_time_block(mlflow_log_metric_mock):
    """ Tests MetricsLogger().log_time_block() """
    metrics_logger = MetricsLogger()

    with metrics_logger.log_time_block("foo_metric"):
        time.sleep(0.01)

    mlflow_log_metric_mock.assert_called_once()
