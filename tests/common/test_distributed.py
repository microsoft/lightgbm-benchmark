"""Tests src/common/io.py"""
import os
import pytest
from unittest.mock import call, Mock, patch
import time
import json

from common.distributed import MultiNodeScript, mpi_config_class
from test_component import (
    assert_runnable_script_properties,
    assert_runnable_script_metrics
)

class FakeMultiNodeScript(MultiNodeScript):
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
@patch('common.distributed.MPIHandler')
def test_multi_node_script(mpi_handler_mock, mlflow_start_run_mock, mlflow_set_tags_mock, mlflow_log_metric_mock, mlflow_end_run_mock):
    # fake mpi initialization + config
    mpi_handler_mock().mpi_config.return_value = mpi_config_class(
        1, # world_size
        0, # world_rank
        False, # mpi_available
        True, # main_node
    )

    # then just run main
    test_component = FakeMultiNodeScript.main(
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


class FailingMultiNodeScript(MultiNodeScript):
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

@patch('common.distributed.MPIHandler')
def test_multi_node_script_failure(mpi_handler_mock):
    # fake mpi initialization + config
    mpi_handler_mock().mpi_config.return_value = mpi_config_class(
        1, # world_size
        0, # world_rank
        False, # mpi_available
        True, # main_node
    )

    # just run main
    with pytest.raises(Exception) as e_test:
        test_component = FailingMultiNodeScript.main(
            [
                "foo.py",
                "--verbose", "True",
                "--custom_properties", json.dumps({'benchmark_name':'unittest'})
            ]
        )
