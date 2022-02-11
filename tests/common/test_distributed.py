"""Tests src/common/io.py"""
import os
import pytest
from unittest.mock import call, Mock, patch
import time
import json

from common.distributed import MultiNodeScript, MultiNodeMPIDriver, multinode_config_class
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
@patch('mlflow.log_artifact')
@patch('mlflow.log_artifact')
@patch('mlflow.log_metric')
@patch('mlflow.set_tags')
@patch('mlflow.start_run')
@patch('common.distributed.MultiNodeMPIDriver')
def test_multi_node_script(mpi_driver_mock, mlflow_start_run_mock, mlflow_set_tags_mock, mlflow_log_metric_mock, mlflow_log_artifact_mock, mlflow_log_figure_mock, mlflow_end_run_mock):
    # fake mpi initialization + config
    mpi_driver_mock().get_multinode_config.return_value = multinode_config_class(
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

    mlflow_log_artifact_mock.assert_called_once() # perf data exported in json
    mlflow_log_figure_mock.assert_called_once() # perf plot


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

@patch('common.distributed.MultiNodeMPIDriver')
def test_multi_node_script_failure(mpi_driver_mock):
    # fake mpi initialization + config
    mpi_driver_mock().get_multinode_config.return_value = multinode_config_class(
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


def test_mpi_driver_mpi_init():
    """Tests the MultiNodeMPIDriver class"""
    # create MPI module mock
    mpi_module_mock = Mock()
    mpi_module_mock.COMM_WORLD = Mock()
    mpi_module_mock.COMM_WORLD.Get_size.return_value = 10
    mpi_module_mock.COMM_WORLD.Get_rank.return_value = 3
    mpi_module_mock.THREAD_MULTIPLE = 3

    # patch _mpi_import to return our MPI module mock
    with patch.object(MultiNodeMPIDriver, "_mpi_import") as mpi_import_mock:
        mpi_import_mock.return_value = mpi_module_mock

        mpi_handler = MultiNodeMPIDriver(mpi_init_mode=3) # MPI.THREAD_MULTIPLE
        mpi_handler.initialize()
        mpi_config = mpi_handler.get_multinode_config()
        mpi_handler.finalize()

        # test this random config
        assert mpi_config.world_rank == 3
        assert mpi_config.world_size == 10
        assert mpi_config.multinode_available == True
        assert mpi_config.main_node == False

def test_mpi_driver_no_mpi_init():
    """Tests the MultiNodeMPIDriver class"""
    # create MPI module mock
    mpi_module_mock = Mock()
    mpi_module_mock.COMM_WORLD = Mock()
    mpi_module_mock.COMM_WORLD.Get_size.return_value = 10 # different value just to make the point
    mpi_module_mock.COMM_WORLD.Get_rank.return_value = 3 # different value just to make the point
    mpi_module_mock.THREAD_MULTIPLE = 3

    # patch _mpi_import to return our MPI module mock
    with patch.object(MultiNodeMPIDriver, "_mpi_import") as mpi_import_mock:
        with patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "6", "OMPI_COMM_WORLD_RANK": "2"}):
            mpi_import_mock.return_value = mpi_module_mock

            mpi_handler = MultiNodeMPIDriver(mpi_init_mode=None)
            mpi_handler.initialize()
            mpi_config = mpi_handler.get_multinode_config()
            mpi_handler.finalize()

            # test this random config
            assert mpi_config.world_rank == 2
            assert mpi_config.world_size == 6
            assert mpi_config.multinode_available == True
            assert mpi_config.main_node == False
