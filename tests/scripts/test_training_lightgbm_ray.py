"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from scripts.training.lightgbm_ray import train
from common.distributed import multinode_config_class

# IMPORTANT: see conftest.py for fixtures

@patch('ray.shutdown') # patching this to avoid ray.shutdown() call, using ray_init_fixture instead
@patch('ray.init') # patching this to avoid ray.init() call, using ray_init_fixture instead
@patch('common.distributed.MultiNodeMPIDriver') # patching this to avoid mpi.init() call
def test_lightgbm_ray_train(mpi_driver_mock, ray_init_mock, ray_shutdown_mock, ray_init_fixture, temporary_dir, regression_train_sample, regression_test_sample):
    """Tests src/scripts/training/lightgbm_ray/train.py"""
    # fake mpi initialization + config
    mpi_driver_mock().get_multinode_config.return_value = multinode_config_class(
        1, # world_size
        0, # world_rank
        False, # mpi_available
        True, # main_node
    )

    model_dir = os.path.join(temporary_dir, "model")
    model_filename = 'custom_model_filename.txt'
    # create test arguments for the script
    objective_argument = "regression"

    script_args = [
        "train.py",
        "--train", regression_train_sample,
        "--label_column", "label",
        "--test", regression_test_sample,
        "--export_model", model_dir,
        "--model_filename", model_filename,        
        "--objective", objective_argument,
        "--boosting_type", "gbdt",
        "--tree_learner", "serial",
        "--metric", "rmse",
        "--num_iterations", "5",
        "--num_leaves", "10",
        "--min_data_in_leaf", "255",
        "--learning_rate", "0.3",
        "--max_bin", "16",
        "--feature_fraction", "0.15",
        "--device_type", "cpu",
        "--cluster_auto_setup", "False"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        train.main()

    # test expected outputs
    assert os.path.isfile(
        os.path.join(model_dir, model_filename)
    ), f"Script train.py should generate a {model_filename} output file but did not"
