"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from scripts.training.ray_tune import raytune
from common.distributed import multinode_config_class

# IMPORTANT: see conftest.py for fixtures

@patch('ray.shutdown') # patching this to avoid ray.shutdown() call, using ray_init_fixture instead
@patch('ray.init') # patching this to avoid ray.init() call, using ray_init_fixture instead
@patch('common.distributed.MultiNodeMPIDriver') # patching this to avoid mpi.init() call
def test_ray_tune(mpi_driver_mock, ray_init_mock, ray_shutdown_mock, ray_init_fixture, temporary_dir, regression_train_sample, regression_test_sample):
    """Tests src/scripts/training/ray_tune/raytune.py"""
    # fake mpi initialization + config
    mpi_driver_mock().get_multinode_config.return_value = multinode_config_class(
        1, # world_size
        0, # world_rank
        False, # mpi_available
        True, # main_node
    )

    output_file = os.path.join(temporary_dir, "output")

    # create test arguments for the script
    objective_argument = "regression"

    script_args = [
        "raytune.py",
        "--train", regression_train_sample,
        "--test", regression_test_sample,
        "--output_path", output_file,
        "--objective", objective_argument,
        "--boosting_type", "gbdt",
        "--tree_learner", "serial",
        "--header", "True",
        "--metric", "rmse",
        "--mode", "min",
        "--num_samples", "4",
        "--num_iterations", "randint(10,20)", # bayesian does not support categorical
        "--num_leaves", "10",
        "--min_data_in_leaf", "255",
        "--learning_rate", "0.3",
        "--max_bin", "16",
        "--feature_fraction", "0.15",
        "--device_type", "cpu",
        "--time_budget", "120"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        raytune.main()

    # test expected outputs
    assert os.path.isfile(
        os.path.join(output_file, "best_result.csv")
    ), "Script raytune.py should generate a best_result.csv output file but did not"

@patch('ray.shutdown') # patching this to avoid ray.shutdown() call, using ray_init_fixture instead
@patch('ray.init') # patching this to avoid ray.init() call, using ray_init_fixture instead
@patch('common.distributed.MultiNodeMPIDriver') # patching this to avoid mpi.init() call
def test_ray_tune_ASHA(mpi_driver_mock, ray_init_mock, ray_shutdown_mock, ray_init_fixture, temporary_dir, regression_train_sample, regression_test_sample):
    """Tests src/scripts/training/ray_tune/raytune.py"""
    # fake mpi initialization + config
    mpi_driver_mock().get_multinode_config.return_value = multinode_config_class(
        1, # world_size
        0, # world_rank
        False, # mpi_available
        True, # main_node
    )

    output_file = os.path.join(temporary_dir, "output")

    # create test arguments for the script
    objective_argument = "regression"

    script_args = [
        "raytune.py",
        "--train", regression_train_sample,
        "--test", regression_test_sample,
        "--output_path", output_file,
        "--objective", objective_argument,
        "--boosting_type", "gbdt",
        "--tree_learner", "serial",
        "--header", "True",
        "--metric", "rmse",
        "--mode", "min",
        "--num_samples", "4",
        "--num_iterations", "randint(10,20)", # bayesian does not support categorical
        "--num_leaves", "10",
        "--min_data_in_leaf", "255",
        "--learning_rate", "0.3",
        "--max_bin", "16",
        "--feature_fraction", "0.15",
        "--device_type", "cpu",
        "--time_budget", "120",
        "--scheduler", "ASHAScheduler"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        raytune.main()

    # test expected outputs
    assert os.path.isfile(
        os.path.join(output_file, "best_result.csv")
    ), "Script raytune.py should generate a best_result.csv output file but did not"

@patch('ray.shutdown') # patching this to avoid ray.shutdown() call, using ray_init_fixture instead
@patch('ray.init') # patching this to avoid ray.init() call, using ray_init_fixture instead
@patch('common.distributed.MultiNodeMPIDriver') # patching this to avoid mpi.init() call
def test_ray_tune_PBT(mpi_driver_mock, ray_init_mock, ray_shutdown_mock, ray_init_fixture, temporary_dir, regression_train_sample, regression_test_sample):
    """Tests src/scripts/training/ray_tune/raytune.py"""
    # fake mpi initialization + config
    mpi_driver_mock().get_multinode_config.return_value = multinode_config_class(
        1, # world_size
        0, # world_rank
        False, # mpi_available
        True, # main_node
    )

    output_file = os.path.join(temporary_dir, "output")

    # create test arguments for the script
    objective_argument = "regression"

    script_args = [
        "raytune.py",
        "--train", regression_train_sample,
        "--test", regression_test_sample,
        "--output_path", output_file,
        "--objective", objective_argument,
        "--boosting_type", "gbdt",
        "--tree_learner", "serial",
        "--header", "True",
        "--metric", "rmse",
        "--mode", "min",
        "--num_samples", "4",
        "--num_iterations", "randint(10,20)", # bayesian does not support categorical
        "--num_leaves", "10",
        "--min_data_in_leaf", "255",
        "--learning_rate", "0.3",
        "--max_bin", "16",
        "--feature_fraction", "0.15",
        "--device_type", "cpu",
        "--time_budget", "120",
        "--scheduler", "PopulationBasedTraining"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        raytune.main()

    # test expected outputs
    assert os.path.isfile(
        os.path.join(output_file, "best_result.csv")
    ), "Script raytune.py should generate a best_result.csv output file but did not"

@patch('ray.shutdown') # patching this to avoid ray.shutdown() call, using ray_init_fixture instead
@patch('ray.init') # patching this to avoid ray.init() call, using ray_init_fixture instead
@patch('common.distributed.MultiNodeMPIDriver') # patching this to avoid mpi.init() call
def test_ray_tune_blendsearch(mpi_driver_mock, ray_init_mock, ray_shutdown_mock, ray_init_fixture, temporary_dir, regression_train_sample, regression_test_sample):
    """Tests src/scripts/training/ray_tune/raytune.py"""
    # fake mpi initialization + config
    mpi_driver_mock().get_multinode_config.return_value = multinode_config_class(
        1, # world_size
        0, # world_rank
        False, # mpi_available
        True, # main_node
    )

    output_file = os.path.join(temporary_dir, "output")

    # create test arguments for the script
    objective_argument = "regression"

    script_args = [
        "raytune.py",
        "--train", regression_train_sample,
        "--test", regression_test_sample,
        "--output_path", output_file,
        "--objective", objective_argument,
        "--boosting_type", "gbdt",
        "--tree_learner", "serial",
        "--header", "True",
        "--metric", "rmse",
        "--mode", "min",
        "--num_samples", "4",
        "--num_iterations", "randint(10,20)", # bayesian does not support categorical
        "--num_leaves", "10",
        "--min_data_in_leaf", "255",
        "--learning_rate", "0.3",
        "--max_bin", "16",
        "--feature_fraction", "0.15",
        "--device_type", "cpu",
        "--time_budget", "120",
        "--search_alg", "BlendSearch",
        "--low_num_iterations", '10'
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        raytune.main()

    # test expected outputs
    assert os.path.isfile(
        os.path.join(output_file, "best_result.csv")
    ), "Script raytune.py should generate a best_result.csv output file but did not"

@patch('ray.shutdown') # patching this to avoid ray.shutdown() call, using ray_init_fixture instead
@patch('ray.init') # patching this to avoid ray.init() call, using ray_init_fixture instead
@patch('common.distributed.MultiNodeMPIDriver') # patching this to avoid mpi.init() call
def test_ray_tune_bohb(mpi_driver_mock, ray_init_mock, ray_shutdown_mock, ray_init_fixture, temporary_dir, regression_train_sample, regression_test_sample):
    """Tests src/scripts/training/ray_tune/raytune.py"""
    # fake mpi initialization + config
    mpi_driver_mock().get_multinode_config.return_value = multinode_config_class(
        1, # world_size
        0, # world_rank
        False, # mpi_available
        True, # main_node
    )

    output_file = os.path.join(temporary_dir, "output")

    # create test arguments for the script
    objective_argument = "regression"

    script_args = [
        "raytune.py",
        "--train", regression_train_sample,
        "--test", regression_test_sample,
        "--output_path", output_file,
        "--objective", objective_argument,
        "--boosting_type", "gbdt",
        "--tree_learner", "serial",
        "--header", "True",
        "--metric", "rmse",
        "--mode", "min",
        "--num_samples", "4",
        "--num_iterations", "randint(10,20)", # bayesian does not support categorical
        "--num_leaves", "10",
        "--min_data_in_leaf", "255",
        "--learning_rate", "0.3",
        "--max_bin", "16",
        "--feature_fraction", "0.15",
        "--device_type", "cpu",
        "--time_budget", "120",
        "--search_alg", "TuneBOHB",
        "--scheduler" "HyperBandForBOHB"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        raytune.main()

    # test expected outputs
    assert os.path.isfile(
        os.path.join(output_file, "best_result.csv")
    ), "Script raytune.py should generate a best_result.csv output file but did not"

@patch('ray.shutdown') # patching this to avoid ray.shutdown() call, using ray_init_fixture instead
@patch('ray.init') # patching this to avoid ray.init() call, using ray_init_fixture instead
@patch('common.distributed.MultiNodeMPIDriver') # patching this to avoid mpi.init() call
def test_ray_tune_optuna_asha(mpi_driver_mock, ray_init_mock, ray_shutdown_mock, ray_init_fixture, temporary_dir, regression_train_sample, regression_test_sample):
    """Tests src/scripts/training/ray_tune/raytune.py"""
    # fake mpi initialization + config
    mpi_driver_mock().get_multinode_config.return_value = multinode_config_class(
        1, # world_size
        0, # world_rank
        False, # mpi_available
        True, # main_node
    )

    output_file = os.path.join(temporary_dir, "output")

    # create test arguments for the script
    objective_argument = "regression"

    script_args = [
        "raytune.py",
        "--train", regression_train_sample,
        "--test", regression_test_sample,
        "--output_path", output_file,
        "--objective", objective_argument,
        "--boosting_type", "gbdt",
        "--tree_learner", "serial",
        "--header", "True",
        "--metric", "rmse",
        "--mode", "min",
        "--num_samples", "4",
        "--num_iterations", "randint(10,20)", # bayesian does not support categorical
        "--num_leaves", "10",
        "--min_data_in_leaf", "255",
        "--learning_rate", "0.3",
        "--max_bin", "16",
        "--feature_fraction", "0.15",
        "--device_type", "cpu",
        "--time_budget", "120",
        "--search_alg", "OptunaSearch",
        "--scheduler" "ASHAScheduler"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        raytune.main()

    # test expected outputs
    assert os.path.isfile(
        os.path.join(output_file, "best_result.csv")
    ), "Script raytune.py should generate a best_result.csv output file but did not"