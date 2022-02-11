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

# IMPORTANT: see conftest.py for fixtures

def test_ray_tune(temporary_dir, regression_train_sample, regression_test_sample):
    """Tests src/scripts/training/ray_tune/raytune.py"""
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
        "--num_trees", "choice([30,40,50,60])",
        "--num_leaves", "10",
        "--min_data_in_leaf", "255",
        "--learning_rate", "0.3",
        "--max_bin", "16",
        "--feature_fraction", "0.15",
        "--device_type", "cpu",
        "--num_samples", "5",
        "--time_budget", "120"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        raytune.main()

    # test expected outputs
    assert os.path.isfile(
        os.path.join(output_file, "best_result.csv")
    ), "Script raytune.py should generate a best_result.csv output file but did not"