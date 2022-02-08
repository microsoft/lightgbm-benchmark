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

# IMPORTANT: see conftest.py for fixtures

def test_lightgbm_python_train(temporary_dir, regression_train_sample, regression_test_sample):
    """Tests src/scripts/training/lightgbm_ray/train.py"""
    model_dir = os.path.join(temporary_dir, "model")

    # create test arguments for the script
    objective_argument = "regression"

    script_args = [
        "train.py",
        "--train", regression_train_sample,
        "--label_column", "label",
        "--test", regression_test_sample,
        "--export_model", model_dir,
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
        "--device_type", "cpu"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        train.main()

    # test expected outputs
    assert os.path.isfile(
        os.path.join(model_dir, "model.txt")
    ), "Script train.py should generate a model.txt output file but did not"
