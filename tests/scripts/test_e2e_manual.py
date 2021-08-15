"""Tests src/scripts/ end-to-end LightGBM manual benchmark"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from scripts.generate_data import generate
from scripts.lightgbm_python import train
from scripts.lightgbm_python import score


def generate_data(output_dir, task_type):
    """Tests src/scripts/generate_data/generate.py"""
    script_args = [
        "generate.py",
        "--train_samples",
        "100",
        "--test_samples",
        "10",
        "--inferencing_samples",
        "100",
        "--n_features",
        "40",
        "--n_informative",
        "10",
        "--random_state",
        "5",
        "--output",
        output_dir,
        "--type",
        "regression",
    ]
    if task_type == "classification":
        script_args.extend(["--n_redundant", "5"])

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        generate.main()

    assert os.path.isfile(
        os.path.join(output_dir, "inference.txt")
    ), "Script generate.py should generate inference.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_dir, "train.txt")
    ), "Script generate.py should generate train.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_dir, "test.txt")
    ), "Script generate.py should generate test.txt under --output dir but did not"

def lightgbm_python_train(train_path, test_path, model_path, task_type):
    """Tests src/scripts/lightgbm_python/train.py"""
    script_args = [
        "train.py",
        "--train", train_path,
        "--test", test_path,
        "--export_model", model_path,
        "--objective", task_type,
        "--boosting_type", "gbdt",
        "--tree_learner", "serial",
        "--metric", "rmse",
        "--num_trees", "5",
        "--num_leaves", "10",
        "--min_data_in_leaf", "1",
        "--learning_rate", "0.3",
        "--max_bin", "16",
        "--feature_fraction", "0.15"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        train.main()

    assert os.path.isfile(
        model_path
    ), "Script train.py should generate a model output file but did not"


def lightgbm_python_score(data_path, model_path, predictions_path):
    """Tests src/scripts/lightgbm_python/score.py"""
    script_args = [
        "score.py",
        "--data", data_path,
        "--model", model_path,
        #"--output", predictions_path
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        score.main()

    #assert os.path.isfile(
    #    predictions_path
    #), "Script score.py should generate a predictions output file but did not"


def test_end_to_end_lightgbm_python(temporary_dir):
    data_dir = os.path.join(temporary_dir, "data")
    task_type = "regression"

    generate_data(data_dir, task_type)

    model_dir = os.path.join(temporary_dir, "model")
    lightgbm_python_train(
        train_path = os.path.join(data_dir, "train.txt"),
        test_path = os.path.join(data_dir, "test.txt"),
        model_path = model_dir,
        task_type = task_type
    )

    lightgbm_python_score(
        data_path = os.path.join(data_dir, "inference.txt"),
        model_path = model_dir,
        predictions_path = os.path.join(data_dir, "predictions.txt")
    )
