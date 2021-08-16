"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from scripts.generate_data import generate
from scripts.lightgbm_python import train
from scripts.lightgbm_python import score


def generate_data(output_train,
                  output_test,
                  output_inference,
                  task_type,
                  mlflow_start_run_mock,
                  mlflow_end_run_mock,
                  mlflow_log_metric_mock,
                  mlflow_set_tags_mock,
                  mlflow_log_params):
    """Tests src/scripts/generate_data/generate.py"""
    # create test arguments for the script
    script_args = [
        "generate.py",
        "--train_samples", "100",
        "--test_samples", "10",
        "--inferencing_samples", "100",
        "--n_features", "40",
        "--n_informative", "10",
        "--random_state", "5",
        "--output_train", output_train,
        "--output_test", output_test,
        "--output_inference", output_inference,
        "--type", "regression",
    ]
    if task_type == "classification":
        script_args.extend(["--n_redundant", "5"])

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        generate.main()

    # test expected outputs
    assert os.path.isfile(
        os.path.join(output_train, "train.txt")
    ), "Script generate.py should generate train.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_test, "test.txt")
    ), "Script generate.py should generate test.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_inference, "inference.txt")
    ), "Script generate.py should generate inference.txt under --output dir but did not"


def lightgbm_python_train(train_dir,
                          test_dir,
                          model_dir,
                          task_type,
                          mlflow_start_run_mock,
                          mlflow_end_run_mock,
                          mlflow_log_metric_mock,
                          mlflow_set_tags_mock,
                          mlflow_log_params):
    """Tests src/scripts/lightgbm_python/train.py"""
    # create test arguments for the script
    if task_type == "regression":
        objective_argument = "regression"
    elif task_type == "classification":
        objective_argument = "binary"
    else:
        raise NotImplementedError("task_type is not supported in those unit tests (yet)")

    script_args = [
        "train.py",
        "--train", train_dir,
        "--test", test_dir,
        "--export_model", model_dir,
        "--objective", objective_argument,
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

    # test expected outputs
    assert os.path.isfile(
        os.path.join(model_dir, "model.txt")
    ), "Script train.py should generate a model.txt output file but did not"


def lightgbm_python_score(inference_dir,
                          model_dir,
                          predictions_dir,
                          mlflow_start_run_mock,
                          mlflow_end_run_mock,
                          mlflow_log_metric_mock,
                          mlflow_set_tags_mock,
                          mlflow_log_params):
    """Tests src/scripts/lightgbm_python/score.py"""
    # create test arguments for the script
    script_args = [
        "score.py",
        "--data", inference_dir,
        "--model", model_dir,
        "--output", predictions_dir
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        score.main()

    # test expected outputs
    # NOTE: work in progress
    #assert os.path.isfile(
    #    predictions_path
    #), "Script score.py should generate a predictions output file but did not"

@patch('mlflow.start_run')
@patch('mlflow.end_run')
@patch('mlflow.log_metric')
@patch('mlflow.set_tags')
@patch('mlflow.log_params')
@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_end_to_end_lightgbm_python(mlflow_start_run_mock,
                                    mlflow_end_run_mock,
                                    mlflow_log_metric_mock,
                                    mlflow_set_tags_mock,
                                    mlflow_log_params,
                                    temporary_dir,
                                    task_type
                                    ):
    """ Tests each script in a sequence """
    # create a directory for each i/o of the pipeline
    train_dir = os.path.join(temporary_dir, "train")
    test_dir = os.path.join(temporary_dir, "test")
    inference_dir = os.path.join(temporary_dir, "inference")
    model_dir = os.path.join(temporary_dir, "model")
    predictions_dir = os.path.join(temporary_dir, "predictions")

    generate_data(
        train_dir,
        test_dir,
        inference_dir,
        task_type,
        mlflow_start_run_mock,
        mlflow_end_run_mock,
        mlflow_log_metric_mock,
        mlflow_set_tags_mock,
        mlflow_log_params
    )

    lightgbm_python_train(
        train_dir,
        test_dir,
        model_dir, # creates model.txt file inside model_dir
        task_type,
        mlflow_start_run_mock,
        mlflow_end_run_mock,
        mlflow_log_metric_mock,
        mlflow_set_tags_mock,
        mlflow_log_params
    )

    lightgbm_python_score(
        inference_dir,
        model_dir,
        predictions_dir,
        mlflow_start_run_mock,
        mlflow_end_run_mock,
        mlflow_log_metric_mock,
        mlflow_set_tags_mock,
        mlflow_log_params
    )
