"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from scripts.data_processing.generate_data import generate


@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_generate_data(temporary_dir, task_type):
    """Tests src/scripts/data_processing/generate_data/generate.py"""
    output_train = os.path.join(temporary_dir, "train")
    output_test = os.path.join(temporary_dir, "test")
    output_inference = os.path.join(temporary_dir, "inference")

    # create test arguments for the script
    script_args = [
        "generate.py",
        "--train_samples", "100",
        "--train_partitions", "10",
        "--test_samples", "10",
        "--test_partitions", "2",
        "--inferencing_samples", "100",
        "--inferencing_partitions", "5",
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
        os.path.join(output_train, "train_0.txt")
    ), "Script generate.py should generate train_0.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_test, "test_0.txt")
    ), "Script generate.py should generate test_0.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_inference, "inference_0.txt")
    ), "Script generate.py should generate inference_0.txt under --output dir but did not"
