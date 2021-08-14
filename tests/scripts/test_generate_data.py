"""Tests src/scripts/generate_data/generate.py"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch
from scripts.generate_data import generate


@pytest.fixture()
def temporary_dir():
    """Creates a temporary directory for the tests below"""
    temp_directory = tempfile.TemporaryDirectory()
    yield temp_directory.name
    temp_directory.cleanup()


@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_generate_data(temporary_dir, task_type):
    """Tests src/scripts/generate_data/generate.py"""
    output_dir = os.path.join(temporary_dir, "output")

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
