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


def test_generate_regression_data(temporary_dir):
    """Tests src/scripts/data_processing/generate_data/generate.py"""
    task_type = "regression"
    output_train = os.path.join(temporary_dir, task_type, "train")
    output_test = os.path.join(temporary_dir, task_type, "test")
    output_inference = os.path.join(temporary_dir, task_type, "inference")
    output_header = os.path.join(temporary_dir, task_type, "header")

    # create test arguments for the script
    script_args = [
        "generate.py",
        "--train_samples", "100",
        "--train_partitions", "4",
        "--test_samples", "10",
        "--test_partitions", "2",
        "--inferencing_samples", "100",
        "--inferencing_partitions", "1",
        "--n_features", "40",
        "--n_informative", "10",
        "--random_state", "5",
        "--output_train", output_train,
        "--output_test", output_test,
        "--output_inference", output_inference,
        "--output_header", output_header,
        "--type", task_type,
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
        os.path.join(output_train, "train_1.txt")
    ), "Script generate.py should generate train_1.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_train, "train_2.txt")
    ), "Script generate.py should generate train_2.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_train, "train_3.txt")
    ), "Script generate.py should generate train_3.txt under --output dir but did not"
    assert not os.path.isfile(
        os.path.join(output_train, "train_4.txt")
    ), "Script generate.py should NOT generate train_4.txt under --output dir but DID"

    assert os.path.isfile(
        os.path.join(output_test, "test_0.txt")
    ), "Script generate.py should generate test_0.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_test, "test_1.txt")
    ), "Script generate.py should generate test_1.txt under --output dir but did not"
    assert not os.path.isfile(
        os.path.join(output_test, "test_2.txt")
    ), "Script generate.py should NOT generate test_2.txt under --output dir but DID"

    assert os.path.isfile(
        os.path.join(output_inference, "inference_0.txt")
    ), "Script generate.py should generate inference_0.txt under --output dir but did not"
    assert not os.path.isfile(
        os.path.join(output_inference, "inference_1.txt")
    ), "Script generate.py should NOT generate inference_1.txt under --output dir but DID"

    assert os.path.isfile(
        os.path.join(output_header, "header.txt")
    ), "Script generate.py should generate inference_0.txt under --output dir but did not"

def test_generate_classification_data(temporary_dir):
    """Tests src/scripts/data_processing/generate_data/generate.py"""
    task_type = "classification"
    output_train = os.path.join(temporary_dir, task_type, "train")
    output_test = os.path.join(temporary_dir, task_type, "test")
    output_inference = os.path.join(temporary_dir, task_type, "inference")
    output_header = os.path.join(temporary_dir, task_type, "header")

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
        "--output_header", output_header,
        "--type", task_type,
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
    assert not os.path.isfile(
        os.path.join(output_train, "train_1.txt")
    ), "Script generate.py should NOT generate train_1.txt under --output dir but DID"

    assert os.path.isfile(
        os.path.join(output_test, "test_0.txt")
    ), "Script generate.py should generate test_0.txt under --output dir but did not"
    assert not os.path.isfile(
        os.path.join(output_test, "test_1.txt")
    ), "Script generate.py should NOT generate test_1.txt under --output dir but DID"

    assert os.path.isfile(
        os.path.join(output_inference, "inference_0.txt")
    ), "Script generate.py should generate inference_0.txt under --output dir but did not"
    assert not os.path.isfile(
        os.path.join(output_inference, "inference_1.txt")
    ), "Script generate.py should NOT generate inference_1.txt under --output dir but DID"

    assert os.path.isfile(
        os.path.join(output_header, "header.txt")
    ), "Script generate.py should generate inference_0.txt under --output dir but did not"

def test_generate_ranking_data(temporary_dir):
    """Tests src/scripts/data_processing/generate_data/generate.py"""
    task_type = "lambdarank"
    output_train = os.path.join(temporary_dir, task_type, "train")
    output_test = os.path.join(temporary_dir, task_type, "test")
    output_inference = os.path.join(temporary_dir, task_type, "inference")
    output_header = os.path.join(temporary_dir, task_type, "header")

    # create test arguments for the script
    script_args = [
        "generate.py",
        "--train_samples", "100",
        "--train_partitions", "4",
        "--test_samples", "10",
        "--test_partitions", "2",
        "--inferencing_samples", "100",
        "--inferencing_partitions", "1",
        "--n_features", "40",
        "--n_informative", "10",
        "--random_state", "5",
        "--output_train", output_train,
        "--output_test", output_test,
        "--output_inference", output_inference,
        "--output_header", output_header,
        "--type", task_type,
    ]
    if task_type == "classification":
        script_args.extend(["--n_redundant", "5"])

    if task_type == "lambdarank":
        script_args.extend(["--docs_per_query", "5"])
        script_args.extend(["--n_label_classes", "3"])

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        generate.main()

    # test expected outputs
    assert os.path.isfile(
        os.path.join(output_train, "train_0.txt")
    ), "Script generate.py should generate train_0.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_train, "train_1.txt")
    ), "Script generate.py should generate train_1.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_train, "train_2.txt")
    ), "Script generate.py should generate train_2.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_train, "train_3.txt")
    ), "Script generate.py should generate train_3.txt under --output dir but did not"
    assert not os.path.isfile(
        os.path.join(output_train, "train_4.txt")
    ), "Script generate.py should NOT generate train_4.txt under --output dir but DID"

    assert os.path.isfile(
        os.path.join(output_test, "test_0.txt")
    ), "Script generate.py should generate test_0.txt under --output dir but did not"
    assert os.path.isfile(
        os.path.join(output_test, "test_1.txt")
    ), "Script generate.py should generate test_1.txt under --output dir but did not"
    assert not os.path.isfile(
        os.path.join(output_test, "test_2.txt")
    ), "Script generate.py should NOT generate test_2.txt under --output dir but DID"

    assert os.path.isfile(
        os.path.join(output_inference, "inference_0.txt")
    ), "Script generate.py should generate inference_0.txt under --output dir but did not"
    assert not os.path.isfile(
        os.path.join(output_inference, "inference_1.txt")
    ), "Script generate.py should NOT generate inference_1.txt under --output dir but DID"

    assert os.path.isfile(
        os.path.join(output_header, "header.txt")
    ), "Script generate.py should generate header.txt under --output dir but did not"