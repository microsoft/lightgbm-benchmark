"""
test src/scripts/partition_data/partition.py
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from scripts.data_processing.lightgbm_data2bin import data2bin

# IMPORTANT: see conftest.py for fixtures

def test_lightgbm_data2bin(temporary_dir, regression_train_sample, regression_test_sample):
    """Tests src/scripts/data_processing/lightgbm_data2bin/data2bin.py"""
    binary_train_data_dir = os.path.join(temporary_dir, "binary_train_data")
    binary_test_data_dir = os.path.join(temporary_dir, "binary_test_data")

    # create test arguments for the script

    script_args = [
        "data2bin.py",
        "--train", regression_train_sample,
        "--test", regression_test_sample,
        "--output_train", binary_train_data_dir,
        "--output_test", binary_test_data_dir,
        "--header", "True",
        "--label_column", "name:label",
        "--max_bin", "255",
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        data2bin.main()

    assert os.path.isfile(os.path.join(binary_train_data_dir, "train.bin"))
    assert os.path.isfile(os.path.join(binary_test_data_dir, "test_0.bin"))
