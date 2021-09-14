"""
test src/scripts/partition_data/partition.py
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from scripts.partition_data import partition

# IMPORTANT: see conftest.py for fixtures

def test_lightgbm_python_train(temporary_dir, regression_train_sample):
    """Tests src/scripts/partition_data/partition.py"""
    partitioned_data_dir = os.path.join(temporary_dir, "partitioned_data")

    # create test arguments for the script

    script_args = [
        "partition.py",
        "--input", regression_train_sample,
        "--output", partitioned_data_dir,
        "--mode", "roundrobin",

        # regression_train_sample has 100 sample, splitting in 5 x 20
        "--number", "5",
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        partition.main()

    assert os.path.isdir(partitioned_data_dir)

    # test expected outputs
    for expected_file in ["part_{:06d}".format(i) for i in range(5)]:
        assert os.path.isfile(
            os.path.join(partitioned_data_dir, expected_file)
        ), "Script partition.py should generate partitioned data in --output, but no output files were found"
