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

def verify_partitioned_files(partitioned_data_dir, expected_file_count, expected_file_length):
    """Utility for testing outputs"""
    assert os.path.isdir(partitioned_data_dir)

    for expected_file in ["part_{:06d}".format(i) for i in range(expected_file_count)]:
        assert os.path.isfile(
            os.path.join(partitioned_data_dir, expected_file)
        ), "Script partition.py should generate partitioned data in --output, but no output files were found"

        # open file in read mode
        with open(os.path.join(partitioned_data_dir, expected_file), 'r') as i_file:
            for count, line in enumerate(i_file):
                pass

        assert (count+1) == expected_file_length # expected size of each chunk

def test_partition_data_roundrobin(temporary_dir, regression_train_sample):
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

    verify_partitioned_files(
        partitioned_data_dir,
        expected_file_count=5,
        expected_file_length=20
    )


def test_partition_data_chunk(temporary_dir, regression_train_sample):
    """Tests src/scripts/partition_data/partition.py"""
    partitioned_data_dir = os.path.join(temporary_dir, "partitioned_data")

    # create test arguments for the script

    script_args = [
        "partition.py",
        "--input", regression_train_sample,
        "--output", partitioned_data_dir,
        "--mode", "chunk",

        # regression_train_sample has 100 sample, splitting in 20 x 5
        "--number", "5",
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        partition.main()

    verify_partitioned_files(
        partitioned_data_dir,
        expected_file_count=20,
        expected_file_length=5
    )

