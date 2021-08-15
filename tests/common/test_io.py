"""Tests src/common/io.py"""
import os
import pytest

from common.io import input_file_path


def test_input_file_path_success(temporary_dir):
    """Tests input_file_path function with a unique file"""
    test1_dir = os.path.join(temporary_dir, "test1")
    os.makedirs(test1_dir, exist_ok=False)
    with open(os.path.join(test1_dir, "random.out"), "w") as out_file:
        out_file.write("something")
    
    full_file_path = os.path.join(test1_dir, "random.out")
    assert input_file_path(test1_dir) == full_file_path
    assert input_file_path(full_file_path) == full_file_path

def test_input_file_path_failure(temporary_dir):
    """Tests input_file_path function when 2 files are provided (should except)"""
    test2_dir = os.path.join(temporary_dir, "test2")
    os.makedirs(test2_dir, exist_ok=False)

    with open(os.path.join(test2_dir, "random1.out"), "w") as out_file:
        out_file.write("something")
    with open(os.path.join(test2_dir, "random2.out"), "w") as out_file:
        out_file.write("something else")
    
    with pytest.raises(Exception):
        assert input_file_path(test2_dir)
