"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
from unittest.mock import patch

from scripts.model_transformation.treelite_compile import compile_treelite
from scripts.inferencing.treelite_python import score

# IMPORTANT: see conftest.py for fixtures

def test_treelist_inferencing_script(temporary_dir, regression_inference_sample, regression_model_sample):
    # create a directory for each output
    predictions_dir = os.path.join(temporary_dir, "predictions")

    if sys.platform == "linux" or sys.platform == "linux2":
        toolchain = "gcc"
    elif sys.platform == "win32":
        toolchain = "msvc"
    else:
        raise NotImplementedError(f"unit test doesn't know how to handle toolchain for platform {sys.platform}")

    script_args = [
        "compile_treelite.py",
        "--model", regression_model_sample,
        "--model_format", "lightgbm",
        "--toolchain", toolchain,
        "--so_path", os.path.join(temporary_dir, "mymodel.so")
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        compile_treelite.main()
    
    script_args = [
        "score.py",
        "--so_path", os.path.join(temporary_dir, "mymodel.so"),
        "--data", regression_inference_sample,
        "--output", predictions_dir,
        "--nthreads", "1",
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        score.main()

    # test expected outputs
    #assert os.path.isfile(os.path.join(predictions_dir, "predictions.txt"))
