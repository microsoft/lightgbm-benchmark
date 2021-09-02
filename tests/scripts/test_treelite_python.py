"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
from unittest.mock import patch

from scripts.treelite_python import compile_treelite
from scripts.treelite_python import score

# IMPORTANT: see conftest.py for fixtures

def test_treelist_inferencing_script(temporary_dir, regression_inference_sample, regression_model_sample):
    # create a directory for each output
    predictions_dir = os.path.join(temporary_dir, "predictions")

    script_args = [
        "compile_treelite.py",
        "--model", regression_model_sample,
        "--model_format", "lightgbm",
        "--toolchain", "gcc",
        "--so_path", os.path.join(temporary_dir, "mymodel.so")
        #"--toolchain", "msvc",
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        compile_treelite.main()
    
    script_args = [
        "score.py",
        "--so_path", os.path.join(temporary_dir, "mymodel.so")
        "--data", regression_inference_sample,
        "--output", predictions_dir,
        "--nthreads", "1",
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        score.main()

    # test expected outputs
    #assert os.path.isfile(os.path.join(predictions_dir, "predictions.txt"))
