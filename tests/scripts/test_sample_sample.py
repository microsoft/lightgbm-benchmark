"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from scripts.sample import sample

# IMPORTANT: see conftest.py for fixtures

def test_sample_inferencing_script(temporary_dir, regression_inference_sample, regression_model_sample):
    # create a directory for each output
    predictions_dir = os.path.join(temporary_dir, "predictions")

    script_args = [
        "score.py",
        "--data", regression_inference_sample,
        "--model", regression_model_sample,
        "--output", predictions_dir
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        sample.main()

    # test expected outputs
    assert os.path.isfile(os.path.join(predictions_dir, "predictions.txt"))