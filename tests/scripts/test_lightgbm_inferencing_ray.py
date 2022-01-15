"""
Test LightGBM Inferencing using Ray
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from scripts.inferencing.lightgbm_ray import score

# IMPORTANT: see conftest.py for fixtures

def test_lightgbm_ray_score(temporary_dir, regression_model_sample, regression_inference_sample):
    """Tests src/scripts/inferencing/lightgbm_ray/score.py"""
    predictions_dir = os.path.join(temporary_dir, "predictions")

    # create test arguments for the script
    script_args = [
        "score.py",
        "--data", regression_inference_sample,
        "--model", regression_model_sample,
        "--output", predictions_dir
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        score.main()

    # test expected outputs
    assert os.path.isfile(os.path.join(predictions_dir, "predictions.txt"))
