"""
Test LightGBM Inferencing using Ray
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from scripts.inferencing.lightgbm_ray import score
from common.distributed import multinode_config_class
# IMPORTANT: see conftest.py for fixtures

@patch('ray.shutdown')
@patch('ray.init')
@patch('common.distributed.MultiNodeMPIDriver')
def test_lightgbm_ray_score(mpi_driver_mock, ray_init_mock, ray_shutdown_mock, ray_init_fixture, temporary_dir, regression_model_sample, regression_inference_sample):
    """Tests src/scripts/inferencing/lightgbm_ray/score.py"""
    # fake mpi initialization + config
    mpi_driver_mock().get_multinode_config.return_value = multinode_config_class(
        1, # world_size
        0, # world_rank
        False, # mpi_available
        True, # main_node
    )

    predictions_dir = os.path.join(temporary_dir, "predictions")

    # create test arguments for the script
    script_args = [
        "score.py",
        "--data", regression_inference_sample,
        "--model", regression_model_sample,
        "--output", predictions_dir, 
        "--cluster_auto_setup", "False"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        score.main()

    # test expected outputs
    assert os.path.isfile(os.path.join(predictions_dir, "predictions.txt"))
