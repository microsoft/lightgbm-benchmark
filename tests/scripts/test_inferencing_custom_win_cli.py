"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch, Mock, call

from scripts.inferencing.custom_win_cli import score

# IMPORTANT: see conftest.py for fixtures

@patch('mlflow.log_metric')
@patch('scripts.inferencing.custom_win_cli.score.subprocess_run')
def test_lightgbm_c_api_score(subprocess_run_mock, mlflow_log_metric_mock, temporary_dir, regression_model_sample, regression_inference_sample):
    """Tests src/scripts/inferencing/lightgbm_c_api/score.py"""
    predictions_dir = os.path.join(temporary_dir, "predictions")

    # create a first mock for the return of subprocess
    subprocess_call_handle_mock = Mock()
    subprocess_call_handle_mock.returncode = 0
    subprocess_call_handle_mock.stderr = "# empty logs"
    subprocess_call_handle_mock.stdout = "# empty logs"

    # feed that mock into a subprocess.run() mock
    subprocess_run_mock.return_value = subprocess_call_handle_mock

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

    # no outputs (yet?)
    # assert os.path.isfile(os.path.join(predictions_dir, "predictions.txt"))

    metric_calls = mlflow_log_metric_mock.call_args_list
    # 12 = 11 perf metrics + 1 time_inferencing
    assert mlflow_log_metric_mock.call_count == 12
    assert len(metric_calls) == 12
