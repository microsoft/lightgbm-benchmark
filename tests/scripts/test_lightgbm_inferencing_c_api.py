"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch, Mock, call

from scripts.inferencing.lightgbm_c_api import score

# IMPORTANT: see conftest.py for fixtures

@patch('mlflow.log_metric')
@patch('mlflow.set_tags')
@patch('scripts.inferencing.lightgbm_c_api.score.subprocess_run')
@patch('scripts.inferencing.lightgbm_c_api.score.locate_lightgbm_benchmark_binaries')
def test_lightgbm_c_api_score(locate_binaries_mock, subprocess_run_mock, mlflow_set_tags_mock, mlflow_log_metric_mock, temporary_dir, regression_model_sample, regression_inference_sample):
    """Tests src/scripts/inferencing/lightgbm_c_api/score.py"""
    predictions_dir = os.path.join(temporary_dir, "predictions")
    locate_binaries_mock.return_value = "fake_cli.exe"

    # create a first mock for the return of subprocess
    subprocess_call_handle_mock = Mock()
    subprocess_call_handle_mock.returncode = 0
    subprocess_call_handle_mock.stderr = "# empty logs"
    subprocess_call_handle_mock.stdout = """
# fake logs for parsing metrics from C API binaries (cli)
ROW line=0 label=0.42 null_elem=3 prediction=0.45 time_usecs=45.2
ROW line=1 label=0.42 null_elem=3 prediction=0.45 time_usecs=45.3
ROW line=3 label=0.42 null_elem=3 prediction=0.45 time_usecs=45.4
METRIC foo=342.0
PROPRETY foo2=bar2
"""
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

    # test arguments
    assert isinstance(subprocess_run_mock.call_args.args[0], list), "first argument of subprocess.run() should be a list"
    assert "fake_cli.exe" in subprocess_run_mock.call_args.args[0][0], "first element in subprocess.run() command should contain return value of locate_lightgbm_benchmark_binaries()"

    # test expected outputs
    assert os.path.isfile(os.path.join(predictions_dir, "predictions.txt"))

    metric_calls = mlflow_log_metric_mock.call_args_list
    # 32 = 18 perf metrics + 13 inference metrics + 1 custom (foo)
    assert mlflow_log_metric_mock.call_count == 32
    assert len(metric_calls) == 32
