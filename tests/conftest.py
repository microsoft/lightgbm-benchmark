""" Add src/ to path """
import os
import sys
import logging
import pytest
import tempfile

LIGHTGBM_BENCHMARK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)

if LIGHTGBM_BENCHMARK_ROOT not in sys.path:
    logging.info(f"Adding {LIGHTGBM_BENCHMARK_ROOT} to path")
    sys.path.append(str(LIGHTGBM_BENCHMARK_ROOT))

@pytest.fixture()
def temporary_dir():
    """Creates a temporary directory for the tests below"""
    temp_directory = tempfile.TemporaryDirectory()
    yield temp_directory.name
    temp_directory.cleanup()


TEST_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

@pytest.fixture()
def regression_train_sample():
    return os.path.join(TEST_DATA_ROOT, "regression", "train")

@pytest.fixture()
def regression_test_sample():
    return os.path.join(TEST_DATA_ROOT, "regression", "test")

@pytest.fixture()
def regression_inference_sample():
    return os.path.join(TEST_DATA_ROOT, "regression", "inference")

@pytest.fixture()
def regression_model_sample():
    return os.path.join(TEST_DATA_ROOT, "regression", "model")
