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

from common.pipelines import aml_connection_config
from common.paths import CONFIG_PATH

@pytest.fixture()
def config_directory():
    """Returns path to configuration files"""
    return CONFIG_PATH

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

# add cli options to connect to AzureML

def pytest_addoption(parser):
    parser.addoption("--aml_subscription_id", action="store")
    parser.addoption("--aml_resource_group", action="store")
    parser.addoption("--aml_workspace_name", action="store")
    parser.addoption("--aml_auth", action="store")
    parser.addoption("--aml_tenant", action="store")

@pytest.fixture(scope='session')
def aml_config(request):
    # creates some aml config for unit tests that require connectivity (tests/pipelines)
    return aml_connection_config(
        request.config.option.aml_subscription_id, # subscription_id: str = MISSING
        request.config.option.aml_resource_group, # resource_group: str = MISSING
        request.config.option.aml_workspace_name, # workspace_name: str = MISSING
        request.config.option.aml_auth, # tenant: Optional[str] = None
        request.config.option.aml_tenant, # auth: str = "interactive"
        False
    )
