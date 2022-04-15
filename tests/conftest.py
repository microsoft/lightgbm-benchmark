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
    """
    Creates some aml config for unit tests that require connectivity (tests/pipelines)

    NOTE: will except with AssertionError and fail test if not provided properly
    """
    subscription_id = request.config.option.aml_subscription_id or os.environ.get('AML_SUBSCRIPTION_ID')
    resource_group = request.config.option.aml_resource_group or os.environ.get('AML_RESOURCE_GROUP')
    workspace_name = request.config.option.aml_workspace_name or os.environ.get('AML_WORKSPACE_NAME')
    auth = request.config.option.aml_auth or os.environ.get('AML_AUTH') or "interactive"
    tenant = request.config.option.aml_tenant or os.environ.get('AML_TENANT')

    test_config = []
    if subscription_id is None:
        test_config.append("To run this unit test, you need to provide a subscription through --aml_subscription_id or env var AML_SUBSCRIPTION_ID")
    if resource_group is None:
        test_config.append("To run this unit test, you need to provide a subscription through --aml_resource_group or env var AML_RESOURCE_GROUP")
    if workspace_name is None:
        test_config.append("To run this unit test, you need to provide a subscription through --aml_workspace_name or env var AML_WORKSPACE_NAME")
    assert (not test_config), "\n".join(test_config)

    return aml_connection_config(
        subscription_id,
        resource_group,
        workspace_name,
        auth,
        tenant,
        False # force auth
    )

@pytest.fixture(scope='session', autouse=True)
def ray_init_fixture(request):
    # using this fixture to avoid re-running ray.init() in each script it is required
    # you'll also need to patch ray.init() and ray.shutdown()
    logging.getLogger().warning("Running ray.init() from ray_init_fixture")
    import ray
    return ray.init()
