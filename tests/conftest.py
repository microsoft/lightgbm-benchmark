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