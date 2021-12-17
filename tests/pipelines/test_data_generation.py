"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from pipelines.azureml.data_generation import main

def test_data_generation_main(aml_config, config_directory):
    # create test arguments for the script
    script_args = [
        "src/pipelines/data_generation.py",
        "--exp-config", os.path.join(config_directory, "experiments", "data-generation.yaml"),
        f"aml.subscription_id={aml_config.subscription_id}",
        f"aml.resource_group={aml_config.resource_group}",
        f"aml.workspace_name={aml_config.workspace_name}",
        f"aml.tenant={aml_config.tenant}",
        f"aml.auth={aml_config.auth}",
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        main()
