"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

from pipelines.azureml.lightgbm_inferencing import main

def test_lightgbm_inferencing(aml_config, config_directory):
    # create test arguments for the script
    script_args = [
        "src/pipelines/lightgbm_inferencing.py",
        "--exp-config", os.path.join(config_directory, "experiments", "lightgbm-inferencing.yaml"),
        f"aml.subscription_id={aml_config.subscription_id}",
        f"aml.resource_group={aml_config.resource_group}",
        f"aml.workspace_name={aml_config.workspace_name}",
        f"aml.tenant={aml_config.tenant}",
        f"aml.auth={aml_config.auth}",
        "+run.validate=True",
        "+run.submit=False"
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        main()
