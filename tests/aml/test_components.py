# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
PyTest suite for testing if each run.py is aligned with module specification:

> Status: this code relates to the _recipe_ and is a _proposition_
"""
import pytest
import os

from shrike.pipeline.testing.components import (
    component_spec_yaml_exists_and_is_parsable,
)
from shrike.pipeline.testing.components import component_uses_private_acr
from shrike.pipeline.testing.components import component_uses_private_python_feed
from shrike.pipeline.testing.components import component_run_py_import
from shrike.pipeline.testing.components import component_run_get_arg_parser
from shrike.pipeline.testing.components import (
    if_arguments_from_component_spec_match_script_argparse,
)

COMPONENT_ROOT_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "scripts")
)

# modules that should ALSO pass advanced tests (design pattern)
COMPONENT_SPEC_FILES = [
    "sample/spec.yaml",

    "data_processing/generate_data/spec.yaml",
    "data_processing/lightgbm_data2bin/spec.yaml",
    "data_processing/partition_data/spec.yaml",

    "training/lightgbm_python/spec.yaml",
    "training/lightgbm_ray/spec.yaml",
    "training/ray_tune/spec.yaml",

    "model_transformation/treelite_compile/spec.yaml",

    "inferencing/lightgbm_python/spec.yaml",
    "inferencing/lightgbm_c_api/spec.yaml",
    "inferencing/lightgbm_ray/spec.yaml",
    "inferencing/custom_win_cli/spec.yaml",
    "inferencing/treelite_python/spec.yaml",
]


### BASIC TESTS ###
# for basic module designs (minimal wrappers)

@pytest.mark.parametrize("component_spec_path", COMPONENT_SPEC_FILES)
def test_component_run_py_import(component_spec_path):
    """Try importing run.py, just to check if basic script passes syntax/imports checks"""
    component_run_py_import(
        os.path.join(COMPONENT_ROOT_FOLDER, component_spec_path)
    )


@pytest.mark.parametrize("component_spec_path", COMPONENT_SPEC_FILES)
def test_component_spec_yaml_exists_and_is_parsable(component_spec_path):
    """Try loading and parsing the component spec yaml file"""
    component_spec_yaml_exists_and_is_parsable(
        os.path.join(COMPONENT_ROOT_FOLDER, component_spec_path)
    )


### ADVANCED TESTS ###
# for module implementing full design pattern (get_arg_parser())

@pytest.mark.parametrize("component_spec_path", COMPONENT_SPEC_FILES)
def test_component_run_get_arg_parser(component_spec_path):
    """Tests if component run.py has function get_arg_parser(parser)"""
    component_run_get_arg_parser(
        os.path.join(COMPONENT_ROOT_FOLDER, component_spec_path)
    )


@pytest.mark.parametrize("component_spec_path", COMPONENT_SPEC_FILES)
def test_if_arguments_from_component_spec_match_script_argparse(component_spec_path):
    """Tests alignment between module_spec arguments and script parser arguments"""
    if_arguments_from_component_spec_match_script_argparse(
        os.path.join(COMPONENT_ROOT_FOLDER, component_spec_path)
    )


# NOTE: this test has been disabled because it requires exception re-throw in compliant_handle()
# @pytest.mark.parametrize("module", MODULE_MANIFEST_ADVANCED)
# def test_script_main_with_synthetic_arguments(mocker, module):
#    """Tests alignment between module_spec arguments and script parser arguments"""
#    script_main_with_synthetic_arguments(module, mocker)
