"""Tests src/common/aml.py"""
import os
import pytest
from unittest.mock import call, Mock, patch
import time
import json

from common.aml import apply_sweep_settings
from common.aml import format_run_name
from common.aml import load_dataset_from_data_input_spec
from common.aml import dataset_from_dstore_path


def test_format_run_name():
    """ Tests format_run_name() """
    test_cases = [
        {
            'input':   "run name foo",
            'expected':"run_name_foo"
        },
        {
            'input':   "abcd01234",
            'expected':"abcd01234"
        },
        {
            'input':   "gen 1000samples+3train*foo",
            'expected':"gen_1000samples_3train_foo"
        },
        {
            'input': "a"*1000,
            'expected': "a"*255
        }
    ]

    for test_case in test_cases:
        assert format_run_name(test_case['input']) == test_case['expected']
