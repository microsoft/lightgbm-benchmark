"""Tests src/common/metrics.py"""
import os
from azureml.train.hyperdrive import (
    choice,
    quniform,
    qloguniform,
    qnormal,
    qlognormal,
    uniform,
    loguniform,
    normal,
    lognormal
)

from common.sweep import SweepParameterParser

### UNIT TESTS ###

def test_parse_choice():
    parsed_param = SweepParameterParser._parse_choice("choice(0.1, 0.2, 0.3)")
    expected_param = choice(0.1, 0.2, 0.3)
    assert parsed_param == expected_param

def test_parse_uniform():
    parsed_param = SweepParameterParser._parse_uniform("uniform(0, 100)")
    expected_param = uniform(0, 100)
    assert parsed_param == expected_param

def test_parse_loguniform():
    parsed_param = SweepParameterParser._parse_loguniform("loguniform(0.1, 0.4)")
    expected_param = loguniform(0.1,0.4)
    assert parsed_param == expected_param

def test_parse_normal():
    parsed_param = SweepParameterParser._parse_normal("normal(0.1, 0.4)")
    expected_param = normal(0.1,0.4)
    assert parsed_param == expected_param

def test_parse_lognormal():
    parsed_param = SweepParameterParser._parse_lognormal("lognormal(0.1, 0.4)")
    expected_param = lognormal(0.1,0.4)
    assert parsed_param == expected_param


def test_parse_quniform():
    parsed_param = SweepParameterParser._parse_quniform("quniform(0, 100, 5)")
    expected_param = quniform(0, 100, 5)
    assert parsed_param == expected_param

def test_parse_qloguniform():
    parsed_param = SweepParameterParser._parse_qloguniform("qloguniform(0, 100, 5)")
    expected_param = qloguniform(0, 100, 5)
    assert parsed_param == expected_param

def test_parse_qnormal():
    parsed_param = SweepParameterParser._parse_qnormal("qnormal(0, 100, 5)")
    expected_param = qnormal(0, 100, 5)
    assert parsed_param == expected_param

def test_parse_qlognormal():
    parsed_param = SweepParameterParser._parse_qlognormal("qlognormal(0, 100, 5)")
    expected_param = qlognormal(0, 100, 5)
    assert parsed_param == expected_param


def test_sweep_parameter_parsing():
    sweep_parser = SweepParameterParser(
        tunable_parameters=[
            # those are keys and their default values
            "num_trees",
            "num_leaves",
            "min_data_in_leaf",
            "learning_rate",
            "max_bin",
            "feature_fraction"
        ],
        cli_prefix="--",
        parameter_sampling="random"
    )

    cli_parser = sweep_parser.get_arg_parser()

    args, unknown_args = cli_parser.parse_known_args([
        "--num_trees", "uniform(0.0005, 0.005)",
        "--num_leaves", "choice(0, 100, 120, 140, 180)",
        "--learning_rate", "0.32"
    ])

    # returns dictionaries
    sweep_parser.parse_from_argparse(args)
    tunable_params = sweep_parser.get_tunable_params()
    fixed_params = sweep_parser.get_fixed_params()

    assert tunable_params == {
        "num_trees" : uniform(0.0005, 0.005),
        "num_leaves" : choice(0, 100, 120, 140, 180)
    }
    assert fixed_params == {
        "learning_rate" : 0.32
    }

def test_sweep_parameter_from_argparse():
    sweep_parser = SweepParameterParser(
        tunable_parameters=[
            # those are keys and their default values
            "num_trees",
            "num_leaves",
            "min_data_in_leaf",
            "learning_rate",
            "max_bin",
            "feature_fraction"
        ],
        cli_prefix="--",
        parameter_sampling="random"
    )

    cli_parser = sweep_parser.get_arg_parser()

    args, unknown_args = cli_parser.parse_known_args([
        "--num_trees", "uniform(0.0005, 0.005)",
        "--num_leaves", "choice(0, 100, 120, 140, 180)",
        "--learning_rate", "0.32"
    ])

    # returns dictionaries
    sweep_parser.parse_from_argparse(args)
    tunable_params = sweep_parser.get_tunable_params()
    fixed_params = sweep_parser.get_fixed_params()

    assert tunable_params == {
        "num_trees" : uniform(0.0005, 0.005),
        "num_leaves" : choice(0, 100, 120, 140, 180)
    }
    assert fixed_params == {
        "learning_rate" : 0.32
    }

def test_sweep_parameter_from_dict():
    sweep_parser = SweepParameterParser(
        tunable_parameters=[
            # those are keys and their default values
            "num_trees",
            "num_leaves",
            "min_data_in_leaf",
            "learning_rate",
            "max_bin",
            "feature_fraction"
        ],
        cli_prefix=None,
        parameter_sampling="random"
    )

    parameter_dict = {
        "num_trees" : "uniform(0.0005, 0.005)",
        "num_leaves" : "choice(0, 100, 120, 140, 180)",
        "learning_rate" : "0.32"
    }

    # returns dictionaries
    sweep_parser.parse_from_dict(parameter_dict)
    tunable_params = sweep_parser.get_tunable_params()
    fixed_params = sweep_parser.get_fixed_params()

    assert tunable_params == {
        "num_trees" : uniform(0.0005, 0.005),
        "num_leaves" : choice(0, 100, 120, 140, 180)
    }
    assert fixed_params == {
        "learning_rate" : 0.32
    }