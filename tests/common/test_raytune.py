"""Tests src/common/metrics.py"""
import numpy as np
from ray.tune import (
    uniform,
    choice,
)

from common.raytune_param import RayTuneParameterParser

### UNIT TESTS ###


def assert_dist_equal(parsed, expected):
    np.random.seed(0)
    parsed_sample = parsed.sample(100)
    np.random.seed(0)
    expected_sample = expected.sample(100)
    assert parsed_sample == expected_sample


def test_raytune_parameter_parsing():
    lgbm_params = {"train": "regression_train_sample",
        "test": "regression_test_sample",
        "output_path": "./model_dir",
        "objective": "regression",
        "boosting_type": "gbdt",
        "tree_learner": "serial",
        "metric": "rmse",
        "mode": "min",
        "num_samples": "4",
        "num_iterations": "choice([30,40,50,60])",
        "num_leaves": "20",
        "min_data_in_leaf":"31",
        "learning_rate": "uniform(0.0005, 0.005)",
        "max_bin":"255",
        "feature_fraction": "1",
        }

    raytune_param_parser = RayTuneParameterParser(
        tunable_parameters=[
            "num_iterations",
            "num_leaves",
            "min_data_in_leaf",
            "learning_rate",
            "max_bin",
            "feature_fraction"
        ],
    )

    # parser gonna parse
    raytune_param_parser.parse_from_dict(lgbm_params)

    # update get_tunable_params and get_fixed_params
    tunable_params = raytune_param_parser.get_tunable_params()
    fixed_params = raytune_param_parser.get_fixed_params()

    parsed_choice = tunable_params["num_iterations"]
    expected_choice = choice([30, 40, 50, 60])

    parsed_uniform = tunable_params["learning_rate"]
    expected_uniform = uniform(0.0005, 0.005)

    assert_dist_equal(parsed_choice, expected_choice)
    assert_dist_equal(parsed_uniform, expected_uniform)

    assert fixed_params['num_leaves'] == 20
