"""Tests src/common/math.py"""
import os
import pytest
import numpy as np

from common.math import bootstrap_ci

def test_bootstrap_ci_fixed_seed():
    """Testing the bootstrap_ci method, but we can't have a non-deterministic test here. """
    sample_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 5.0])
    operators={
        'mean':np.mean,
        'p90': (lambda x : np.percentile(x, 90)),
        'p99': (lambda x : np.percentile(x, 99)),
    }

    np.random.seed(404) # fixed const

    # because we're fixing the seed, we can actually go deeper
    expected_values = {
        'mean': (0.30000000000000004, 0.99395, 2.1624999999999996),
        'p90': (0.5, 2.36413, 5.0),
        'p99': (0.593, 3.469213, 5.0),
    }

    returned_values = bootstrap_ci(
        sample_data,
        iterations=1000,
        operators=operators
    )

    assert returned_values == expected_values


def test_bootstrap_ci_no_seed():
    """Testing the bootstrap_ci method, but we can't have a non-deterministic test here. """
    sample_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 5.0])
    operators={
        'mean':np.mean,
        'p90': (lambda x : np.percentile(x, 90)),
        'p99': (lambda x : np.percentile(x, 99)),
    }

    np.random.seed(None) # not const

    returned_values = bootstrap_ci(
        sample_data,
        iterations=1000,
        operators=operators
    )

    for key in operators:
        # check type
        assert key in returned_values
        assert isinstance(returned_values[key], tuple)
        assert len(returned_values[key]) == 3

        # basic interval over
        ci_left, ci_mean, ci_right = returned_values[key]
        assert ci_left <= ci_mean
        assert ci_mean <= ci_right

        # because it's a bootstrap, these are supposed to be true
        assert min(sample_data) <= ci_left
        assert ci_right <= max(sample_data)
