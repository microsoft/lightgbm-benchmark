"""Tests src/common/data.py"""
import os
import pytest
from unittest.mock import call, Mock, patch

from common.data import RegressionDataGenerator

def test_regression_data_generator():
    """Tests format of outputs of RegressionDataGenerator"""
    generator = RegressionDataGenerator(
        batch_size=64,
        n_features=100,
        n_informative=50,
        bias=1.0,
        noise=1.0,
        seed=4
    )

    for i in range(10):
        batch = generator.generate()

        assert batch is not None
        assert isinstance(batch, tuple)
        assert len(batch) == 2

        X, y = batch
        assert X is not None
        assert y is not None

        assert X.shape == (64, 100)
        assert y.shape == (64,)

def test_regression_data_generator_reproducibility():
    """Tests initializing generator with seeds"""
    generator1 = RegressionDataGenerator(
        batch_size=64,
        n_features=100,
        n_informative=50,
        bias=1.0,
        noise=1.0,
        seed=4
    )
    X1,y1 = generator1.generate()

    generator2 = RegressionDataGenerator(
        batch_size=64,
        n_features=100,
        n_informative=50,
        bias=1.0,
        noise=1.0,
        seed=5
    )
    X2,y2 = generator2.generate()

    generator3 = RegressionDataGenerator(
        batch_size=64,
        n_features=100,
        n_informative=50,
        bias=1.0,
        noise=1.0,
        seed=4 # <<< Equal to generator 1
    )
    X3,y3 = generator3.generate()

    # if using same seed twice, should be equal strictly
    assert (X1 == X3).all()
    assert (y1 == y3).all()

    # if using different seeds, likely to be different
    assert (X1 != X2).all()
    assert (y1 != y2).all()
