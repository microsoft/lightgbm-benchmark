"""Tests src/common/io.py"""
import os
import pytest
from unittest.mock import call, Mock, patch

from common.lightgbm_utils import LightGBMCallbackHandler
from lightgbm.callback import CallbackEnv

def test_lightgbm_callback_handler():
    metrics_logger = Mock()

    callback_handler = LightGBMCallbackHandler(
        metrics_logger, metrics_prefix=None, metrics_suffix=None
    )

    # namedtuple
    # see https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/callback.html
    callback_env = CallbackEnv(
        None, # model
        {"foo_param": 0.32}, # params
        3, # iteration
        0, # begin_iteration
        5, # end_iteration
        [
            # list of tuples
            (
                "valid_0", # dataset name
                "rmse", # evaluation name
                12345.0, # result
                None, # _
            ),
            (
                "valid_0", # dataset name
                "l2", # evaluation name
                3456.0, # result
                None, # _
            )
        ]
    )
    callback_handler.callback(callback_env)

    metrics_logger.log_metric.assert_has_calls(
        [
            call(key="valid_0.rmse", value=12345.0, step=3),
            call(key="valid_0.l2", value=3456.0, step=3)
        ],
        any_order=True
    )
