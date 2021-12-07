"""Tests src/common/metrics.py"""
import os
import pytest
from unittest.mock import call, Mock, patch
import time

from common.perf import PerformanceReportingThread

def verify_all_perf_report_keys(perf_report):
    """Helper test function, tests all keys in perf report"""
    assert isinstance(perf_report, dict)

    required_keys = [
        "cpu_pct_per_cpu_avg",
        "cpu_pct_per_cpu_min",
        "cpu_pct_per_cpu_max",
        "mem_percent",
        "disk_usage_percent",
        "disk_io_read_mb",
        "disk_io_write_mb",
        "net_io_lo_sent_mb",
        "net_io_ext_sent_mb",
        "net_io_lo_recv_mb",
        "net_io_ext_recv_mb"
    ]

    for key in required_keys:
        assert key in perf_report, f"key {key} should be in the perf report, but instead we find: {list(perf_report.keys())}"
        assert isinstance(perf_report[key], float) # all metrics are float so far\
    
    assert "not_in_perf_report" not in perf_report


def test_perf_report_run_as_thread():
    """ Tests PerformanceReportingThread() as a thread """
    # creating a mock to provide as callback
    call_on_loop_method = Mock()
    call_on_exit_method = Mock()

    perf_report_thread = PerformanceReportingThread(initial_time_increment=2.0)
    perf_report_thread.call_on_loop = call_on_loop_method
    perf_report_thread.call_on_exit = call_on_exit_method

    perf_report_thread.start() # will engage in first loop and sleep 2.0
    time.sleep(0.5) # will wait to be in the middle of that loop
    perf_report_thread.finalize()

    # on exit not called in this one
    call_on_exit_method.assert_called_once()

    # get all mock calls
    callback_call_args = call_on_loop_method.call_args_list

    assert len(callback_call_args) == 1 # just called once
    assert len(callback_call_args[0].args) == 1 # only 1 argument
    
    perf_report = callback_call_args[0].args[0]
    verify_all_perf_report_keys(perf_report)
