# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Helps with reporting performance metrics (cpu/mem utilization).
Needs to be implemented in the rest of the code.
"""
import logging
import threading
import time
import psutil


class PerformanceReportingThread(threading.Thread):
    """Thread to report performance (cpu/mem/net)"""
    def __init__(self,
                 initial_time_increment=1.0,
                 callback_on_loop=None,
                 callback_on_exit=None,
                 callback_tags={}):
        """Constructor
        
        Args:
            initial_time_increment (float): how much time to sleep between perf readings
            callback_on_loop (func): function to call when a perf reading is issued
            callback_on_exit (func): function to call when thread is finalized
            callback_tags (dict): keyword args to add to call to callback_on_loop
        """
        threading.Thread.__init__(self)
        self.killed = False # flag, set to True to kill from the inside

        self.logger = logging.getLogger(__name__)

        # time between perf reports
        self.time_increment = initial_time_increment

        # set callbacks
        self.callback_on_loop = callback_on_loop
        self.callback_on_exit = callback_on_exit
        self.callback_tags = callback_tags


    #####################
    ### RUN FUNCTIONS ###
    #####################

    def run(self):
        """Run function of the thread, while(True)"""
        while not(self.killed):
            if self.time_increment >= 1.0: # cpu_percent.interval already consumes 1sec
                time.sleep(self.time_increment - 1.0) # will double every time report_store_max_length is reached
            self._run_loop()

        self.callback_on_exit(**self.callback_tags)

    def _run_loop(self):
        """What to run within the while(not(killed))"""
        perf_report = {}

        # CPU UTILIZATION
        cpu_utilization = psutil.cpu_percent(interval=1.0, percpu=True) # will take 1 sec to return
        perf_report["cpu_pct_per_cpu_avg"] = sum(cpu_utilization) / len(cpu_utilization)
        perf_report["cpu_pct_per_cpu_min"] = min(cpu_utilization)
        perf_report["cpu_pct_per_cpu_max"] = max(cpu_utilization)

        # MEM UTILIZATION
        perf_report["mem_percent"] = psutil.virtual_memory().percent

        # DISK UTILIZAITON
        perf_report["disk_usage_percent"] = psutil.disk_usage('/').percent
        perf_report["disk_io_read_mb"] = (psutil.disk_io_counters(perdisk=False).read_bytes / (1024 * 1024))
        perf_report["disk_io_write_mb"] = (psutil.disk_io_counters(perdisk=False).write_count / (1024 * 1024))

        # NET I/O SEND/RECV
        net_io_counters = psutil.net_io_counters(pernic=True)
        net_io_lo_identifiers = []
        net_io_ext_identifiers = []

        for key in net_io_counters:
            if 'loopback' in key.lower():
                net_io_lo_identifiers.append(key)
            elif key.lower() == 'lo':
                net_io_lo_identifiers.append(key)
            else:
                net_io_ext_identifiers.append(key)
        
        lo_sent_mb = sum(
            [
                net_io_counters.get(key).bytes_sent
                for key in net_io_lo_identifiers
            ]
        ) / (1024 * 1024)

        ext_sent_mb = sum(
            [
                net_io_counters.get(key).bytes_sent
                for key in net_io_ext_identifiers
            ]
        ) / (1024 * 1024)

        lo_recv_mb = sum(
            [
                net_io_counters.get(key).bytes_recv
                for key in net_io_lo_identifiers
            ]
        ) / (1024 * 1024)

        ext_recv_mb = sum(
            [
                net_io_counters.get(key).bytes_recv
                for key in net_io_ext_identifiers
            ]
        ) / (1024 * 1024)

        perf_report["net_io_lo_sent_mb"] = lo_sent_mb
        perf_report["net_io_ext_sent_mb"] = ext_sent_mb
        perf_report["net_io_lo_recv_mb"] = lo_recv_mb
        perf_report["net_io_ext_recv_mb"] = ext_recv_mb

        # add a timestamp
        perf_report["timestamp"] = time.time()

        # END OF REPORT
        self.callback_on_loop(perf_report, **self.callback_tags)

    def finalize(self):
        """Ask the thread to finalize (clean)"""
        self.killed = True
        self.join()


class PerformanceMetricsCollector():
    """Collects performance metrics from PerformanceReportingThread"""
    def __init__(self, max_length=10000, report_each_step=False, report_at_finalize=True, report_thread=None):
        self.logger = logging.getLogger(__name__)

        self.perf_reports = {}
        self.perf_reports_freqs = {}
        self.perf_reports_counters = {}

        self.max_length = (max_length//2 + max_length%2) * 2 # has to be dividable by 2
        self.report_each_step = report_each_step
        self.report_at_finalize = report_at_finalize

        if report_thread:
            self.report_thread = report_thread
        else:
            self.report_thread = PerformanceReportingThread(
                initial_time_increment=1.0,
                callback_on_loop=self.append_perf_metrics,
                callback_tags={'node':0}
            )
            

    def start(self):
        self.report_thread.start()
    
    def finalize(self):
        self.report_thread.finalize()

    def append_perf_metrics(self, perf_metrics, node=0):
        if node not in self.perf_reports:
            self.perf_reports[node] = []
            self.perf_reports_freqs[node] = 1
            self.perf_reports_counters[node] = 0

        self.perf_reports_counters[node] += 1

        if not (self.perf_reports_counters[node] % self.perf_reports_freqs[node]):
            # if we've decided to skip this one
            return

        self.perf_reports[node].append((time.time(), perf_metrics))

        if len(self.perf_reports[node]) >= self.max_length:
            # trim the report by half
            self.perf_reports[node] = [
                self.perf_reports[node][i]
                for i in range(0, self.max_length, 2)
            ]
            self.perf_reports_freqs[node] *= 2 # we'll start accepting reports only 1 out of 2
            self.logger.warning(f"Perf report store reached max, increasing freq to {self.perf_reports_freqs[node]}")
