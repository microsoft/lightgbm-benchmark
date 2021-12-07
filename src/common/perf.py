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
                 initial_time_increment=1.0):
        """Constructor"""
        threading.Thread.__init__(self)
        self.killed = False # flag, set to True to kill from the inside

        self.logger = logging.getLogger(__name__)

        # time between perf reports
        self.time_increment = initial_time_increment


    ##################
    ### CALL BACKS ###
    ##################

    def call_on_loop(self, perf_report):
        """You need to implement this"""
        pass

    def call_on_exit(self):
        """You need to implement this"""
        pass

    #####################
    ### RUN FUNCTIONS ###
    #####################

    def run(self):
        """Run function of the thread, while(True)"""
        while not(self.killed):
            if self.time_increment >= 1.0: # cpu_percent.interval already consumes 1sec
                time.sleep(self.time_increment - 1.0) # will double every time report_store_max_length is reached
            self._run_loop()

        self.call_on_exit()

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

        # END OF REPORT
        self.call_on_loop(perf_report)

    def finalize(self):
        """Ask the thread to finalize (clean)"""
        self.killed = True
        self.join()
