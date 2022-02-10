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
import numpy as np


class PerformanceReportingThread(threading.Thread):
    """Thread to report performance (cpu/mem/net)"""
    def __init__(self,
                 initial_time_increment=1.0,
                 cpu_interval=1.0,
                 callback_on_loop=None,
                 callback_on_exit=None):
        """Constructor
        
        Args:
            initial_time_increment (float): how much time to sleep between perf readings
            cpu_interval (float): interval to capture cpu utilization
            callback_on_loop (func): function to call when a perf reading is issued
            callback_on_exit (func): function to call when thread is finalized
        """
        threading.Thread.__init__(self)
        self.killed = False # flag, set to True to kill from the inside

        self.logger = logging.getLogger(__name__)

        # time between perf reports
        self.time_increment = initial_time_increment
        self.cpu_interval = cpu_interval

        # set callbacks
        self.callback_on_loop = callback_on_loop
        self.callback_on_exit = callback_on_exit


    #####################
    ### RUN FUNCTIONS ###
    #####################

    def run(self):
        """Run function of the thread, while(True)"""
        while not(self.killed):
            if self.time_increment >= self.cpu_interval: # cpu_percent.interval already consumes 1sec
                time.sleep(self.time_increment - self.cpu_interval) # will double every time report_store_max_length is reached
            self._run_loop()

        if self.callback_on_exit:
            self.callback_on_exit()

    def _run_loop(self):
        """What to run within the while(not(killed))"""
        perf_report = {}

        # CPU UTILIZATION
        cpu_utilization = psutil.cpu_percent(interval=self.cpu_interval, percpu=True) # will take 1 sec to return
        perf_report["cpu_pct_per_cpu_avg"] = sum(cpu_utilization) / len(cpu_utilization)
        perf_report["cpu_pct_per_cpu_min"] = min(cpu_utilization)
        perf_report["cpu_pct_per_cpu_max"] = max(cpu_utilization)

        # MEM UTILIZATION
        perf_report["mem_percent"] = psutil.virtual_memory().percent

        # DISK UTILIZAITON
        perf_report["disk_usage_percent"] = psutil.disk_usage('/').percent
        perf_report["disk_io_read_mb"] = (psutil.disk_io_counters(perdisk=False).read_bytes / (1024 * 1024))
        perf_report["disk_io_write_mb"] = (psutil.disk_io_counters(perdisk=False).write_bytes / (1024 * 1024))

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
        if self.callback_on_loop:
            self.callback_on_loop(perf_report)

    def finalize(self):
        """Ask the thread to finalize (clean)"""
        self.killed = True
        self.join()


class PerformanceMetricsCollector():
    """Collects performance metrics from PerformanceReportingThread
    Limits all values to a maximum length"""
    def __init__(self, max_length=1000):
        """Constructor
        
        Args:
            max_length (int): maximum number of perf reports to keep
        """
        self.logger = logging.getLogger(__name__)

        # create a thread to generate reports regularly
        self.report_thread = PerformanceReportingThread(
            initial_time_increment=1.0,
            cpu_interval=1.0,
            callback_on_loop=self.append_perf_metrics
        )

        self.perf_reports = [] # internal storage
        self.perf_reports_freqs = 1 # frequency to skip reports from thread
        self.perf_reports_counter = 0 # how many reports we had so far

        self.max_length = (max_length//2 + max_length%2) * 2 # has to be dividable by 2


    def start(self):
        """Start collector perf metrics (start internal thread)"""
        self.logger.info(f"Starting perf metric collector (max_length={self.max_length})")
        self.report_thread.start()

    def finalize(self):
        """Stop collector perf metrics (stop internal thread)"""
        self.logger.info(f"Finalizing perf metric collector (length={len(self.perf_reports)})")
        self.report_thread.finalize()

    def append_perf_metrics(self, perf_metrics):
        """Add a perf metric report to the internal storage"""
        self.perf_reports_counter += 1

        if (self.perf_reports_counter % self.perf_reports_freqs):
            # if we've decided to skip this one
            return

        self.perf_reports.append(perf_metrics)

        if len(self.perf_reports) > self.max_length:
            # trim the report by half
            self.perf_reports = [
                self.perf_reports[i]
                for i in range(0, self.max_length, 2)
            ]
            self.perf_reports_freqs *= 2 # we'll start accepting reports only 1 out of 2
            self.logger.warning(f"Perf report store reached max, increasing freq to {self.perf_reports_freqs}")


class PerfReportPlotter():
    PERF_DISK_NET_PLOT_BINS = 50
    PERF_DISK_NET_PLOT_FIGSIZE = (16,8)

    """Once collected all perf reports from all nodes"""
    def __init__(self, metrics_logger):
        self.all_reports = {}
        self.metrics_logger = metrics_logger

    def add_perf_reports(self, perf_reports, node):
        """Add a set of reports from a given node"""
        self.all_reports[node] = perf_reports

    def report_nodes_perf(self):
        """Report performance via metrics/plots for all nodes (loop)"""
        # Currently reporting one metric per node
        for node in self.all_reports:
            self.report_node_perf(node)

    def diff_at_partitions(self, a, p, t):
        """Compute the difference end-begin for all partitions in an array.

        Args:
            a (np.array): a np array of increasing values
            p (int): a given number of partitions
            t (np.array): np array of timestamps for a
        
        Returns:
            diff_part (np.array): dimension p
        """
        if len(a) > (2*p):
            partitioned_diff = []
            partitioned_time = []
            index_partitions = np.array_split(np.arange(len(a)), p)
            print(f"i={index_partitions}")
            for part in index_partitions:
                partitioned_diff.append(a[part[-1]] - a[part[0]]) # diff between last and first value of partition
                partitioned_time.append(t[part[0]])
            return np.array(partitioned_diff), np.array(partitioned_time)
        else:
            return np.ediff1d(a), t[1::]

    def report_node_perf(self, node):
        """Report performance via metrics/plots for one given node"""
        # CPU UTILIZATION
        cpu_avg_utilization = [ report["cpu_pct_per_cpu_avg"] for report in self.all_reports[node] ]
        self.metrics_logger.log_metric(
            "max_t_(cpu_pct_per_cpu_avg)",
            max(cpu_avg_utilization),
            step=node
        )
        self.metrics_logger.log_metric(
            "cpu_avg_utilization_pct",
            sum(cpu_avg_utilization)/len(cpu_avg_utilization),
            step=node
        )
        self.metrics_logger.log_metric(
            "cpu_avg_utilization_at100_pct",
            sum( [ utilization >= 100.0 for utilization in cpu_avg_utilization])/len(cpu_avg_utilization)*100.0,
            step=node
        )
        self.metrics_logger.log_metric(
            "cpu_avg_utilization_over80_pct",
            sum( [ utilization >= 80.0 for utilization in cpu_avg_utilization])/len(cpu_avg_utilization)*100.0,
            step=node
        )
        self.metrics_logger.log_metric(
            "cpu_avg_utilization_over40_pct",
            sum( [ utilization >= 40.0 for utilization in cpu_avg_utilization])/len(cpu_avg_utilization)*100.0,
            step=node
        )
        self.metrics_logger.log_metric(
            "cpu_avg_utilization_over20_pct",
            sum( [ utilization >= 20.0 for utilization in cpu_avg_utilization])/len(cpu_avg_utilization)*100.0,
            step=node
        )
        self.metrics_logger.log_metric(
            "max_t_(cpu_pct_per_cpu_min)",
            max([ report["cpu_pct_per_cpu_min"] for report in self.all_reports[node] ]),
            step=node
        )
        self.metrics_logger.log_metric(
            "max_t_(cpu_pct_per_cpu_max)",
            max([ report["cpu_pct_per_cpu_max"] for report in self.all_reports[node] ]),
            step=node
        )

        # "CPU HOURS"
        job_internal_cpu_hours = (time.time() - self.all_reports[node][0]["timestamp"]) * psutil.cpu_count() / 60 / 60
        self.metrics_logger.log_metric(
            "node_cpu_hours",
            job_internal_cpu_hours,
            step=node
        )
        self.metrics_logger.log_metric(
            "node_unused_cpu_hours",
            job_internal_cpu_hours * (100.0 - sum(cpu_avg_utilization)/len(cpu_avg_utilization)) / 100.0,
            step=node
        )

        # MEM
        mem_utilization = [ report["mem_percent"] for report in self.all_reports[node] ]
        self.metrics_logger.log_metric(
            "max_t_(mem_percent)",
            max([ report["mem_percent"] for report in self.all_reports[node] ]),
            step=node
        )

        # DISK
        self.metrics_logger.log_metric(
            "max_t_disk_usage_percent",
            max([ report["disk_usage_percent"] for report in self.all_reports[node] ]),
            step=node
        )

        disk_io_read = np.array([ report["disk_io_read_mb"] for report in self.all_reports[node] ])
        self.metrics_logger.log_metric(
            "total_disk_io_read_mb",
            np.max(disk_io_read),
            step=node
        )
        disk_io_write = np.array([ report["disk_io_write_mb"] for report in self.all_reports[node] ])
        self.metrics_logger.log_metric(
            "total_disk_io_write_mb",
            np.max(disk_io_write),
            step=node
        )

        # NET I/O
        self.metrics_logger.log_metric(
            "total_net_io_lo_sent_mb",
            max([ report["net_io_lo_sent_mb"] for report in self.all_reports[node] ]),
            step=node
        )
        net_io_ext_sent = np.array([ report["net_io_ext_sent_mb"] for report in self.all_reports[node] ])
        self.metrics_logger.log_metric(
            "total_net_io_ext_sent_mb",
            np.max(net_io_ext_sent),
            step=node
        )
        self.metrics_logger.log_metric(
            "total_net_io_lo_recv_mb",
            max([ report["net_io_lo_recv_mb"] for report in self.all_reports[node] ]),
            step=node
        )
        net_io_ext_recv = np.array([ report["net_io_ext_recv_mb"] for report in self.all_reports[node] ])
        self.metrics_logger.log_metric(
            "total_net_io_ext_recv_mb",
            np.max(net_io_ext_recv),
            step=node
        )

        # then plot everything
        timestamps = np.array([ report["timestamp"] for report in self.all_reports[node] ])
        self.plot_all_perf(node, timestamps, disk_io_read, disk_io_write, net_io_ext_sent, net_io_ext_recv, cpu_avg_utilization, mem_utilization)

    def plot_all_perf(self, node, timestamps, disk_io_read, disk_io_write, net_io_ext_sent, net_io_ext_recv, cpu_utilization, mem_utilization):
        """Plots mem/cpy/disk/net using matplotlib.
        
        Args:
            node (int): node id
            timestamps (np.array): timestamps for all following arrays
            disk_io_read (np.array): disk read cumulative sum (increasing)
            disk_io_write (np.array): disk write cumulative sum (increasing)
            net_io_ext_sent (np.array): net sent cumulative sum (increasing)
            net_io_ext_recv (np.array): net recv cumulative sum (increasing)
            cpu_utilization (np.array): cpu utilization
            mem_utilization (np.array): mem utilization
        
        Returns:
            None
        
        NOTE: we're expecting all arrays to have same length.
        """
        timestamps = timestamps - timestamps[0]
        disk_io_read = disk_io_read - disk_io_read[0] # cumsum starting at 0
        disk_io_write = disk_io_write - disk_io_write[0] # cumsum starting at 0
        net_io_ext_sent = net_io_ext_sent - net_io_ext_sent[0] # cumsum starting at 0
        net_io_ext_recv = net_io_ext_recv - net_io_ext_recv[0] # cumsum starting at 0

        # import matploblib just-in-time
        import matplotlib.pyplot as plt
        #plt.switch_backend('agg')

        # show the distribution prediction latencies
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=PerfReportPlotter.PERF_DISK_NET_PLOT_FIGSIZE)
        
        #ax = axes[0]
        ax.set_xlabel('job time')
        ax.set_ylabel('mb')

        reduced_disk_io_read, reduced_timestamps = self.diff_at_partitions(disk_io_read, PerfReportPlotter.PERF_DISK_NET_PLOT_BINS, timestamps)
        reduced_disk_io_write, _ = self.diff_at_partitions(disk_io_write, PerfReportPlotter.PERF_DISK_NET_PLOT_BINS, timestamps)
        reduced_net_io_ext_sent, _ = self.diff_at_partitions(net_io_ext_sent, PerfReportPlotter.PERF_DISK_NET_PLOT_BINS, timestamps)
        reduced_net_io_ext_recv, _ = self.diff_at_partitions(net_io_ext_recv, PerfReportPlotter.PERF_DISK_NET_PLOT_BINS, timestamps)

        # identify a good normalization value for disk+io plots
        max_disk_net_normalization_value = max(
            max(reduced_disk_io_read),
            max(reduced_disk_io_write),
            max(reduced_net_io_ext_sent),
            max(reduced_net_io_ext_recv),
        ) * 1.05 # 5% more to keep some visual margin on the plot

        width = max(reduced_timestamps) / len(reduced_timestamps) / 5
        ax.bar(reduced_timestamps, reduced_disk_io_read, width=width, label=f"disk read", color="springgreen")
        ax.bar(reduced_timestamps + width, reduced_disk_io_write, width=width, label=f"disk write", color="darkgreen")
        ax.bar(reduced_timestamps + 2*width, reduced_net_io_ext_sent, width=width, label=f"net recv", color="hotpink")
        ax.bar(reduced_timestamps + 3*width, reduced_net_io_ext_recv, width=width, label=f"net sent", color="purple")
        ax.set_ylim([0.0, max_disk_net_normalization_value])
        ax.legend(loc='upper left')

        ax2 = ax.twinx()
        #ax2 = axes[1]
        ax2.set_ylabel('%')
        ax2.plot(timestamps, cpu_utilization, label="cpu", color="r")
        ax2.plot(timestamps, mem_utilization, label="mem", color="b")
        ax2.set_ylim([0.0, 100.0])
        ax2.legend(loc='upper right')
        #ax.set_title(f"Disk+Net I/O normalized plot (max = {max_disk_net_normalization_value:.2f}mb")

        #plt.xlabel("job time")
        #plt.ylim(0.0, max_disk_net_normalization_value)
        plt.legend(loc='best')

        plt.show()

        # record in mlflow
        self.metrics_logger.log_figure(fig, f"disk_and_net_plot_node{node}.png")
