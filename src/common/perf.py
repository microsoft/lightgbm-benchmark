# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Helps with reporting performance metrics (cpu/mem utilization)
"""
import logging
import threading
import time
import psutil
import matplotlib.pyplot as plt


class PerformanceReportingThread(threading.Thread):
    """Thread to report performance (cpu/mem/net)"""
    def __init__(self, metrics_logger=None, store_max_length=10000, initial_time_increment=1.0, report_each_step=False, report_at_finalize=True, world_rank=0):
        """Constructor

        Args:
            metrics_logger (common.metrics.MetricsLogger): class to log metrics using MLFlow
        """
        threading.Thread.__init__(self)
        self.killed = False # flag, set to True to kill from the inside

        self.logger = logging.getLogger(__name__)
        self.metrics_logger = metrics_logger

        self.report_store = []
        self.report_store_max_length = (store_max_length//2 + store_max_length%2) * 2 # needs to be dividable by 2
        self.time_increment = initial_time_increment

        self.report_each_step = report_each_step
        self.report_at_finalize = report_at_finalize

        self.world_rank = 0 # for mpi

        # NOTE: do not use matplotlib gui backeng in a thread
        plt.switch_backend('agg')


    #####################
    ### RUN FUNCTIONS ###
    #####################

    def store_report(self, report):
        self.report_store.append((time.time(), report))

        if len(self.report_store) >= self.report_store_max_length:
            # trim the report by half
            self.report_store = [
                self.report_store[i]
                for i in range(0, self.report_store_max_length, 2)
            ]
            self.time_increment *= 2
            self.logger.warning(f"Perf report store reached max, increasing time_increment to {self.time_increment}")

    def run(self):
        """Run function"""
        while not(self.killed):
            if self.time_increment >= 1.0: # cpu_percent.interval already consumes 1sec
                time.sleep(self.time_increment - 1.0) # will double every time report_store_max_length is reached

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

            if self.report_each_step:
                self.logger.info(f"PERF={perf_report}")

            self.store_report(perf_report)

        if self.report_at_finalize:
            self.plot_all_reports()

    def plot_all_reports(self):
        custom_fig_size=(16,4)

        report_plots_specs = [
            {
                "file_key":"cpu_pct",
                "keys":["cpu_pct_per_cpu_avg","cpu_pct_per_cpu_min","cpu_pct_per_cpu_max"],
                "title":"CPU utilization %",
                "xlabel":"step",
                "ylabel":"%",
                "bottom":0.0,
                "top":100.0
            },
            {
                "file_key":"mem_pct",
                "keys":"mem_percent",
                "title":"MEM utilization %",
                "xlabel":"step",
                "ylabel":"%",
                "bottom":0.0,
                "top":100.0
            },
            {
                "file_key":"disk_usage",
                "keys":"disk_usage_percent",
                "title":"DISK usage %",
                "xlabel":"step",
                "ylabel":"%",
                "bottom":0.0,
                "top":100.0
            },
            {
                "file_key":"disk_io",
                "keys":["disk_io_read_mb", "disk_io_write_mb"],
                "title":"DISK I/O",
                "xlabel":"step",
                "ylabel":"MB"
            },
            {
                "file_key":"net_io",
                "keys":["net_io_lo_sent_mb","net_io_ext_sent_mb","net_io_lo_recv_mb","net_io_ext_recv_mb"],
                "title":"NET I/O",
                "xlabel":"step",
                "ylabel":"MB"
            }
        ]

        for entry in report_plots_specs:
            fig = plt.figure(figsize=custom_fig_size)
            ax = fig.add_subplot(111)

            # get and plot all value keys
            if isinstance(entry["keys"], list):
                # if multiple keys specified as lies
                for key in entry["keys"]:
                    # get each one distinctly
                    values = [
                        report.get(key, None)
                        for report_time, report in self.report_store
                    ]
                    # then plot
                    ax.plot(values, label=key)

                # don't forget the legend
                ax.legend()
            else:
                # if only one key
                values = [
                    report.get(entry["keys"], None)
                    for report_time, report in self.report_store
                ]
                ax.plot(values)
                # no legend needed here

            ax.set_ylim(bottom=entry.get("bottom", None), top=entry.get("top", None))
            ax.set_title(entry.get("title", None))
            plt.xlabel(entry.get("xlabel", "step"))
            plt.ylabel(entry.get("ylabel", None))

            # record in mlflow
            if self.metrics_logger:
                self.metrics_logger.log_figure(fig, f"perf_node_{self.world_rank}_{entry['file_key']}.png")

    def finalize(self):
        self.killed = True
        self.join()


if __name__ == "__main__":
    # just for testing locally
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    from metrics import MetricsLogger

    metrics_logger = MetricsLogger("fake.foo")

    logger.info("TEST")
    perf_report = PerformanceReportingThread(
        metrics_logger=metrics_logger,
        store_max_length=10,
        initial_time_increment=1,
        report_each_step=True,
        report_at_finalize=True
    )
    perf_report.start()

    time.sleep(10)

    perf_report.finalize()

    metrics_logger.close()