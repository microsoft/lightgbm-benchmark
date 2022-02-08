Each component in this repository is reporting performance metrics in MLFlow.

The list of available metrics is detailed below:

| Metric | Level | Description |
| :-- | :-- | :-- |
| `cpu_avg_utilization_over20_pct` | One value per node | How much time every cpu of that node are utilized more than 20%, over the total job time. |
| `cpu_avg_utilization_over40_pct` | One value per node | How much time are all cpus are utilized more than 40%, over the total job time. |
| `cpu_avg_utilization_over80_pct` | One value per node | ow much time are all cpus are utilized more than 80%, over the total job time. |
| `cpu_avg_utilization_at100_pct` | One value per node | How much time are all cpus fully utilized at 100%, over the total job time. |
| `cpu_avg_utilization_pct` | One value per node | How much are every cpu utilized on average during the entire job. |
| `max_t_cpu_pct_per_cpu_avg` | One value per node | Maximum value taken by the **average cpu utilization** over the entire job time. |
| `max_t_cpu_pct_per_cpu_max` | One value per node | Maximum value taken by the **maximum cpu utilization** over the entire job time. |
| `max_t_cpu_pct_per_cpu_min` | One value per node | Maximum value taken by the **minimum cpu utilization** over the entire job time. |
| `node_cpu_hours` | One value per node | `time * #cpus` |
| `node_unused_cpu_hours` | One value per node | `time * #cpus * (1 - cpu_avg_utilization_pct)` |
| `max_t_mem_percent` | One value per node | Maximum value taken by **memory utilization** over the entire job time. |
| `max_t_disk_usage_percent` | One value per node | Maximum value taken by **disk usage** over the entire job time. |
| `disk_io_read_mb` | One value per node | Total disk **data read** in MB (max value at the end of job). |
| `disk_io_write_mb` | One value per node | Total disk **data write** in MB (max value at the end of job). |
| `net_io_lo_sent_mb` | One value per node | Total net data **sent on loopback** device (max value at the end of job). |
| `net_io_ext_sent_mb` | One value per node | Total net data **sent on external** device (max value at the end of job). |
| `net_io_lo_recv_mb` | One value per node | Total net data **received on loopback** device (max value at the end of job). |
| `net_io_ext_recv_mb` | One value per node | Total net data **received on external** device (max value at the end of job). |

::: src.common.perf
