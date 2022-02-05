Each component in this repository is reporting performance metrics in MLFlow.

The list of available metrics is detailed below:

| Metric | Level | Description |
| :-- | :-- | :-- |
| `cpu_avg_utilization_over20_pct` | One value per node | How much time every cpu of that node are utilized more than 20%, over the total job time. |
| `cpu_avg_utilization_over40_pct` | One value per node | How much time are all cpus are utilized more than 40%, over the total job time. |
| `cpu_avg_utilization_over80_pct` | One value per node | ow much time are all cpus are utilized more than 80%, over the total job time. |
| `cpu_avg_utilization_at100_pct` | One value per node | How much time are all cpus fully utilized at 100%, over the total job time. |
| `cpu_avg_utilization_pct` | One value per node | How much are every cpu utilized on average during the entire job. |
| `max_t_cpu_pct_per_cpu_avg` | One value per node | Maximum value taken by the average cpu utilization over the entire job time. |
| `max_t_cpu_pct_per_cpu_min` | One value per node | Maximum value taken by the minimum cpu utilization over the entire job time. |
| `max_t_cpu_pct_per_cpu_min` | One value per node | Maximum value taken by the maximum cpu utilization over the entire job time. |
| `node_cpu_hours` | One value per node | `time * #cpus` |
| `node_unused_cpu_hours` | One value per node | `time * #cpus * (1 - cpu_avg_utilization_pct)` |

::: src.common.perf
