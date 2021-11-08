# LightGBM Inferencing Benchmark

!!! note
    The report below has been automatically generated with results from the [lightgbm-benchmark repo](https://github.com/microsoft/lightgbm-benchmark).

!!! warning
    This is work in progress, to check out current work items check the [project open inferencing issues](https://github.com/microsoft/lightgbm-benchmark/issues?q=is%3Aissue+is%3Aopen+label%3Ainferencing-benchmark). In particular, **do not trust these numbers yet** until we've removed this header!

## Variants

| variant_id        |   index | framework       | version          | build                                                      |   cpu count |   num threads | machine   | system   |
|:------------------|--------:|:----------------|:-----------------|:-----------------------------------------------------------|------------:|--------------:|:----------|:---------|
| lightgbm#0        |       0 | lightgbm        | PYTHON_API.3.3.0 | default                                                    |          16 |             1 | x86_64    | Linux    |
| lightgbm#1        |       1 | lightgbm        | C_API.3.3.0      | default                                                    |          16 |             1 | x86_64    | Linux    |
| lightgbm#2        |       2 | lightgbm        | C_API.3.2.1      | docker/lightgbm-v3.2.1/linux_cpu_mpi_build.dockerfile      |          16 |             1 | x86_64    | Linux    |
| lightgbm#3        |       3 | lightgbm        | C_API.3.2.1      | docker/lightgbm-custom/v321_patch_cpu_mpi_build.dockerfile |          16 |             1 | x86_64    | Linux    |
| treelite_python#4 |       4 | treelite_python | 1.3.0            | default                                                    |          16 |             1 | x86_64    | Linux    |

## Metric time_inferencing per prediction (usecs)

| inferencing task config                |   lightgbm#0 |   lightgbm#1 |   lightgbm#2 |   lightgbm#3 |   treelite_python#4 |
|:---------------------------------------|-------------:|-------------:|-------------:|-------------:|--------------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |       6.6782 |      1.64384 |     0.674529 |      0.6571  |            0.244832 |
| 10 trees<br/>31 leaves<br/>100 cols    |      10.2975 |      1.12449 |     1.80957  |      1.01547 |            0.376225 |
| 10 trees<br/>31 leaves<br/>1000 cols   |      41.6222 |      2.31878 |     1.13139  |      0.97278 |            1.8821   |
| 100 trees<br/>31 leaves<br/>10 cols    |      22.618  |     12.0014  |    12.6983   |      5.91319 |            5.57137  |
| 100 trees<br/>31 leaves<br/>100 cols   |      24.7328 |     11.9549  |    11.4674   |      5.51865 |            6.09224  |
| 100 trees<br/>31 leaves<br/>1000 cols  |      51.3106 |     12.1792  |     5.52488  |     13.0928  |            9.02305  |
| 1000 trees<br/>31 leaves<br/>10 cols   |     110.798  |     70.9829  |    66.5644   |     79.937   |           32.0208   |
| 1000 trees<br/>31 leaves<br/>100 cols  |     161.674  |    181.517   |   156.472    |    175.729   |           98.3616   |
| 1000 trees<br/>31 leaves<br/>1000 cols |     177.93   |    153.365   |   156.771    |    154.881   |           98.6686   |
| 5000 trees<br/>31 leaves<br/>10 cols   |     372.706  |    366.744   |   336.039    |    317.522   |          226.819    |
| 5000 trees<br/>31 leaves<br/>100 cols  |     473.022  |    441.521   |   444.257    |    482.049   |          311.862    |
| 5000 trees<br/>31 leaves<br/>1000 cols |     689.319  |    643.467   |   731.767    |    674.668   |          543.936    |

## Percentile metrics for each variant

Some variants above report percentile metrics. Those are reported by computing inferencing latency per request batch (currently, batch size = 1, and number of threads = 1). Not all variants provide those (work in progress).


### lightgbm#1

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |       1.401 |      2.1    |       2.801 |
| 10 trees<br/>31 leaves<br/>100 cols    |       1     |      1.3    |       3.7   |
| 10 trees<br/>31 leaves<br/>1000 cols   |       1.9   |      2.3    |       5     |
| 100 trees<br/>31 leaves<br/>10 cols    |      10.4   |     11      |      28.899 |
| 100 trees<br/>31 leaves<br/>100 cols   |      11.1   |     11.801  |      24.5   |
| 100 trees<br/>31 leaves<br/>1000 cols  |      11.2   |     12      |      24.299 |
| 1000 trees<br/>31 leaves<br/>10 cols   |      61.799 |     90.6099 |     158.4   |
| 1000 trees<br/>31 leaves<br/>100 cols  |     166.798 |    218.607  |     295.797 |
| 1000 trees<br/>31 leaves<br/>1000 cols |     143.299 |    178.398  |     259.699 |
| 5000 trees<br/>31 leaves<br/>10 cols   |     368.715 |    395.217  |     417.117 |
| 5000 trees<br/>31 leaves<br/>100 cols  |     427.81  |    493.111  |     728.425 |
| 5000 trees<br/>31 leaves<br/>1000 cols |     617.894 |    741.442  |     967.791 |

### lightgbm#2

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |       0.7   |       0.8   |       0.9   |
| 10 trees<br/>31 leaves<br/>100 cols    |       1.6   |       2.1   |       4.401 |
| 10 trees<br/>31 leaves<br/>1000 cols   |       1.1   |       1.2   |       1.4   |
| 100 trees<br/>31 leaves<br/>10 cols    |      11.3   |      12     |      29.601 |
| 100 trees<br/>31 leaves<br/>100 cols   |      10     |      11.2   |      30.901 |
| 100 trees<br/>31 leaves<br/>1000 cols  |       5.5   |       5.6   |       6.3   |
| 1000 trees<br/>31 leaves<br/>10 cols   |      65.199 |      69.799 |      79.199 |
| 1000 trees<br/>31 leaves<br/>100 cols  |     150.099 |     188.598 |     274.501 |
| 1000 trees<br/>31 leaves<br/>1000 cols |     149.699 |     188.898 |     258.097 |
| 5000 trees<br/>31 leaves<br/>10 cols   |     320.597 |     400.807 |     537.207 |
| 5000 trees<br/>31 leaves<br/>100 cols  |     443.101 |     452.601 |     461.901 |
| 5000 trees<br/>31 leaves<br/>1000 cols |     706.195 |     836.994 |    1072.75  |

### lightgbm#3

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |       0.6   |      0.7    |       2.6   |
| 10 trees<br/>31 leaves<br/>100 cols    |       0.9   |      1.1    |       3.2   |
| 10 trees<br/>31 leaves<br/>1000 cols   |       1     |      1.1    |       1.3   |
| 100 trees<br/>31 leaves<br/>10 cols    |       5.8   |      5.9    |       8.1   |
| 100 trees<br/>31 leaves<br/>100 cols   |       5.4   |      5.5    |       7.6   |
| 100 trees<br/>31 leaves<br/>1000 cols  |      12.2   |     13.1    |      25.7   |
| 1000 trees<br/>31 leaves<br/>10 cols   |      70.6   |     89.5991 |     194.5   |
| 1000 trees<br/>31 leaves<br/>100 cols  |     174.997 |    176.997  |     189.897 |
| 1000 trees<br/>31 leaves<br/>1000 cols |     145.052 |    182.705  |     265.01  |
| 5000 trees<br/>31 leaves<br/>10 cols   |     309.496 |    366.595  |     474.694 |
| 5000 trees<br/>31 leaves<br/>100 cols  |     441.699 |    594.399  |     756.604 |
| 5000 trees<br/>31 leaves<br/>1000 cols |     670.905 |    690.206  |     711.207 |
