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
| lightgbm#2        |       2 | lightgbm        | C_API.3.3.0      | docker/lightgbm-custom/v330_patch_cpu_mpi_build.dockerfile |          16 |             1 | x86_64    | Linux    |
| lightgbm#3        |       3 | lightgbm        | C_API.3.2.1      | docker/lightgbm-v3.2.1/linux_cpu_mpi_build.dockerfile      |          16 |             1 | x86_64    | Linux    |
| lightgbm#4        |       4 | lightgbm        | C_API.3.2.1      | docker/lightgbm-custom/v321_patch_cpu_mpi_build.dockerfile |          16 |             1 | x86_64    | Linux    |
| treelite_python#5 |       5 | treelite_python | 1.3.0            | default                                                    |          16 |             1 | x86_64    | Linux    |

## Metric time_inferencing per prediction (usecs)

| inferencing task config                |   lightgbm#0 |   lightgbm#1 |   lightgbm#2 |   lightgbm#3 |   lightgbm#4 |   treelite_python#5 |
|:---------------------------------------|-------------:|-------------:|-------------:|-------------:|-------------:|--------------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |      6.71442 |      1.27191 |      1.88084 |      1.97014 |      1.50457 |            0.299835 |
| 10 trees<br/>31 leaves<br/>100 cols    |     10.0109  |      1.87281 |      1.89273 |      1.51227 |      1.93901 |            0.465536 |
| 10 trees<br/>31 leaves<br/>1000 cols   |     37.308   |      4.32708 |      4.70362 |      7.06888 |      4.72284 |            2.08173  |
| 100 trees<br/>31 leaves<br/>10 cols    |     18.8272  |     12.7087  |     14.9646  |     10.8278  |     16.6011  |            5.27241  |
| 100 trees<br/>31 leaves<br/>100 cols   |     23.524   |      9.6317  |     11.2825  |     15.0675  |     13.3228  |            7.3904   |
| 100 trees<br/>31 leaves<br/>1000 cols  |     45.8476  |     14.3042  |     18.5159  |     15.6538  |     14.9914  |            7.93605  |
| 1000 trees<br/>31 leaves<br/>10 cols   |    113.854   |     95.4644  |    104.575   |     93.1975  |    107.137   |           28.5369   |
| 1000 trees<br/>31 leaves<br/>100 cols  |    173.506   |    136.601   |    137.953   |    137.349   |    165.446   |           96.1941   |
| 1000 trees<br/>31 leaves<br/>1000 cols |    178.49    |    143.14    |    143.734   |    146.814   |    149.186   |           98.9669   |
| 5000 trees<br/>31 leaves<br/>10 cols   |    395.046   |    394.296   |    425.493   |    326.193   |    443.607   |          251.199    |
| 5000 trees<br/>31 leaves<br/>100 cols  |    467.79    |    459.998   |    535.714   |    537.431   |    450.346   |          295.183    |
| 5000 trees<br/>31 leaves<br/>1000 cols |    645.185   |    580.791   |    574.005   |    643.234   |    591.006   |          442.544    |

## Percentile metrics for each variant

Some variants above report percentile metrics. Those are reported by computing inferencing latency per request batch (currently, batch size = 1, and number of threads = 1). Not all variants provide those (work in progress).


### lightgbm#1

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |      1.3    |       1.5   |       1.6   |
| 10 trees<br/>31 leaves<br/>100 cols    |      1.8    |       2     |       3.1   |
| 10 trees<br/>31 leaves<br/>1000 cols   |      4.201  |       4.5   |       5.6   |
| 100 trees<br/>31 leaves<br/>10 cols    |     12.6    |      13.8   |      19.1   |
| 100 trees<br/>31 leaves<br/>100 cols   |      9.501  |      10     |      12.802 |
| 100 trees<br/>31 leaves<br/>1000 cols  |     14.301  |      15.601 |      25.001 |
| 1000 trees<br/>31 leaves<br/>10 cols   |     95.1015 |      98.801 |     108.803 |
| 1000 trees<br/>31 leaves<br/>100 cols  |    131.001  |     145.6   |     215.101 |
| 1000 trees<br/>31 leaves<br/>1000 cols |    142.601  |     145.202 |     157.302 |
| 5000 trees<br/>31 leaves<br/>10 cols   |    383.404  |     430.905 |     584.61  |
| 5000 trees<br/>31 leaves<br/>100 cols  |    448.404  |     504.305 |     633.407 |
| 5000 trees<br/>31 leaves<br/>1000 cols |    557.003  |     640.203 |     836.145 |

### lightgbm#2

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |       1.8   |       2.1   |     2.601   |
| 10 trees<br/>31 leaves<br/>100 cols    |       1.9   |       2     |     2.10001 |
| 10 trees<br/>31 leaves<br/>1000 cols   |       4.7   |       4.901 |     5.4     |
| 100 trees<br/>31 leaves<br/>10 cols    |      13.7   |      15.4   |    37.204   |
| 100 trees<br/>31 leaves<br/>100 cols   |      10.8   |      12.901 |    17.301   |
| 100 trees<br/>31 leaves<br/>1000 cols  |      17.7   |      19.001 |    31.4     |
| 1000 trees<br/>31 leaves<br/>10 cols   |     104.003 |     108.703 |   122.603   |
| 1000 trees<br/>31 leaves<br/>100 cols  |     132.501 |     149.701 |   221.015   |
| 1000 trees<br/>31 leaves<br/>1000 cols |     138.702 |     160.802 |   219.107   |
| 5000 trees<br/>31 leaves<br/>10 cols   |     425.024 |     463.626 |   496.927   |
| 5000 trees<br/>31 leaves<br/>100 cols  |     508.705 |     588.917 |   946.39    |
| 5000 trees<br/>31 leaves<br/>1000 cols |     550.905 |     624.606 |   810.269   |

### lightgbm#3

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |       1.8   |       2.3   |       3.1   |
| 10 trees<br/>31 leaves<br/>100 cols    |       1.5   |       1.6   |       1.9   |
| 10 trees<br/>31 leaves<br/>1000 cols   |       6.3   |       7.2   |      23.901 |
| 100 trees<br/>31 leaves<br/>10 cols    |      10.8   |      11.6   |      12.6   |
| 100 trees<br/>31 leaves<br/>100 cols   |      14.3   |      15.7   |      29.903 |
| 100 trees<br/>31 leaves<br/>1000 cols  |      15.1   |      16.2   |      27.201 |
| 1000 trees<br/>31 leaves<br/>10 cols   |      85.301 |     109.901 |     168.301 |
| 1000 trees<br/>31 leaves<br/>100 cols  |     132.401 |     149.601 |     201.402 |
| 1000 trees<br/>31 leaves<br/>1000 cols |     146.202 |     148.903 |     161.503 |
| 5000 trees<br/>31 leaves<br/>10 cols   |     312.703 |     354.715 |     505.311 |
| 5000 trees<br/>31 leaves<br/>100 cols  |     537.638 |     582.651 |     608.343 |
| 5000 trees<br/>31 leaves<br/>1000 cols |     641.307 |     654.907 |     667.409 |

### lightgbm#4

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |       1.3   |       1.7   |       2.7   |
| 10 trees<br/>31 leaves<br/>100 cols    |       1.8   |       2.2   |       2.6   |
| 10 trees<br/>31 leaves<br/>1000 cols   |       4.7   |       4.9   |       5.3   |
| 100 trees<br/>31 leaves<br/>10 cols    |      15.7   |      17.2   |      34.9   |
| 100 trees<br/>31 leaves<br/>100 cols   |      12.201 |      13.501 |      48.706 |
| 100 trees<br/>31 leaves<br/>1000 cols  |      14.901 |      16.101 |      24.701 |
| 1000 trees<br/>31 leaves<br/>10 cols   |      97.301 |     136.401 |     201.902 |
| 1000 trees<br/>31 leaves<br/>100 cols  |     164.901 |     170.101 |     182.801 |
| 1000 trees<br/>31 leaves<br/>1000 cols |     148.403 |     151.003 |     166.205 |
| 5000 trees<br/>31 leaves<br/>10 cols   |     439.327 |     492.54  |     602.444 |
| 5000 trees<br/>31 leaves<br/>100 cols  |     439.432 |     490.245 |     605.846 |
| 5000 trees<br/>31 leaves<br/>1000 cols |     571.902 |     640.112 |     827.614 |
