# LightGBM Inferencing Benchmark

!!! note
    The report below has been automatically generated with results from the [lightgbm-benchmark repo](https://github.com/microsoft/lightgbm-benchmark).

!!! warning
    This is work in progress, to check out current work items check the [project open issues](https://github.com/microsoft/lightgbm-benchmark/issues). In particular, **do not trust these numbers yet** until we've removed this header!

## Variants

| variant_id        |   index | framework       | version          | build                                                      |   cpu count |   num threads | machine   | system   |
|:------------------|--------:|:----------------|:-----------------|:-----------------------------------------------------------|------------:|--------------:|:----------|:---------|
| lightgbm#0        |       0 | lightgbm        | PYTHON_API.3.3.0 | default                                                    |          16 |             1 | x86_64    | Linux    |
| lightgbm#1        |       1 | lightgbm        | PYTHON_API.3.3.0 | docker/lightgbm-v3.2.1/linux_cpu_mpi_pip.dockerfile        |          16 |             1 | x86_64    | Linux    |
| lightgbm#2        |       2 | lightgbm        | PYTHON_API.3.2.1 | docker/lightgbm-v3.2.1/linux_cpu_mpi_build.dockerfile      |          16 |             1 | x86_64    | Linux    |
| lightgbm#3        |       3 | lightgbm        | PYTHON_API.3.2.1 | docker/lightgbm-custom/v321_patch_cpu_mpi_build.dockerfile |          16 |             1 | x86_64    | Linux    |
| lightgbm#4        |       4 | lightgbm        | C_API.3.3.0      | default                                                    |          16 |             1 | x86_64    | Linux    |
| treelite_python#5 |       5 | treelite_python | 1.3.0            | default                                                    |          16 |             1 | x86_64    | Linux    |

## Metric time_inferencing per prediction (usecs)

| inferencing task config                |   lightgbm#0 |   lightgbm#1 |   lightgbm#2 |   lightgbm#3 |   lightgbm#4 |   treelite_python#5 |
|:---------------------------------------|-------------:|-------------:|-------------:|-------------:|-------------:|--------------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |      6.72381 |       6.7342 |      4.35991 |      4.15237 |     0.674118 |            0.299048 |
| 10 trees<br/>31 leaves<br/>100 cols    |     10.1787  |       9.9839 |      7.9644  |      7.34487 |     0.668166 |            0.426555 |
| 10 trees<br/>31 leaves<br/>1000 cols   |     37.1217  |      33.1649 |     35.9022  |     35.936   |     1.00468  |            1.824    |
| 100 trees<br/>31 leaves<br/>10 cols    |     16.3346  |      16.3827 |     12.9529  |      9.94401 |     5.64863  |            5.06129  |
| 100 trees<br/>31 leaves<br/>100 cols   |     17.4502  |      19.2278 |     17.4101  |     14.6763  |     5.3586   |            5.68655  |
| 100 trees<br/>31 leaves<br/>1000 cols  |     45.7731  |      46.275  |     44.7125  |     40.2211  |    11.0882   |            7.38578  |
| 1000 trees<br/>31 leaves<br/>10 cols   |    106.716   |     100.041  |     96.568   |     54.452   |    68.0959   |           31.8616   |
| 1000 trees<br/>31 leaves<br/>100 cols  |    166.423   |     170.721  |    161.044   |     83.6609  |   159.991    |           99.5954   |
| 1000 trees<br/>31 leaves<br/>1000 cols |    182.831   |     181.864  |    177.292   |    110.828   |   168.757    |           98.1796   |
| 5000 trees<br/>31 leaves<br/>10 cols   |    370.63    |     381.474  |    376.225   |    192.508   |   336.046    |          186.961    |
| 5000 trees<br/>31 leaves<br/>100 cols  |    472.504   |     472.018  |    485.115   |    225.766   |   470.001    |          295.55     |
| 5000 trees<br/>31 leaves<br/>1000 cols |    662.376   |     644.987  |    650.918   |    369.316   |   673.113    |          441.634    |

## Percentile metrics for each variant

### lightgbm#4

| inferencing task config                |   metrics.batch_time_inferencing_p50_usecs |   metrics.batch_time_inferencing_p90_usecs |   metrics.batch_time_inferencing_p99_usecs |
|:---------------------------------------|-------------------------------------------:|-------------------------------------------:|-------------------------------------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |                                      0.6   |                                      0.7   |                                      2.7   |
| 10 trees<br/>31 leaves<br/>100 cols    |                                      0.6   |                                      0.8   |                                      1.1   |
| 10 trees<br/>31 leaves<br/>1000 cols   |                                      1     |                                      1.1   |                                      1.201 |
| 100 trees<br/>31 leaves<br/>10 cols    |                                      5.6   |                                      5.7   |                                      7.9   |
| 100 trees<br/>31 leaves<br/>100 cols   |                                      5.3   |                                      5.4   |                                      6.501 |
| 100 trees<br/>31 leaves<br/>1000 cols  |                                     10.401 |                                     11.2   |                                     23.402 |
| 1000 trees<br/>31 leaves<br/>10 cols   |                                     67.597 |                                     69.097 |                                     80.796 |
| 1000 trees<br/>31 leaves<br/>100 cols  |                                    158.206 |                                    164.505 |                                    186.506 |
| 1000 trees<br/>31 leaves<br/>1000 cols |                                    168.095 |                                    169.695 |                                    182.696 |
| 5000 trees<br/>31 leaves<br/>10 cols   |                                    333.61  |                                    345.821 |                                    359.91  |
| 5000 trees<br/>31 leaves<br/>100 cols  |                                    468.213 |                                    479.114 |                                    488.814 |
| 5000 trees<br/>31 leaves<br/>1000 cols |                                    648.023 |                                    681.324 |                                   1249.85  |
