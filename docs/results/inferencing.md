# LightGBM Inferencing Benchmark

!!! note
    The report below has been automatically generated with results from the [lightgbm-benchmark repo](https://github.com/microsoft/lightgbm-benchmark).

!!! warning
    This is work in progress, to check out current work items check the [project open issues](https://github.com/microsoft/lightgbm-benchmark/issues). In particular, **do not trust these numbers yet** until we've removed this header!

## Variants

| variant_id        | framework       | version   | build                                      |   cpu count |   num threads | machine   | system   |
|:------------------|:----------------|:----------|:-------------------------------------------|------------:|--------------:|:----------|:---------|
| lightgbm_python#0 | lightgbm_python | 3.2.1     | dockers/lightgbm_cpu_mpi_pip.dockerfile    |          32 |             1 | x86_64    | Linux    |
| lightgbm_python#1 | lightgbm_python | 3.2.1.99  | dockers/lightgbm_cpu_mpi_build.dockerfile  |          32 |             1 | x86_64    | Linux    |
| lightgbm_python#2 | lightgbm_python | 3.2.1.99  | dockers/lightgbm_cpu_mpi_custom.dockerfile |          32 |             1 | x86_64    | Linux    |
| treelite_python#3 | treelite_python | 1.3.0     | n/a                                        |          32 |             1 | x86_64    | Linux    |

## Metric time_inferencing per prediction (usecs)

| inferencing task config                |   lightgbm_python#0 |   lightgbm_python#1 |   lightgbm_python#2 |   treelite_python#3 |
|:---------------------------------------|--------------------:|--------------------:|--------------------:|--------------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |             5.92084 |             8.58464 |             7.55572 |            0.543094 |
| 10 trees<br/>31 leaves<br/>100 cols    |             9.27417 |            12.2316  |            11.2963  |            0.571966 |
| 10 trees<br/>31 leaves<br/>1000 cols   |            41.4401  |            40.9806  |            46.0711  |            1.76368  |
| 100 trees<br/>31 leaves<br/>10 cols    |            17.1347  |            19.7504  |            12.405   |            5.38526  |
| 100 trees<br/>31 leaves<br/>100 cols   |            21.5201  |            22.5409  |            16.6037  |            5.50926  |
| 100 trees<br/>31 leaves<br/>1000 cols  |            54.0431  |            52.1384  |            49.4389  |            8.50625  |
| 1000 trees<br/>31 leaves<br/>10 cols   |           117.764   |           117.32    |            69.3599  |           38.3443   |
| 1000 trees<br/>31 leaves<br/>100 cols  |           203.721   |           184.02    |           112.265   |          101.245    |
| 1000 trees<br/>31 leaves<br/>1000 cols |           214.57    |           214.075   |           127.218   |          116.414    |
| 5000 trees<br/>31 leaves<br/>10 cols   |           427.844   |           438.51    |           238.599   |          219.13     |
| 5000 trees<br/>31 leaves<br/>100 cols  |           550.605   |           569.618   |           305.591   |          355.015    |
| 5000 trees<br/>31 leaves<br/>1000 cols |           812.568   |           735.613   |           506.837   |          523.521    |