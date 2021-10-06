# LightGBM Inferencing Benchmark

!!! note
    The report below has been automatically generated with results from the [lightgbm-benchmark repo](https://github.com/microsoft/lightgbm-benchmark).

!!! warning
    This is work in progress, to check out current work items check the [project open issues](https://github.com/microsoft/lightgbm-benchmark/issues). In particular, **do not trust these numbers yet** until we've removed this header!

## Variants

| variant_id        | framework       | version   | build                                      | os    |
|:------------------|:----------------|:----------|:-------------------------------------------|:------|
| lightgbm_python#0 | lightgbm_python | 3.2.1     | dockers/lightgbm_cpu_mpi_pip.dockerfile    | Linux |
| lightgbm_python#2 | lightgbm_python | 3.2.1.99  | dockers/lightgbm_cpu_mpi_custom.dockerfile | Linux |
| lightgbm_python#1 | lightgbm_python | 3.2.1.99  | dockers/lightgbm_cpu_mpi_build.dockerfile  | Linux |
| treelite_python#3 | treelite_python | 1.3.0     | n/a                                        | Linux |

## Metric time_inferencing per prediction (usecs)

| inferencing task config                |   lightgbm_python#0 |   lightgbm_python#1 |   lightgbm_python#2 |   treelite_python#3 |
|:---------------------------------------|--------------------:|--------------------:|--------------------:|--------------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |             4.46353 |             6.86219 |             6.30934 |            0.362396 |
| 10 trees<br/>31 leaves<br/>100 cols    |             8.64604 |             9.04746 |             9.81698 |            0.518346 |
| 10 trees<br/>31 leaves<br/>1000 cols   |            36.27    |            39.7902  |            39.3422  |            1.89013  |
| 100 trees<br/>31 leaves<br/>10 cols    |            15.1912  |            15.9602  |            13.2037  |            5.15461  |
| 100 trees<br/>31 leaves<br/>100 cols   |            19.3119  |            21.9855  |            13.4163  |            5.64711  |
| 100 trees<br/>31 leaves<br/>1000 cols  |            44.6079  |            48.9864  |            48.1159  |            6.98378  |
| 1000 trees<br/>31 leaves<br/>10 cols   |           110.55    |           111.39    |            59.1312  |           32.3749   |
| 1000 trees<br/>31 leaves<br/>100 cols  |           161.12    |           183.413   |            95.6368  |           96.5918   |
| 1000 trees<br/>31 leaves<br/>1000 cols |           190.81    |           199.568   |           120.469   |          105.153    |
| 5000 trees<br/>31 leaves<br/>10 cols   |           406.119   |           400.482   |           243.836   |          189.392    |
| 5000 trees<br/>31 leaves<br/>100 cols  |           519.317   |           516.659   |           301.009   |          288.075    |
| 5000 trees<br/>31 leaves<br/>1000 cols |           699.416   |           714.399   |           378.438   |          444.761    |