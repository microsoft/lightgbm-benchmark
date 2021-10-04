# LightGBM Inferencing Benchmark

> The report below has been automatically generated with results from the [lightgbm-benchmark repo](https://github.com/microsoft/lightgbm-benchmark).

IMPORTANT: This is work in progress, to check out current work items check the [project open issues](https://github.com/microsoft/lightgbm-benchmark/issues). In particular, **do not trust these numbers yet** until we've removed this header!

## Variants

| Index | Framework | Version | Build |
| :-- | :-- | :-- | :-- |
| lightgbm_python_1| lightgbm_python| 3.2.1| [dockers/lightgbm_cpu_mpi_pip.dockerfile](https://github.com/microsoft/lightgbm-benchmark/tree/main/src/scripts/lightgbm_python/dockers/lightgbm_cpu_mpi_pip.dockerfile)|
| lightgbm_python_2| lightgbm_python| 3.2.1.99| [dockers/lightgbm_cpu_mpi_build.dockerfile](https://github.com/microsoft/lightgbm-benchmark/tree/main/src/scripts/lightgbm_python/dockers/lightgbm_cpu_mpi_build.dockerfile)|
| lightgbm_python_3| lightgbm_python| 3.2.1.99| [dockers/lightgbm_cpu_mpi_custom.dockerfile](https://github.com/microsoft/lightgbm-benchmark/tree/main/src/scripts/lightgbm_python/dockers/lightgbm_cpu_mpi_custom.dockerfile)|
| treelite_python_4| treelite_python| 1.3.0| n/a|


## Metric time_inferencing per prediction (usecs)

| &nbsp; | lightgbm_python_1 | lightgbm_python_2 | lightgbm_python_3 | treelite_python_4 |
| :-- | :-- | :-- | :-- | :-- |
| trees=10<br/>leaves=31<br/>cols=10<br/> | 5.49 | 7.67 | 7.41 | 0.34 |
| trees=10<br/>leaves=31<br/>cols=100<br/> | 8.88 | 10.04 | 10.81 | 0.36 |
| trees=10<br/>leaves=31<br/>cols=1000<br/> | 40.33 | 37.42 | 43.68 | 1.88 |
| trees=100<br/>leaves=31<br/>cols=10<br/> | 15.40 | 16.83 | 11.96 | 6.20 |
| trees=100<br/>leaves=31<br/>cols=100<br/> | 19.44 | 19.14 | 14.66 | 6.12 |
| trees=100<br/>leaves=31<br/>cols=1000<br/> | 51.15 | 47.06 | 45.47 | 7.82 |
| trees=1000<br/>leaves=31<br/>cols=10<br/> | 122.77 | 127.34 | 64.13 | 33.89 |
| trees=1000<br/>leaves=31<br/>cols=100<br/> | 185.39 | 179.37 | 110.73 | 101.45 |
| trees=1000<br/>leaves=31<br/>cols=1000<br/> | 204.33 | 193.04 | 140.70 | 106.86 |
| trees=5000<br/>leaves=31<br/>cols=10<br/> | 425.67 | 473.33 | 217.00 | 201.87 |
| trees=5000<br/>leaves=31<br/>cols=100<br/> | 514.21 | 497.09 | 300.97 | 289.12 |
| trees=5000<br/>leaves=31<br/>cols=1000<br/> | 720.98 | 658.46 | 429.75 | 467.27 |
