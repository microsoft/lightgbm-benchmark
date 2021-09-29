# LightGBM Inferencing Benchmark

> The report below has been automatically generated with results from the [lightgbm-benchmark repo](https://github.com/microsoft/lightgbm-benchmark).

IMPORTANT: This is work in progress, to check out current work items check the [project open issues](https://github.com/microsoft/lightgbm-benchmark/issues). In particular, **do not trust these numbers yet** until we've removed this header!

## Variants

| Index | Framework | Version | Build |
| :-- | :-- | :-- | :-- |
| lightgbm_python_1 | lightgbm_python | 3.2.1 | dockers/lightgbm_cpu_mpi_pip.dockerfile |
| lightgbm_python_2 | lightgbm_python | 3.2.1.99 | dockers/lightgbm_cpu_mpi_build.dockerfile |
| lightgbm_python_3 | lightgbm_python | 3.2.1.99 | dockers/lightgbm_cpu_mpi_custom.dockerfile |
| treelite_python_4 | treelite_python | 1.3.0 | n/a |


## Metric time_inferencing

| &nbsp; | lightgbm_python_1 | lightgbm_python_2 | lightgbm_python_3 | treelite_python_4 |
| :-- | :-- | :-- | :-- | :-- |
| trees=10<br/>leaves=31<br/>cols=10<br/> | 0.0482 | 0.0706 | 0.0677 | 0.0026 |
| trees=10<br/>leaves=31<br/>cols=100<br/> | 0.0831 | 0.1265 | 0.1093 | 0.0045 |
| trees=10<br/>leaves=31<br/>cols=1000<br/> | 0.3558 | 0.3697 | 0.3563 | 0.0192 |
| trees=100<br/>leaves=31<br/>cols=10<br/> | 0.2453 | 0.1682 | 0.1224 | 0.0512 |
| trees=100<br/>leaves=31<br/>cols=100<br/> | 0.1840 | 0.2266 | 0.1429 | 0.0677 |
| trees=100<br/>leaves=31<br/>cols=1000<br/> | 0.4872 | 0.4730 | 0.4142 | 0.0733 |
| trees=1000<br/>leaves=31<br/>cols=10<br/> | 1.0601 | 1.0230 | 0.5993 | 0.5955 |
| trees=1000<br/>leaves=31<br/>cols=100<br/> | 1.6479 | 1.6528 | 0.9629 | 0.9816 |
| trees=1000<br/>leaves=31<br/>cols=1000<br/> | 1.8647 | 2.0542 | 1.2856 | 0.9696 |
| trees=5000<br/>leaves=31<br/>cols=10<br/> | 4.3415 | 4.4693 | 2.4208 | 1.9794 |
| trees=5000<br/>leaves=31<br/>cols=100<br/> | 5.5063 | 4.7652 | 2.6773 | 3.4036 |
| trees=5000<br/>leaves=31<br/>cols=1000<br/> | 6.8401 | 7.1698 | 4.4330 | 4.4953 |
