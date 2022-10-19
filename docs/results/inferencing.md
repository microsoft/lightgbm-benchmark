# LightGBM Inferencing Benchmark

!!! note
    The report below has been automatically generated with results from the [lightgbm-benchmark repo](https://github.com/microsoft/lightgbm-benchmark).  

!!! warning
    This is work in progress, to check out current work items check the [project open inferencing issues](https://github.com/microsoft/lightgbm-benchmark/issues?q=is%3Aissue+is%3Aopen+label%3Ainferencing-benchmark). In particular, **do not trust these numbers yet** until we've removed this header!

## Variants

| variant_id                 |   index | framework                | version          | build                                                      |   cpu count |   num threads | machine   | system   |
|:---------------------------|--------:|:-------------------------|:-----------------|:-----------------------------------------------------------|------------:|--------------:|:----------|:---------|
| lightgbm#0                 |       0 | lightgbm                 | PYTHON_API.3.3.0 | default                                                    |          16 |             1 | x86_64    | Linux    |
| lightgbm#1                 |       1 | lightgbm                 | C_API.3.3.0      | default                                                    |          16 |             1 | x86_64    | Linux    |
| lightgbm#2                 |       2 | lightgbm                 | C_API.3.3.0      | docker/lightgbm-custom/v330_patch_cpu_mpi_build.dockerfile |          16 |             1 | x86_64    | Linux    |
| lightgbm#3                 |       3 | lightgbm                 | C_API.3.2.1      | docker/lightgbm-v3.2.1/linux_cpu_mpi_build.dockerfile      |          16 |             1 | x86_64    | Linux    |
| lightgbm#4                 |       4 | lightgbm                 | C_API.3.2.1      | docker/lightgbm-custom/v321_patch_cpu_mpi_build.dockerfile |          16 |             1 | x86_64    | Linux    |
| onnx#5                     |       5 | onnx                     | ONNXRT.1.12.1    | default                                                    |          16 |             1 | x86_64    | Linux    |
| onnx_batch#6               |       6 | onnx_batch               | ONNXRT.1.12.1    | default                                                    |          16 |             1 | x86_64    | Linux    |
| onnx_multithreaded#7       |       7 | onnx_multithreaded       | ONNXRT.1.12.1    | default                                                    |          16 |             - | x86_64    | Linux    |
| onnx_multithreaded_batch#8 |       8 | onnx_multithreaded_batch | ONNXRT.1.12.1    | default                                                    |          16 |             - | x86_64    | Linux    |
| treelite_python#9          |       9 | treelite_python          | 2.1.0            | default                                                    |          16 |             1 | x86_64    | Linux    |

## Metric time_inferencing per prediction (usecs)

| inferencing task config                |   lightgbm#0 |   lightgbm#1 |   lightgbm#2 |   lightgbm#3 |   lightgbm#4 |    onnx#5 |   onnx_batch#6 |   onnx_multithreaded#7 |   onnx_multithreaded_batch#8 |   treelite_python#9 |
|:---------------------------------------|-------------:|-------------:|-------------:|-------------:|-------------:|----------:|---------------:|-----------------------:|-----------------------------:|--------------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |      6.95305 |      1.11553 |      1.19408 |      1.15504 |      1.12653 |   7.62398 |      0.0969134 |                21.4563 |                     0.198045 |            0.303221 |
| 10 trees<br/>31 leaves<br/>100 cols    |      9.9608  |      1.57071 |      1.81644 |      1.55628 |      1.73756 |   7.67336 |      0.149622  |                22.5913 |                     0.303975 |            0.449347 |
| 10 trees<br/>31 leaves<br/>1000 cols   |     36.8206  |      3.97296 |      4.00286 |      4.35525 |      4.56862 |   7.65319 |      1.23701   |                21.9663 |                     1.03079  |            1.90513  |
| 100 trees<br/>31 leaves<br/>10 cols    |     16.081   |     10.3246  |     11.2351  |     10.4623  |     10.411   |  12.9457  |      0.489068  |                27.8963 |                     0.518232 |            5.12872  |
| 100 trees<br/>31 leaves<br/>100 cols   |     18.419   |     10.2733  |      9.27452 |     10.6115  |     10.4095  |  13.1084  |      0.691856  |                26.6879 |                     0.637577 |            5.73254  |
| 100 trees<br/>31 leaves<br/>1000 cols  |     45.0129  |     12.6701  |     11.4707  |     12.7013  |     12.794   |  11.9506  |      2.29946   |                28.9509 |                     1.98307  |            7.35011  |
| 1000 trees<br/>31 leaves<br/>10 cols   |     97.3209  |     97.622   |    103.892   |     95.7561  |     97.6808  |  18.3931  |      3.95854   |                40.0455 |                     4.24206  |           33.3337   |
| 1000 trees<br/>31 leaves<br/>100 cols  |    154.284   |    146.32    |    154.788   |    149.401   |    149.942   |  20.4271  |      5.15573   |                40.3441 |                     4.93979  |           96.6871   |
| 1000 trees<br/>31 leaves<br/>1000 cols |    165.235   |    140.012   |    150.223   |    143.748   |    141.769   |  20.1743  |     11.7819    |                36.897  |                    12.1277   |          101.73     |
| 5000 trees<br/>31 leaves<br/>10 cols   |    376.015   |    407.244   |    373.407   |    366.11    |    383.453   |  43.7589  |     10.8586    |                85.8648 |                    10.1721   |          219.653    |
| 5000 trees<br/>31 leaves<br/>100 cols  |    421.179   |    465.234   |    482.583   |    468.308   |    473.928   | 104.56    |     24.15      |               156.015  |                    24.2661   |          300.779    |
| 5000 trees<br/>31 leaves<br/>1000 cols |    644.905   |    587.578   |    581.033   |    625.28    |    598.814   |  94.8404  |     58.758     |               127.584  |                    58.3206   |          416.228    |

## Percentile metrics for each variant

Some variants above report percentile metrics. Those are reported by computing inferencing latency per request batch (currently, batch size = 1, and number of threads = 1). Not all variants provide those (work in progress).


### lightgbm#1

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |       1.1   |       1.2   |       1.399 |
| 10 trees<br/>31 leaves<br/>100 cols    |       1.6   |       1.7   |       1.9   |
| 10 trees<br/>31 leaves<br/>1000 cols   |       3.9   |       4.2   |       4.5   |
| 100 trees<br/>31 leaves<br/>10 cols    |      10.3   |      11     |      11.601 |
| 100 trees<br/>31 leaves<br/>100 cols   |      10.2   |      10.7   |      11.1   |
| 100 trees<br/>31 leaves<br/>1000 cols  |      12.601 |      13.001 |      13.6   |
| 1000 trees<br/>31 leaves<br/>10 cols   |      96     |     102.001 |     114.201 |
| 1000 trees<br/>31 leaves<br/>100 cols  |     145.899 |     150.599 |     161.099 |
| 1000 trees<br/>31 leaves<br/>1000 cols |     139.124 |     142.024 |     154.528 |
| 5000 trees<br/>31 leaves<br/>10 cols   |     405.801 |     424.302 |     444.202 |
| 5000 trees<br/>31 leaves<br/>100 cols  |     464.302 |     476.601 |     490.101 |
| 5000 trees<br/>31 leaves<br/>1000 cols |     585.368 |     600.169 |     611.8   |

### lightgbm#2

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |       1.2   |       1.3   |       1.5   |
| 10 trees<br/>31 leaves<br/>100 cols    |       1.8   |       1.9   |       2.1   |
| 10 trees<br/>31 leaves<br/>1000 cols   |       3.9   |       4.2   |       4.8   |
| 100 trees<br/>31 leaves<br/>10 cols    |      11.1   |      12     |      13.8   |
| 100 trees<br/>31 leaves<br/>100 cols   |       9.3   |       9.601 |      10     |
| 100 trees<br/>31 leaves<br/>1000 cols  |      11.399 |      11.799 |      13.401 |
| 1000 trees<br/>31 leaves<br/>10 cols   |     103.501 |     108.1   |     116.9   |
| 1000 trees<br/>31 leaves<br/>100 cols  |     154.296 |     159.296 |     170.495 |
| 1000 trees<br/>31 leaves<br/>1000 cols |     149.602 |     152.301 |     164.802 |
| 5000 trees<br/>31 leaves<br/>10 cols   |     372.405 |     389.205 |     405.207 |
| 5000 trees<br/>31 leaves<br/>100 cols  |     481.504 |     496.705 |     510.607 |
| 5000 trees<br/>31 leaves<br/>1000 cols |     578.888 |     596.699 |     618.387 |

### lightgbm#3

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |       1.1   |       1.3   |       1.5   |
| 10 trees<br/>31 leaves<br/>100 cols    |       1.5   |       1.7   |       1.9   |
| 10 trees<br/>31 leaves<br/>1000 cols   |       4.3   |       4.5   |       4.9   |
| 100 trees<br/>31 leaves<br/>10 cols    |      10.401 |      11.2   |      11.8   |
| 100 trees<br/>31 leaves<br/>100 cols   |      10.601 |      11.001 |      11.401 |
| 100 trees<br/>31 leaves<br/>1000 cols  |      12.601 |      13     |      13.6   |
| 1000 trees<br/>31 leaves<br/>10 cols   |      95.5   |      99     |     108.8   |
| 1000 trees<br/>31 leaves<br/>100 cols  |     149     |     153.8   |     164.202 |
| 1000 trees<br/>31 leaves<br/>1000 cols |     142.699 |     145.799 |     158.899 |
| 5000 trees<br/>31 leaves<br/>10 cols   |     363.53  |     384.032 |     427.939 |
| 5000 trees<br/>31 leaves<br/>100 cols  |     466.461 |     479.863 |     501.27  |
| 5000 trees<br/>31 leaves<br/>1000 cols |     622.902 |     637.601 |     650.101 |

### lightgbm#4

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |       1.1   |       1.3   |       1.4   |
| 10 trees<br/>31 leaves<br/>100 cols    |       1.7   |       1.9   |       2     |
| 10 trees<br/>31 leaves<br/>1000 cols   |       4.5   |       4.8   |       5.2   |
| 100 trees<br/>31 leaves<br/>10 cols    |      10.4   |      11.1   |      11.9   |
| 100 trees<br/>31 leaves<br/>100 cols   |      10.4   |      10.8   |      11.3   |
| 100 trees<br/>31 leaves<br/>1000 cols  |      12.798 |      13.099 |      13.598 |
| 1000 trees<br/>31 leaves<br/>10 cols   |      97.302 |     101.201 |     111.002 |
| 1000 trees<br/>31 leaves<br/>100 cols  |     149.489 |     154.29  |     165.188 |
| 1000 trees<br/>31 leaves<br/>1000 cols |     141.2   |     143.601 |     156.5   |
| 5000 trees<br/>31 leaves<br/>10 cols   |     382.303 |     398.402 |     413.602 |
| 5000 trees<br/>31 leaves<br/>100 cols  |     472.51  |     485.21  |     499.01  |
| 5000 trees<br/>31 leaves<br/>1000 cols |     596.097 |     611.307 |     625.896 |

### onnx#5

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |      7.51   |      7.6    |      8.88   |
| 10 trees<br/>31 leaves<br/>100 cols    |      7.5998 |      7.6798 |      8.8698 |
| 10 trees<br/>31 leaves<br/>1000 cols   |      7.59   |      7.6901 |      8.91   |
| 100 trees<br/>31 leaves<br/>10 cols    |     12.85   |     13.09   |     14.6201 |
| 100 trees<br/>31 leaves<br/>100 cols   |     12.9402 |     13.6202 |     14.7802 |
| 100 trees<br/>31 leaves<br/>1000 cols  |     11.8401 |     12.09   |     13.4901 |
| 1000 trees<br/>31 leaves<br/>10 cols   |     18.0601 |     19.2001 |     21.3902 |
| 1000 trees<br/>31 leaves<br/>100 cols  |     20.1093 |     21.0993 |     23.7093 |
| 1000 trees<br/>31 leaves<br/>1000 cols |     19.6325 |     21.1828 |     23.7534 |
| 5000 trees<br/>31 leaves<br/>10 cols   |     43.3894 |     45.2993 |     47.5894 |
| 5000 trees<br/>31 leaves<br/>100 cols  |    104.27   |    111.281  |    118.342  |
| 5000 trees<br/>31 leaves<br/>1000 cols |     94.5217 |     97.9918 |    101.332  |

### onnx_multithreaded#7

| inferencing task config                |   p50_usecs |   p90_usecs |   p99_usecs |
|:---------------------------------------|------------:|------------:|------------:|
| 10 trees<br/>31 leaves<br/>10 cols     |     21.8309 |     22.6609 |     24.3009 |
| 10 trees<br/>31 leaves<br/>100 cols    |     21.985  |     23.721  |     45.6108 |
| 10 trees<br/>31 leaves<br/>1000 cols   |     22.2599 |     23.7498 |     25.5398 |
| 100 trees<br/>31 leaves<br/>10 cols    |     26.3017 |     29.4928 |     42.4632 |
| 100 trees<br/>31 leaves<br/>100 cols   |     25.7001 |     28.0201 |     42.1234 |
| 100 trees<br/>31 leaves<br/>1000 cols  |     27.34   |     29.691  |     38.3805 |
| 1000 trees<br/>31 leaves<br/>10 cols   |     36.7701 |     40.1901 |    105.035  |
| 1000 trees<br/>31 leaves<br/>100 cols  |     36.8403 |     39.6914 |    109.843  |
| 1000 trees<br/>31 leaves<br/>1000 cols |     33.7296 |     36.1596 |     68.4439 |
| 5000 trees<br/>31 leaves<br/>10 cols   |     72.6305 |     91.5047 |    400.135  |
| 5000 trees<br/>31 leaves<br/>100 cols  |    122.421  |    173.977  |    828.446  |
| 5000 trees<br/>31 leaves<br/>1000 cols |    101.62   |    130.733  |    732.035  |