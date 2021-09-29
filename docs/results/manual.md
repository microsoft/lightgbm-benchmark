# Latest Benchmark results

## STANDARD_DS14_V2

Here's some manual results for [Standard DS14-4_v2](https://docs.microsoft.com/en-us/azure/virtual-machines/dv2-dsv2-series-memory#dsv2-series-11-15) (4 vcpus, 112 GiB memory), Linux (ubuntu 20.04) with Premium SSD LRS.

```
Train data shape: (100000, 4001)
Test data shape: (10000, 4001)
Inference data shape: (100000, 4000)
--- time elapsed: data_generation = 48.887274 s [tags: {'task': 'generate'}]
--- time elapsed: data_saving = 202.063839 s [tags: {'task': 'generate'}]
--- time elapsed: data_loading = 64.472545 s [tags: {'framework': 'lightgbm_python', 'task': 'train', 'lightgbm_version': '3.2.1'}]
--- time elapsed: training = 186.153282 s [tags: {'framework': 'lightgbm_python', 'task': 'train', 'lightgbm_version': '3.2.1'}]
--- time elapsed: data_loading = 174.565443 s [tags: {'framework': 'lightgbm_python', 'task': 'score', 'lightgbm_version': '3.2.1'}]
--- time elapsed: inferencing = 7.100806 s [tags: {'framework': 'lightgbm_python', 'task': 'score', 'lightgbm_version': '3.2.1'}]
```
