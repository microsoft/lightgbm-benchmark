# Reporting for the benchmark

This document details a proposed framework to report values (parameters, metrics) so they can be compared or aggregated during the benchmark analysis.

This reporting is facilitated by a library under `/src/common/metrics.py`. This page first introduces the specifications of the reporting for each benchmark script, then documents the common library functions to implement this reporting.

## Specifications of reporting

As mentioned in the [project definition](Benchmark-project.md), we'd like to address three benchmarking scenarios:
1. Training framework comparison (lightgbm versus other ML frameworks)
2. Lightgbm performance and scalability (lightgbm on different compute types)
3. Lightgbm "master" vs lightgbm "custom" (measuring progress of lightgbm versions)

In order to do support those, we propose to report 3 kind of content:
- **properties**: used to segment the analysis, they will be properties of the script (framework, version) or properties of the environment (VM types, dependencies, compilation settings, etc).
- **parameters**: in particular for training, any relevant parameter passed to the script (ex: learning rate).
- **metrics**: measures taken during the script, in particular various execution times or custom validation metrics (ex: RMSE).

For all scripts, we'd like to have a minimal set of typical properties, parameters and metrics that each script will report. See `/src/scripts/lightgbm_python/train.py` for an example implementation of all of those.

The following tables details each reporting entry, with their type and description.

### Common properties

The purpose of properties is to let us segment the benchmarking analysis. For instance, comparing different frameworks against one another, or compare two lightgbm versions. Some of those properties can be reported by the scripts themselves (ex: python api version), some others will have to be reported by the orchestrator (ex: VM type on which the script is run).

| Entry | Type | Description |
| :-- | :-- | :-- |
| `task` | property | the task of the script, picked in  ` ['generate', 'train', 'score']` |
| `framework` | property | an identifier for the ML algorithm being benchmarked (ex: `lightgbm_python`, `treelite`). |
| `framework_version` | property | the version of the framework (ex: `"3.2.1"`). |
| `environment` | property | Optional: log relevant dependencies and their version numbers as a dictionary. |

In order to facilitate recording all those, we could add as many system information we could get from [python modules like `platform`](https://www.geeksforgeeks.org/get-your-system-information-using-python-script/).

To learn how to report properties, see common library below.

### Common metrics

The common metrics capture various times that we'll compare accross frameworks. If possible, we'd like the training and inferencing times to be distinct from data loading. If that's not possible, then to not report any data loading time and we'll figure out how to compare those during analysis.

| Entry | Type | Description |
| :-- | :-- | :-- |
| `time_data_loading` | metric | time for loading the data before executing the task |
| `time_data_generation` | metric | time for generating data (for task `generate`) |
| `time_training` | metric | time for training on previously loaded data (for task `training`) |
| `time_inferencing` | metric | time for inferencing on previously loaded data (for task `inferencing`) |

To learn how to implement reporting those metrics, see common libary below.

### Parameters

There's no common parameters yet. You can report anything as parameters. See how below.

## Using common report library

To use the common report library, first import:

```python
from common.metrics import MetricsLogger
```

Then, a typical logging session works as follows.

### 1. Open a session

```python
# initializes reporting of metrics with a session name
metrics_logger = MetricsLogger("lightgbm_python.score")
```

### 2. Add common properties

Make sure to provide the properties expected per specifications above.

```python
# add the common properties to the session
metrics_logger.set_properties(
    task = 'score',
    framework = 'lightgbm_python',
    framework_version = lightgbm.__version__
)
```

You can capture all relevant platform/system info by using helper code function:

```python
# will capture platform info and record as properties
metrics_logger.set_platform_properties()
```

Optionally, you can provide custom properties using json (for instance from CLI arguments), and report those using:

```python
# logger will parse the json
metrics_logger.set_properties_from_json(json_string)
```

### 3. Add any parameters

Any keyword arg of log_parameters() is submitted as a parameter in mlflow.

```python
metrics_logger.log_parameters(**lgbm_params)
```

#### 4. Compute wall-time using with statement

To compute wall time, the `MetricsLogger` class provide a helper method you can use within a `with` statement:

```python
with metrics_logger.log_time_block("time_training"):
    # anything within this code block will count in wall time
    booster = lightgbm.train(
        lgbm_params,
        train_data,
        valid_sets = val_data
    )

# anything outside of that will not count
```

This will record a metric `"time_training"` measuring the time spent for the execution of this code block (only).