# Reporting for the benchmark

This document details a proposed framework to report values (parameters, metrics) so they can be compared or aggregated during the benchmark analysis.

This reporting is facilitated by a library under `/src/common/metrics.py`. This page first introduces the specifications of the reporting for each benchmark script, then documents the common library functions to implement this reporting.

## Specifications of reporting for each script

For all scripts, we'd like to have a minimal set of typical properties, parameters and metrics that each script will report. See `/src/scripts/lightgbm_python/train.py` for an example implementation of all of those.

The following tables details each reporting entry, with its type and description.

### Common properties

| Entry | Type | Description |
| :-- | :-- | :-- |
| `task` | property | the task of the script, picked in  ` ['generate', 'train', 'score']` |
| `framework` | property | an identifier for the framework in the benchmark (ex: `lightgbm_python`, `treelite`). |
| `framework_version` | property | the version of the framework (ex: `"3.2.1"`). |

To learn how to report properties, see common library below.

### Common metrics

| Entry | Type | Description |
| :-- | :-- | :-- |
| `data_loading` | metric | loading data before executing the task |
| `data_generation` | metric | generating synthetic data |
| `training` | metric | wall time for training |
| `inferencing` | metric | wall time for inferencing |

To learn how to report metrics, see common libary below.

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

### 3. Add any parameters

Any keyword arg of log_parameters() is submitted as a parameter in mlflow.

```python
metrics_logger.log_parameters(**lgbm_params)
```

#### 4. Compute wall-time using with statement

To compute wall time, the `MetricsLogger` class provide a helper method you can use within a `with` statement:

```python
with metrics_logger.log_time_block("training"):
    # anything within this code block will count in wall time
    booster = lightgbm.train(
        lgbm_params,
        train_data,
        valid_sets = val_data
    )

# anything outside of that will not count
```

This will record a metric `"data_loading"` measuring the time spent for the execution of this code block (only).