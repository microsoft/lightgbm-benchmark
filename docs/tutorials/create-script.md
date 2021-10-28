# Tutorial: Create a script for the benchmark project

**Motivations** - The scripts of the lightgbm-benchmark rely on a helper class that is structuring and standardizing the behavior of each script (init/close of MLFlow, logging system metrics and properties, etc).

## A. Create a python script

Copy the folder under `src/scripts/sample` into another folder `src/scripts/my_component/`, and start editing `run.py`.

### Class constructor

1. Rename class `SampleClassScript` to something meaningful, ex: `MyComponentScript`.

2. In the class constructor, make sure to modify `task`, `framework` and `framework_version` to match with your script.

    ```python
    class SampleClassScript(RunnableScript):
        def __init__(self):
            super().__init__(
                task = "train", # train, score, or anything meaningful
                framework = "sample", # name of your framework (ex: lightgbm_python)
                framework_version = "0.0.1" # any version string (ex: lightgbm.__version__)
            )
    ```

    !!! hint
        There are 2 other arguments to this `__init__()` call:  
        - `metrics_prefix (str)` will allow you to add a prefix to all metrics recorded in your script.  
        - `do_not_log_properties (bool)` will prevent properties to be logged.  
          
        Both have their value in MPI scripts, where you might want to add a prefix to the metrics produced by a given node. You will also want to record MLFlow properties only on the main node in order to avoid conflicts.

3. At the end of the script, you'll notice those two functions below. Please leave them here as we'll use them for unit testing later. Just **modify the name of the class** to match with yours.

    ```python
    def get_arg_parser(parser=None):
        """ To ensure compatibility with shrike unit tests """
        return SampleClassScript.get_arg_parser(parser)

    def main(cli_args=None):
        """ To ensure compatibility with shrike unit tests """
        SampleClassScript.main(cli_args)

    if __name__ == "__main__":
        main()
    ```

!!! success
    At this point, you should be able to run `python src/scripts/my_component/run.py -h` and see the default arguments from the sample.

### Script arguments using argparse

Modify the method `get_arg_parser()` to add your own custom arguments. We rely on classic [argparse](https://docs.python.org/3/library/argparse.html) here and you'll find plenty of examples in the other scripts.

!!! warning
    The class `RunnableScript` will already have generic arguments for your script to be compatible with the benchmark pipelines (ex: verbose logging, custom tags, etc). It is important that you add those generic arguments inside your function.

```python
@classmethod
def get_arg_parser(cls, parser=None):
    """Adds component/module arguments to a given argument parser.

    Args:
        parser (argparse.ArgumentParser): an argument parser instance

    Returns:
        ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # important: add generic arguments
    parser = RunnableScript.get_arg_parser(parser)

    # add custom argument parsing definition here

    # be sure to return the parser
    return parser
```

!!! success
    At this point, you should be able to run `python src/scripts/my_component/run.py -h` and see your own arguments in the middle of the generic helper arguments.

### Most important: your `run()` method

Modify the method `run()` to implement your own behavior. This method is given important arguments:
- an `args` namespace as parsed from the command line arguments,
- a `logger` that the helper class initializes for you,
- a `metrics_logger` to log lines and metrics (see [methods of `metrics_logger` in `src/common/metrics.py`](../references/common/metrics.md#src.common.metrics.MetricsLogger)),
- and a list of `unknown_args` (not recognized by parser).

See in the sample script for a couple examples of those methods and how to benefit from them during your run.

```python
def run(self, args, logger, metrics_logger, unknown_args):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script
        logger (logging.Logger): a logger for this script (logging.getLogger() with initialization)
        metrics_logger (common.metrics.MetricLogger)
        unknown_args (list[str]): list of arguments not recognized during argparse
    """
    # record relevant parameters
    metrics_logger.log_parameters(
        num_threads=args.num_threads
    )

    logger.info("Running command {}".format(" ".join(lightgbm_predict_command)))

    with metrics_logger.time_block("example_time_block"):
        print("Hello World!")
```

!!! success
    At this point, you should be able to run your script locally on some sample data, depending on your script goal.

### Test your metrics with MLFlow

List all local experiments

```bash
mlflow experiments --list
```

List your local runs

```bash
mlflow runs list --experiment-id  {{YOUR_EXPERIMENT_ID}}
```

Get details of the run

```bash
mlflow runs describe --run-id {{YOUR_RUN_ID}}
```

```json
{
    "info": {
        "artifact_uri": "file:///somepath",
        "end_time": 1635378123882,
        "experiment_id": "0",
        "lifecycle_stage": "active",
        "run_id": "d8efd2858ebc4ab5a39f15338bebe4b9",
        "run_uuid": "d8efd2858ebc4ab5a39f15338bebe4b9",
        "start_time": 1635378123755,
        "status": "FINISHED",
        "user_id": "username"
    },
    "data": {
        "metrics": {
            "time_data_loading": 0.032999277114868164,
            "time_inferencing": 0.0
        },
        "params": {
            "num_threads": "1"
        },
        "tags": {
            "cpu_count": "8",
            "framework": "treelite_python",
            "framework_version": "1.3.0",
            "machine": "AMD64",
            "mlflow.source.name": "score.py",
            "mlflow.source.type": "LOCAL",
            "mlflow.user": "username",
            "processor": "Intel64 Family 6 Model 94 Stepping 3, GenuineIntel",
            "system": "Windows",
            "system_version": "10.0.19043",
            "task": "score"
        }
    }
}```
