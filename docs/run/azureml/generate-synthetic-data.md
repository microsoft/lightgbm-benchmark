# Generate synthetic data for benchmarking

**Objectives** - By following this tutorial, you will be able to:

- generate multiple synthetic regression datasets to benchmark lightgbm training or inferencing

**Requirements** - To enjoy this tutorial, you need to:

- have an existing [AzureML workspace with relevant compute resource](azure-setup.md).
- have [setup your local environment](local-setup.md) to run our benchmarking pipelines.

## Check out the generation configuration

Open the file under `conf/experiments/data-generation.yaml`. It contains in particular a section `data_generation_config` that we will look more closely in this section.

The following yaml section contains the parameters to run a pipeline that will automatically generate synthetic data for various tasks, at various sizes.

```yaml
experiment:
  name: "data_generation_dev"

data_generation_config:
  # name of your particular benchmark
  benchmark_name: "benchmark-dev" # override this with a unique name

  # DATA
  tasks:
    - task: "regression"
      train_samples: 100000
      test_samples: 10000
      inferencing_samples: 10000
      n_features: 10
      n_informative: 10
    - task: "regression"
      train_samples: 100000
      test_samples: 10000
      inferencing_samples: 10000
      n_features: 100
      n_informative: 100
    - task: "regression"
      train_samples: 100000
      test_samples: 10000
      inferencing_samples: 10000
      n_features: 1000
      n_informative: 1000

  register_outputs: false
  register_outputs_prefix: "data-synthetic"
```

In particular, the configuration consists in listing `tasks` which are made of key data generation arguments:

```yaml
    - task: <regression or classification>
      train_samples: <number of training rows>
      test_samples: <number of testing rows>
      inferencing_samples: <number of inferencing rows>
      n_features: <number of features>
      n_informative: <how many features are informative>
```

!!! note
    This actually corresponds to a dataclass `data_generation_task` detailed in [src/common/tasks.py](../../references/common/tasks.md).  
    All the items are required except `n_informative`.

The current list corresponds to the default settings required to run the [training benchmark pipeline](benchmark-training.md) and the [inferencing benchmark pipeline](benchmark-inferencing.md).

The option `register_outputs` can be turned to `true` if you want the pipeline to automatically register its outputs with a naming convention `{prefix}-{task}-{n_features}cols-{samples}samples-{train|test|inference}` that will be used in the next steps.

```yaml
  register_outputs: false
  register_outputs_prefix: "data-synthetic"
```

## Run the pipeline

!!! warning
    For this section, we'll use `custom` as the name for the AzureML reference config files you created during [local setup](local-setup.md).

Running the data generation pipeline consists in launching a python script with the pipeline configuration file.

```bash
python pipelines/azureml/pipelines/data_generation.py --exp-config pipelines/azureml/conf/experiments/data-generation.yaml
```

The python script will build a pipeline based on the collection of manual scripts, each running in its own python environment. The configuration for the parameters from each scripts will be provided from the configuration file in `conf/experiments/data-generation.yaml`.

Running the python command should open a browser to your workspace opening the experiment view.

To activate output registration, you can either modify it in `data-generation.yaml`, or override it from the command line:

```bash
python pipelines/azureml/pipelines/data_generation.py --exp-config pipelines/azureml/conf/experiments/data-generation.yaml data_generation.register_outputs=True
```

To find the resulting datasets, go into your workspace under the **Datasets** tab.

## Next Steps

When the pipeline completes and you can see the registered dataset in your workspace, you are now ready to test running the [training benchmark pipeline](benchmark-training.md).
