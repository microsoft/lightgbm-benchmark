# Run LightGBM Training on your own data in AzureML

**Objectives** - By following this tutorial, you will be able to:

- run lightgbm training pipeline on your own train/test data in AzureML

**Requirements** - To enjoy this tutorial, you need to:
- have installed the [local python requirements](../install.md).
- have an existing [AzureML workspace with relevant compute resource](azure-setup.md).
- have [edited your config files](local-setup.md) to run the pipelines in your workspace.

## Get your data into AzureML

There are two ways you could simply get your data into your AzureML workspace.

- Option A: [Upload your local data into AzureML](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui)
- Option B: [Create a dataset from an existing storage](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui)

For each of those, you need to create a File dataset with your training and testing data, each provided as one unique file.

## Run training on your train/test datasets

1\. Check out the file `conf/experiments/lightgbm_training/cpu.yaml (see below):

``` yaml
{!./conf/experiments/lightgbm_training/cpu.yaml!}
```

2\. Modify the lines below to reflect the name of your input train/test datasets:

```yaml
# list all the train/test pairs to train on
tasks:
  - train:
      name: "NAME OF YOUR TRAINING DATASET HERE"
    test:
      name: "NAME OF YOUR TESTING DATASET HERE"
```

!!! hint
    `tasks` is actually a list, if you provide multiple entries, the pipeline will train one model per train/test pair.

4\. If you want the pipeline to save your model as a dataset, turn `register_model` to True and uncomment the lines below to name the output according to the naming convention:

```yaml
lightgbm_training_config:
  reference:
    # model registration
    # naming convention: "{register_model_prefix}-{task_key}-{num_iterations}trees-{num_leaves}leaves-{register_model_suffix}"
    output:
      register_model: False
      #register_model_prefix: "model"
      #register_model_suffix: null
```

!!! hint
    you can decide to register the output of the pipeline later manually from the AzureML portal.

5\. Run the training pipeline:

```bash
python src/pipelines/azureml/lightgbm_training.py --exp-config conf/experiments/lightgbm_training/cpu.yaml
```

That's it.

## Options to modify the training parameters

The benchmark training pipeline is entirely configurable. There are a few key parameters in the config yaml file that will provide interesting training scenarios. We've provided a couple of typical setups in distinct config files. Feel free to explore all of them and come up with your own set of parameters.

### Scalable multi node training using mpi

!!! hint
    Check out example config file `conf/experiments/lightgbm_training/cpu.yaml`.

To enable multi-node training, simple modify the number of nodes under `runtime`:

```yaml
lightgbm_training_config:
  reference:
    runtime:
      nodes: 1
```

When running the pipeline, it will automatically partition the data to match with the number of nodes, and create multi-node training provisioning the required number of nodes.

```bash
python src/pipelines/azureml/lightgbm_training.py --exp-config conf/experiments/lightgbm_training/cpu.yaml
```

### Gpu training (experimental)

!!! hint
    Check out example config file `conf/experiments/lightgbm_training/gpu.yaml`.

To enable gpu training, modify the options below to build a GPU-ready docker image and turn on gpu in LightGBM training:

```yaml
lightgbm_training_config:
  reference:
    training:
      device_type: "gpu"
    runtime:
      build: "docker/lightgbm-v3.3.3/linux_gpu_pip.dockerfile"
```

When running the pipeline, it will automatically run on the gpu cluster you've named in your `compute/custom.yaml` file.

```bash
python src/pipelines/azureml/lightgbm_training.py --exp-config conf/experiments/lightgbm_training/gpu.yaml
```

### Running a custom lightgbm build (experimental)

!!! hint
    Check out example config file `conf/experiments/lightgbm_training/cpu-custom.yaml`.

To enable training on a custom build, modify the options below:

```yaml
lightgbm_training_config:
  reference:
    runtime:
      build: "dockers/lightgbm_cpu_mpi_custom.dockerfile" # relative to lightgbm_python folder
```

When running the pipeline, it will build the container from this custom dockerfile and use it to run your job.

```bash
python src/pipelines/azureml/lightgbm_training.py --exp-config conf/experiments/lightgbm_training/cpu-custom.yaml
```

### Hyperarameter search using Sweep

AzureML has a feature to [tune model hyperparameters](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/tune-model-hyperparameters), we've implemented it in this training pipeline.

!!! hint
    Check out example config file `conf/experiments/lightgbm_training/sweep.yaml`.

To enable parameter sweep, just change the "sweepable" parameters (see below) to use the [syntax allowed by AzureML sweep](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#define-the-search-space):

```yaml
lightgbm_training_config:
  reference:
    training:
      # "sweepable" training parameters
      num_iterations: "choice(100, 200)"
      num_leaves: "choice(10,20,30)"
      min_data_in_leaf: 20
      learning_rate: 0.1
      max_bin: 255
      feature_fraction: 1.0
```

Running the pipeline with this config will automatically try multiple values for the parameters and return the best model.

```bash
python src/pipelines/azureml/lightgbm_training.py --exp-config conf/experiments/lightgbm_training/sweep.yaml
```

You can also modify the parameters of Sweep itself, see [documentation on the role of each of those settings](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters):

```yaml
lightgbm_training_config:
  reference:
    sweep:
        #primary_metric: "node_0/valid_0.rmse" # if you comment it out, will use "node_0/valid_0.METRIC"
        goal: "minimize"
        algorithm: "random"
        early_termination:
          policy_type: "median_stopping"
          evaluation_interval: 1
          delay_evaluation: 5
          truncation_percentage: 20
        limits:
          max_total_trials: 100
          max_concurrent_trials: 10
          timeout_minutes: 60
```

## Running multiple variants of training parameters

The training pipeline allows you do benchmark multiple variants of the training parameters.

The structure of `lightgbm_training_config` settings relies on 3 main sections:
- `tasks` : a list of train/test dataset pairs
- `reference_training`: parameters used as reference for lightgbm training
- `variants`: a list of parameter overrides that apply on top of `reference_training` parameters.

So you can create as many tasks and variants as you'd like and run them all into one single pipeline.

An example use case is training on cpu versus gpu. See the example file [training-cpu-vs-gpu.yaml](https://github.com/microsoft/lightgbm-benchmark/tree/main/conf/experiments/benchmarks/training-cpu-num-trees.yaml). In this file, the variant just consists in overriding the `num_iterations`:

```yaml
lightgbm_training_config:
  benchmark_name: "benchmark-cpu-num-trees"

  # list all the train/test pairs to train on
  tasks:
    - train:
        name: "data-synthetic-regression-10cols-100000samples-train"
      test::
        name: "data-synthetic-regression-10cols-10000samples-test"
      task_key: "synthetic-regression-10cols" # optional, user to register outputs
    - train:
        name: "data-synthetic-regression-100cols-100000samples-train"
      test::
        name: "data-synthetic-regression-100cols-10000samples-test"
      task_key: "synthetic-regression-100cols" # optional, user to register outputs
    - train:
        name: "data-synthetic-regression-1000cols-100000samples-train"
      test::
        name: "data-synthetic-regression-1000cols-10000samples-test"
      task_key: "synthetic-regression-1000cols" # optional, user to register outputs

  # reference settings for the benchmark
  # all variants will be based on this
  reference:
    # lots of other params here
    training:
      num_iterations: 100

  # variant settings override what is in reference_training
  variants:
    - training:
        num_iterations: 10
    - training:
        num_iterations: 1000
    - training:
        num_iterations: 5000
```
