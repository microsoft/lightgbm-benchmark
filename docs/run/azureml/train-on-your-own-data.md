# Run LightGBM Training on your own data in AzureML

**Objectives** - By following this tutorial, you will be able to:

- run lightgbm training pipeline on your own train/test data in AzureML

**Requirements** - To enjoy this tutorial, you need to:
- have installed the [local python requirements](../install.md).
- have an existing [AzureML workspace with relevant compute resource](setup.md).
- have [edited your config files](setup.md) to run the pipelines in your workspace.

## Get your data into AzureML

There are two ways you could simply get your data into your AzureML workspace.

### Option A: Upload your local data into AzureML

### Option B: Create a dataset from an existing storage

## Run training on your train/test datasets

> NOTE: We are using [Shrike](https://github.com/Azure/shrike/tree/main/shrike) to build and submit our pipelines. You can find more documentation on the arguments to configure a pipeline run from the command line in the [Shrike docs](https://azure.github.io/shrike/pipeline/configure-aml-pipeline/).

1. Go into the subdirectory `pipelines/azureml/`

2. Check out the file `conf/experiments/lightgbm_training/cpu.yaml (see below)

    ```yaml
    {!./pipelines/azureml/conf/experiments/lightgbm_training/cpu.yaml!}
    ```

3. Modify the lines below to reflect the name of your input train/test datasets:

    ```yaml
    train_dataset: "NAME OF YOUR TRAINING DATASET HERE"
    test_dataset: "NAME OF YOUR TESTING DATASET HERE"
    ```

4. If you want the pipeline to save your model as a dataset, uncomment the line below and name the output accordingly:

    ```yaml
    training_register_model_as: "my-custom-lightgbm-model"
    ```

    > NOTE: you can decide to register the output of the pipeline later manually from the AzureML portal.

5. Run the training pipeline:

    ```bash
    python pipelines/lightgbm_training.py --config-dir ./conf --config-name experiments/lightgbm_training/cpu run.submit=True aml=myaml compute=myaml
    ```

That's it.

## Options to modify the training parameters

The benchmark training pipeline is entirely configurable. There are a few key parameters in the config yaml file that will provide interesting training scenarios. We've provided a couple of typical setups in distinct config files. Feel free to explore all of them and come up with your own set of parameters.

### Scalable multi node training using mpi

> Check out example config file `conf/experiments/lightgbm_training/cpu.yaml`.

To enable multi-node training, simple modify the number of nodes under:

```yaml
lightgbm_training:
    nodes: 1
```

When running the pipeline, it will automatically partition the data to match with the number of nodes, and create multi-node training provisioning the required number of nodes.

```bash
python pipelines/lightgbm_training.py --config-dir ./conf --config-name experiments/lightgbm_training/cpu run.submit=True aml=myaml compute=myaml
```

### Gpu training (experimental)

> Check out example config file `conf/experiments/lightgbm_training/gpu.yaml`.

To enable gpu training, modify the options below:

```yaml
lightgbm_training:
    device_type: "gpu"
```

When running the pipeline, it will automatically run on the gpu cluster you've named in your `compute/myaml.yaml` file.

```bash
python pipelines/lightgbm_training.py --config-dir ./conf --config-name experiments/lightgbm_training/gpu run.submit=True aml=myaml compute=myaml
```

### Running a custom lightgbm build (experimental)

> Check out example config file `conf/experiments/lightgbm_training/cpu-custom.yaml`.

To enable gpu training, modify the options below:

```yaml
lightgbm_training:
    override_docker: "../../../../../src/scripts/lightgbm_python/dockers/lightgbm_cpu_mpi_custom.dockerfile"
```

When running the pipeline, it will build the container from this custom dockerfile and use it to run your job.

```bash
python pipelines/lightgbm_training.py --config-dir ./conf --config-name experiments/lightgbm_training/cpu-custom run.submit=True aml=myaml compute=myaml
```

### Hyperarameter search using Sweep

AzureML has a feature to [tune model hyperparameters](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/tune-model-hyperparameters), we've implemented it in this training pipeline.

> Check out example config file `conf/experiments/lightgbm_training_sweep/cpu.yaml`.

To enable parameter sweep, just change the "sweepable" parameters (see below) to use the [syntax allowed by AzureML sweep](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#define-the-search-space):

```yaml
lightgbm_training:
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
python pipelines/lightgbm_training.py --config-dir ./conf --config-name experiments/lightgbm_training/sweep run.submit=True aml=myaml compute=myaml
```

You can also modify the parameters of Sweep itself, see [documentation on the role of each of those settings](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters):

```yaml
lightgbm_training:
    # SWEEP
    sweep_algorithm: "random"
    sweep_goal: "minimize"
    sweep_max_total_trials: 10
    sweep_max_concurrent_trials: 10
    sweep_timeout_minutes: 60
```
