# Run inferencing benchmark pipeline and analyze results

**Objectives** - By following this tutorial, you will be able to:

- run lightgbm inferencing on multiple datasets with multiple configuration variants

**Requirements** - To enjoy this tutorial, you need to:

- have an existing [AzureML workspace with relevant compute resource](azure-setup.md).
- have [setup your local environment](local-setup.md) to run our benchmarking pipelines.
- if you plan to use your own data, you'll need to [upload custom inferencing data and model](upload-your-data.md) into your AzureML workspace
- if you plan to use synthetic data, you should first [run the data generation pipeline](generate-synthetic-data.md) and the [training benchmark pipeline](benchmark-training.md)

## Check out the inferencing configuration

Open the file under `pipelines/azureml/conf/experiments/lightgbm-inferencing.yaml`. It contains in particular a section `lightgbm_inferencing` that we will look more closely in this section.

The `lightgbm_inferencing:` yaml section contains the parameters to run in parallel several inferencing variants on a given set of inferencing data and models.

```yaml
lightgbm_inferencing:
  # name of your particular benchmark
  benchmark_name: "lightgbm-inferencing-dev" # override this with a unique name

  # list all the data/model pairs to run inferencing with
  tasks:
    - data:
        name: "data-synthetic-regression-100cols-10000samples-inference"
      model:
        name: "model-synthetic-regression-100cols-10trees-31leaves"

  # list all inferencing frameworks and their builds
  variants:
    - framework: lightgbm_python
    - framework: lightgbm_python
      build: docker/lightgbm-v3.2.1/linux_cpu_mpi_build.dockerfile # relative to repo root
    - framework: lightgbm_python
      build: docker/lightgbm-custom/v321_patch_cpu_mpi_build.dockerfile # relative to repo root
    - framework: lightgbm_cli
    - framework: lightgbm_c_api
    - framework: treelite_python
```

### List of inferencing tasks

The configuration consists in listing `tasks` which are made of pairs of `data` and `model`. You have two possibilities here:

!!! note
    This actually corresponds to dataclasses `inferencing_task` and `data_input_spec` detailed in [src/common/tasks.py](../../references/common/tasks.md).

#### Option 1 : use a named registered dataset

If your data and model have been uploaded (or generated) and registered as datasets in AzureML, you can provide the name for each :

```yaml
    - data:
        name: "data-synthetic-regression-100cols-10000samples-inference"
      model:
        name: "model-synthetic-regression-100cols-10trees-31leaves"
```

Datasets have versions in AzureML. You can specify which version you want to use under `data` by adding a `version:` field.

#### Option 2 : use a datastore + path

If your data or model is stored in an external storage, attached to your workspace as a datastore, you can provide the path to the data (works with model as well) :

```yaml
    - data:
        datastore: "externalstorage"
        path: "/custom/test/inferencing/data.txt"
      model:
        datastore: "externalstorage"
        path: "/custom/test/model/model100trees.txt"
```

### List of inferencing tasks

For each task, the pipeline will run inferencing "variants", which are different frameworks you can compare.

!!! note
    This actually corresponds to dataclass `inferencing_variants` detailed in [src/common/tasks.py](../../references/common/tasks.md).

```yaml
  # list all inferencing frameworks and their builds
  variants:
    - framework: lightgbm_python
    - framework: lightgbm_python
      build: docker/lightgbm-v3.2.1/linux_cpu_mpi_build.dockerfile # relative to repo root
    - framework: lightgbm_python
      build: docker/lightgbm-custom/v321_patch_cpu_mpi_build.dockerfile # relative to repo root
    - framework: lightgbm_cli
    - framework: lightgbm_c_api
    - framework: treelite_python
```

We only support a limited set of frameworks. Each framework should correspond to a folder under `src/scripts/inferencing/`. For those of those framework, the pipeline will automatically add model transformation steps (ex: treelite needs to pre-compile the model).

For each framework, you can override the build by specifying a dockerfile under the `build:` field.

## Run the pipeline

!!! warning
    To execute, run from the `pipelines/azureml/` subdirectory.  
    For this section, we'll use `myaml` as the name for the AzureML reference config files you created during [local setup](local-setup.md).

### Running a partial benchmark for testing

Running the pipeline consists in launching a python script with the pipeline configuration file.

```bash
python pipelines/lightgbm_inferencing.py --config-dir ./conf --config-name experiments/lightgbm-inferencing run.submit=True aml=myaml compute=myaml
```

The python script will build a pipeline based on the collection of manual scripts, each running in its own python environment. The configuration for the parameters from each scripts will be provided from the configuration file in `conf/experiments/lightgbm-inferencing.yaml`.

Running the python command should open a browser to your workspace opening the experiment view.

### Running the FULL benchmark pipeline

The configuration we explored above is a reduction of the full benchmark configuration that you will find under `pipelines/azureml/conf/experiments/benchmarks/lightgbm-inferencing.yaml`. **IMPORTANT**: notice the change of path here, under `benchmarks/`.

The list in this benchmark config file corresponds to the default settings used when running the [data generation pipeline](generate-synthetic-data.md) and the [training benchmark pipeline](benchmark-training.md).

```bash
python pipelines/lightgbm_inferencing.py --config-dir ./conf --config-name experiments/benchmarks/lightgbm-inferencing run.submit=True aml=myaml compute=myaml
```

!!! important
    In the configuration file, set a distinct unique `benchmark_name` to be able to analyze the results of **your** run distinctly.

## Analyze the results of a full run

WORK IN PROGRESS
