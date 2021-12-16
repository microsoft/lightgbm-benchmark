# Local Setup: run a sample benchmark pipeline on AzureML

**Objectives** - By following this tutorial, you will be able to setup resources in Azure to be able to run the pipelines in this repo.:

**Requirements** - To enjoy this tutorial, you first need to:
- install the [local python requirements](../install.md).
- provision [Azure resources first](azure-setup.md), and have a working AzureML workspace.

## A. Edit config files to point to your AzureML workspace

To be able to submit the benchmark pipelines in AzureML, you need to edit some configuration files with the right references to connect to your AzureML resources.

1. Edit file under `conf/aml/custom.yaml` to match with your AzureML workspace references:

    ```yaml
    # @package _group_
    subscription_id: TODO
    resource_group: TODO
    workspace_name: TODO
    tenant: TODO
    auth: "interactive"
    ```

2. Edit file under `conf/compute/custom.yaml` to match with the name of your compute targets in AzureML. Check below for reference. If you haven't created a gpu cluster, you can leave the config file as is for the gpu lines.

    ```yaml
    # @package _group_
    linux_cpu: "cpu-cluster"
    linux_gpu: "linux-gpu-nv6"
    windows_cpu: "win-cpu"
    ```

!!! note
    Configs the repo asusme you use `custom` as name to find your aml/compute config. If in the future you have multiple aml/compute configs (ex: `myotheraml.yaml`), when you'll want to run a pipeline, use arguments `aml=myotheraml compute=myotheraml` to override.

## B. Verify your setup: run a sample pipeline in your workspace

Running a pipeline consists in launching a python script with a pipeline configuration file.

For instance, when you run:
```bash
python pipelines/azureml/pipelines/data_generation.py --exp-config pipelines/azureml/conf/experiments/data-generation.yaml
```

The python script will build a pipeline based on the collection of manual scripts, each running in its own python environment. The configuration for the parameters from each scripts will be provided from the configuration file in `conf/experiments/data-generation.yaml`.

```yaml
{!./conf/experiments/data-generation.yaml!}
```

Running the python command should open a browser to your workspace opening the experiment view.
