# Run a sample benchmark pipeline on AzureML

**Objectives** - By following this tutorial, you will be able to setup resources in Azure to be able to run the pipelines in this repo.:

**Requirements** - To enjoy this tutorial, you need to have a working Azure account and subscription (see [how to create one]()).

## A. Create an AzureML workspace

1. If you don't have one already, [create an AzureML workspace](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources).

2. In that workspace, you will need to [create compute clusters](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#cluster). Here's what we recommend to get you started with LightGBM training using the pipelines of this repo. The names below are indicative, but we'll keep refering to those in our docs.

    | Cluster Name | SKU | Node count | Description |
    | :-- | :-- | :-- | :-- |
    | `cpu-cluster` | Standard_DS3_v2 | 4 | A cluster for simple jobs, running on cheap VMs. |
    | `linux-cpu-d32sv3` | Standard_D32s_v3 | 10 | A cluster for LightGBM itself, with a more powerful yet affordable VM. Feel free to provision more or less. |
    | `linux-gpu-nv6` | Standard_NV6 (gpu) | 1 | Optional: for trying out gpu lightgbm training (work in progress) |

    IMPORTANT: Whenever you create those, set the minimum number of nodes to 0 so that unused clusters will automatically size down and reduce costs.

## B. Edit config files to point to your AzureML workspace

To be able to submit the benchmark pipelines in AzureML, you need to edit some configuration files with the right references to connect to your AzureML resources.

1. Copy file `/pipelines/azureml/conf/aml/sample.yaml` into a file of your own in the same directory (ex: `myaml.yaml`).

2. Edit this file to match with your AzureML workspace references:

    ```yaml
    # @package _group_
    subscription_id: TODO
    resource_group: TODO
    workspace_name: TODO
    tenant: TODO
    auth: "interactive"
    ```

3. Copy file `/pipelines/azureml/conf/compute/sample.yaml` into a file of your own in the same directory (ex: `myaml.yaml`).

4. Edit this file to match with the name of your compute targets in AzureML. Check below for reference. If you haven't created a gpu cluster, you can leave the config file as is for the gpu lines.

    ```yaml
    # @package _group_
    default_compute_target: cpu-cluster # name of default target
    linux_cpu_dc_target: linux-cpu-d32sv3
    linux_cpu_prod_target: linux-cpu-d32sv3
    linux_gpu_dc_target: linux-gpu-nv6
    linux_gpu_prod_target: linux-gpu-nv6
    linux_input_mode: "mount"
    linux_output_mode: "mount"
    windows_cpu_dc_target: "windows-cpu-d32sv3" # (work in progress)
    windows_cpu_prod_target: "windows-cpu-d32sv3" # (work in progress)
    compliant_datastore: "workspaceblobstore"
    noncompliant_datastore: "workspaceblobstore"
    ```

From now on, when you'll want to run a pipeline, use arguments `aml=myaml compute=myaml` to run it in your AzureML workspace.

## Verify your setup: run a sample pipeline in your workspace

> We are using [Shrike](https://github.com/Azure/shrike/tree/main/shrike) to build and submit our pipelines. You can find more documentation on the arguments to configure a pipeline run from the command line in the [Shrike docs](https://azure.github.io/shrike/pipeline/configure-aml-pipeline/).

Running a pipeline consists in launching a python script with a pipeline configuration file.

> :**IMPORTANT**: to execute, run from the `pipelines/azureml/` subdirectory

For instance, when you run:
```bash
python pipelines/data_generation.py --config-dir ./conf --config-name experiments/data-generation run.submit=True aml=myaml compute=myaml
```

The python script will build a pipeline based on the collection of manual scripts, each running in its own python environment. The configuration for the parameters from each scripts will be provided from the configuration file in `conf/experiments/data-generation.yaml`.

```yaml
{!./pipelines/azureml/conf/experiments/data-generation.yaml!}
```

Running the python command should open a browser to your workspace opening the experiment view:

![](../../../.media/data_generation_pipeline.png)

> In the future, if you don't want to add `aml=myaml compute=myaml` every single time, you can modify the `defaults` section of the yaml config:
>
> ```yaml
> defaults:
>   - aml: myaml
>   - compute: myaml
>   - modules: benchmark
> ```
