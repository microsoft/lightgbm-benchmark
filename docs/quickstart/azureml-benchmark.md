# Run benchmark pipeline on AzureML

**Objectives** - By following this tutorial, you will be able to:

- run lightgbm inferencing benchmark pipeline in AzureML

**Requirements** - To enjoy this tutorial, you need to have an existing AzureML workspace (see [instructions here](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources) to create one).

## Configure you AzureML workspace references

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

4. Edit this file to match with the name of your compute targets in AzureML:

    ```yaml
    # @package _group_
    default_compute_target: cpu-cluster # name of default target
    linux_cpu_dc_target: cpu-cluster
    linux_cpu_prod_target: cpu-cluster
    linux_gpu_dc_target: cpu-cluster
    linux_gpu_prod_target: cpu-cluster
    linux_input_mode: "mount"
    linux_output_mode: "mount"
    windows_cpu_dc_target: "cpu-win-cluster"
    windows_cpu_prod_target: "cpu-win-cluster"
    compliant_datastore: "workspaceblobstore" # where intermediary output is written
    noncompliant_datastore: "workspaceblobstore"
    ```

Now, when you'll want to run a pipeline, use arguments `aml=myaml compute=myaml` to run it in your AzureML workspace.

## Run a pipeline

We are using [Shrike](https://github.com/Azure/shrike/tree/main/shrike) to build and submit our pipelines. You can find more documentation on the arguments to configure a pipeline run from the command line in the [Shrike docs](https://azure.github.io/shrike/pipeline/configure-aml-pipeline/).

Running a pipeline consists in launching a python script with a pipeline configuration file.

> :**IMPORTANT**: to execute, run from the `pipelines/azureml/` subdirectory

For instance, when you run:
```bash
python pipelines/lightgbm_e2e_synthetic.py --config-dir ./conf --config-name experiments/benchmark-e2e-synthetic run.submit=True
```

The python script will build a pipeline based on the collection of manual scripts, each running in its own python environment. The configuration for the parameters from each scripts will be provided from the configuration file in `conf/experiments/benchmark-e2e-synthetic.yaml`.

```yaml
{!./pipelines/azureml/conf/experiments/benchmark-e2e-synthetic.yaml!}
```

To modify the behavior of the pipeline, modify the parameters in this file accordingly.