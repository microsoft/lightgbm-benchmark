# Upload data sources into AzureML to run the benchmark

**Objectives** - By following this tutorial, you will be able to:

- upload sample or custom data into AzureML
- have a train/test dataset ready to run a LightGBM training

**Requirements** - To enjoy this tutorial, you need to:
- have an existing [AzureML workspace with relevant compute resource](azure-setup.md).
- have installed the [az ml cli](../install.md) (python and az ml cli).

## Get data into AzureML

There are multiple ways to get your data into your AzureML workspace. Here's a couple:

- Option A: use the `az ml cli` to upload files from the commandline
- Option B: use the AzureML UI to [upload your local data](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui)
- Option C: use the AzurEML UI to [create a dataset from an existing storage](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui)

Options B and C are documented in the AzureML documentation (links above).

We'll show option A in the following, as **we provide some yaml templates** to upload standard datasets into your provisioned AzureML workspace for running our benchmark.

!!! warning
    The `data/` folder of our repository has been added to `.gitignore` to avoid uploading your own data in git. But **please be careful** when adding your own data into the repository folder to not commit is mistakenly.

## Upload a sample dataset using `az ml cli`

Our repo has a minimal set of sample data we use for unit testing. We'll demo how to add those to your workspace as a way to show how to upload your own files using the command line.

In a terminal:

1\. If you haven't already, connect by typing

```bash
az login
```

To avoid having to add your workspace/resource group every time, set those as defaults:

```bash
az account set --subscription <subscription ID>
az configure --defaults workspace=<Azure Machine Learning workspace name> group=<resource group>
```

2\. From the repository root, type:

```bash
# to upload dummy train dataset
az ml dataset create --file data/sample/unittests-regression-train.yml

# to upload dummy test dataset
az ml dataset create --file data/sample/unittests-regression-test.yml
```

This will use the sample config file below to create a dataset uploading the file specified in `local_path` into your workspace.

``` yaml
{!./data/sample/unittests-regression-train.yml!}
```

3\. To find it in the AzureML UI, get into your workspace under the **Datasets** tab.

You'll now be able to consume this data as an input of [lightgbm training](train-on-your-data.md) or inferencing pipelines.

Feel free to edit this sample file to upload your own data into AzureML from local files and folders.

## Upload standard benchmark datasets into AzureML

Work in progress, feel free to contribute to the [discussion on this topic in the github repo](https://github.com/microsoft/lightgbm-benchmark/discussions/131).