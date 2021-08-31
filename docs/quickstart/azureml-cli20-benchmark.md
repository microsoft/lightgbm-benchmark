# Run benchmark in AzureML (with CLI 2.0)

**Objectives** - By following this tutorial, you will be able to:

- Run the LightGBM benchmark scripts on the AzureML platform (CLI 2.0 edition)

**Requirements** - To enjoy this tutorial, you need to be able to:

- Install Azure CLI, AzureML extension and enable private features (see below).
- Create or access an AzureML workspace (see [instructions](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources).

## Install requirements

1. Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) (version >= 2.27.0)

2. Install [Azure ML extension](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli)

3. Activate the preview features by setting environment variable `AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true` (see [instructions](https://github.com/Azure/azureml-previews/tree/main/previews/pipelines#how-to-get-started)).

4. If you don't have an AzureML workspace yet, create one following [those instructions](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources).

## Prepare for running the benchmark

### A. Set your default azure references

Before you get started, we recommend you to set your Azure CLI on your specific subscription, resource gropu and workspace:

```
az account set --subscription [SUBSCRIPTION]
az configure --defaults group=[RESOURCE GROUP] workspace=[WORKSPACE]
```

### B. Publish the modules in your workspace

```
az ml component create --file ./pipelines/azureml_cli20/components/lightgbm_python_score.yml
```

### C. Create the datasets

### Option 1: upload manually using AzureML UI

### Option 2: upload manually using CLI

```bash
az ml data create --file ./pipelines/azureml_cli/inference_data.yml
```

### Option 3: generate in AzureML

(Work in progress)

## Run the benchmark

```
az ml job create --file ./pipelines/azureml_cli20/pipelines/score_ab.yml --web
```
