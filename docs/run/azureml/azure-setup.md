# Provision an AzureML workspace to run the LightGBM benchmark

**Objectives** - By following this tutorial, you will be able to setup the Azure resources you need to run the pipelines in this repo.

**Requirements** - To enjoy this tutorial, you need to have a working Azure account and subscription (see [how to create one](https://azure.microsoft.com/en-us/free/), or how to get [Visual Studio Enterprise benefits](https://docs.microsoft.com/en-us/visualstudio/subscriptions/vs-azure)).

## Option A. Create an AzureML workspace (manual route)

1. If you don't have one already, [create an AzureML workspace](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources).

2. In that workspace, you will need to [create compute clusters](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#cluster). Here's what we recommend to get you started with LightGBM training using the pipelines of this repo. The names below are indicative, but we'll keep refering to those in our docs.

    | Cluster Name | SKU | Node count | Description |
    | :-- | :-- | :-- | :-- |
    | `cpu-cluster` | Standard_DS3_v2 | 4 | A cluster for simple jobs, running on cheap VMs. |
    | `linux-cpu-d32sv3` | Standard_D32s_v3 | 10 | A cluster for LightGBM itself, with a more powerful yet affordable VM. Feel free to provision more or less. |
    | `linux-gpu-nv6` | Standard_NV6 (gpu) | 1 | Optional: for trying out gpu lightgbm training (work in progress) |

    IMPORTANT: Whenever you create those, set the minimum number of nodes to 0 so that unused clusters will automatically size down and reduce costs.

## Option B. Create an AzureML workspace for LightGBM using an ARM template

Work in progress, feel free to contribute to the [discussion on this topic in the github repo](https://github.com/microsoft/lightgbm-benchmark/discussions/133).
