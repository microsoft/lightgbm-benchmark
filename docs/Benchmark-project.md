# LightGBM benchmark project: goals and methodology

The LightGBM benchmark aims at providing tools and automation to compare implementations of lightgbm and other boosting-tree-based algorithms for both training and inferencing. The focus is on production use cases, and the evaluation on both model quality (validation metrics) and computing performance (training speed, compute hours, inferencing latency, etc). We'll focus on LightGBM first, and expand it to other learners as we go along.

The goal is to support the community of developers of LightGBM by providing tools and a methodology for evaluating new releases of LightGBM on a standard and reproducible benchmark.

## Implementation Goals

In addition of the description above, the code provided in this repo follows three goals, in the decreasing order of priority:

1. **Automation of routines** - The code should provide means to automate usual benchmark routines such as reporting metrics, orchestrating the benchmark on multiple data sources and multiple configurations, comparing/analyzing multiple runs, etc.

2. **Reproducible and self-service** - The benchmark code should allow users to reproduce the results on their end, or adapt it to another environment to run the full benchmark on their data.

3. **Act as reference** - The scripts provided in the benchmark could be used as reference / standard implementations in python for training and inferencing. The code should be readable, documented and unit tested. It should make the best and up-to-date use of the python libraries it proposes to evaluate.

## Running the benchmark

We propose the benchmark to be available both for manual or orchestrated (cloud-based) run.

- **Manual** - Users can reproduce the benchmark results while running the code locally to a given VM.

- **Orchestrated** - Users can run the benchmark in a cloud-based ML orchestration platform (ex: AzureML).

We also want to identify how to enable users to run the benchmark on custom lightgbm builds (either local or from a github branch), so that they can assess the performance of their changes, or use different compilation options.

## Data and tasks

The users should be able to run the benchmark on multiple data sources. Not only we want to provide synthetic and public data benchmarks, but we also propose to make sure users can use the benchmark on their on data sets. This can be facilitated by using a cloud-based ML platform like AzureML.

- **Synthetic Data** - Generated fake data, in particular to test scalability, inferencing latency.

- **Public Data** - Example public datasets used in the community and literature to benchmark the learners.

- **Production Data** - Provide the ability for users to run the benchmark on their own data.

## Code Structure

To allow for modularity and standardization, we propose to structure the benchmark into a collection of independent python scripts (specs in a separate document). Each script will have its own environment, inputs, outputs and parameters. All scripts will eventually draw from common helper classes to support implementation routines (reporting, argument parsing, etc).

## Benchmarking scenarios and reporting

The benchmark could be use in the following comparison scenarios:

- **Training framework comparison** - Compare lightgbm versus other training (or inferencing) ML framework. This comparison will be mainly based on model quality, but also training computing performance. It will run those framework on typical public datasets.
- **Lightgbm performance and scalability** - Compare lightgbm (stable or custom) to itself while changing compiling options, VMs and environments. This will likely use synthetic data or large public datasets. This is where allowing users to run on their own production data makes more sense.
- **Lightgbm "master" vs lightgbm "custom"** - Compare any new implementation of lightgbm against the current stable release. This comparison will ensure model quality is equivalent, and will focus on compute performance.

For each of those scenarios, we'll need to report both the variables subject to change (framework, lightgbm version, VMs, dependencies, etc) and the metrics produces during training or inferencing. A list of those variables and metrics is provided in the [Reporting Guide](Reporting-guide.md).

## Phases and Milestones

We propose the following process to iterate on implementation.

### 1. Foundations phase

In this first phase, we'll implement the foundations to allow the benchmark to grow later. We'll start with:
- a couple minimal scripts for generating synthetic data, training lightgbm and inferencing
- some common libraries around metrics reporting
- a first benchmark pipeline in either AzureML SDK1.5 or SDK2.0 to run end-to-end and report metrics
- enabling both manual and orchestrated runs of the benchmark
- a build with a first set of unit tests
- documentation of the goals, developer guide and common routines

**Milestone**: an end-to-end run of the benchmark both locally (VM) and in the cloud, reporting a minimal set of wall time metrics, running on synthetic data, producing numbers we manually report in the repo as markdown.

### 2. Standardization phase

This phase will see the implementation of features to automate or speed-up typical user routines. In particular, we anticipate supporting the following use cases:
- running the benchmark on custom lightgbm builds and/or branches
- analyzing the benchmark results in a systematic script for crunching numbers and compile a readable report
- a nice report format for comparing pairs of lightgbm implementations (ex: master vs branch), or multiple implementations against one.
- systematic testing of every script in a consistent unit test methodology
- identifying ways to standardize code to avoid duplication

**Milestone**: an end-to-end benchmark run producing report automatically, comparing pairs of variants of lightgbm (ex: stable vs custom build).

### 3. Expansion phase

In this phase, we will expand the number of benchmark scripts and pipelines to cover the following:
- implementation of multi-node lightgbm training and inferencing
- implementation of xgboost, catboost, and others
- implementation of more public datasets, data processors

**Milestone**: an end-to-end benchmark pipeline running on 3+ public datasets, comparing 3+ tree-based frameworks.

## Appendix

### Example existing benchmarks

- https://github.com/catboost/benchmarks
- https://github.com/guolinke/boosting_tree_benchmarks
- https://sites.google.com/view/lauraepp/benchmarks/xgb-vs-lgb-feb-2017
- https://github.com/Laurae2/gbt_benchmarks
