# Developer guide

The following document details the proposed structure for this repo, the benchmark scripts and libraries.

!!! hint
    For a quickstart: please check out `/src/scripts/sample/sample.py` for a full sample python script implementing the recommendations below.

## General design and motivations

The goal of the benchmark repo is to provide scripts to compare tree-based machine learning framework on similar tasks and environment.

The benchmark is made of a collection of scripts. These scripts can be run locally, or manually from within a remote VM. But these scripts will eventually be submitted as a pipeline or workflow chaining scripts in a sequence: data generation, data processing, training, inferencing.

In order to do that, we need each step for benchmarking a given framework to be implemented as a single script with clear inputs, outputs and parameters. We organize those into sub-directories under `/src/scripts/`.

For the benchmark to automate the analysis of the results and the production of a benchmark report, we need each script to report a common set of metrics or tags.

## Proposed structure of the repo

| Directory | Content |
| :-- | :-- |
| `/.github/workflows/` | [github workflows](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions) for this project |
| `/docs/` | Documentation in markdown |
| `/pipelines/` | (coming) Scripts or config files for running the benchmark in the cloud (ex: AzureML) |
| `/src/` | &nbsp; |
| `/src/common/` | Common libraries for all benchmark scripts |
| `/src/scripts/` | Benchmark scripts |
| `/tests/` | Unit tests using pytest |

## Proposed structure of the scripts

**Scripts will be written in python**. Even if the module is calling a CLI command (ex: lightgbm.exe), writing it in python will allow us to wrap it with metrics reporting, consistent unit tests, etc.

Scripts will have a minimum of **three functions `get_arg_parser()`, `run()` and `main()`** (see below code template). The value of this code template is:
- this template is the current structure expected by the [shrike](https://pypi.org/project/shrike/) contrib package, letting us automate a lot of the unit testing, integration testing, etc,
- we can create unit tests easily to test all script arguments for consistency,
- the arg parsers can also be built in conjonction with other parsers (ex: a super script adding all the arguments from multiple scripts).

The `main()` function is likely to be the same in each script. The alternative could be to create a script class with pre-implemented `main()` method. But each script could become a little less readable for beginner users wanting to run and edit those scripts.

Each script should be runnable on its own, and provide a clear set of usage instructions when using `-h` option.

## Scripts inputs, outputs and parameters

Each script `get_arg_parser()` function builds an `argparse` for the script which details the inputs, outputs and parameters.

**For inputs and outputs, we propose to use directories exclusively**. The goal is to be compatible with pipeline orchestration platforms (like AzureML) where inputs can be either file paths or directory paths, depending on the platform.

For this, each script input will be considered a distinct directory containing at least one file (several if needed). Each script output will be provided as a directory in which the script will create one or multiple files on its own.

Two helper functions in the common library will help to handle that type of i/o arguments consistently through the benchmark.

## Common libraries

We'll build common libraries under `/src/common/` to help script developers to achieve common routines in a standard and consistent manner.

For instance:
- mlflow metrics recording
- managing typical arguments (ex: input/output folders with single file)

## Reporting standards

Each script takes care of reporting relevant properties, parameters and metrics. How to set those is detailed in the reporting guide (work in progress).
