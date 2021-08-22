# Developer guide

The following document details the proposed structure for this repo, the benchmark scripts and libraries.

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

## Python code template

Here's a proposed python script template.

```python
# <<<any imports>>>

# let's add the right PYTHONPATH for common module
# (assuming your script is located at /src/scripts/SOMETHING/foo.py)
COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if COMMON_ROOT not in sys.path:
    print(f"Adding {COMMON_ROOT} to PYTHONPATH")
    sys.path.append(str(COMMON_ROOT))

# before doing local import
from common.metrics import MetricsLogger
from common.io import input_file_path

def get_arg_parser(parser=None):
    """Adds component/module arguments to a given argument parser.

    Args:
        parser (argparse.ArgumentParser): an argument parser instance

    Returns:
        ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the module
    if parser is None:
        parser = argparse.ArgumentParser(__doc__)

    # <<< whatever parsing building is needed>>>

    return parser


def run(args, unknown_args=[]):
    """Run script with arguments (the core of the component)

    Args:
        args (argparse.namespace): command line arguments provided to script, see get_arg_parser()
        unknown_args (list[str]): list of arguments not known by parser
    """
    # <<<whatever needs to be implemented here>>>
    pass


def main(cli_args=None):
    """ Component main function, parses arguments and executes run() function.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # construct arg parser
    parser = get_arg_parser()
 
    # if argument parsing fails, or if unknown arguments, will except
    args, unknown_args = parser.parse_known_args(cli_args)

    # run the actual thing
    run(args, unknown_args)


if __name__ == "__main__":
    main()    
```

## Reporting standards

Each script takes care of reporting relevant properties, parameters and metrics. How to set those is detailed in the reporting guide (work in progress).
