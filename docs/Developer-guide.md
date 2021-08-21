# Developer guide

The following document details the proposed structure for this repo, the benchmark scripts and libraries.

## Context and goal

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

## Proposed structure of the benchmark scripts

**Scripts will be written in python**. Even if the module is calling a CLI command (ex: lightgbm.exe), writing it in python will allow us to wrap it with metrics reporting, consistent unit tests, etc.

Scripts will have a minimum of **three functions `get_arg_parser()`, `run()` and `main()`** (see below code template). The value of this code template is:
- we can create unit tests easily to test all script arguments for consistency,
- the arg parsers can also be built in conjonction with other parsers (ex: a super script adding all the arguments from multiple scripts),
- this template is the current structure expected by the [shrike](https://pypi.org/project/shrike/) contrib package, letting us automate a lot of the unit testing, integration testing, etc.

The `main()` function is likely to be the same in each script. The alternative could be to create a script class with pre-implemented `main()` method. But each script could become a little less readable for beginner users wanting to run and edit those scripts.

Each script should be runnable on its own, and provide a clear set of usage instructions when using `-h` option.

## Common libraries

We'll build common libraries under `/src/common/` to help script developers to achieve common routines in a standard and consistent manner.

For instance:
- mlflow metrics recording
- managing typical arguments (ex: input folders with single file)

## Python code template

Here's a proposed python script template.

```python
# <<<any imports>>>

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
