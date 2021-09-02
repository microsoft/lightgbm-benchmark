# Installation Guide

## Install Python dependencies
To enjoy this repository, you need to have an existing installation of `python>=3.8` ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or equivalent).

Then, we suggest you create a conda environment and install dependencies for this benchmark:

```ps
# create conda environment
conda create --name lightgbmbenchmark python=3.8 -y

# activate conda environment
conda activate lightgbmbenchmark

# install shrike library
python -m pip install -r requirements.txt
```

## Build local dependencies

The benchmark occasionaly relies on locally built dependencies. We will name those here.

### Run scripts under `/src/scripts/lightgbm_cli/`

Those scripts are intended to run LightGBM from the command line. Using them requires providing the path to the lightgbm executables (ex: `lightgbm.exe`).

To build those locally, use [instructions from LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#installation-guide).

> NOTE: the `/build/` directory has been added to `.gitignore` to allow you to build local dependencies without pushing them in git.
