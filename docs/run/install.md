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


### Run lightgbm train locally (requires mpi)

Our lightgbm training script is distributed-ready, and currently using mpi. To be able to use this locally, either for debugging or for benchmarking, you'll need to [install LightGBM with mpi support](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-mpi-version).

One easy way is to install lightgbm with mpi option (requires cmake and other build tools):

```
pip install --upgrade pip setuptools wheel
pip install cmake==3.21.0
pip install lightgbm==3.2.1 --install-option=--mpi
```

### Run scripts under `/src/scripts/lightgbm_cli/`

Those scripts are intended to run LightGBM from the command line. Using them requires providing the path to the lightgbm executables (ex: `lightgbm.exe`).

To build those locally, use [instructions from LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#installation-guide).

!!! note
    The `/build/` directory has been added to `.gitignore` to allow you to build local dependencies without pushing them in git.