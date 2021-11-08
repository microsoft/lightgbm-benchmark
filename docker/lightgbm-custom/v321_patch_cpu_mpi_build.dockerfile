FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210615.v1
LABEL lightgbmbenchmark.linux.cpu.mpi.build.version="3.2.1-patch/20211108.1"

# Those arguments will NOT be used by AzureML
# they are here just to allow for lightgbm-benchmark build to actually check
# dockerfiles in a PR against their actual branch
ARG lightgbm_branch=tags/v3.2.1
ARG lightgbm_benchmark_branch=main

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/lightgbm

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=20.2.4

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN HOROVOD_WITH_TENSORFLOW=1 \
    pip install 'pandas>=1.1,<1.2' \
                'numpy>=1.10,<1.20' \
                'scipy~=1.5.0' \
                'scikit-learn~=0.24.1' \
                'azureml-core==1.35.0' \
                'azureml-defaults==1.35.0' \
                'azureml-mlflow==1.35.0' \
                'azureml-telemetry==1.35.0' \
                'mpi4py==3.1.1'

# install lightgbm with pip
RUN pip install --upgrade pip setuptools wheel && \
    pip install 'cmake==3.21.0'

# Clone lightgbm official repository (master branch)
RUN git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    git checkout ${lightgbm_branch}

# Download and apply a particular patch
RUN cd /LightGBM && \
    wget https://raw.githubusercontent.com/microsoft/lightgbm-benchmark/32cbb007b61f5bed89af1423c7da250607726a35/pipelines/azureml_sdk15/components/lightgbm_python_custom/lightgbm_custom.python.patch && \
    git apply --whitespace=fix ./lightgbm_custom.python.patch

# Build lightgbm with custom patch applied
RUN cd /LightGBM/python-package && \
    python setup.py install --mpi

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
