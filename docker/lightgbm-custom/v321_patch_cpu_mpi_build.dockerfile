FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20221129.v1
LABEL lightgbmbenchmark.linux.cpu.mpi.build.version="3.2.1-patch/20211109.1"

# Those arguments will NOT be used by AzureML
# they are here just to allow for lightgbm-benchmark build to actually check
# dockerfiles in a PR against their actual branch
ARG lightgbm_version="3.2.1"
ARG lightgbm_benchmark_branch=main

RUN apt-get update && \
    apt-get -y install build-essential cmake

# LIGHTGBM EXEC AND LIBRARY

# Clone lightgbm official repository (master branch)
RUN git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    git checkout tags/v${lightgbm_version}

# Download and apply a particular patch
RUN cd /LightGBM && \
    wget https://raw.githubusercontent.com/microsoft/lightgbm-benchmark/32cbb007b61f5bed89af1423c7da250607726a35/pipelines/azureml_sdk15/components/lightgbm_python_custom/lightgbm_custom.python.patch && \
    git apply --whitespace=fix ./lightgbm_custom.python.patch

# https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#build-lightgbm
RUN cd /LightGBM && \
    mkdir build && \
    cd build && \
    cmake -DUSE_MPI=ON .. && \
    make -j$(nproc)

# Prepend path to LightGBM LIB
ENV PATH /LightGBM:$PATH

# building lightgbm-benchmark binaries
RUN git clone --recursive https://github.com/microsoft/lightgbm-benchmark.git && \
    cd lightgbm-benchmark && \
    git checkout ${lightgbm_benchmark_branch}

# assuming lightgbm lib+includes are installed on the system
RUN cd /lightgbm-benchmark/src/binaries/ && \
    mkdir build && \
    cd build && \
    cmake -DLIGHTGBM_CLONE=/LightGBM -DUSE_LIGHTGBM_V321_PARSER=ON .. && \
    cmake --build . --target lightgbm_predict --config Release

# provide env variable with path to built binaries
ENV LIGHTGBM_BENCHMARK_BINARIES_PATH /lightgbm-benchmark/src/binaries/build
RUN ls -l $LIGHTGBM_BENCHMARK_BINARIES_PATH

## ANACONDA ENVIRONMENT

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/lightgbm

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=22.2.2

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN HOROVOD_WITH_TENSORFLOW=1 \
    pip install 'pandas>=1.1,<1.2' \
                'numpy>=1.10,<1.20' \
                'matplotlib==3.4.3' \
                'scipy~=1.5.0' \
                'scikit-learn~=0.24.1' \
                'azureml-core==1.35.0' \
                'azureml-defaults==1.35.0' \
                'azureml-mlflow==1.35.0' \
                'azureml-telemetry==1.35.0' \
                'mpi4py==3.1.1'

RUN pip install --upgrade pip setuptools wheel && \
    pip install 'cmake==3.21.0'

# Install LightGBM Python API from build
RUN cd /LightGBM/python-package/ && \
    python setup.py install --precompile

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
