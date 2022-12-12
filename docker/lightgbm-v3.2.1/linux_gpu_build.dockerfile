FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04:20221129.v1
LABEL lightgbmbenchmark.linux.gpu.build.version="3.2.1/20211108.1"

# Those arguments will NOT be used by AzureML
# they are here just to allow for lightgbm-benchmark build to actually check
# dockerfiles in a PR against their actual branch
ARG lightgbm_version="3.2.1"

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/lightgbm

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=22.2.2

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

RUN apt-get update -y

RUN apt-get install --no-install-recommends nvidia-375 -y && \
    apt-get install --no-install-recommends nvidia-opencl-icd-375 nvidia-opencl-dev opencl-headers -y

RUN apt-get install --no-install-recommends git cmake build-essential libboost-dev libboost-system-dev libboost-filesystem-dev -y

# Clone lightgbm official repository (master branch)
RUN git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    git checkout tags/v${lightgbm_version}

# https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#build-lightgbm
RUN cd /LightGBM && \
    mkdir build && \
    cd build && \
    cmake -DUSE_GPU=ON -DUSE_MPI=ON .. && \
    make -j$(nproc)

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

RUN pip install --upgrade pip setuptools wheel && \
    pip install 'cmake==3.21.0' 

# Install LightGBM Python API from build
RUN cd /LightGBM/python-package/ && \
    python setup.py install --precompile

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
