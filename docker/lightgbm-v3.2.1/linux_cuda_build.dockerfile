FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04
LABEL lightgbmbenchmark.linux.cuda.build.version="3.2.1/20211029.1"

#################################################################################################################
#           Global Path Setting
#################################################################################################################

ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/lib

ENV OPENCL_LIBRARIES /usr/local/cuda/lib64
ENV OPENCL_INCLUDE_DIR /usr/local/cuda/include

#################################################################################################################
#           SYSTEM
#################################################################################################################
# update: downloads the package lists from the repositories and "updates" them to get information on the newest versions of packages and their
# dependencies. It will do this for all repositories and PPAs.

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        git \
        vim \
        mercurial \
        subversion \
        cmake \
        libboost-dev \
        libboost-system-dev \
        libboost-filesystem-dev \
        gcc \
        g++

# Add OpenCL ICD files for LightGBM
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

#################################################################################################################
#           LightGBM
#################################################################################################################

# Clone lightgbm official repository (master branch)
RUN git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    git checkout tags/v3.2.1

# https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#build-lightgbm
# see https://github.com/microsoft/LightGBM/blob/master/docker/gpu/dockerfile.gpu
RUN cd /LightGBM && \
    mkdir build && \
    cd build && \
    cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. && \
    make OPENCL_HEADERS=/usr/local/cuda-10.1/targets/x86_64-linux/include LIBOPENCL=/usr/local/cuda-10.1/targets/x86_64-linux/lib

#################################################################################################################
#           CONDA
#################################################################################################################

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
                'azureml-core==1.30.0' \
                'azureml-defaults==1.30.0' \
                'azureml-mlflow==1.30.0' \
                'azureml-telemetry==1.30.0' \
                'mpi4py==3.1.1'

RUN pip install --upgrade pip setuptools wheel && \
    pip install 'cmake==3.21.0' 

# https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#install-python-interface-optional
RUN cd /LightGBM/python-package/ && \
    python setup.py install --precompile

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
