FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
LABEL lightgbmbenchmark.linux.cpu.mpi.pip.version="3.3.1/20211210.1"

# https://github.com/microsoft/lightgbm-transform/blob/main/docs/Installation-Guide.rst
# Install CMake, gcc, g++, boost.
RUN apt-get update && apt-get -y upgrade && DEBIAN_FRONTEND="noninteractive" apt-get install -y libboost-all-dev gcc g++ wget cmake git curl libtinfo5

# Install LLVM with RTTI feature.
WORKDIR /root
RUN wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-3.5.1.tar.gz && tar zxvf llvmorg-3.5.1.tar.gz
WORKDIR /root/llvm-project-llvmorg-3.5.1/llvm
RUN mkdir build && cd build && cmake -DLLVM_REQUIRES_RTTI=1 .. && make -j4 && make install

# Install bond.
WORKDIR /root
RUN git clone --recursive https://github.com/microsoft/bond.git
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y clang zlib1g-dev
RUN curl -sSL https://get.haskellstack.org/ | sh
WORKDIR /root/bond
RUN mkdir build && cd build && cmake -DBOND_ENABLE_GRPC=FALSE .. && make -j4 && make install


# Those arguments will NOT be used by AzureML
# they are here just to allow for lightgbm-benchmark build to actually check
# dockerfiles in a PR against their actual branch
ARG lightgbm_version="3.3.1"

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
                'matplotlib==3.4.3' \
                'scipy~=1.5.0' \
                'scikit-learn~=0.24.1' \
                'azureml-core==1.35.0' \
                'azureml-defaults==1.35.0' \
                'azureml-mlflow==1.35.0' \
                'azureml-telemetry==1.35.0' \
                'mpi4py==3.1.1'

# install lightgbm with mpi
RUN pip install --upgrade pip setuptools wheel && \
    pip install 'cmake==3.21.0' && \
    pip install lightgbm-transform==${lightgbm_version}

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
