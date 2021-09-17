FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/lightgbm

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=20.2.4

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

RUN apt-get update -y

RUN apt-get install --no-install-recommends nvidia-375 -y && \
    apt-get install --no-install-recommends nvidia-opencl-icd-375 nvidia-opencl-dev opencl-headers -y

RUN apt-get install --no-install-recommends git cmake build-essential libboost-dev libboost-system-dev libboost-filesystem-dev -y

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

# Clone lightgbm official repository (master branch)
RUN git clone --recursive https://github.com/microsoft/LightGBM

# https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#build-lightgbm
RUN cd /LightGBM && \
    mkdir build && \
    cd build && \
    cmake -DUSE_GPU=1 -DUSE_MPI=ON .. && \
    make -j$(nproc)

# https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#install-python-interface-optional
RUN cd /LightGBM/python-package/ && \
    python setup.py install --precompile

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH