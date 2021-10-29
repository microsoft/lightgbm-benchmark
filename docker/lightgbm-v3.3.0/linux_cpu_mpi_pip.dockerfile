FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210615.v1
LABEL lightgbmbenchmark.linux.cpu.mpi.pip.version="3.3.0/20211029.1"

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

# install lightgbm with mpi
RUN pip install --upgrade pip setuptools wheel && \
    pip install 'cmake==3.21.0' && \
    pip install 'lightgbm==3.3.0' --install-option=--mpi

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
