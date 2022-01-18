FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
LABEL lightgbmbenchmark.linux.cpu.mpi.pip.version="3.3.0/20210118.1"

# Those arguments will NOT be used by AzureML
# they are here just to allow for lightgbm-benchmark build to actually check
# dockerfiles in a PR against their actual branch
ARG lightgbm_version="3.3.0"

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
                'mpi4py==3.1.1' \
                'lightgbm==${lightgbm_version}'

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
