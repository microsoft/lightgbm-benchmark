FROM mcr.microsoft.com/azureml/windows-servercore-1809:latest
LABEL lightgbmbenchmark.windows.cpu.mpi.pip.version="3.3.0/20211108.1"

# Those arguments will NOT be used by AzureML
# they are here just to allow for lightgbm-benchmark build to actually check
# dockerfiles in a PR against their actual branch
ARG lightgbm_version="3.3.0"

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/lightgbm

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=20.2.4

# Prepend path to AzureML conda environment
#ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install 'pandas>=1.1,<1.2' \
                'numpy>=1.10,<1.20' \
                'scipy~=1.5.0' \
                'scikit-learn~=0.24.1' \
                'azureml-core==1.35.0' \
                'azureml-defaults==1.35.0' \
                'azureml-mlflow==1.35.0' \
                'azureml-telemetry==1.35.0' \
                lightgbm==$($Env:lightgbm_version)
