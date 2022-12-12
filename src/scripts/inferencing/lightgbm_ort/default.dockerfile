FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/lightgbm

ARG lightgbm_version="3.3.0"

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=22.2.2

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN HOROVOD_WITH_TENSORFLOW=1 \
    pip install 'pandas==1.5.2' \
                'numpy==1.23.5' \
                'matplotlib==3.6.2' \
                'scipy~=1.5.0' \
                'scikit-learn~=0.24.1' \
                'azureml-core==1.35.0' \
                'azureml-defaults==1.35.0' \
                'azureml-mlflow==1.35.0' \
                'azureml-telemetry==1.35.0' \
                'mpi4py==3.1.1' \
                'onnxruntime==1.12.1' \
                'onnxmltools==1.11.1' \
                'onnxconverter-common==1.12.2'

# install lightgbm with mpi
RUN pip install lightgbm==${lightgbm_version} \
    pip install 'protobuf==3.20.3'