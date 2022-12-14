FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20221129.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/lightgbm

ARG lightgbm_version="3.3.3"

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=22.2.2

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install 'numpy==1.23.5' \
                'matplotlib==3.6.2' \
                'psutil==5.9.4'\
                'azureml-mlflow==1.48.0' \
                'onnxruntime==1.13.1' \
                'onnxmltools==1.11.1' \
                'onnxconverter-common==1.13.0'

# install lightgbm with mpi
RUN pip install lightgbm==${lightgbm_version} \
    pip install 'protobuf==3.20.3'