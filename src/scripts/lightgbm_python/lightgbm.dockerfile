FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210615.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/lightgbm

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.7 pip=20.2.4

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN HOROVOD_WITH_TENSORFLOW=1 \
    pip install 'pandas>=1.1,<1.2' \
                'numpy>=1.10,<1.20' \
                'scipy~=1.5.0' \
                'scikit-learn~=0.24.1' \
                'dask~=2021.6.0' \
                'distributed~=2021.6.0' \
                'dask-ml~=1.9.0' \
                'azureml-core==1.30.0' \
                'azureml-defaults==1.30.0' \
                'azureml-mlflow==1.30.0' \
                'azureml-telemetry==1.30.0' \
                'mpi4py==3.1.1'

RUN pip install --upgrade pip setuptools wheel && \
    pip install 'cmake==3.21.0' 

# Clone lightgbm official repository (master branch)
RUN mkdir LightGBM && \
    cd LightGBM && \
    git clone --recursive https://github.com/microsoft/LightGBM.git

# Build lightgbm with custom patch applied
RUN cd LightGBM/LightGBM/python-package && \
    python setup.py install --mpi

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
