FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20211012.v1

RUN apt-get update && \
    apt-get -y install build-essential cmake

# LIGHTGBM EXEC AND LIBRARY

# Clone lightgbm official repository (master branch)
RUN git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    git checkout tags/v3.3.0

# https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#build-lightgbm
RUN cd /LightGBM && \
    mkdir build && \
    cd build && \
    cmake -DUSE_MPI=ON .. && \
    make -j$(nproc)

# Prepend path to LightGBM LIB
ENV PATH /LightGBM:$PATH

# building lightgbm-benchmark binaries
RUN git clone --recursive https://github.com/microsoft/lightgbm-benchmark.git

# assuming lightgbm lib+includes are installed on the system
RUN cd /lightgbm-benchmark/src/binaries/ && \
    mkdir build && \
    cd build && \
    cmake -DLIGHTGBM_INC=/LightGBM/include -DLIGHTGBM_LIB=/LightGBM .. && \
    cmake --build . --target lightgbm_predict --config Release

# provide env variable with path to built binaries
ENV LIGHTGBM_BENCHMARK_BINARIES_PATH /lightgbm-benchmark/src/binaries/build
RUN ls -l $LIGHTGBM_BENCHMARK_BINARIES_PATH

## ANACONDA ENVIRONMENT

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
                'azureml-core==1.35.0' \
                'azureml-defaults==1.35.0' \
                'azureml-mlflow==1.35.0' \
                'azureml-telemetry==1.35.0' \
                'mpi4py==3.1.1'

RUN pip install --upgrade pip setuptools wheel && \
    pip install 'cmake==3.21.0'

# https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#install-python-interface-optional
RUN cd /LightGBM/python-package/ && \
    python setup.py install --precompile