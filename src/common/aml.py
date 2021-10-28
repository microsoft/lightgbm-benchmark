# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script contains methods to handle connection to AzureML,
such as registering Datasets or obtaining a Dataset handler from a given workspace.
"""
import logging
from azureml.core import Datastore, Dataset


def dataset_from_dstore_path(workspace, datastore, datastore_path, validate=True):
    """ Obtains a local reference for a given datastore and path  

    Args:
        datastore (str): name of the AzureML datastore
        datastore_path (str): path in datastore to register as Dataset
        validate (bool): validate files exist or not

    Returns:
        azureml.core.Dataset: registered Dataset object
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Connecting to Datastore {datastore}...")
    datastore = Datastore.get(workspace, datastore)

    logger.info(f"Reading path {datastore_path}...")
    remote_ds_path = [(datastore, datastore_path)]

    logger.info(f"Registering as dataset...")
    remote_dataset = Dataset.File.from_files(path=remote_ds_path, validate=validate)

    return remote_dataset


def load_dataset_from_data_input_spec(workspace, data_input_spec):
    """ Loads a dataset based on config object data_input_spec (see tasks.py data_input_spec)
    
    Args:
        workspace (azureml.core.Workspace): connector to an AzureML workspace
        data_input_spec (OmegaConf.DictConfig): config Hydra dataclass data_input_spec (see tasks.py)

    Returns:
        azureml.core.Dataset: registered Dataset object
    """
    logger = logging.getLogger(__name__)

    if data_input_spec.name:
        logger.info(f"Reading dataset from name={data_input_spec.name} version={data_input_spec.version}")
        loaded_dataset = Dataset.get_by_name(workspace, name=data_input_spec.name, version=data_input_spec.version)
    elif data_input_spec.uuid:
        logger.info(f"Reading dataset from uuid")
        loaded_dataset = Dataset.get_by_id(workspace, id=data_input_spec.uuid)
    elif data_input_spec.datastore and data_input_spec.path:        
        logger.info(f"Connecting to Datastore {data_input_spec.datastore}...")
        datastore = Datastore.get(workspace, data_input_spec.datastore)

        logger.info(f"Reading path {data_input_spec.path}...")
        remote_ds_path = [(datastore, data_input_spec.path)]

        logger.info(f"Registering as dataset...")
        loaded_dataset = Dataset.File.from_files(path=remote_ds_path, validate=data_input_spec.validate)
    else:
        raise ValueError("To load a dataset using data_input_spec, you need to provide either a name, a uuid or a datastore+path (provided config = {data_input_spec})")

    return loaded_dataset
