import logging
from azureml.core import Datastore, Dataset

def dataset_from_dstore_path(workspace, datastore, datastore_path):
    """ Obtains a local reference for a given datastore and path
    
    Args:
        datastore (str)
        datastore_path (str)

    Returns:
        remote_dataset (azureml.core.Dataset)
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Connecting to Datastore {datastore}...")
    datastore = Datastore.get(workspace, datastore)

    logger.info(f"Reading path {datastore_path}...")
    remote_ds_path = [(datastore, datastore_path)]

    logger.info(f"Registering as dataset...")
    remote_dataset = Dataset.File.from_files(path=remote_ds_path, validate=True)

    return remote_dataset