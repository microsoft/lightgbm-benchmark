# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script contains methods to handle connection to AzureML,
such as registering Datasets or obtaining a Dataset handler from a given workspace.
"""
import logging
import re
from azure.ai.ml.entities import Data
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

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

def load_dataset_from_data_input_spec(workspace,data_input_spec):
    """ Loads a dataset based on config object data_input_spec (see tasks.py data_input_spec)
    
    Args:
        workspace (azureml.core.Workspace): connector to an AzureML workspace
        data_input_spec (OmegaConf.DictConfig): config Hydra dataclass data_input_spec (see tasks.py)

    Returns:
        azureml.core.Dataset: registered Dataset object
    """
    logger = logging.getLogger(__name__)

    if data_input_spec.path:
        logger.info(f"Reading dataset from path={data_input_spec.path}")
        loaded_dataset = Input(path=data_input_spec.path)
    else:
        raise ValueError("Input path issue (provided path = {data_input_spec.path})")

    return loaded_dataset


def apply_sweep_settings(step, sweep_settings_config):
    """Applies the settings to a sweep step based on a config dataclass.

    Args:
        step (PipelineStep): the instance of the step
        sweep_settings_config (OmegaConf.DictConfig): schema specified in src.common.tasks.sweep_runsettings
    """
    if (not sweep_settings_config.primary_metric) or (not sweep_settings_config.goal):
        raise ValueError("in sweep settings, you need to provide a primary_metric and a goal settings.")
    else:
        step.runsettings.sweep.objective.configure(
            primary_metric = sweep_settings_config.primary_metric,
            goal = sweep_settings_config.goal,
        )

    if not sweep_settings_config.algorithm:
        raise ValueError("in sweep settings, you need to provide an algorithm setting.")
    else:
        step.runsettings.sweep.algorithm = sweep_settings_config.algorithm

    if sweep_settings_config.limits:
        step.runsettings.sweep.limits.configure(
            max_total_trials = sweep_settings_config.limits.max_total_trials,
            max_concurrent_trials = sweep_settings_config.limits.max_concurrent_trials,
            timeout_minutes = sweep_settings_config.limits.timeout_minutes,
        )
    
    if sweep_settings_config.early_termination:
        if sweep_settings_config.early_termination.policy_type == "median_stopping":
            step.runsettings.sweep.early_termination.configure(
                policy_type="median_stopping",
                evaluation_interval=sweep_settings_config.early_termination.evaluation_interval,
                delay_evaluation=sweep_settings_config.early_termination.delay_evaluation
            )
        elif sweep_settings_config.early_termination.policy_type == "bandit":
            step.runsettings.sweep.early_termination.configure(
                policy_type="bandit",
                slack_factor=sweep_settings_config.early_termination.slack_factor,
                evaluation_interval=sweep_settings_config.early_termination.evaluation_interval,
                delay_evaluation=sweep_settings_config.early_termination.delay_evaluation
            )
        elif sweep_settings_config.early_termination.policy_type == "truncation_selection":
            step.runsettings.sweep.early_termination.configure(
                policy_type="truncation_selection",
                truncation_percentage=sweep_settings_config.early_termination.truncation_percentage,
                evaluation_interval=sweep_settings_config.early_termination.evaluation_interval,
                delay_evaluation=sweep_settings_config.early_termination.delay_evaluation
            )
        elif sweep_settings_config.early_termination.policy_type == "default":
            pass
        elif sweep_settings_config.early_termination.policy_type == None:
            pass
        else:
            raise NotImplementedError(f"sweep settings early_termination policy_type={sweep_settings_config.early_termination.policy_type} is not implemented.")


def format_run_name(run_name: str):
    """Formats a run name to fit with AzureML constraints.
    
    Args:
        run_name (str): string to be formatted.

    Returns:
        formatted_run_name (str)
    
    Notes:
        Node name must start with a letter,
        and can only contain letters, numbers,
        underscores, within 1-255 characters.
    """
    # removing all chars not allowed, use underscore instead
    formatted_run_name = re.sub(r'[^a-zA-Z0-9_]', '_', run_name)

    # cutting to first 255 chars
    if len(formatted_run_name) > 255:
        formatted_run_name = formatted_run_name[0:255]

    return formatted_run_name
