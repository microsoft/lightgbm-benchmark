from dataclasses import dataclass
from omegaconf import MISSING
from typing import Any, Optional

@dataclass
class data_input_spec:
    # NOTE: Union is not supported in Hydra/OmegaConf
    # specify either by dataset name and version
    name: Optional[str] = None
    version: Optional[str] = None
    # or by uuid (non-registered)
    uuid: Optional[str] = None
    # or by datastore+path
    datastore: Optional[str] = None
    path: Optional[str] = None
    validate: bool = True

@dataclass
class inferencing_task:
    data: data_input_spec = MISSING
    model: data_input_spec = MISSING
    task_key: Optional[str] = None
    predict_disable_shape_check: bool = False

@dataclass
class inferencing_variants:
    framework: str = MISSING
    build: Optional[str] = None

@dataclass
class data_generation_task:
    task: str = MISSING
    task_key: Optional[str] = None
    train_samples: int = MISSING
    train_partitions: int = 1
    test_samples: int = MISSING
    test_partitions: int = 1
    inferencing_samples: int = MISSING
    inferencing_partitions: int = 1
    n_features: int = MISSING
    n_informative: Optional[int] = None

@dataclass
class training_task:
    train: data_input_spec = MISSING
    test: data_input_spec = MISSING
    # provide a key for internal tagging + reporting
    task_key: Optional[str] = None


@dataclass
class sweep_early_termination_settings:
    policy_type: str = 'default' # truncation_selection | median_stopping | bandit
    evaluation_interval: Optional[int] = None
    delay_evaluation: Optional[int] = None

    # truncation settings
    truncation_percentage: Optional[int] = None # for truncation_selection

    # bandit settings
    slack_factor: Optional[float] = None

@dataclass
class sweep_limits_settings:
    max_total_trials: int = MISSING
    max_concurrent_trials: Optional[int] = None # must be between 1 and 100
    timeout_minutes: Optional[int] = None

@dataclass
class sweep_settings:
    # TODO: add all parameters from shrike https://github.com/Azure/shrike/blob/387fadb47d69e46bd7e5ac6f243250dc6044afaa/shrike/pipeline/pipeline_helper.py#L809
    # goal settings
    primary_metric: Optional[str] = None
    goal: Optional[str] = None
    algorithm: str = "random"

    early_termination: Optional[sweep_early_termination_settings] = None
    limits: Optional[sweep_limits_settings] = None

@dataclass
class lightgbm_training_variant_parameters:
    # fixed training parameters
    objective: str = MISSING
    metric: str = MISSING
    boosting: str = MISSING
    tree_learner: str = MISSING

    # sweepable training parameters
    # NOTE: need to be str so they can be parsed (ex: 'choice(100,200)')
    num_iterations: str = MISSING
    num_leaves: str = MISSING
    min_data_in_leaf: str = MISSING
    learning_rate: str = MISSING
    max_bin: str = MISSING
    feature_fraction: str = MISSING
    label_gain: Optional[str] = None
    custom_params: Optional[Any] = None

    # COMPUTE
    device_type: str = "cpu"
    verbose: bool = False

@dataclass
class lightgbm_training_data_variant_parameters:
    # FILE OPTIONS
    auto_partitioning: bool = True
    pre_convert_to_binary: bool = False # doesn't work with partitioned data (yet)

    # input parameters
    header: bool = False
    label_column: Optional[str] = "0"
    group_column: Optional[str] = None
    construct: bool = True

@dataclass
class lightgbm_training_environment_variant_parameters:
    # COMPUTE
    nodes: int = 1
    processes: int = 1
    target: Optional[str] = None
    build: Optional[str] = None

@dataclass
class lightgbm_training_output_variant_parameters:
    register_model: bool = False # "{register_model_prefix}-{task_key}-{num_iterations}trees-{num_leaves}leaves-{register_model_suffix}"
    register_model_prefix: Optional[str] = None
    register_model_suffix: Optional[str] = None

@dataclass
class training_variant:
    # three below are mandatory sections of the variant config
    data: lightgbm_training_data_variant_parameters = MISSING
    training: lightgbm_training_variant_parameters = MISSING
    runtime: lightgbm_training_environment_variant_parameters = MISSING

    # two below are optional
    sweep: Optional[sweep_settings] = None
    output:Optional[lightgbm_training_output_variant_parameters] = None
