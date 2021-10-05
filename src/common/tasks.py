from dataclasses import dataclass
from omegaconf import MISSING
from typing import Optional

@dataclass
class inferencing_task:
    dataset: str = MISSING
    model: str = MISSING
    predict_disable_shape_check: bool = False

@dataclass
class inferencing_variants:
    framework: str = MISSING
    build: Optional[str] = None
    os: str = "Linux" # linux or windows, linux by default

@dataclass
class data_generation_task:
    task: str = MISSING
    train_samples: int = MISSING
    test_samples: int = MISSING
    inferencing_samples: int = MISSING
    n_features: int = MISSING
    n_informative: Optional[int] = None

@dataclass
class training_task:
    train_dataset: str = MISSING
    train_dataset_version: Optional[str] = None
    test_dataset: str = MISSING
    test_dataset_version: Optional[str] = None

@dataclass
class training_variant:
    # TRAINING
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

    # COMPUTE
    device_type: str = "cpu"
    nodes: int = 1
    processes: int = 1
    target: Optional[str] = None
    override_docker: Optional[str] = None
    override_os: Optional[str] = None

    # SWEEP
    # TODO: add all parameters from shrike https://github.com/Azure/shrike/blob/387fadb47d69e46bd7e5ac6f243250dc6044afaa/shrike/pipeline/pipeline_helper.py#L809
    sweep_algorithm: str = "random"
    sweep_goal: str = "minimize"
    sweep_max_total_trials: Optional[int] = None
    sweep_max_concurrent_trials: Optional[int] = None
    sweep_timeout_minutes: Optional[int] = None

    # OUTPUT REGISTRATION
    register_model_as: Optional[str] = None
