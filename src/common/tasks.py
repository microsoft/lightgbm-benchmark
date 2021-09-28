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
