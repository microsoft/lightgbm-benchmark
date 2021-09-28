from dataclasses import dataclass
from omegaconf import MISSING
from typing import Optional

@dataclass
class synthetic_data_config:
    task: str = MISSING
    train_samples: int = MISSING
    test_samples: int = MISSING
    inferencing_samples: int = MISSING
    n_features: int = MISSING
    n_informative: Optional[int] = None
