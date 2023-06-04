"""Configuration for readmission_model v1d77y2023."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class ReadmissionModelConfig_v1d77y2023:
    name: str = "readmission_model"
    version: str = "1.77.0"
    num_layers: int = 2
    hidden_dim: int = 64
    learning_rate: float = 0.000100
    batch_size: int = 16
    max_epochs: int = 50
    dropout: float = 0.1
    checkpoint_dir: Path = Path("checkpoints/readmission_model/v1d77y2023")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
