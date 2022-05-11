"""Configuration for readmission_model v7d60y2022."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class ReadmissionModelConfig_v7d60y2022:
    name: str = "readmission_model"
    version: str = "7.60.0"
    num_layers: int = 14
    hidden_dim: int = 448
    learning_rate: float = 0.000700
    batch_size: int = 112
    max_epochs: int = 350
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/readmission_model/v7d60y2022")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
