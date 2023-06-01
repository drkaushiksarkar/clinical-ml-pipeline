"""Configuration for phenotyper v5d75y2023."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class PhenotyperConfig_v5d75y2023:
    name: str = "phenotyper"
    version: str = "5.75.0"
    num_layers: int = 10
    hidden_dim: int = 320
    learning_rate: float = 0.000500
    batch_size: int = 80
    max_epochs: int = 250
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/phenotyper/v5d75y2023")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
