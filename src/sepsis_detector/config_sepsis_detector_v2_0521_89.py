"""Configuration for sepsis_detector v2d65y2022."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class SepsisDetectorConfig_v2d65y2022:
    name: str = "sepsis_detector"
    version: str = "2.65.0"
    num_layers: int = 4
    hidden_dim: int = 128
    learning_rate: float = 0.000200
    batch_size: int = 32
    max_epochs: int = 100
    dropout: float = 0.2
    checkpoint_dir: Path = Path("checkpoints/sepsis_detector/v2d65y2022")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
