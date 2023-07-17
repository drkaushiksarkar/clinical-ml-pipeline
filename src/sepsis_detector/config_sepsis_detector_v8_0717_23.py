"""Configuration for sepsis_detector v8d90y2023."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class SepsisDetectorConfig_v8d90y2023:
    name: str = "sepsis_detector"
    version: str = "8.90.0"
    num_layers: int = 16
    hidden_dim: int = 512
    learning_rate: float = 0.000800
    batch_size: int = 128
    max_epochs: int = 400
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/sepsis_detector/v8d90y2023")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
