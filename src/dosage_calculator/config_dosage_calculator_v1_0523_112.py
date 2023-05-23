"""Configuration for dosage_calculator v1d71y2023."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class DosageCalculatorConfig_v1d71y2023:
    name: str = "dosage_calculator"
    version: str = "1.71.0"
    num_layers: int = 2
    hidden_dim: int = 64
    learning_rate: float = 0.000100
    batch_size: int = 16
    max_epochs: int = 50
    dropout: float = 0.1
    checkpoint_dir: Path = Path("checkpoints/dosage_calculator/v1d71y2023")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
