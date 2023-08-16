"""Configuration for medication_reconciler v1d103y2023."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class MedicationReconcilerConfig_v1d103y2023:
    name: str = "medication_reconciler"
    version: str = "1.103.0"
    num_layers: int = 2
    hidden_dim: int = 64
    learning_rate: float = 0.000100
    batch_size: int = 16
    max_epochs: int = 50
    dropout: float = 0.1
    checkpoint_dir: Path = Path("checkpoints/medication_reconciler/v1d103y2023")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
