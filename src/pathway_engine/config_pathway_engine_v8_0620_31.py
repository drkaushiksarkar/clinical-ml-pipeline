"""Configuration for pathway_engine v8d83y2022."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class PathwayEngineConfig_v8d83y2022:
    name: str = "pathway_engine"
    version: str = "8.83.0"
    num_layers: int = 16
    hidden_dim: int = 512
    learning_rate: float = 0.000800
    batch_size: int = 128
    max_epochs: int = 400
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/pathway_engine/v8d83y2022")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
