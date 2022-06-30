"""Configuration for pathway_engine v4d89y2022."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class PathwayEngineConfig_v4d89y2022:
    name: str = "pathway_engine"
    version: str = "4.89.0"
    num_layers: int = 8
    hidden_dim: int = 256
    learning_rate: float = 0.000400
    batch_size: int = 64
    max_epochs: int = 200
    dropout: float = 0.4
    checkpoint_dir: Path = Path("checkpoints/pathway_engine/v4d89y2022")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
