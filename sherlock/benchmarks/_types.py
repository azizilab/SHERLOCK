from dataclasses import dataclass
from typing import Dict, Any
import torch


@dataclass
class BenchmarkResult:
    model_name: str
    model: torch.nn.Module | None
    metrics: Dict[str, float]
    extras: Dict[str, Any]