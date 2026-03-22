# sherlock/benchmarks/models/_base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

@dataclass
class ForwardOut:
    x_mu: torch.Tensor
    x_theta: Optional[torch.Tensor]
    z: Optional[torch.Tensor]
    mu_q: Optional[torch.Tensor]
    logvar_q: Optional[torch.Tensor]
    extras: Dict[str, Any]

class BasePerturbModel(nn.Module):
    def forward(self, x: torch.Tensor, p: torch.Tensor) -> ForwardOut:
        raise NotImplementedError

    def loss(self, x: torch.Tensor, p: torch.Tensor, *, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    @torch.no_grad()
    def predict_mu(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.forward(x, p).x_mu

    @torch.no_grad()
    def latent(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.forward(x, p).z