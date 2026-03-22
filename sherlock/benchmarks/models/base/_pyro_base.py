from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

class BasePyroPerturbModel(nn.Module):
    """
    Pyro models should implement:
      - model(*batch)
      - guide(*batch)
      - predict_mu(*batch): return predicted mean expression (for r2/ate)
      - latent(*batch): optional (for analysis)
    """
    def model(self, *args, **kwargs):
        raise NotImplementedError

    def guide(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def predict_mu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def latent(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError