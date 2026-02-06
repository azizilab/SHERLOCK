from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass
from typing import Dict, Any

from ._datasets import PerturbBenchmarkDataset
from .models import MODEL_REGISTRY



@dataclass
class BenchmarkResult:
    model_name: str
    model: torch.nn.Module | None
    metrics: Dict[str, float]
    extras: Dict[str, Any]


def run_benchmark(
    adata,
    *,
    model: str,
    batch_size: int = 4096,
    val_frac: float = 0.1,
    latent_dim: int = 16,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    lr: float = 5e-4,
    num_epochs: int = 50,
    validate_every: int = 5,
    patience: int = 20,
    device: str | torch.device = "cpu",
    seed: int = 0,
    treat_effect=None,
) -> BenchmarkResult:
    model = model.lower()

    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown benchmark model '{model}'. "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    entry = MODEL_REGISTRY.get(model)

    if entry is None or entry.get("model") is None or entry.get("trainer") is None:
        raise NotImplementedError(f"Benchmark model '{model}' not implemented yet.")

    if isinstance(device, str):
        device = torch.device(device)

    # Dataset
    ds = PerturbBenchmarkDataset(adata)

    # Split
    n = len(ds)
    n_val = int(round(val_frac * n))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


    entry = MODEL_REGISTRY.get(model)

    if entry is None:
        raise NotImplementedError(f"Benchmark model '{model}' not implemented yet.")
    
    ModelClass = entry["model"]
    TrainerClass = entry["trainer"]

    net = ModelClass(
        input_dim=adata.shape[1],
        latent_dim=latent_dim,
        perturbs=len(ds.perturbation_dict),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    trainer = TrainerClass(
        cVAE=net,
        dataloader=train_loader,
        val_dataloader=val_loader,
        treat_effect=treat_effect,
        lr=lr,
        num_epochs=num_epochs,
        validate_every=validate_every,
        patience=patience,
        device=device,
    )

    trained_model = trainer.fit()

    metrics = {
        "r2": float(trainer._last_valid.get("R2", np.nan)),
        "ate": float(trainer._last_valid.get("ATE", np.nan)),
        "val_loss": float(trainer._last_valid.get("val_loss", np.nan)),
    }

    return BenchmarkResult(
        model_name=model,
        model=trained_model,
        metrics=metrics,
        extras={
            "n_train": n_train,
            "n_val": n_val,
            "n_perts": len(ds.perturbation_dict),
            "ntc_idx": ds.ntc_idx,
        },
    )
