# sherlock/benchmarks/_metrics.py
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import torch


def lognorm_ln(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Library-size normalize to 1e4 then ln(1+x).
    Input/Output: torch tensors
    """
    lib = x.sum(dim=1, keepdim=True).clamp_min(eps)
    x = 1e4 * x / lib
    return torch.log(x + 1.0)


def compute_ate_from_preds_ln(
    preds_ln: np.ndarray,
    p_all: np.ndarray,
    ds,
    treat_effect,
) -> float:
    """
    Pearson corr between predicted ATE matrix (pert x gene) and ground-truth treat_effect (AnnData-like).
    Returns np.nan if treat_effect is None or required dataset fields missing.
    """
    if treat_effect is None:
        return float("nan")

    pert_dict = getattr(ds, "perturbation_dict", {})
    ntc_idx = getattr(ds, "ntc_idx", None)

    if ntc_idx is None or len(pert_dict) == 0:
        return float("nan")

    if preds_ln.size == 0 or p_all.size == 0:
        return float("nan")

    n_perts = len(pert_dict)
    n_genes = preds_ln.shape[1]

    # NTC mean
    ntc_mask = (p_all == ntc_idx)
    if not ntc_mask.any():
        return float("nan")
    ntc_mean = preds_ln[ntc_mask].mean(axis=0)

    # Effects per perturbation
    effect = np.zeros((n_perts, n_genes), dtype=np.float32)
    for pert_name, j in pert_dict.items():
        if j == ntc_idx:
            continue
        mask = (p_all == j)
        if not mask.any():
            continue
        effect[j] = preds_ln[mask].mean(axis=0) - ntc_mean

    # Build names aligned to treat_effect
    idx_to_pert = {j: name for name, j in pert_dict.items()}
    row_names = [idx_to_pert[j] for j in range(len(pert_dict))]

    effect_df = pd.DataFrame(
        effect,
        index=row_names,
        columns=treat_effect.var_names,
    )

    # Align to GT
    effect_aligned = effect_df.loc[treat_effect.obs_names, treat_effect.var_names]
    x = effect_aligned.values.ravel()
    y = np.asarray(treat_effect.X).ravel()

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])