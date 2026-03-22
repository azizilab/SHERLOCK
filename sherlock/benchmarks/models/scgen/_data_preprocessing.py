from __future__ import annotations

from anndata import AnnData
import scanpy as sc
import numpy as np


def preprocess_for_scgen(
    adata: AnnData,
    pert_col: str,
    control_label: str,
    min_cells: int = 30,
    target_sum: float = 1e4,
    inplace: bool = False,
) -> AnnData:
    """
    Prepare an AnnData object for scGen.

    Steps:
    1. Filter perturbation groups with too few cells, keeping control.
    2. Normalize total counts per cell.
    3. Apply log1p transform.
    """
    if not inplace:
        adata = adata.copy()

    if pert_col not in adata.obs.columns:
        raise KeyError(f"Column '{pert_col}' not found in adata.obs.")

    labels = adata.obs[pert_col].astype(str)
    if control_label not in labels.unique():
        raise ValueError(f"Control label '{control_label}' not found in adata.obs['{pert_col}'].")

    # Filter perturbations with too few cells (always keep control)
    counts = labels.value_counts()
    keep_labels = counts[counts >= min_cells].index.tolist()
    if control_label not in keep_labels:
        keep_labels.append(control_label)

    adata = adata[labels.isin(keep_labels)].copy()

    # Normalize per cell
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # Log transform
    sc.pp.log1p(adata)

    return adata