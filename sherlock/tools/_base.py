"""
Abstract base class for perturbation models.

Subclasses must implement:
    get_z(x, p=None, **kwargs)     → (B, d)
    get_recon(x, p=None, **kwargs) → (B, G)

The concrete eval() handles storing z in adata.obsm, x_pred in
adata.layers, and z_corr / perts in adata.uns via get_corr().
Model-specific extras are added via the overridable _eval() hook.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    import anndata


class PerturbModelBase(ABC, nn.Module):
    """Abstract base for all perturbation models in the sherlock package."""

    # ── abstract interface ────────────────────────────────────────────

    @abstractmethod
    @torch.no_grad()
    def get_z(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Mean posterior latent z.

        Parameters
        ----------
        x  : (B, G) raw counts
        p  : (B,)   integer perturbation indices (optional; model-dependent)
        **kwargs : any additional model-specific inputs

        Returns
        -------
        z_mean : (B, d)
        """
        ...

    @abstractmethod
    @torch.no_grad()
    def get_recon(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Reconstructed expected counts.

        Parameters
        ----------
        x  : (B, G) raw counts (used for library-size scaling)
        p  : (B,)   integer perturbation indices (optional; model-dependent)
        **kwargs : any additional model-specific inputs

        Returns
        -------
        mu : (B, G) expected counts at observed library size
        """
        ...

    # ── shared utility ────────────────────────────────────────────────

    @torch.no_grad()
    def get_corr(
        self,
        z: np.ndarray,
        p_indices: np.ndarray,
        n_perturbs: int,
    ) -> np.ndarray:
        """
        Pseudobulk z correlation matrix over perturbation classes.

        Averages z over cells sharing the same perturbation index, then
        computes a Pearson correlation matrix across those mean vectors.

        Parameters
        ----------
        z          : (N, d) latent embeddings
        p_indices  : (N,)   integer perturbation index per cell
        n_perturbs : total classes (including NTC)

        Returns
        -------
        corr : (n_perturbs, n_perturbs)
        """
        d = z.shape[1]
        z_bar = np.zeros((n_perturbs, d), dtype=np.float32)
        for pidx in range(n_perturbs):
            mask = p_indices == pidx
            if mask.any():
                z_bar[pidx] = z[mask].mean(0)
        return np.corrcoef(z_bar).astype(np.float32)

    # ── inference loop (overridable) ──────────────────────────────────

    def _run_inference(
        self,
        dataset: Dataset,
        device: torch.device,
        batch_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the get_z / get_recon forward passes over *dataset*.

        Override in subclasses if the model needs a different batch format
        (e.g. models that require additional inputs per batch).

        Returns
        -------
        z_all    : (N, d)
        recon_all: (N, G)
        p_all    : (N,)  integer perturbation indices
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        z_parts, recon_parts, p_parts = [], [], []

        with torch.no_grad():
            for x, p in loader:
                x = x.to(device, dtype=torch.float32)
                p = p.to(device)
                z_parts.append(self.get_z(x, p).cpu().numpy())
                recon_parts.append(self.get_recon(x, p).cpu().numpy())
                p_parts.append(p.cpu().numpy())

        return (
            np.concatenate(z_parts,     axis=0).astype(np.float32),
            np.concatenate(recon_parts, axis=0).astype(np.float32),
            np.concatenate(p_parts,     axis=0),
        )

    # ── universal evaluation ──────────────────────────────────────────

    def eval(
        self,
        adata: anndata.AnnData | None = None,
        obsm_key: str = "z",
        uns_key: str = "results",
        device: torch.device | None = None,
        batch_size: int = 1024,
        dataset: Dataset | None = None,
    ):
        """
        Evaluate on *adata*, writing results in-place.

        Called with *no arguments* it falls back to nn.Module.eval()
        (toggles training mode).  Called with *adata* it runs:

          1. _run_inference() → z, x_pred via get_z() / get_recon()
             stored in adata.obsm[obsm_key] and adata.layers["x_pred"]
          2. get_corr()        → adata.uns[uns_key]["z_corr"]
          3. _eval() hook      → model-specific extras merged into uns

        Parameters
        ----------
        adata      : AnnData to evaluate on.
        obsm_key   : key in adata.obsm to store z.
        uns_key    : key in adata.uns to store the results dict.
        device     : torch device; defaults to cpu.
        batch_size : inference batch size.
        dataset    : dataset to use; if None, PerturbSimpleDataset(adata)
                     is constructed automatically.
        """
        if adata is None:
            return super().eval()

        from ._datasets import PerturbSimpleDataset
        from .._configs import get_config

        if device is None:
            device = torch.device("cpu")

        super().eval()
        self.to(device)

        if dataset is None:
            p_key = get_config("pert_key")
            ntc_label = get_config("ntc_label")
            dataset = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label)

        z_all, recon_all, p_all = self._run_inference(dataset, device, batch_size)

        adata.obsm[obsm_key] = z_all
        adata.layers["x_pred"] = recon_all

        idx2pert = dataset.idx_to_pert()
        perts = np.array([idx2pert[i] for i in range(dataset.n_perturbs)])

        uns_data: dict = {
            "z_corr": self.get_corr(z_all, p_all, dataset.n_perturbs),
            "perts": perts,
        }

        uns_data.update(self._eval(adata, obsm_key=obsm_key, device=device))
        adata.uns[uns_key] = uns_data

    def _eval(
        self,
        adata,
        obsm_key: str,
        device: torch.device,
    ) -> dict:
        """
        Hook for model-specific evaluation metrics.

        Called at the end of eval() after z, x_pred and z_corr are stored.
        Override in subclasses to add model-specific results (accuracy, gate
        weights, explained variance, counterfactuals, …).  The returned dict
        is merged into adata.uns[uns_key].  Default returns {}.
        """
        return {}
