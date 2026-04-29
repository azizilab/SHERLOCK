"""
ContrastiveVIWrapper — wraps scvi.external.ContrastiveVI in PerturbModelBase.

Weinberger et al. (2023) "Isolating salient variations of interest in
single-cell data with contrastiveVI." Nature Methods.

The model separates expression variation into:
  - Background latent z: shared variation captured by both NTC and perturbed cells
  - Salient latent s: perturbation-specific variation; NTC cells are trained with s = 0

This wrapper maps the scvi interface onto PerturbModelBase so that the standard
sherlock eval / clustering / counterfactual pipeline works unchanged.

Mapping:
  get_z(x, p)          → salient mean  E[q(s|x)]         (perturbation-specific)
  get_recon(x, p)      → decode cat([E[q(z|x)], E[q(s|x)]])
  _get_rho_embed       → pseudobulk mean of s per perturbation
  _abduct_ntc(x_ntc)   → background mean E[q(z|x_NTC)]
  _get_all_pert_shifts → per-perturbation mean salient ŝ_p (precomputed in _eval)
  _apply_shift(u, m_p) → cat([u, m_p])   — forms full decoder input [z, s]
  _decode_to_expr      → pads s=0 for NTC baseline; uses full [z, s] for CF
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ._base import PerturbModelBase


class ContrastiveVIWrapper(PerturbModelBase):
    """
    Wraps ``scvi.external.ContrastiveVI`` for compatibility with the sherlock
    ``PerturbModelBase`` interface.

    The salient latent ``s`` plays the role of ``z`` in the base class:
    it is the perturbation-specific representation stored in ``adata.obsm``
    and used for all downstream evaluation.  The background latent ``z`` is
    used internally for counterfactual abduction from NTC cells.

    Parameters
    ----------
    scvi_model          : trained ``ContrastiveVI`` instance
    n_background_latent : dimensionality of the background latent space
    n_salient_latent    : dimensionality of the salient latent space
    """

    def __init__(
        self,
        scvi_model,
        n_background_latent: int,
        n_salient_latent: int,
    ):
        nn.Module.__init__(self)
        # Store without registering scvi internals as nn.Module sub-modules
        object.__setattr__(self, "_scvi_model", scvi_model)
        self.n_background_latent = n_background_latent
        self.n_salient_latent    = n_salient_latent
        self._pert_salient_means: np.ndarray | None = None

    @property
    def _module(self):
        return object.__getattribute__(self, "_scvi_model").module

    def to(self, *args, **kwargs):
        """Move both the wrapper and the underlying scvi module to the target device."""
        self._module.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def _batch_idx(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(n, 1, dtype=torch.long, device=device)

    # ── PerturbModelBase interface ─────────────────────────────────────────────

    def checkpoint_ctor_args(self) -> dict:
        return {
            "n_background_latent": self.n_background_latent,
            "n_salient_latent":    self.n_salient_latent,
        }

    @torch.no_grad()
    def get_z(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Salient mean E[q(s|x)] — the perturbation-specific representation."""
        log_x     = torch.log1p(x)
        batch_idx = self._batch_idx(x.shape[0], x.device)
        s_mean, _, _ = self._module.s_encoder(log_x, batch_idx)
        return s_mean

    @torch.no_grad()
    def get_recon(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Decode cat([E[q(z|x)], E[q(s|x)]]) → expected NB counts."""
        log_x     = torch.log1p(x)
        batch_idx = self._batch_idx(x.shape[0], x.device)
        z_mean, _, _ = self._module.z_encoder(log_x, batch_idx)
        s_mean, _, _ = self._module.s_encoder(log_x, batch_idx)
        library   = torch.log(x.sum(1).clamp_min(1e-8)).unsqueeze(1)
        zs        = torch.cat([z_mean, s_mean], dim=1)
        _, _, px_rate, _ = self._module.decoder("gene", zs, library, batch_idx)
        return px_rate

    def _get_rho_embed(
        self,
        pert_indices: np.ndarray,
        z: np.ndarray,
        p_indices: np.ndarray,
        all_perts: np.ndarray,
    ) -> np.ndarray:
        """Pseudobulk mean of salient s per perturbation."""
        P = len(pert_indices)
        d = z.shape[1]
        embed = np.zeros((P, d), dtype=np.float32)
        for j, pidx in enumerate(pert_indices):
            mask = p_indices == pidx
            if mask.any():
                embed[j] = z[mask].mean(0)
        return embed

    # ── counterfactual hooks ───────────────────────────────────────────────────

    @torch.no_grad()
    def _abduct_ntc(self, x_ntc: torch.Tensor) -> torch.Tensor:
        """Abduct background latent z for NTC cells."""
        log_x     = torch.log1p(x_ntc)
        batch_idx = self._batch_idx(x_ntc.shape[0], x_ntc.device)
        z_mean, _, _ = self._module.z_encoder(log_x, batch_idx)
        return z_mean

    @torch.no_grad()
    def _apply_shift(self, u: torch.Tensor, m_p: torch.Tensor) -> torch.Tensor:
        """
        Form full decoder input cat([background_z, salient_s_p]).

        Parameters
        ----------
        u   : (K, n_background_latent) — background centroids from NTC
        m_p : (1, n_salient_latent)    — mean salient for perturbation p
        """
        return torch.cat([u, m_p.expand(u.shape[0], -1)], dim=1)

    @torch.no_grad()
    def _get_all_pert_shifts(self, device: torch.device) -> torch.Tensor | None:
        """
        Per-perturbation mean salient latent, shape (P, n_salient_latent).
        Precomputed by _eval() after full-dataset inference.
        """
        if self._pert_salient_means is None:
            return None
        return torch.tensor(self._pert_salient_means, device=device)

    @torch.no_grad()
    def _decode_to_expr(self, z: torch.Tensor, lib_size: float) -> torch.Tensor:
        """
        Decode from full [background, salient] input or background-only (s padded to 0).

        Called with background-only centroids (K, n_background) for the NTC baseline;
        called with cat([z, s]) of shape (K, n_background + n_salient) for CF decoding.
        """
        B         = z.shape[0]
        batch_idx = self._batch_idx(B, z.device)
        if z.shape[1] == self.n_background_latent:
            s_zero = torch.zeros(B, self.n_salient_latent, dtype=z.dtype, device=z.device)
            zs = torch.cat([z, s_zero], dim=1)
        else:
            zs = z
        lib_log = torch.full((B, 1), float(np.log(lib_size + 1e-8)), dtype=z.dtype, device=z.device)
        _, _, px_rate, _ = self._module.decoder("gene", zs, lib_log, batch_idx)
        return px_rate

    # ── model-specific eval hook ───────────────────────────────────────────────

    @torch.no_grad()
    def _eval(self, adata, obsm_key: str, device: torch.device) -> dict:
        """
        Precompute per-perturbation mean salient vectors for counterfactual use.
        Called by PerturbModelBase.eval() after obsm[obsm_key] has been populated.
        """
        from ._datasets import PerturbSimpleDataset
        from .._configs import get_config

        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")
        ds = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label)

        s_all = adata.obsm[obsm_key]  # (N, n_salient_latent) — stored by base eval()
        p_all = ds.P_indices           # (N,) integer perturbation index per cell

        means = np.zeros((ds.n_perturbs, s_all.shape[1]), dtype=np.float32)
        for pidx in range(ds.n_perturbs):
            mask = p_all == pidx
            if mask.any():
                means[pidx] = s_all[mask].mean(0)

        self._pert_salient_means = means
        return {}
