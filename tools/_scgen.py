"""
ScGen — sherlock wrapper for the scGen algorithm.

Lotfollahi et al. (2019) "scGen predicts single-cell perturbation responses."
Nature Methods 16, 715–721. https://doi.org/10.1038/s41592-019-0494-8
GitHub: https://github.com/theislab/scgen

Algorithm
---------
1. Train a standard unconditional VAE on all cells (no perturbation conditioning).
   Loss: MSE(log1p(x), decode(z)) + kl_weight · KL[q(z|x) ∥ N(0,I)]

2. Post-training, estimate a per-perturbation shift in latent space:
       Δ_p = mean_z(perturbed) − mean_z(NTC)
   (scgen calls this 'delta' in its predict() method)

3. ATE prediction for a control cell i under perturbation p:
       z_cf = encode(x_i^NTC) + Δ_p
       x_cf = decode(z_cf)

Note on causality
-----------------
scGen is an **associational** ATE estimator, not a Pearl-style structural
counterfactual.  Δ_p is derived from the *observed* marginal shift in latent
space between perturbed and control populations; it does not perform
abduction–action–prediction at the individual level.  Using mean(encode(x_p))
to define the shift constitutes data leakage in the Pearl counterfactual sense:
the shift vector is estimated from the very population whose counterfactual
we claim to predict.  Results should be interpreted as population-level ATE
estimates, not individual counterfactuals.

Implementation note
-------------------
The upstream scgen package (v2.1.0, pip install scgen) imports from
``scvi._compat`` which was removed in scvi-tools ≥ 1.0.  This file
reimplements the SCGENVAE architecture and algorithm directly from the scgen
source (https://github.com/theislab/scgen) using scvi.nn building blocks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi.nn import Encoder, FCLayers

from ._base import PerturbModelBase


class SCGENModel(PerturbModelBase):
    """
    ScGen-style unconditional VAE with vector-arithmetic ATE prediction.

    Architecture matches SCGENVAE from the scgen repo:
    - Encoder : log1p(x) → (q_mean, q_var, z)  via scvi.nn.Encoder
    - Decoder : z → FCLayers → linear → log1p-scale prediction
    - Loss    : MSE(log1p(x), decode(z)) + kl_weight · KL   (scgen original)

    After training, per-perturbation delta vectors are fitted by calling
    fit_deltas(adata, device) and stored as a non-trainable buffer so they
    are preserved by the checkpoint utilities.

    Parameters
    ----------
    input_dim    : G — number of input genes
    n_perturbs   : total perturbations including NTC (index 0)
    latent_dim   : latent space dimensionality (scgen default: 100)
    n_hidden     : hidden units per MLP layer   (scgen default: 800)
    n_layers     : number of hidden MLP layers  (scgen default: 2)
    dropout_rate : dropout probability          (scgen default: 0.2)
    kl_weight    : weight on the KL term        (scgen default: 0.00005)
    """

    def __init__(
        self,
        input_dim: int,
        n_perturbs: int,
        latent_dim: int = 100,
        n_hidden: int = 800,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        kl_weight: float = 0.00005,
    ):
        super().__init__()
        self.input_dim    = input_dim
        self.n_perturbs   = n_perturbs
        self.n_latent     = latent_dim
        self.n_hidden     = n_hidden
        self.n_layers     = n_layers
        self.dropout_rate = dropout_rate
        self.kl_weight    = kl_weight

        # Encoder — matches SCGENVAE.z_encoder
        self.z_encoder = Encoder(
            input_dim,
            latent_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution="normal",
            use_batch_norm=True,
            return_dist=True,
            activation_fn=nn.LeakyReLU,
        )

        # Decoder — matches DecoderSCGEN from scgen._base_components
        # FCLayers: latent_dim → n_hidden  (n_layers hidden layers)
        # linear_out: n_hidden → input_dim
        self.decoder = FCLayers(
            n_in=latent_dim,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            activation_fn=nn.LeakyReLU,
        )
        self.decoder_out = nn.Linear(n_hidden, input_dim)

        # Per-perturbation delta vectors (fitted post-training, not trained).
        # Row 0 (NTC) is zero by convention.
        self.register_buffer("pert_deltas", torch.zeros(n_perturbs, latent_dim))

    # ── decoder ───────────────────────────────────────────────────────────────

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """z → log1p-scale prediction (matches SCGENVAE.generative output)."""
        return self.decoder_out(self.decoder(z))

    # ── training loss ─────────────────────────────────────────────────────────

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        scgen loss (Lotfollahi et al. 2019, Eq. 1):
            L = 0.5 · Σ_g (log1p(x_g) − x̂_g)²  +  0.5 · kl_weight · KL

        No perturbation label is used — the VAE is unconditional.
        """
        log_x = torch.log1p(x)
        qz, z = self.z_encoder(log_x)
        x_hat = self._decode(z)

        rl  = ((log_x - x_hat) ** 2).sum(dim=1)                          # MSE sum over genes
        kld = kl(qz, Normal(torch.zeros_like(qz.loc),
                             torch.ones_like(qz.scale))).sum(dim=1)       # KL

        return (0.5 * rl + 0.5 * (kld * self.kl_weight)).mean()

    # ── PerturbModelBase interface ────────────────────────────────────────────

    def checkpoint_ctor_args(self) -> dict:
        return dict(
            input_dim    = self.input_dim,
            n_perturbs   = self.n_perturbs,
            latent_dim   = self.n_latent,
            n_hidden     = self.n_hidden,
            n_layers     = self.n_layers,
            dropout_rate = self.dropout_rate,
            kl_weight    = self.kl_weight,
        )

    @torch.no_grad()
    def get_z(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Encoder posterior mean E[q(z|x)]."""
        qz, _ = self.z_encoder(torch.log1p(x))
        return qz.loc

    @torch.no_grad()
    def get_recon(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Encode → decode → count-space prediction (expm1 of log1p output)."""
        qz, _ = self.z_encoder(torch.log1p(x))
        return torch.expm1(self._decode(qz.loc)).clamp(min=0.0)

    def _get_rho_embed(
        self,
        pert_indices: np.ndarray,
        z: np.ndarray,
        p_indices: np.ndarray,
        all_perts: np.ndarray,
    ) -> np.ndarray:
        """Pseudobulk mean of encoder z per perturbation."""
        embed = np.zeros((len(pert_indices), z.shape[1]), dtype=np.float32)
        for j, pidx in enumerate(pert_indices):
            mask = p_indices == pidx
            if mask.any():
                embed[j] = z[mask].mean(0)
        return embed

    # ── delta fitting (post-training) ─────────────────────────────────────────

    @torch.no_grad()
    def fit_deltas(
        self,
        adata,
        device: torch.device,
        batch_size: int = 1024,
    ) -> None:
        """
        Compute per-perturbation population-level shift vectors from training data.

            Δ_p = mean(encode(x_p)) − mean(encode(x_NTC))

        Matches scgen's predict() / _avg_vector() logic.  See note on causality
        in the module docstring — this is associational, not a Pearl counterfactual.

        Results are stored in self.pert_deltas and saved with the checkpoint.
        """
        from torch.utils.data import DataLoader
        from ._datasets import PerturbSimpleDataset
        from .._configs import get_config

        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")
        ds        = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label)

        self.to(device)
        self.eval()

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        z_parts, p_parts = [], []
        for x_batch, p_batch in loader:
            x_batch = x_batch.to(device, dtype=torch.float32)
            qz, _ = self.z_encoder(torch.log1p(x_batch))
            z_parts.append(qz.loc.cpu().numpy())
            p_parts.append(p_batch.cpu().numpy())

        z_all = np.concatenate(z_parts, axis=0)   # (N, d)
        p_all = np.concatenate(p_parts, axis=0)   # (N,)

        ntc_idx    = ds.ntc_idx
        ntc_mean   = z_all[p_all == ntc_idx].mean(0).astype(np.float32)

        deltas = np.zeros((ds.n_perturbs, self.n_latent), dtype=np.float32)
        for pidx in range(ds.n_perturbs):
            if pidx == ntc_idx:
                continue
            mask = p_all == pidx
            if mask.any():
                deltas[pidx] = z_all[mask].mean(0).astype(np.float32) - ntc_mean

        self.pert_deltas.copy_(torch.tensor(deltas, device=device))

    # ── counterfactual hooks ──────────────────────────────────────────────────

    @torch.no_grad()
    def _abduct_ntc(self, x_ntc: torch.Tensor) -> torch.Tensor:
        """Encode NTC cells → posterior mean z (abduction step)."""
        qz, _ = self.z_encoder(torch.log1p(x_ntc))
        return qz.loc

    @torch.no_grad()
    def _get_all_pert_shifts(self, device: torch.device) -> torch.Tensor:
        """Return precomputed Δ_p vectors, shape (P, latent_dim)."""
        return self.pert_deltas.to(device)

    @torch.no_grad()
    def _decode_to_expr(self, z: torch.Tensor, lib_size: float) -> torch.Tensor:
        """Decode z → count-space expression (expm1 of log1p-scale output)."""
        return torch.expm1(self._decode(z)).clamp(min=0.0)

    # ── model-specific eval hook ──────────────────────────────────────────────

    @torch.no_grad()
    def _eval(self, adata, obsm_key: str, device: torch.device) -> dict:
        """Refresh delta vectors after encoding the full dataset."""
        self.fit_deltas(adata, device)
        return {}
