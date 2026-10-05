"""
ContrastiveVI — sherlock's combined generative model + evaluation class.

Weinberger et al. (2023) "Isolating salient variations of interest in
single-cell data with contrastiveVI." Nature Methods.

Inherits from both scvi's ContrastiveVAE (generative model) and
PerturbModelBase (sherlock evaluation interface), so the trained object
can be used directly for eval/counterfactual without a wrapper.

Paper variable naming vs. scvi implementation:
  Paper z  ↔  scvi z  (background, z_encoder)    — shared across all cells
  Paper t  ↔  scvi s  (salient,     s_encoder)    — perturbation-specific
  Paper s  ↔  scvi batch_index (observed covariate) — batch, NOT perturbation

The perturbation identity has NO explicit slot in the generative model;
perturbation effects are captured entirely by the continuous salient t.

Extension over vanilla ContrastiveVI:
  Replaces the universal N(0,I) prior on t for target cells with a
  learned per-perturbation prior p(t|p) = N(mu_p, diag(exp(log_var_p))).
  Initialized to N(0,I) so training starts identical to vanilla.

Counterfactual ATE:
  1. Abduct z from NTC:  E[q(z | x_NTC)]
  2. Apply pert p's prior mean mu_p as the salient shift
  3. NTC baseline: t = 0 (Dirac prior for background cells)
  4. Decode with batch_index = 0

Perturbation index convention (must match PerturbSimpleDataset):
  NTC = 0, non-NTC labels sorted alphabetically starting at 1.
  Enforced by registering _sherlock_pert_idx (float32 P_indices) as a
  continuous covariate in run(), giving CONT_COVS_KEY access in loss().
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from scvi._constants import REGISTRY_KEYS
from scvi.external.contrastivevi._module import ContrastiveVAE
from scvi.module.base import LossOutput

from ._base import PerturbModelBase


class ContrastiveVI(ContrastiveVAE, PerturbModelBase):
    """
    Sherlock ContrastiveVI: learned per-perturbation salient prior.

    Parameters
    ----------
    n_perturbs : total perturbations including NTC (index 0)
    **kwargs   : forwarded verbatim to ContrastiveVAE (n_input, n_batch, …)
    """

    def __init__(self, n_perturbs: int, combinatorial: bool = False, **kwargs) -> None:
        ContrastiveVAE.__init__(self, **kwargs)
        self.n_perturbs = n_perturbs
        self.combinatorial = combinatorial
        n_s = self.n_salient_latent
        # Per-perturbation prior; init = N(0,I) matches vanilla ContrastiveVI
        self.pert_mu      = nn.Parameter(torch.zeros(n_perturbs, n_s))
        self.pert_log_var = nn.Parameter(torch.zeros(n_perturbs, n_s))  # log sigma^2
        self._pert_t_means: np.ndarray | None = None  # diagnostic: empirical posterior means

    # ── helpers ──────────────────────────────────────────────────────────────

    def _batch_idx(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(n, 1, dtype=torch.long, device=device)

    # ── PerturbModelBase interface ────────────────────────────────────────────

    def checkpoint_ctor_args(self) -> dict:
        return {
            "n_perturbs":          self.n_perturbs,
            "combinatorial":       self.combinatorial,
            "n_input":             self.n_input,
            "n_batch":             self.n_batch,
            "n_hidden":            self.n_hidden,
            "n_background_latent": self.n_background_latent,
            "n_salient_latent":    self.n_salient_latent,
            "n_layers":            self.n_layers,
            "dropout_rate":        self.dropout_rate,
            "use_observed_lib_size": self.use_observed_lib_size,
            "wasserstein_penalty": self.wasserstein_penalty,
        }

    @torch.no_grad()
    def get_z(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Salient mean E[q(t|x)] via s_encoder — perturbation-specific representation."""
        log_x     = torch.log1p(x)
        batch_idx = self._batch_idx(x.shape[0], x.device)
        t_mean, _, _ = self.s_encoder(log_x, batch_idx)
        return t_mean

    @torch.no_grad()
    def get_recon(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Decode cat([E[q(z|x)], E[q(t|x)]]) → expected NB counts."""
        log_x     = torch.log1p(x)
        batch_idx = self._batch_idx(x.shape[0], x.device)
        z_mean, _, _ = self.z_encoder(log_x, batch_idx)
        t_mean, _, _ = self.s_encoder(log_x, batch_idx)
        library   = torch.log(x.sum(1).clamp_min(1e-8)).unsqueeze(1)
        zt        = torch.cat([z_mean, t_mean], dim=1)
        _, _, px_rate, _ = self.decoder("gene", zt, library, batch_idx)
        return px_rate

    def _get_rho_embed(
        self,
        pert_indices: np.ndarray,
        z: np.ndarray,
        p_indices: np.ndarray,
        all_perts: np.ndarray,
    ) -> np.ndarray:
        """Pseudobulk mean of salient t per perturbation."""
        P     = len(pert_indices)
        embed = np.zeros((P, z.shape[1]), dtype=np.float32)
        for j, pidx in enumerate(pert_indices):
            mask = p_indices == pidx
            if mask.any():
                embed[j] = z[mask].mean(0)
        return embed

    # ── counterfactual hooks ──────────────────────────────────────────────────

    @torch.no_grad()
    def _abduct_ntc(self, x_ntc: torch.Tensor) -> torch.Tensor:
        """E[q(z | x_NTC)] — background latent for NTC cells."""
        log_x     = torch.log1p(x_ntc)
        batch_idx = self._batch_idx(x_ntc.shape[0], x_ntc.device)
        z_mean, _, _ = self.z_encoder(log_x, batch_idx)
        return z_mean

    @torch.no_grad()
    def _apply_shift(self, u: torch.Tensor, m_p: torch.Tensor) -> torch.Tensor:
        """cat([z_background, t_pert]) → full decoder input."""
        return torch.cat([u, m_p.expand(u.shape[0], -1)], dim=1)

    @torch.no_grad()
    def _get_all_pert_shifts(self, device: torch.device) -> torch.Tensor | None:
        """Learned prior means mu_p for each perturbation, shape (P, n_salient)."""
        return self.pert_mu.detach().to(device)

    @torch.no_grad()
    def _decode_to_expr(self, z: torch.Tensor, lib_size: float) -> torch.Tensor:
        """
        Decode z (K, n_background) or full zt (K, n_background+n_salient) → counts.
        Background-only input pads t=0 for the NTC baseline.
        """
        B         = z.shape[0]
        batch_idx = self._batch_idx(B, z.device)
        if z.shape[1] == self.n_background_latent:
            t_pad = torch.zeros(B, self.n_salient_latent, dtype=z.dtype, device=z.device)
            zt    = torch.cat([z, t_pad], dim=1)
        else:
            zt = z
        lib_log = torch.full(
            (B, 1), float(np.log(lib_size + 1e-8)), dtype=z.dtype, device=z.device
        )
        _, _, px_rate, _ = self.decoder("gene", zt, lib_log, batch_idx)
        return px_rate

    # ── ContrastiveVAE loss override ──────────────────────────────────────────

    def loss(
        self,
        concat_tensors: dict,
        inference_outputs: dict,
        generative_outputs: dict,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """
        Replaces KL(q(t|x_target) || N(0,I)) with
        KL(q(t|x_target) || N(mu_p, exp(log_var_p))) for target cells.

        Perturbation indices come from CONT_COVS_KEY[:, 0], which run()
        populates with PerturbSimpleDataset.P_indices cast to float32.
        """
        bg = concat_tensors["background"]
        tg = concat_tensors["target"]

        B = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(bg, B)
        self._reduce_tensors_to_min_batch_size(tg, B)

        bg_losses = self._generic_loss(bg, inference_outputs["background"], generative_outputs["background"])
        tg_losses = self._generic_loss(tg, inference_outputs["target"],     generative_outputs["target"])

        # ── per-perturbation KL for target cells ─────────────────────────
        p0 = tg[REGISTRY_KEYS.CONT_COVS_KEY][:, 0].long()

        qs_m    = inference_outputs["target"]["qs_m"]                           # (B, n_s)
        qs_v    = inference_outputs["target"]["qs_v"]                           # (B, n_s) variance
        prior_m = self.pert_mu[p0]                                              # (B, n_s)
        prior_v = self.pert_log_var[p0].exp().clamp(min=1e-4)                   # (B, n_s) variance

        # combinatorial: second component in CONT_COVS column 1; -1 = absent.
        # p(t | p1, p2) = N(mu_p1 + mu_p2, var_p1 + var_p2) by sum-of-Gaussians
        if self.combinatorial and tg[REGISTRY_KEYS.CONT_COVS_KEY].shape[1] > 1:
            p1 = tg[REGISTRY_KEYS.CONT_COVS_KEY][:, 1].long()                  # (B,) -1 = absent
            has_p1 = (p1 >= 0).float().unsqueeze(-1)                            # (B, 1)
            prior_m = prior_m + self.pert_mu[p1.clamp(min=0)] * has_p1
            prior_v = prior_v + self.pert_log_var[p1.clamp(min=0)].exp().clamp(min=1e-4) * has_p1

        # KL(N(qs_m, qs_v) || N(prior_m, prior_v))
        kl_s = 0.5 * (
            qs_v / prior_v
            + (qs_m - prior_m).pow(2) / prior_v
            - 1.0
            + prior_v.log()
            - qs_v.clamp(min=1e-8).log()
        ).sum(-1)                                                                # (B,)

        # ── total loss (mirrors ContrastiveVAE.loss structure) ────────────
        reconst = bg_losses["recon_loss"] + tg_losses["recon_loss"]
        kl_z    = bg_losses["kl_z"]       + tg_losses["kl_z"]
        kl_l    = bg_losses["kl_library"] + tg_losses["kl_library"]

        wass = (
            inference_outputs["background"]["qs_m"].norm(dim=-1).pow(2)
            + inference_outputs["background"]["qs_v"].sum(-1)
        )

        loss = torch.mean(
            reconst
            + kl_weight * (self.wasserstein_penalty * wass + kl_z + kl_s)
            + kl_l
        )

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst,
            kl_local={
                "kl_divergence_l": kl_l,
                "kl_divergence_z": kl_z,
                "kl_divergence_s": kl_s,
            },
            extra_metrics={"wasserstein_loss_sum": wass.sum()},
        )

    # ── model-specific eval hook ──────────────────────────────────────────────

    @torch.no_grad()
    def _eval(self, adata, obsm_key: str, device: torch.device) -> dict:
        """
        Stores empirical posterior means of t per perturbation in _pert_t_means
        for diagnostics.  Counterfactuals use pert_mu (the learned prior mean).
        """
        from ._datasets import PerturbSimpleDataset
        from .._configs import get_config

        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")
        ds        = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label,
                                          combinatorial=self.combinatorial)

        t_all = adata.obsm[obsm_key]   # (N, n_salient) — E[q(t|x)] per cell
        p_all = ds.P_indices            # (N,) or (N, 2) for combinatorial

        means = np.zeros((ds.n_perturbs, t_all.shape[1]), dtype=np.float32)
        if self.combinatorial:
            # use only single-pert cells (col 1 == -1); index from col 0
            single_mask = p_all[:, 1] == -1
            p_single    = p_all[single_mask, 0]
            t_single    = t_all[single_mask]
            for pidx in range(ds.n_perturbs):
                mask = p_single == pidx
                if mask.any():
                    means[pidx] = t_single[mask].mean(0)
        else:
            for pidx in range(ds.n_perturbs):
                mask = p_all == pidx
                if mask.any():
                    means[pidx] = t_all[mask].mean(0)

        self._pert_t_means = means
        return {}
