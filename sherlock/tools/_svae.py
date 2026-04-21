"""
sVAE — Sparse Mechanism Shift VAE
Lopez et al. (2023) "Learning Causal Representations of Single Cells
via Sparse Mechanism Shift Modeling."  CLeaR 2023.
https://proceedings.mlr.press/v213/lopez23a/
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

from ._base import PerturbModelBase


# ── GumbelSigmoid ─────────────────────────────────────────────────────────────

class GumbelSigmoid(nn.Module):
    """
    Per-perturbation per-latent-dim stochastic binary mask via Gumbel-sigmoid.

    Adapted from Lopez et al. (2023) / Genentech/sVAE _utils.py.
    Uses a straight-through estimator so gradients flow through the hard mask.

    Parameters
    ----------
    n_perturbs : number of distinct perturbation labels
    latent_dim : latent space dimensionality
    tau        : temperature (lower → harder samples; 1.0 default)
    drawhard   : if True, round y_soft → {0,1} with straight-through gradient
    """

    def __init__(
        self,
        n_perturbs: int,
        latent_dim: int,
        tau: float = 1.0,
        drawhard: bool = True,
    ):
        super().__init__()
        self.tau = tau
        self.drawhard = drawhard
        self._frozen = False
        # Initialised to 5 → P(mask=1) ≈ 0.993.
        # The Beta sparsity prior drives log_alpha negative during training.
        self.log_alpha = nn.Parameter(torch.full((n_perturbs, latent_dim), 5.0))
        self.register_buffer("fixed_mask", torch.ones(n_perturbs, latent_dim))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Sample a mask for a batch of perturbation indices.

        Parameters
        ----------
        actions : (B,) long tensor of perturbation indices

        Returns
        -------
        mask : (B, latent_dim) — hard {0,1} via straight-through (or soft)
        """
        if self._frozen:
            return self.fixed_mask[actions]

        la = self.log_alpha[actions]                          # (B, d)
        u = torch.zeros_like(la).uniform_().clamp_(1e-6, 1.0 - 1e-6)
        logistic_noise = u.log() - (1.0 - u).log()
        y_soft = torch.sigmoid((la + logistic_noise) / self.tau)

        if self.drawhard:
            y_hard = (y_soft > 0.5).to(y_soft.dtype)
            # Straight-through: hard sample in forward, soft in backward
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def get_proba(self) -> torch.Tensor:
        """Marginal P(mask_kd = 1) = σ(log_alpha_kd).  Shape: (P, d)."""
        return torch.sigmoid(self.log_alpha)

    def freeze(self) -> None:
        """Binarise at the 0.5 threshold and freeze for deterministic inference."""
        proba = self.get_proba()
        self.fixed_mask.copy_((proba > 0.5).to(proba.dtype))
        self._frozen = True

    def set_temperature(self, tau: float) -> None:
        self.tau = float(tau)


# ── MLP helper ────────────────────────────────────────────────────────────────

def _mlp(
    input_dim: int,
    output_dim: int,
    n_hidden: int,
    n_layers: int,
    dropout: float,
) -> nn.Sequential:
    """Build a BatchNorm–ReLU–Dropout MLP."""
    layers: list[nn.Module] = []
    prev = input_dim
    for _ in range(n_layers):
        layers += [
            nn.Linear(prev, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        prev = n_hidden
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ── sVAE ──────────────────────────────────────────────────────────────────────

class sVAE(PerturbModelBase):
    """
    Sparse Mechanism Shift VAE (Lopez et al., CLeaR 2023).

    Parameters
    ----------
    input_dim           : number of input genes (G)
    latent_dim          : latent space dimensionality (d)
    n_perturbs          : total perturbation labels including NTC (NTC = index 0)
    n_hidden            : hidden units per MLP layer
    n_layers            : number of hidden MLP layers in encoder and decoder
    dropout_rate        : dropout probability
    sparse_mask_penalty : λ in Beta(1, λ) prior; higher → sparser masks
    tau                 : GumbelSigmoid temperature (lower → harder samples)
    beta                : β-VAE weight on the KL(z) divergence term
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        n_perturbs: int = 1,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        sparse_mask_penalty: float = 10.0,
        tau: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_perturbs = n_perturbs
        self.sparse_mask_penalty = sparse_mask_penalty
        self.beta = beta

        # ── Encoder q(z|x) — does NOT condition on perturbation label ─────
        self.encoder = _mlp(input_dim, 2 * latent_dim, n_hidden, n_layers, dropout_rate)

        # ── Decoder p(x|z) ────────────────────────────────────────────────
        self.decoder = _mlp(latent_dim, input_dim, n_hidden, n_layers, dropout_rate)

        # ── NB dispersion per gene (unconstrained; softplus at use time) ──
        self.px_r = nn.Parameter(torch.zeros(input_dim))

        # ── Per-perturbation mean shift μ_d ∈ R^d ─────────────────────────
        # Initialised near zero; sparsity prior keeps them small
        self.action_prior_mean = nn.Parameter(
            torch.randn(n_perturbs, latent_dim) * 0.01
        )

        # ── Beta-prior logit weights w_d  (p_d = σ(w_d)) ─────────────────
        self.action_prior_logit_weight = nn.Parameter(
            torch.ones(n_perturbs, latent_dim)
        )

        # ── GumbelSigmoid mask m_d ────────────────────────────────────────
        self.gumbel_action = GumbelSigmoid(n_perturbs, latent_dim, tau=tau)

    # ── Pyro generative model ─────────────────────────────────────────────────

    def model(self, x: torch.Tensor, p: torch.Tensor) -> None:
        pyro.module("svae", self)

        N = x.shape[0]
        theta = F.softplus(self.px_r) + 1e-3            # (G,) NB dispersion

        # ── Global sparsity prior on mask probabilities ────────────────────
        # Beta(1, λ) prior encourages q_proba → 0 (sparse masks).
        # logp_mask is added as a factor so the ELBO includes it.
        q_proba = self.gumbel_action.get_proba().clamp(1e-6, 1.0 - 1e-6)  # (P, d)
        prior_w = torch.ones_like(q_proba)
        logp_mask = torch.distributions.Beta(
            prior_w, prior_w * self.sparse_mask_penalty
        ).log_prob(q_proba).sum()
        pyro.factor("mask_sparsity", logp_mask)

        with pyro.plate("cells", N):
            # Sparse prior: p(z|d) = N(μ_d ⊙ m_d, I)
            mu_d  = self.action_prior_mean[p]      # (N, d)
            mask  = self.gumbel_action(p)           # (N, d) straight-through
            prior_loc = mu_d * mask                 # (N, d)

            z = pyro.sample(
                "z",
                dist.Normal(prior_loc, torch.ones_like(prior_loc)).to_event(1),
            )

            # Decode: μ = library × softmax(decoder(z))
            library = x.sum(dim=-1, keepdim=True)      # (N, 1)
            scale   = F.softmax(self.decoder(z), dim=-1)  # (N, G)
            mu      = library * scale                      # (N, G)

            logits_nb = torch.log(mu + 1e-6) - torch.log(theta)
            pyro.sample(
                "x_obs",
                dist.NegativeBinomial(total_count=theta, logits=logits_nb).to_event(1),
                obs=x,
            )

    # ── Pyro variational posterior ────────────────────────────────────────────

    def guide(self, x: torch.Tensor, p: torch.Tensor) -> None:
        pyro.module("svae", self)

        N     = x.shape[0]
        x_norm = torch.log1p(x)                      # log(1 + counts)
        out   = self.encoder(x_norm)
        z_mu, z_log_sigma = out.chunk(2, dim=-1)
        z_sigma = F.softplus(z_log_sigma) + 1e-4

        with pyro.plate("cells", N):
            pyro.sample("z", dist.Normal(z_mu, z_sigma).to_event(1))

    # ── PerturbModelBase interface ────────────────────────────────────────────

    @torch.no_grad()
    def get_z(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Posterior mean μ_z from the encoder.

        The perturbation label *p* is not used — the encoder is
        perturbation-agnostic by design.
        """
        x_norm = torch.log1p(x)
        z_mu, _ = self.encoder(x_norm).chunk(2, dim=-1)
        return z_mu

    @torch.no_grad()
    def get_recon(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Decode posterior mean z → expected NB mean (counts scale)."""
        z       = self.get_z(x)
        library = x.sum(dim=-1, keepdim=True)
        scale   = F.softmax(self.decoder(z), dim=-1)
        return library * scale

    # ── sVAE-specific utilities ───────────────────────────────────────────────

    @torch.no_grad()
    def _eval(self, adata, obsm_key: str, device: torch.device) -> dict:
        from ._datasets import PerturbSimpleDataset
        from .._configs import get_config

        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")

        dataset      = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label)
        idx_to_pert  = dataset.idx_to_pert()
        ntc_idx      = dataset.ntc_idx

        # Strict hard-mask evaluation to avoid soft-gating inflation.ß.
        proba = self.gumbel_action.get_proba().detach().cpu()              # (P_all, d)
        hard_mask = (proba > 0.45).to(proba.dtype)                          # (P_all, d)
        means = self.action_prior_mean.detach().cpu()                      # (P_all, d)
        eff = means * hard_mask                                            # (P_all, d)

        # keep only non-NTC rows, in dataset index order
        non_ntc_items = sorted((i, n) for i, n in idx_to_pert.items() if i != ntc_idx)
        non_ntc_idx   = [i for i, _ in non_ntc_items]
        non_ntc_names = [n for _, n in non_ntc_items]

        eff_non_ntc = eff[non_ntc_idx].numpy()                             # (P, d)
        P = eff_non_ntc.shape[0]

        if P >= 2:
            X = eff_non_ntc - eff_non_ntc.mean(axis=1, keepdims=True)
            row_norm = np.linalg.norm(X, axis=1)
            nz = row_norm > 1e-12

            rho_corr = np.zeros((P, P), dtype=np.float32)
            if int(nz.sum()) >= 2:
                Xn = X[nz] / row_norm[nz][:, None]
                Cnz = np.clip(Xn @ Xn.T, -1.0, 1.0).astype(np.float32)
                nz_idx = np.where(nz)[0]
                rho_corr[np.ix_(nz_idx, nz_idx)] = Cnz
            np.fill_diagonal(rho_corr, 1.0)
            rho_perts = np.array(non_ntc_names)
        else:
            rho_corr = np.empty((0, 0), dtype=np.float32)
            rho_perts = np.array([], dtype=object)

        return {
            "rho_corr":  rho_corr,
            "rho_perts": rho_perts,
        }

    @torch.no_grad()
    def get_mask(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return mask probabilities (P × d) or the binarised hard mask.

        Parameters
        ----------
        deterministic : if True return (proba > 0.5).float() — the hard mask
        """
        proba = self.gumbel_action.get_proba()
        if deterministic:
            return (proba > 0.5).float()
        return proba
