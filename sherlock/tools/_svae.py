"""
sVAE — Sparse Mechanism Shift VAE
Lopez et al. (2023) "Learning Causal Representations of Single Cells
via Sparse Mechanism Shift Modeling."  CLeaR 2023.

Faithful PyTorch reimplementation of Genentech/sVAE SpikeSlabVAEModule
(_module.py) and GumbelSigmoid (_utils.py), wrapped in PerturbModelBase
for benchmarking. Uses scvi's Encoder and DecoderSCVI with NegativeBinomial
likelihood. Training is done via a plain PyTorch Adam optimizer (no Pyro).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi.nn import DecoderSCVI, Encoder
from scvi.distributions import NegativeBinomial

from ._base import PerturbModelBase


# ── GumbelSigmoid ─────────────────────────────────────────────────────────────
# Exact copy of Genentech/sVAE svae/_utils.py

class GumbelSigmoid(nn.Module):
    def __init__(self, num_action, num_latent, freeze=False, drawhard=True, tau=1):
        super(GumbelSigmoid, self).__init__()
        self.shape = (num_action, num_latent)
        self.freeze = freeze
        self.drawhard = drawhard
        self.log_alpha = nn.Parameter(torch.zeros(self.shape))
        self.tau = tau
        # useful to make sure these parameters will be pushed to the GPU
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.register_buffer("fixed_mask", torch.ones(self.shape))
        self.reset_parameters()

    # changed this to draw one action per minibatch sample...
    def forward(self, action):
        bs = action.shape[0]
        if self.freeze:
            y = self.fixed_mask[action, :]
            return y
        else:
            shape = tuple([bs] + [self.shape[1]])
            logistic_noise = (
                self.sample_logistic(shape)
                .type(self.log_alpha.type())
                .to(self.log_alpha.device)
            )
            y_soft = torch.sigmoid((self.log_alpha[action] + logistic_noise) / self.tau)

            if self.drawhard:
                y_hard = (y_soft > 0.5).type(y_soft.type())

                # This weird line does two things:
                #   1) at forward, we get a hard sample.
                #   2) at backward, we differentiate the gumbel sigmoid
                y = y_hard.detach() - y_soft.detach() + y_soft

            else:
                y = y_soft

            return y

    def get_proba(self):
        """Returns probability of getting one"""
        if self.freeze:
            return self.fixed_mask
        else:
            return torch.sigmoid(self.log_alpha)

    def reset_parameters(self):
        torch.nn.init.constant_(
            self.log_alpha, 5
        )  # 5)  # will yield a probability ~0.99. Inspired by DCDI

    def sample_logistic(self, shape):
        u = self.uniform.sample(shape)
        return torch.log(u) - torch.log(1 - u)

    def threshold(self):
        proba = self.get_proba()
        self.fixed_mask.copy_((proba > 0.5).type(proba.type()))
        self.freeze = True


# ── sVAE ──────────────────────────────────────────────────────────────────────

class sVAE(PerturbModelBase):
    """
    Sparse Mechanism Shift VAE (Lopez et al., CLeaR 2023).

    Faithful PyTorch reimplementation of Genentech/sVAE SpikeSlabVAEModule,
    wrapped in PerturbModelBase. Architecture mirrors the reference exactly:
    scvi Encoder → z, scvi DecoderSCVI → NegativeBinomial likelihood,
    GumbelSigmoid sparse mechanism shift prior on latent z.

    Parameters
    ----------
    input_dim           : G — number of input genes
    n_perturbs          : number of perturbation labels (n_labels in reference)
    latent_dim          : latent space dimensionality
    n_hidden            : hidden units per MLP layer
    n_layers            : number of hidden MLP layers in encoder and decoder
    dropout_rate        : dropout probability
    sparse_mask_penalty : Beta(1, λ) prior; higher λ → sparser masks
    beta                : KL weight applied during warmup phase
    """

    def __init__(
        self,
        input_dim: int,
        n_perturbs: int,
        latent_dim: int = 10,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        sparse_mask_penalty: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.n_latent = latent_dim
        self.n_perturbs = n_perturbs
        self.sparse_mask_penalty = sparse_mask_penalty
        self.beta = beta
        self.warmup = True
        self.use_global_kl = True
        self.use_chem_prior = True

        self.px_r = nn.Parameter(torch.randn(input_dim))

        # z encoder: n_input → n_latent (no batch/covariate conditioning)
        self.z_encoder = Encoder(
            input_dim,
            latent_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution="normal",
            use_batch_norm=True,
            return_dist=True,
        )
        # l encoder: defined to match reference architecture; library is computed
        # deterministically as log(x.sum(1)) in inference (ql = None in reference)
        self.l_encoder = Encoder(
            input_dim,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            return_dist=True,
        )
        # decoder: n_latent → n_input
        self.decoder = DecoderSCVI(
            latent_dim,
            input_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=True,
            scale_activation="softmax",
        )

        # mu_a — per-perturbation latent mean shift
        self.action_prior_mean = nn.Parameter(torch.randn(n_perturbs, latent_dim))
        # p_a — logit weights for Beta prior
        self.action_prior_logit_weight = nn.Parameter(torch.ones(n_perturbs, latent_dim))
        # q_a — GumbelSigmoid binary mask
        self.gumbel_action = GumbelSigmoid(num_action=n_perturbs, num_latent=latent_dim)

    # ── forward methods (matching SpikeSlabVAEModule) ─────────────────────────

    def inference(self, x: torch.Tensor) -> dict:
        """Encoder — matches SpikeSlabVAEModule.inference()."""
        library = torch.log(x.sum(1)).unsqueeze(1)
        x_ = torch.log(1 + x)
        qz, z = self.z_encoder(x_)
        return dict(z=z, qz=qz, library=library)

    def generative(self, z: torch.Tensor, library: torch.Tensor, p: torch.Tensor) -> dict:
        """Generative model — matches SpikeSlabVAEModule.generative()."""
        px_scale, _, px_rate, _ = self.decoder("gene", z, library)
        px_r = torch.exp(self.px_r)
        px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)

        # Priors
        mask = self.gumbel_action(p)
        mean_z = torch.index_select(self.action_prior_mean, 0, p)
        if self.use_chem_prior:
            pz = Normal(mean_z * mask, torch.ones_like(z))
        else:
            pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return dict(px=px, pz=pz)

    def loss(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        kl_weight: float = 1.0,
        n_obs: int = 1,
    ) -> torch.Tensor:
        """
        Training loss — matches SpikeSlabVAEModule.loss().

        Parameters
        ----------
        x         : (B, G) raw counts
        p         : (B,)   integer perturbation indices
        kl_weight : KL annealing weight (1.0 = fully on)
        n_obs     : total dataset size for loss scaling (scVI convention)
        """
        inf = self.inference(x)
        gen = self.generative(inf["z"], inf["library"], p)

        kl_divergence_z = kl(inf["qz"], gen["pz"]).sum(dim=1)
        kl_divergence_l = 0.0

        reconst_loss = -gen["px"].log_prob(x).sum(-1)

        if self.warmup:
            weighted_kl_local = (
                self.beta * kl_weight * kl_divergence_z + kl_divergence_l
            )
        else:
            weighted_kl_local = kl_divergence_z + kl_divergence_l

        q_discrete = self.gumbel_action.get_proba()
        prior_w = torch.ones_like(self.action_prior_logit_weight)
        logp_qw = (
            torch.distributions.Beta(prior_w, prior_w * self.sparse_mask_penalty)
            .log_prob(q_discrete)
            .sum()
        )

        if self.use_global_kl:
            # practical implementation: set p_discrete = q_discrete (see paper)
            kl_global = -logp_qw
            total_loss = (
                n_obs * torch.mean(reconst_loss + weighted_kl_local)
                + kl_weight * kl_global
            )
        else:
            total_loss = n_obs * torch.mean(reconst_loss + weighted_kl_local)

        return total_loss

    # ── reference utility methods ─────────────────────────────────────────────

    def freeze_params(self) -> None:
        """Freeze decoder/encoder/px_r for test-time action-param fine-tuning."""
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.z_encoder.parameters():
            param.requires_grad = False
        self.px_r.requires_grad = False
        self.action_prior_logit_weight.requires_grad = False

        for _, mod in self.decoder.named_modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.momentum = 0
        for _, mod in self.z_encoder.named_modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.momentum = 0

    def reinit_actsparse_and_freeze(self, loc) -> None:
        """Reinit action params for held-out perturbations then binarize mask."""
        with torch.no_grad():
            self.action_prior_mean[loc] = 0
            self.gumbel_action.log_alpha[loc] = 5
        self.gumbel_action.threshold()

    # ── PerturbModelBase interface ────────────────────────────────────────────

    @torch.no_grad()
    def get_z(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Posterior mean μ_z from the encoder (perturbation-agnostic)."""
        x_ = torch.log(1 + x)
        qz, _ = self.z_encoder(x_)
        return qz.loc

    @torch.no_grad()
    def get_recon(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Decode posterior mean z → expected NB counts."""
        x_ = torch.log(1 + x)
        library = torch.log(x.sum(1)).unsqueeze(1)
        qz, _ = self.z_encoder(x_)
        _, _, px_rate, _ = self.decoder("gene", qz.loc, library)
        return px_rate

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
        Mask probabilities (P, d) or binarized hard mask.

        Parameters
        ----------
        deterministic : if True, return (proba > 0.5).float()
        """
        proba = self.gumbel_action.get_proba()
        if deterministic:
            return (proba > 0.5).float()
        return proba
