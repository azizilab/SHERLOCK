"""
cVAE — Conditional VAE with linear perturbation shift.

Generative model:
  p(U)       = N(0, I)
  q_phi(U|x) = N(mu_phi(x), sigma_phi(x))      [encoder — x only]
  z          = U + A[p],  A[NTC] = 0            [linear shift in latent space]
  p_theta(x|z) = NB(f(z))                       [decoder — no perturbation label]

NTC cells (p == 0) receive A = 0, so their latent is just U and the baseline
reconstruction is decode(U).  Counterfactuals follow directly:

  U_i  = E[q_phi(U | x_i^NTC)]
  x_i^cf(p) = E[p_theta(x | U_i + A[p])]

Because z = U + A[p] and the decoder is perturbation-agnostic, this model uses
the PerturbModelBase counterfactual hooks unchanged (no override needed).
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


class cVAE(PerturbModelBase):
    """
    Conditional VAE with linear latent-space perturbation shift.

    The encoder q(U|x) infers a perturbation-free background. Each
    perturbation has a learned shift vector A[p] in latent space; the
    perturbed latent is z = U + A[p]. NTC is pinned at A = 0 via
    padding_idx, so the decoder baseline is always decode(U).

    Parameters
    ----------
    input_dim    : G — number of input genes
    n_perturbs   : number of perturbation labels (NTC always at index 0)
    latent_dim   : latent space dimensionality
    n_hidden     : hidden units per MLP layer
    n_layers     : number of hidden MLP layers
    dropout_rate : dropout probability
    beta         : kept for API parity; not used in loss (pass kl_weight instead)
    """

    def __init__(
        self,
        input_dim: int,
        n_perturbs: int,
        latent_dim: int = 10,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        beta: float = 1.0,
        combinatorial: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_latent = latent_dim
        self.n_perturbs = n_perturbs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.combinatorial = combinatorial

        self.px_r = nn.Parameter(torch.randn(input_dim))

        # per-perturbation latent shift; padding_idx=0 keeps A[NTC] frozen at 0
        self.pert_emb = nn.Embedding(n_perturbs, latent_dim, padding_idx=0)

        # u-encoder: q(U | x) — no perturbation label
        self.u_encoder = Encoder(
            input_dim,
            latent_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution="normal",
            use_batch_norm=True,
            return_dist=True,
        )

        # decoder: p(x | z) — perturbation-agnostic
        self.decoder = DecoderSCVI(
            latent_dim,
            input_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=True,
            scale_activation="softmax",
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor):
        """Return q(U | x) and a reparameterized sample."""
        return self.u_encoder(torch.log1p(x))   # (qu, u)

    # ── training interface ────────────────────────────────────────────────────

    def _pert_shift(self, p: torch.Tensor) -> torch.Tensor:
        """Sum pert_emb over valid components. Works for 1-D and (N,2) p."""
        if p.ndim == 1:
            return self.pert_emb(p.clamp(min=0))
        # combinatorial: sum component embeddings, masking out -1 slots
        a = self.pert_emb(p[:, 0].clamp(min=0))          # always present
        mask1 = (p[:, 1] >= 0).float().unsqueeze(-1)      # 1 if second component exists
        a = a + self.pert_emb(p[:, 1].clamp(min=0)) * mask1
        return a

    def inference(self, x: torch.Tensor, p: torch.Tensor) -> dict:
        """
        Infer q(U | x), shift z = U + A[p].
        NTC (p == 0) has A = 0 via padding_idx, so z_NTC = U.
        For combinatorial p (N,2), shifts for each component are summed.
        """
        library = torch.log(x.sum(1).clamp_min(1e-8)).unsqueeze(1)
        qu, u = self._encode(x)
        a = self._pert_shift(p)
        z = u + a
        return dict(u=u, qu=qu, z=z, library=library)

    def generative(self, z: torch.Tensor, library: torch.Tensor) -> dict:
        """Decode z → NegativeBinomial."""
        px_scale, _, px_rate, _ = self.decoder("gene", z, library)
        px_r = torch.exp(self.px_r)
        return dict(px=NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale))

    def loss(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        kl_weight: float = 1.0,
        n_obs: int = 1,
    ) -> torch.Tensor:
        """ELBO: E[log p(x|z)] - kl_weight * KL[q(U|x) || N(0,I)]"""
        inf = self.inference(x, p)
        gen = self.generative(inf["z"], inf["library"])

        pu = Normal(torch.zeros_like(inf["u"]), torch.ones_like(inf["u"]))
        kl_u = kl(inf["qu"], pu).sum(dim=1)
        reconst_loss = -gen["px"].log_prob(x).sum(-1)

        return n_obs * torch.mean(reconst_loss + kl_weight * kl_u)

    # ── PerturbModelBase interface ────────────────────────────────────────────

    def checkpoint_ctor_args(self) -> dict:
        return {
            "input_dim":     self.input_dim,
            "n_perturbs":    self.n_perturbs,
            "latent_dim":    self.n_latent,
            "n_hidden":      self.n_hidden,
            "n_layers":      self.n_layers,
            "dropout_rate":  self.dropout_rate,
            "beta":          self.beta,
            "combinatorial": self.combinatorial,
        }

    def _get_rho_embed(
        self,
        pert_indices: np.ndarray,
        z: np.ndarray,
        p_indices: np.ndarray,
        all_perts: np.ndarray,
    ) -> np.ndarray:
        """Learned shift vectors A[p] for selected (non-NTC) perturbations."""
        emb = self.pert_emb.weight.detach().cpu().numpy()   # (P_all, latent_dim)
        return emb[pert_indices].astype(np.float32)

    @torch.no_grad()
    def get_z(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        If p is None, return background U = E[q(U | x)].
        If p is given, return z = U + A[p] (summed over components for 2-D p).
        """
        qu, _ = self._encode(x)
        u = qu.loc
        if p is None:
            return u
        return u + self._pert_shift(p)

    @torch.no_grad()
    def get_recon(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Decode z = U + A[p] → expected NB counts."""
        library = torch.log(x.sum(1).clamp_min(1e-8)).unsqueeze(1)
        z = self.get_z(x, p)
        _, _, px_rate, _ = self.decoder("gene", z, library)
        return px_rate

    # ── counterfactual hooks (base class predict_counterfactual_effects) ──────

    @torch.no_grad()
    def _abduct_ntc(self, x_ntc: torch.Tensor) -> torch.Tensor:
        """Abduct background U_i = E[q(U | x_i^NTC)]."""
        qu, _ = self._encode(x_ntc)
        return qu.loc

    @torch.no_grad()
    def _get_all_pert_shifts(self, device: torch.device) -> torch.Tensor:
        """
        Return A[p] for all perturbations, shape (P, d).
        Row 0 (NTC) is zero by padding_idx; base class skips it when iterating
        non-NTC perturbations.
        """
        return self.pert_emb.weight.detach().to(device)

    @torch.no_grad()
    def _decode_to_expr(self, z: torch.Tensor, lib_size: float) -> torch.Tensor:
        lib_log = torch.full(
            (z.shape[0], 1), float(np.log(lib_size + 1e-8)), dtype=z.dtype, device=z.device
        )
        _, _, px_rate, _ = self.decoder("gene", z, lib_log)
        return px_rate

    # ── model-specific eval hook ──────────────────────────────────────────────

    @torch.no_grad()
    def _eval(self, adata, obsm_key: str, device: torch.device) -> dict:  # noqa: ARG002
        """Return perturbation embedding correlation (non-NTC only)."""
        from ._datasets import PerturbSimpleDataset
        from .._configs import get_config

        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")
        ds = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label,
                                  combinatorial=self.combinatorial)
        idx_to_pert = ds.idx_to_pert()
        ntc_idx = ds.ntc_idx

        emb = self.pert_emb.weight.detach().cpu().numpy()
        non_ntc_items = sorted((i, n) for i, n in idx_to_pert.items() if i != ntc_idx)
        non_ntc_idx_  = [i for i, _ in non_ntc_items]
        non_ntc_names = [n for _, n in non_ntc_items]

        rho_corr = np.corrcoef(emb[non_ntc_idx_]).astype(np.float32)
        return {
            "rho_corr":  rho_corr,
            "rho_perts": np.array(non_ntc_names),
        }
