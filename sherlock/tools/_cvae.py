"""
Conditional VAE (cVAE) — integrated sherlock implementation.

Architecture (following SAMS-VAE paper, Appendix A):
  Encoder  q(z | x, d):  mednorm(x) ⊕ emb(p)  →  MLP  →  (μ, log σ)
  Decoder  p(x | z, d):  z ⊕ emb(p)  →  MLP  →  NB logits
  Prior    p(z) = N(0, I)
  Likelihood: Negative-Binomial with learned per-gene dispersion

The perturbation embedding is shared between encoder and decoder.
NTC cells are included in training as a regular perturbation class (index 0).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

from ._base import PerturbModelBase


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(), nn.LayerNorm(hidden_dim)
    ]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.LayerNorm(hidden_dim)]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class cVAE(PerturbModelBase):
    """
    Conditional VAE for single-cell perturbation data.

    Parameters
    ----------
    input_dim  : number of genes.
    latent_dim : latent space dimensionality.
    n_perturbs : total perturbation classes including NTC.
    emb_dim    : perturbation embedding size.
    hidden_dim : MLP hidden layer width.
    n_hidden   : number of hidden layers in each MLP.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_perturbs: int,
        emb_dim: int = 32,
        hidden_dim: int = 400,
        n_hidden: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_perturbs = n_perturbs
        self.emb_dim = emb_dim

        self.pert_emb = nn.Embedding(n_perturbs, emb_dim)
        self.encoder  = _mlp(input_dim + emb_dim, hidden_dim, latent_dim * 2, n_hidden)
        self.decoder  = _mlp(latent_dim + emb_dim, hidden_dim, input_dim, n_hidden)

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _mednorm(x: torch.Tensor) -> torch.Tensor:
        lib = x.sum(dim=1, keepdim=True)
        med = torch.median(lib).item()
        return torch.log1p(x / lib * med)

    def _decode_mu(self, z: torch.Tensor, e: torch.Tensor, library: torch.Tensor) -> torch.Tensor:
        logits_gene = self.decoder(torch.cat([z, e], dim=-1))
        return library * torch.softmax(logits_gene, dim=-1)

    # ── Pyro model / guide ────────────────────────────────────────────

    def model(self, x: torch.Tensor, p: torch.Tensor) -> None:
        pyro.module("cVAE", self)
        device = x.device
        B = x.size(0)

        theta_uncon = pyro.param("theta_uncon", torch.ones(self.input_dim, device=device))
        theta = F.softplus(theta_uncon) + 1e-3

        e = self.pert_emb(p)

        with pyro.plate("cells", B):
            z = pyro.sample(
                "z",
                dist.Normal(
                    torch.zeros(B, self.latent_dim, device=device),
                    torch.ones(B, self.latent_dim, device=device),
                ).to_event(1),
            )
            library = x.sum(-1, keepdim=True)
            mu = self._decode_mu(z, e, library)
            logits_nb = (mu + 1e-6).log() - theta.log()
            pyro.sample(
                "X",
                dist.NegativeBinomial(total_count=theta, logits=logits_nb).to_event(1),
                obs=x.float(),
            )

    def guide(self, x: torch.Tensor, p: torch.Tensor) -> None:
        pyro.module("cVAE", self)
        B = x.size(0)

        x_norm = self._mednorm(x)
        e = self.pert_emb(p)
        mu, logvar = self.encoder(torch.cat([x_norm, e], dim=-1)).chunk(2, dim=-1)
        std = (0.5 * logvar).exp().clamp(min=1e-4)

        with pyro.plate("cells", B):
            pyro.sample("z", dist.Normal(mu, std).to_event(1))

    # ── PerturbModelBase interface ────────────────────────────────────

    @torch.no_grad()
    def get_z(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Mean posterior q(z | x, p)."""
        assert p is not None, "cVAE.get_z requires perturbation indices p"
        x_norm = self._mednorm(x)
        e = self.pert_emb(p)
        mu, _ = self.encoder(torch.cat([x_norm, e], dim=-1)).chunk(2, dim=-1)
        return mu

    @torch.no_grad()
    def get_recon(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Reconstructed expected counts at observed library size."""
        assert p is not None, "cVAE.get_recon requires perturbation indices p"
        z = self.get_z(x, p)
        e = self.pert_emb(p)
        library = x.sum(-1, keepdim=True)
        return self._decode_mu(z, e, library)

    @torch.no_grad()
    def _eval(self, adata, obsm_key: str, device: torch.device) -> dict:
        from ._datasets import PerturbSimpleDataset
        from .._configs import get_config

        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")

        dataset     = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label)
        idx_to_pert = dataset.idx_to_pert()
        ntc_idx     = dataset.ntc_idx

        # learned perturbation embeddings  (P_all, emb_dim)
        emb = self.pert_emb.weight.detach().cpu().numpy()

        non_ntc_items = sorted((i, n) for i, n in idx_to_pert.items() if i != ntc_idx)
        non_ntc_idx   = [i for i, _ in non_ntc_items]
        non_ntc_names = [n for _, n in non_ntc_items]

        rho_corr = np.corrcoef(emb[non_ntc_idx]).astype(np.float32)

        return {
            "rho_corr":  rho_corr,
            "rho_perts": np.array(non_ntc_names),
        }
