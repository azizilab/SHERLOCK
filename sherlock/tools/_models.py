import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
import numpy as np

from ._base import PerturbModelBase
from .._configs import get_config



class HardConcreteGate(nn.Module):
    def __init__(self, shape, init_p=0.5, temperature=2.0 / 3.0, gamma=-0.1, zeta=1.1):
        super().__init__()
        self.shape = shape
        self.gamma = float(gamma)
        self.zeta = float(zeta)

        init_logit = math.log(init_p) - math.log(1.0 - init_p)
        self.log_alpha = nn.Parameter(torch.full(shape, init_logit))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    @torch.no_grad()
    def set_temperature(self, t: float):
        self.temperature.fill_(float(t))

    def _sample_relaxed(self):
        dev = self.log_alpha.device
        u = torch.rand(self.shape, device=dev)
        temp = self.temperature.to(dev)
        s = torch.sigmoid((self.log_alpha + torch.log(u) - torch.log1p(-u)) / temp)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar

    def forward(self, deterministic: bool = False):
        temp = self.temperature.to(self.log_alpha.device)
        if deterministic:
            s = torch.sigmoid(self.log_alpha / temp)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
        else:
            s_bar = self._sample_relaxed()
        z_hard = torch.clamp(s_bar, 0.0, 1.0)
        return z_hard + (s_bar - z_hard).detach()

    def expected_L0(self):
        c = math.log(-self.gamma / self.zeta)
        temp = self.temperature.to(self.log_alpha.device)
        return torch.sigmoid(self.log_alpha - temp * c)

    def l1_logit(self):
        return torch.norm(torch.sigmoid(self.log_alpha), p=1)

    def expected_L2(self, ungated):
        pi = self.expected_L0()
        return (pi * (ungated ** 2)).sum()


class QRCov(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        # Create on default device; module `.to(...)` will move params as needed
        self.U_unproj = nn.Parameter(0.01 * torch.randn(d, r))
        self.log_s = nn.Parameter(torch.linspace(0.0, -2.0, r))

    def _canonicalize(self):
        Q, R = torch.linalg.qr(self.U_unproj, mode="reduced")
        diag = torch.diag(R)
        signs = torch.where(torch.sign(diag) == 0, torch.ones_like(diag), torch.sign(diag))
        Q = Q * signs
        s = torch.exp(self.log_s)
        perm = torch.argsort(s, descending=True)
        return Q[:, perm], s[perm]

    def forward(self):
        Q, s = self._canonicalize()
        return Q @ torch.diag(s)


class VAE(PerturbModelBase):
    def __init__(
        self,
        input_dim,
        latent_dim,
        perturbs,
        conds,
        tau,
        l0_lambda=1.0,
        l1_lambda=1e-3,
        l2_lambda=1e-3,
        H_lambda=1e-1,
        ce_lambda=10,
        ntc_lambda=1e-5,
        pert_lambda=1.0,
        cov_lambda=1e-5,
        gate_row_repulsion_lambda=1e-2,
        gate_init_p=0.5,
        rank=6,
        use_conditions=False,
        shift='linear',
        use_synergy=False,
        synergy_rank=4,
        use_de_align_loss=False,
        de_align_lambda=1e-3,
        treat_effect_map=None,
        combinatorial=False,
    ):
        super().__init__()
        assert shift in ['poe', 'linear'], "shift must be 'poe' or 'linear'"

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.perturbs = perturbs
        self.conds = conds
        self.use_conditions = use_conditions
        self.rank = rank
        self.shift = shift
        self.use_synergy = use_synergy
        self.synergy_rank = synergy_rank
        self.combinatorial = combinatorial

        # reg weights
        self.l0_lambda = float(l0_lambda)
        self.l1_lambda = float(l1_lambda)
        self.l2_lambda = float(l2_lambda)
        self.H_lambda = float(H_lambda)
        self.ce_lambda = float(ce_lambda)
        self.ntc_lambda = float(ntc_lambda)
        self.pert_lambda = float(pert_lambda)
        self.cov_lambda = float(cov_lambda)
        self.gate_row_repulsion_lambda = float(gate_row_repulsion_lambda)
        self.use_de_align_loss = bool(use_de_align_loss)
        self.de_align_lambda = float(de_align_lambda)
        self.treat_effect_map = treat_effect_map

        # embeddings / condition prior params
        self.p_emb = nn.Embedding(perturbs, latent_dim)
        self.c_emb_mu = nn.Embedding(conds, latent_dim)
        self.c_emb_logvar = nn.Embedding(conds, latent_dim)

        # encoders / decoders
        def enc(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, 128), nn.LeakyReLU(), nn.LayerNorm(128),
                nn.Linear(128, 128), nn.LeakyReLU(), nn.LayerNorm(128),
                nn.Linear(128, out_d),
            )

        self.z_encoder = enc(input_dim, latent_dim * 2)
        self.z0_encoder = enc(input_dim, latent_dim * 2)

        self.z_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, input_dim),
        )

        self.rho_enc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, 128), nn.LeakyReLU(),
            nn.Linear(128, perturbs),
        )

        # gating (always on)
        self.tau_init = float(tau)
        self.gate = HardConcreteGate(
            shape=(self.perturbs, self.latent_dim),
            init_p=gate_init_p,
            temperature=self.tau_init if self.tau_init is not None else (2.0 / 3.0),
        )

        # low-rank covariance (params live on module device)
        self.qr = QRCov(self.perturbs, self.rank)

        # KL warmup weight — set externally by VAETrainer each epoch
        self.kl_weight = 1.0
        self.warmup_done = False
        # L0/gate sparsity weight — set externally by VAETrainer via delayed schedule
        # (ramps 0→1 only after KL warmup ends to prevent gate collapse on warmup exit)
        self.l0_weight = 0.0



    # --- helpers ---
    @torch.no_grad()
    def _corr(self, M, eps=1e-8):
        d = torch.sqrt(torch.clamp(torch.diag(M), min=eps))
        denom = torch.outer(d, d)
        return M / torch.clamp(denom, min=eps)

    def _mednorm(self, x: torch.Tensor) -> torch.Tensor:
        lib = x.sum(dim=1, keepdim=True)
        med = torch.median(lib).item()
        return torch.log1p(x / torch.clamp(lib, min=1e-8) * med)

    @torch.no_grad()
    def _get_cov(self):
        L = self.qr().detach()
        sigma_P = pyro.param("sigma_fac").detach().to(L.device)
        P_ = L.size(0)
        eyeP = torch.eye(P_, dtype=L.dtype, device=L.device)
        Sigma_P = L @ L.T + (sigma_P**2) * eyeP
        return Sigma_P
    
    def __poe(self, z0_loc, lin_shift, W_var):
        # PoE mean: z0_loc + alpha * lin_shift, alpha = 1/(1+W_var)
        # z_var is controlled separately via z_var_scale in the model
        alpha = 1.0 / (1.0 + W_var)
        z_loc = z0_loc + alpha * lin_shift
        return z_loc

    def __decode(self, x, z, weight, theta, d_key):
        logits_gene = weight(z)
        total_counts = x.sum(-1, keepdim=True)
        mu_prob = torch.softmax(logits_gene, dim=-1)
        mu = total_counts * mu_prob
        pyro.deterministic(d_key, mu)
        logits_nb = (mu + 1e-6).log() - theta.log()
        return logits_nb

    # --- model/guide ---
    def model(self, x_p, x_ntc, p, c, c_ntc):
        pyro.module("VAE", self)
        device = x_p.device

        theta_uncon = pyro.param("theta_uncon", torch.ones(self.input_dim, device=device))
        theta = F.softplus(theta_uncon) + 1e-3

        L = self.qr()
        sigma = pyro.param("sigma_fac", torch.full((), 0.05, device=device), constraint=constraints.positive)
        row_cov = L @ L.T + sigma.pow(2) * torch.eye(self.perturbs, device=device)

        # mild stabilizers
        pyro.factor("half_normal_energy", -0.5 * (sigma / 0.05) ** 2)
        pyro.factor("cov_ridge", - self.cov_lambda * (L ** 2).sum())

        with pyro.plate("perturbations", self.perturbs):
            rho_single = pyro.sample(
                "rho",
                dist.Normal(torch.zeros_like(self.p_emb.weight), torch.ones_like(self.p_emb.weight)).to_event(1),
                infer={"scale": self.kl_weight},
            )
        chol_P = torch.linalg.cholesky(row_cov)
        A = chol_P @ rho_single  # (P, d)

        W = self.gate(deterministic=False)  # (P, d) in [0,1]
        pyro.deterministic("W", W)

        pi = self.gate.expected_L0()
        expected_l0 = pi.sum()
        expected_l2 = self.gate.expected_L2(A)
        pyro.factor("l0_penalty", - self.l0_weight * self.l0_lambda * expected_l0)
        pyro.factor("l1_penalty", - self.l0_weight * self.l1_lambda * self.gate.l1_logit())
        pyro.factor("l2_penalty", - self.l0_weight * self.l2_lambda * expected_l2)

        # column entropy encouragement
        col_usage = pi.sum(dim=0)
        p_col = col_usage / (col_usage.sum() + 1e-8)
        entropy_col = -(p_col * (p_col + 1e-8).log()).sum()
        pyro.factor("col_entropy", + self.H_lambda * entropy_col)
        # encourage perturbations to use different latent gates
        if self.perturbs > 1:
            W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
            sim = W_norm @ W_norm.T
            off_diag = sim[~torch.eye(self.perturbs, device=device, dtype=torch.bool)]
            repulsion = (off_diag ** 2).mean()
            pyro.factor("gate_row_repulsion", - self.gate_row_repulsion_lambda * repulsion)

        if self.use_synergy:
            syn_scale = pyro.sample(
                "syn_scale",
                dist.HalfNormal(torch.tensor(1.0, device=device)),
                infer={"scale": self.kl_weight},
            )
            U_syn = pyro.sample(
                "U_syn",
                dist.Normal(
                    torch.zeros(self.latent_dim, self.latent_dim, self.synergy_rank, device=device),
                    syn_scale,
                ).to_event(3),
                infer={"scale": self.kl_weight},
            )
            V_syn = pyro.sample(
                "V_syn",
                dist.Normal(
                    torch.zeros(self.latent_dim, self.latent_dim, self.synergy_rank, device=device),
                    syn_scale,
                ).to_event(3),
                infer={"scale": self.kl_weight},
            )

        # ── per-perturbed-cell shift ─────────────────────────────────────
        n_ntc = x_ntc.size(0)
        n     = x_p.size(0)
        n_all = n_ntc + n

        if p.ndim > 1:  # combinatorial
            p_safe      = p.clamp(min=0)
            comb_mask   = (p != -1).float().unsqueeze(-1)              # (n, K, 1)
            lin_shift_p = (A[p_safe] * W[p_safe] * comb_mask).sum(dim=1)  # (n, d)
            W_p_eff     = (W[p_safe] * comb_mask).sum(dim=1).clamp(0, 1)  # (n, d)
        else:
            lin_shift_p = A[p] * W[p]                                  # (n, d)
            W_p_eff     = W[p]                                         # (n, d)

        # ── synergy shift: bilinear pooling on unmasked rho embeddings ───
        if self.use_synergy:
            if p.ndim == 1:
                syn_shift_p = torch.zeros(n, self.latent_dim, device=device)
            else:
                p0 = p[:, 0].clamp(min=0)
                p1 = p[:, 1].clamp(min=0)
                x_syn = A[p0]  # (n, d)
                y_syn = A[p1]  # (n, d)
                Ux = torch.einsum('dir,ni->ndr', U_syn, x_syn)   # (n, d, r)
                Vy = torch.einsum('dir,ni->ndr', V_syn, y_syn)   # (n, d, r)
                syn_shift_p = (Ux * Vy).sum(-1)                  # (n, d)
                valid_mask = (p[:, 1] != -1).float().unsqueeze(-1)
                syn_shift_p = syn_shift_p * valid_mask
            syn_all = torch.cat(
                [torch.zeros(n_ntc, self.latent_dim, device=device), syn_shift_p], dim=0
            )

        # ── n_all tensors: NTC cells first (W=0), perturbed cells second ─
        zeros_ntc     = torch.zeros(n_ntc, self.latent_dim, device=device)
        W_all         = torch.cat([zeros_ntc, W_p_eff],    dim=0)  # (n_all, d)
        lin_shift_all = torch.cat([zeros_ntc, lin_shift_p], dim=0)  # (n_all, d)

        if self.use_conditions:
            z0_loc_p   = self.c_emb_mu(c)
            z0_scale_p = (0.5 * self.c_emb_logvar(c)).exp()
            pyro.factor("do_prior_ridge_mu",     -1e-4 * (z0_loc_p ** 2).mean())
            pyro.factor("do_prior_ridge_logvar", -1e-4 * (self.c_emb_logvar.weight ** 2).mean())
            z0_loc_all   = torch.cat([zeros_ntc,                                          z0_loc_p],   dim=0)
            z0_scale_all = torch.cat([torch.ones(n_ntc, self.latent_dim, device=device), z0_scale_p], dim=0)
        else:
            z0_loc_all   = torch.zeros(n_all, self.latent_dim, device=device)
            z0_scale_all = torch.ones( n_all, self.latent_dim, device=device)

        z_var = pyro.param("z_var_scale", torch.ones(self.latent_dim, device=device), constraint=constraints.positive)

        if self.shift == 'poe':
            #Here to not be in cell plate
            W_scale = pyro.param("W_scale", 0.1 * torch.ones(self.latent_dim, device=device), constraint=constraints.positive)
            pyro.factor("W_scale_prior", -0.5 * (W_scale / 0.1).pow(2).sum())

        # ── cells plate: one z per cell (NTC: W=0 ⟹ z≈z0; perturbed: z=shift(z0)) ─
        with pyro.plate("cells", n_all):
            #TODO this is not proper POE for what we need. should be using z0_mu and z0_std
            #but gradient won't flow this way since its not from pyro.sample. Temp fix using
            #sampled z0 but its not ideal since it won't have the same mean/var as the guide's z0 distribution.
            #A better fix would be to implement the POE logic manually in the guide and model instead of relying on pyro.sample for z0
            z0     = pyro.sample("z0", dist.Normal(z0_loc_all, z0_scale_all).to_event(1), infer={"scale": self.kl_weight})
            z0_mod = z0 * (1.0 - W_all)

            if self.shift == 'poe':
                z_loc = self.__poe(z0_mod, lin_shift_all, W_scale.pow(2))
            elif self.shift == 'linear':
                z_loc = z0_mod + lin_shift_all
            else:
                raise ValueError("Invalid shift type")

            if self.use_synergy:
                z_loc = z_loc + syn_all

            z = pyro.sample("z", dist.Normal(z_loc, torch.sqrt(z_var)).to_event(1), infer={"scale": self.kl_weight})

        # ── observations and CE loss (outside plate) ─────────────────────
        z_ntc_z = z[:n_ntc]
        z_p_z   = z[n_ntc:]

        cls_logits = self.cls_head(z_p_z)
        pyro.deterministic("cls_logits", cls_logits)

        if p.ndim == 1:
            CE_loss = F.cross_entropy(cls_logits, p, reduction="sum")
        else:
            B_ce, Pdim = cls_logits.shape
            target = torch.zeros((B_ce, Pdim), device=device, dtype=cls_logits.dtype)
            mask   = (p != -1)
            target.scatter_(1, p.clamp(min=0), mask.to(target.dtype))
            CE_loss = F.binary_cross_entropy_with_logits(cls_logits, target, reduction="sum")
        pyro.factor("CE_loss", -self.ce_lambda * CE_loss)

        logits_ntc  = self.__decode(x_ntc, z_ntc_z, self.z_decoder, theta, "x_ntc")
        ntc_logprob = dist.NegativeBinomial(total_count=theta, logits=logits_ntc).log_prob(x_ntc.float()).sum()
        pyro.factor("X_ntc_logprob", self.ntc_lambda * ntc_logprob)

        logits_p  = self.__decode(x_p, z_p_z, self.z_decoder, theta, "x_p")
        p_logprob = dist.NegativeBinomial(total_count=theta, logits=logits_p).log_prob(x_p.float()).sum()
        pyro.factor("X_p_logprob", self.pert_lambda * p_logprob)

    def guide(self, x_p, x_ntc, p, c, c_ntc):
        pyro.module("VAE", self)

        x_p   = self._mednorm(x_p)
        x_ntc = self._mednorm(x_ntc)

        n_ntc = x_ntc.size(0)
        n     = x_p.size(0)
        n_all = n_ntc + n

        q_loc = self.rho_enc(self.p_emb.weight)
        with pyro.plate("perturbations", self.perturbs):
            pyro.sample("rho", dist.Normal(q_loc, torch.ones_like(q_loc)).to_event(1), infer={"scale": self.kl_weight})

        if self.use_synergy:
            device = x_p.device
            syn_scale_val = pyro.param(
                "syn_scale_val",
                torch.tensor(0.5, device=device),
                constraint=constraints.positive,
            )
            pyro.sample("syn_scale", dist.Delta(syn_scale_val))

            U_syn_loc = pyro.param(
                "U_syn_loc",
                0.01 * torch.randn(self.latent_dim, self.latent_dim, self.synergy_rank, device=device),
            )
            U_syn_scale = pyro.param(
                "U_syn_scale",
                0.05 * torch.ones(self.latent_dim, self.latent_dim, self.synergy_rank, device=device),
                constraint=constraints.positive,
            )
            V_syn_loc = pyro.param(
                "V_syn_loc",
                0.01 * torch.randn(self.latent_dim, self.latent_dim, self.synergy_rank, device=device),
            )
            V_syn_scale = pyro.param(
                "V_syn_scale",
                0.05 * torch.ones(self.latent_dim, self.latent_dim, self.synergy_rank, device=device),
                constraint=constraints.positive,
            )
            pyro.sample("U_syn", dist.Normal(U_syn_loc, U_syn_scale).to_event(3), infer={"scale": self.kl_weight})
            pyro.sample("V_syn", dist.Normal(V_syn_loc, V_syn_scale).to_event(3), infer={"scale": self.kl_weight})

        # z0: encode all NTC cells; assign each perturbed cell the mean z0 of its condition
        z0_mu_ntc, z0_logvar_ntc = self.z0_encoder(x_ntc).chunk(2, dim=-1)
        z0_std_ntc = (0.5 * z0_logvar_ntc).exp()

        z0_mu_p  = torch.zeros(n, self.latent_dim, device=x_p.device)
        z0_std_p = torch.ones( n, self.latent_dim, device=x_p.device)
        for cond_i in c.unique():
            ntc_mask  = (c_ntc == cond_i)
            pert_mask = (c    == cond_i)
            if ntc_mask.any():
                z0_mu_p[pert_mask]  = z0_mu_ntc[ntc_mask].mean(0)
                z0_std_p[pert_mask] = z0_std_ntc[ntc_mask].mean(0)

        z0_mu_all  = torch.cat([z0_mu_ntc, z0_mu_p],  dim=0)
        z0_std_all = torch.cat([z0_std_ntc, z0_std_p], dim=0)

        # z: encode NTC and perturbed cells through the same z_encoder
        z_mu_ntc, z_logvar_ntc = self.z_encoder(x_ntc).chunk(2, dim=-1)
        z_std_ntc = (0.5 * z_logvar_ntc).exp()
        z_mu_p,   z_logvar_p   = self.z_encoder(x_p).chunk(2, dim=-1)
        z_std_p   = (0.5 * z_logvar_p).exp()

        z_mu_all  = torch.cat([z_mu_ntc, z_mu_p],  dim=0)
        z_std_all = torch.cat([z_std_ntc, z_std_p], dim=0)

        with pyro.plate("cells", n_all):
            pyro.sample("z0", dist.Normal(z0_mu_all, z0_std_all).to_event(1), infer={"scale": self.kl_weight})
            pyro.sample("z",  dist.Normal(z_mu_all,  z_std_all).to_event(1),  infer={"scale": self.kl_weight})

    @torch.no_grad()
    def _abduct_z0(self, x_ntc):
        lib_ntc = x_ntc.sum(dim=1, keepdim=True)
        med_ntc = torch.median(lib_ntc).item()
        x_ntc_n = torch.log1p(x_ntc / lib_ntc * med_ntc)
        z0_mu, _ = self.z0_encoder(x_ntc_n).chunk(2, dim=-1)
        return z0_mu

    # ------------------------------ PerturbModelBase interface ----------------------------- #

    @torch.no_grad()
    def checkpoint_ctor_args(self) -> dict:
        return {
            "input_dim":      self.input_dim,
            "latent_dim":     self.latent_dim,
            "perturbs":       self.perturbs,
            "conds":          self.conds,
            "use_conditions": bool(self.use_conditions),
            "rank":           self.rank,
            "shift":          self.shift,
            "use_synergy":    bool(self.use_synergy),
            "synergy_rank":   self.synergy_rank,
            "combinatorial":  self.combinatorial,
        }

    def get_z(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Mean of q(z | x_p).

        The VAE encoder does not condition on perturbation labels —
        the perturbation effect is captured in the latent shift during
        training and is already baked into the encoder weights.
        """
        lib = x.sum(dim=1, keepdim=True)
        med = torch.median(lib).item()
        x_norm = torch.log1p(x / lib * med)
        z_mu, _ = self.z_encoder(x_norm).chunk(2, dim=-1)
        return z_mu

    @torch.no_grad()
    def get_recon(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Reconstructed expected counts at observed library size."""
        z_mu = self.get_z(x, p)
        logits_gene = self.z_decoder(z_mu)
        mu_prob = F.softmax(logits_gene, dim=-1)
        library = x.sum(dim=-1, keepdim=True)
        return library * mu_prob

    def _get_rho_embed(
        self,
        pert_indices: np.ndarray,
        z: np.ndarray,
        p_indices: np.ndarray,
        all_perts: np.ndarray,
    ) -> np.ndarray:
        """
        A*W action embeddings for the selected perturbations.

        PerturbSimpleDataset sorts non-NTC perts alphabetically, matching
        VAE's internal row order (0..P-1).  We recover VAE rows by ranking
        pert_indices among all non-NTC global indices.
        """
        from .._configs import get_config
        ntc_label    = get_config("ntc_label")
        all_non_ntc  = np.where(all_perts != ntc_label)[0]
        vae_rows     = np.searchsorted(all_non_ntc, pert_indices)

        Sigma_P = self._get_cov().cpu()
        q_rho   = self.rho_enc(self.p_emb.weight).detach().cpu()   # (P_vae, d)
        chol_P  = torch.linalg.cholesky(Sigma_P.to(q_rho.dtype))
        A       = (chol_P @ q_rho).detach().numpy()                 # (P_vae, d)
        W       = self.gate(deterministic=True).detach().cpu().numpy()
        return (A * W)[vae_rows].astype(np.float32)

    # ── counterfactual hooks (PerturbModelBase interface) ─────────────────────

    @torch.no_grad()
    def _abduct_ntc(self, x_ntc: torch.Tensor) -> torch.Tensor:
        return self._abduct_z0(x_ntc)

    @torch.no_grad()
    def _get_all_pert_shifts(self, device: torch.device) -> torch.Tensor:
        """Cholesky + gate computed once; returns A*W of shape (P, d)."""
        L      = self.qr().detach().to(device)
        sigma  = pyro.param("sigma_fac").detach().to(device)
        chol_P = torch.linalg.cholesky(L @ L.T + sigma ** 2 * torch.eye(self.perturbs, device=device))
        A = chol_P @ self.rho_enc(self.p_emb.weight).detach()
        W = self.gate(deterministic=True).detach()

        syn = torch.zeros_like(A)

        out = torch.stack([A * W, W, A, syn], dim=1)  # (P, 4, d) for unpacking in _apply_shift
        return torch.cat([out.new_zeros((1, 4, self.latent_dim)), out], dim=0) #0 is NTC for this function

    @torch.no_grad()
    def _apply_shift(self, u: torch.Tensor, m_p: torch.Tensor) -> torch.Tensor:
        """Apply perturbation shift via PoE or linear."""

        m_p, w_p, a_p, s_p = (
            m_p[:, 0, :],
            m_p[:, 1, :],
            m_p[:, 2, :],
            m_p[:, 3, :],
        )

        u_mod = u * (1.0 - w_p)

        if self.shift == "poe":
            W_scale = pyro.param(
                "W_scale",
                0.1 * torch.ones(self.latent_dim),
                constraint=constraints.positive,
            ).to(u.device)
            return self.__poe(u_mod, m_p, W_scale.pow(2)) + s_p

        elif self.shift == "linear":
            return u_mod + m_p + s_p

        raise ValueError("Invalid shift type")

    @torch.no_grad()
    def _combine_pert_shifts(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """Union-merge stacked (A*W, W, A, syn) shift tensors for combinatorial perturbations."""

        shift = m1[:, 0, :] + m2[:, 0, :]
        gate  = (m1[:, 1, :] + m2[:, 1, :]).clamp(0.0, 1.0)

        a_i = m1[:, 2, :]
        a_j = m2[:, 2, :]

        if self.use_synergy:
            U_syn = pyro.param("U_syn_loc").to(m1.device)
            V_syn = pyro.param("V_syn_loc").to(m1.device)

            Ux = torch.einsum("dir,bi->bdr", U_syn, a_i)
            Vy = torch.einsum("dir,bi->bdr", V_syn, a_j)
            syn = (Ux * Vy).sum(-1)
        else:
            syn = torch.zeros_like(shift)

        # keep raw A slot zero after combining because we only need it before computing syn
        return torch.stack(
            [shift, gate, torch.zeros_like(shift), syn],
            dim=1,
        )

    @torch.no_grad()
    def _decode_to_expr(self, z: torch.Tensor, lib_size: float) -> torch.Tensor:
        return lib_size * torch.softmax(self.z_decoder(z), dim=-1)

    # ------------------------------ PerturbModelBase overrides ---------------------------- #

    @torch.no_grad()
    def _run_inference(self, dataset, device, batch_size=1024):
        """
        NTC cells are encoded via z0_encoder and z_encoder for pre and post shift; perturbed
        cells only via z_encoder. z for NTC should reflect an unshifted state. Both are stored together in
        so adata.obsm['z'].
        """
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        x_list, p_list = [], []
        for x, p in loader:
            x_list.append(x)
            p_list.append(p)

        x_all = torch.cat(x_list, dim=0).to(device, dtype=torch.float32)
        p_all = torch.cat(p_list, dim=0)

        lib_all = x_all.sum(dim=1, keepdim=True)
        med = torch.median(lib_all).item()
        x_norm_all = torch.log1p(x_all / lib_all * med)

        N, G = x_all.shape
        if p_all.ndim == 2:
            ntc_mask = (p_all[:, 0] == dataset.ntc_idx) & (p_all[:, 1] == -1)
        else:
            ntc_mask = p_all == dataset.ntc_idx

        z_out    = torch.empty(N, self.latent_dim)
        recon_out = torch.empty(N, G)

        for encoder, mask in [(self.z_encoder, ntc_mask), (self.z_encoder, ~ntc_mask)]:
            indices = mask.nonzero(as_tuple=True)[0]
            for i in range(0, len(indices), batch_size):
                idx_b  = indices[i : i + batch_size]
                xb     = x_norm_all[idx_b]
                lib_b  = lib_all[idx_b]
                z_mu, _ = encoder(xb).chunk(2, dim=-1)
                logits   = self.z_decoder(z_mu)
                z_out[idx_b]    = z_mu.detach().cpu()
                recon_out[idx_b] = (lib_b * F.softmax(logits, dim=-1)).detach().cpu()

        return (
            z_out.numpy().astype(np.float32),
            recon_out.numpy().astype(np.float32),
            p_all.numpy(),
        )

    # ------------------------------ Sherlock-specific helpers ----------------------------- #
    @torch.no_grad()
    def _counterfactual_effect_size(self, x_ntc_cond, cond_idx, device=None):
        if device is None:
            device = next(self.parameters()).device

        if isinstance(x_ntc_cond, np.ndarray):
            x_ntc = torch.as_tensor(x_ntc_cond, dtype=torch.float32, device=device)
        else:
            x_ntc = x_ntc_cond.to(device=device, dtype=torch.float32)

        n_cells, G = x_ntc.shape
        num_p = self.perturbs

        if n_cells < 2:
            return np.zeros((num_p, G), dtype=np.float32)

        lib = x_ntc.sum(dim=1, keepdim=True) + 1e-8
        var_total = torch.log1p(x_ntc / lib * 1e4).var(dim=0, unbiased=True)  # (G,)

        # abduct backgrounds and shifts once — no repeated Cholesky or re-encoding
        u_ntc  = self._abduct_ntc(x_ntc)            # (n_cells, d)
        shifts = self._get_all_pert_shifts(device)[1:]   # (P, d)

        probs0   = self._decode_to_expr(u_ntc, 1.0)    # (n_cells, G) — proportions
        mu0_norm = torch.log1p(probs0 * 1e4)

        ev_pg = torch.zeros(num_p, G, device=device, dtype=torch.float32)
        for p_idx in range(num_p):
            z_cf     = self._apply_shift(u_ntc, shifts[[p_idx]])    # (n_cells, d)
            probs1   = self._decode_to_expr(z_cf, 1.0)              # (n_cells, G)
            mu1_norm = torch.log1p(probs1 * 1e4)
            mean_shift = (mu1_norm - mu0_norm).mean(dim=0)
            ev_pg[p_idx] = mean_shift.pow(2) / (var_total + 1e-8)

        return ev_pg.cpu().numpy().astype(np.float32)
    
    @staticmethod
    def _silhouette_pair(z_c1: np.ndarray, z_c2: np.ndarray) -> float:
        """
        Mean silhouette score for a 2-condition split in z-space.
        Scale- and rotation-invariant → comparable across model runs.
        Returns float in [-1, 1]: 1=separated, 0=boundary, <0=mixed.
        """
        n1, n2 = len(z_c1), len(z_c2)
        if n1 < 2 or n2 < 2:
            return np.nan
        d11 = np.linalg.norm(z_c1[:, np.newaxis] - z_c1[np.newaxis], axis=-1)
        d22 = np.linalg.norm(z_c2[:, np.newaxis] - z_c2[np.newaxis], axis=-1)
        d12 = np.linalg.norm(z_c1[:, np.newaxis] - z_c2[np.newaxis], axis=-1)
        a1 = d11.sum(axis=1) / (n1 - 1)
        b1 = d12.mean(axis=1)
        s1 = (b1 - a1) / np.maximum(a1, b1)
        a2 = d22.sum(axis=1) / (n2 - 1)
        b2 = d12.mean(axis=0)
        s2 = (b2 - a2) / np.maximum(a2, b2)
        return float(np.concatenate([s1, s2]).mean())

    @torch.no_grad()
    def condition_perturbation_interaction(self, adata, ds, obsm_key):
        """
        Condition-perturbation interaction via silhouette score in z-space.

        score[p, c1, c2] = mean silhouette of cells with perturbation p,
                        labeled by condition (c1 vs c2).

        Range [-1, 1]:
        +1 : conditions perfectly separated for this drug
        0 : conditions on the boundary
        <0 : conditions mixed

        Returns: (P, C, C) float32 array, NaN where a perturbation has <2 cells
                in one of the two conditions.
        """
        p_key = get_config("pert_key")
        treatment_key = get_config("treatment_key")

        z_all = np.array(adata.obsm[obsm_key], dtype=np.float32)
        pert_col = np.array(adata.obs[p_key].values, dtype=str)
        cond_col = np.array(adata.obs[treatment_key].values, dtype=str)

        P, C = self.perturbs, self.conds

        idx_to_pert = {v: k for k, v in ds.perturbation_dict.items()}
        idx_to_cond = {v: k for k, v in ds.condition_dict.items()}
        pert_names = [idx_to_pert[i] for i in range(P)]
        cond_names = [idx_to_cond[i] for i in range(C)]

        valid = ~np.isnan(z_all).any(axis=1)

        score = np.full((P, C, C), np.nan, dtype=np.float32)
        for p, pname in enumerate(pert_names):
            for c1 in range(C):
                for c2 in range(c1 + 1, C):
                    mask_c1 = valid & (pert_col == pname) & (cond_col == cond_names[c1])
                    mask_c2 = valid & (pert_col == pname) & (cond_col == cond_names[c2])
                    s = self._silhouette_pair(z_all[mask_c1], z_all[mask_c2])
                    score[p, c1, c2] = s
                    score[p, c2, c1] = s

        return score.astype(np.float32)

    @torch.no_grad()
    def get_synergy_matrix(self, observed_pairs=None) -> np.ndarray:
        """
        Returns a (P, P) matrix where entry (i, j) = L2 norm of the synergy vector
        for the pair (A[i], A[j]) under the learned low-rank bilinear form.

        Parameters
        ----------
        observed_pairs : set of (int, int) or None
            If provided, only fill entries for observed combinatorial pairs; others are 0.
            If None, fill all off-diagonal entries.
        """
        if not self.use_synergy:
            return np.zeros((self.perturbs, self.perturbs), dtype=np.float32)

        device = next(self.parameters()).device
        L = self.qr().detach().to(device)
        sigma = pyro.param("sigma_fac").detach().to(device)
        chol_P = torch.linalg.cholesky(L @ L.T + sigma ** 2 * torch.eye(self.perturbs, device=device))
        A = (chol_P @ self.rho_enc(self.p_emb.weight).detach())  # (P, d)

        P = self.perturbs
        syn_matrix = np.zeros((P, P), dtype=np.float32)

        for i in range(P):
            for j in range(P):
                if i == j:
                    continue
                if observed_pairs is not None and (i, j) not in observed_pairs:
                    continue
                Ux = torch.einsum('dir,i->dr', pyro.param("U_syn_loc"), A[i])   # (d, r)
                Vy = torch.einsum('dir,i->dr', pyro.param("V_syn_loc"), A[j])   # (d, r)
                syn_vec = (Ux * Vy).sum(-1)                         # (d,)
                syn_matrix[i, j] = float(syn_vec.norm().item())

        return syn_matrix

    @torch.no_grad()
    def _eval(self, adata, obsm_key: str, device: torch.device) -> dict:
        from scipy.stats import spearmanr, pearsonr
        from ._datasets import PerturbMatchingDataset

        p_key         = get_config("pert_key")
        ntc_label     = get_config("ntc_label")
        treatment_key = get_config("treatment_key")

        self.to(device)
        ds = PerturbMatchingDataset(adata, combinatorial=self.combinatorial)
        result = {}

        # ── classifier accuracy ───────────────────────────────────────
        P_sub = adata[adata.obs[p_key].values != ntc_label]
        z_np  = P_sub.obsm[obsm_key]
        z     = torch.as_tensor(z_np, dtype=torch.float32, device=device)
        logits = self.cls_head(z)

        if self.combinatorial:
            p_idx_2d = torch.from_numpy(ds.P_indices).to(device)   # (N, 2)
            target = torch.zeros(z.size(0), self.perturbs, device=device)
            comp_mask = p_idx_2d != -1
            target.scatter_(1, p_idx_2d.clamp(min=0), comp_mask.to(target.dtype))
            pred = (logits.sigmoid() > 0.5).to(target.dtype)
            result["acc"] = (pred == target).all(dim=1).float().mean().item()
        else:
            true_idx = np.array(
                [ds.perturbation_dict[n] for n in P_sub.obs[p_key].values], dtype=np.int64
            )
            y_true = torch.as_tensor(true_idx, dtype=torch.long, device=device)
            result["acc"] = (logits.argmax(dim=-1) == y_true).float().mean().item()

        # ── rho_corr from learned covariance + A*W embed ─────────────
        L        = self.qr().cpu()
        sigma_P  = pyro.param("sigma_fac").cpu().detach()
        Sigma_P  = L @ L.T + sigma_P ** 2 * torch.eye(L.size(0))
        rho_corr = self._corr(Sigma_P).numpy()
        result["rho_corr"] = rho_corr

        q_rho  = self.rho_enc(self.p_emb.weight).detach().cpu()
        chol_P = torch.linalg.cholesky(Sigma_P.to(q_rho.dtype))
        A      = (chol_P @ q_rho).detach().numpy()
        W_np   = self.gate(deterministic=True).detach().cpu().numpy()
        result["rho_embed"] = (A * W_np).astype(np.float32)

        # ── pseudobulk z_corr for non-NTC cells (matches rho_corr dims) ─
        num_p      = self.perturbs
        P_idx      = torch.from_numpy(ds.P_indices)
        z_all_pert = torch.as_tensor(z_np, dtype=torch.float32)
        z_bar      = torch.zeros(num_p, self.latent_dim)

        if self.combinatorial:
            single_mask = P_idx[:, 1] == -1
            P_idx_1d    = P_idx[single_mask, 0]
            z_single    = z_all_pert[single_mask]
            for pidx in range(num_p):
                mask = P_idx_1d == pidx
                if mask.any():
                    z_bar[pidx] = z_single[mask].mean(0)
        else:
            for pidx in range(num_p):
                mask = P_idx == pidx
                if mask.any():
                    z_bar[pidx] = z_all_pert[mask].mean(0)

        nz = z_bar.norm(dim=1) > 1e-12
        z_corr_mat = torch.full((num_p, num_p), float("nan"))
        if nz.sum() >= 2:
            sub_corr = torch.corrcoef(z_bar[nz])
            nz_idx = torch.where(nz)[0]
            z_corr_mat[nz_idx.unsqueeze(1), nz_idx.unsqueeze(0)] = sub_corr
        torch.diagonal(z_corr_mat).fill_(1.0)
        result["z_corr"] = z_corr_mat.numpy()

        # ── z_corr vs rho_corr ────────────────────────────────────────
        def _flat_triu(M):
            M   = torch.tensor(M) if not torch.is_tensor(M) else M
            idx = torch.triu_indices(M.size(0), M.size(1), offset=1)
            return M[idx[0], idx[1]]

        r_s, p_s = spearmanr(_flat_triu(rho_corr), _flat_triu(z_corr_mat))
        r_p, p_p = pearsonr( _flat_triu(rho_corr), _flat_triu(z_corr_mat))
        result["z_rho_corr"] = {
            "spearman_r": r_s, "spearman_p": p_s,
            "pearson_r":  r_p, "pearson_p":  p_p,
        }

        # ── gate weights W ────────────────────────────────────────────
        result["W"] = self.gate(deterministic=True).cpu().detach().numpy()

        # ── explained variance (condition × perturbation × gene) ─────
        x_ntc_mat = ds.X_ntc  # list over cond: cells x genes

        cond_unique = adata.obs[treatment_key].unique()
        conds = torch.tensor([ds.condition_dict[c] for c in cond_unique], dtype=torch.int32)

        ev_list = []
        device = next(self.parameters()).device

        for _, cond_idx in zip(cond_unique, conds):
            x_ntc_cond = x_ntc_mat[cond_idx]
            ev_pg = self._counterfactual_effect_size(
                x_ntc_cond=x_ntc_cond,
                cond_idx=cond_idx,
                device=device,
            )
            ev_list.append(ev_pg)
        result['counterfactual_effect_size'] = np.stack(ev_list, axis=0)  # (C, P, G)

        # Condition-Perturbation Interaction (P, C, C)
        cpi = self.condition_perturbation_interaction(adata, ds, obsm_key)
        result['condition_perturbation_interaction'] = cpi  # (P, C, C)

        result['conditions'] = np.array(cond_unique)
        idx_to_pert = {idx: name for name, idx in ds.perturbation_dict.items()}
        result['rho_perts'] = np.array([idx_to_pert[i] for i in range(len(idx_to_pert))])

        # ── synergy matrix (P, P) + saved params for plot_synergy ────────────
        if self.use_synergy:
            if self.combinatorial:
                P_idx_np = ds.P_indices  # (N, 2)
                combo_mask_syn = P_idx_np[:, 1] != -1
                combos_syn = P_idx_np[combo_mask_syn]
                observed_pairs = {
                    (int(row[0]), int(row[1]))
                    for row in combos_syn
                    if int(row[0]) >= 0 and int(row[1]) >= 0
                }
            else:
                observed_pairs = None
            result['synergy_matrix'] = self.get_synergy_matrix(observed_pairs=observed_pairs)

            # Save the bilinear projection data so plot_synergy needs no model/pyro access
            result['synergy_params'] = {
                'A':         A.astype(np.float32),
                'U_loc':     pyro.param("U_syn_loc").detach().cpu().numpy().astype(np.float32),
                'U_scale':   pyro.param("U_syn_scale").detach().cpu().numpy().astype(np.float32),
                'V_loc':     pyro.param("V_syn_loc").detach().cpu().numpy().astype(np.float32),
                'V_scale':   pyro.param("V_syn_scale").detach().cpu().numpy().astype(np.float32),
                'syn_scale': float(pyro.param("syn_scale_val").detach().cpu().item()),
            }

        return result
