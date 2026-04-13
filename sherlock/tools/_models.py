import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from torch.func import jacrev, vmap
import numpy as np

from ._base import PerturbModelBase


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
        l0_lambda=10.0,
        l1_lambda=1e-3,
        l2_lambda=1e-3,
        H_lambda=1e-3,
        ce_lambda=1.0,
        ntc_lambda=1e-5,
        pert_lambda=1.0,
        cov_lambda=1e-4,
        gate_init_p=0.5,
        rank=6,
        use_conditions=False,
        shift='poe',
        use_synergy=False,
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

        # reg weights
        self.l0_lambda = float(l0_lambda)
        self.l1_lambda = float(l1_lambda)
        self.l2_lambda = float(l2_lambda)
        self.H_lambda = float(H_lambda)
        self.ce_lambda = float(ce_lambda)
        self.ntc_lambda = float(ntc_lambda)
        self.pert_lambda = float(pert_lambda)
        self.cov_lambda = float(cov_lambda)

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

        # synergy: one latent vector per unique unordered pair (p1, p2), p1 < p2
        # Single-perturbation synergy is 0 by definition (not in this set).
        self.use_synergy = use_synergy
        if use_synergy:
            n_combos = perturbs * (perturbs - 1) // 2
            self.n_combos = n_combos
            self.syn_emb = nn.Embedding(n_combos, latent_dim)
            nn.init.zeros_(self.syn_emb.weight)

            # (P, P) lookup: pair_idx[i, j] = combo index for pair (i,j), -1 on diagonal
            pair_idx = torch.full((perturbs, perturbs), -1, dtype=torch.long)
            k = 0
            for i in range(perturbs):
                for j in range(i + 1, perturbs):
                    pair_idx[i, j] = k
                    pair_idx[j, i] = k
                    k += 1
            self.register_buffer("pair_idx", pair_idx)

    # --- helpers ---
    @torch.no_grad()
    def _corr(self, M, eps=1e-8):
        d = torch.sqrt(torch.clamp(torch.diag(M), min=eps))
        denom = torch.outer(d, d)
        return M / torch.clamp(denom, min=eps)

    @torch.no_grad()
    def _get_cov(self):
        L = self.qr().detach()
        sigma_P = pyro.param("sigma_fac").detach()
        P_ = L.size(0)
        eyeP = torch.eye(P_, dtype=L.dtype, device=L.device)
        Sigma_P = L @ L.T + (sigma_P**2) * eyeP
        return Sigma_P
    
    def __poe(self, z0_loc, z0_scale, lin_shift):
        device = z0_loc.device
        z0_var = z0_scale.pow(2)
        W_scale = pyro.param(
            "W_scale",
            0.1 * torch.ones(self.latent_dim, device=device),
            constraint=constraints.positive,
        ).to(z0_loc.device)
        W_var = W_scale.pow(2)
        precision = 1.0 / z0_var + 1.0 / W_var
        z_var = 1.0 / precision
        z_loc = z_var * (z0_loc / z0_var + (z0_loc + lin_shift) / W_var)
        return z_loc, z_var

    def __decode(self, x, z, weight, theta, d_key):
        logits_gene = weight(z)
        total_counts = x.sum(-1, keepdim=True)
        mu_prob = torch.softmax(logits_gene, dim=-1)
        mu = total_counts * mu_prob
        pyro.deterministic(d_key, mu)
        logits_nb = (mu + 1e-6).log() - theta.log()
        return logits_nb

    # --- model/guide ---
    def model(self, x_p, x_ntc, p, c):
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
            )
        chol_P = torch.linalg.cholesky(row_cov)
        A = chol_P @ rho_single  # (P, d)

        W = self.gate(deterministic=False)  # (P, d) in [0,1]
        pyro.deterministic("W", W)

        pi = self.gate.expected_L0()
        expected_l0 = pi.sum()
        expected_l2 = self.gate.expected_L2(A)
        pyro.factor("l0_penalty", - self.l0_lambda * expected_l0)
        pyro.factor("l1_penalty", - self.l1_lambda * self.gate.l1_logit())
        pyro.factor("l2_penalty", - self.l2_lambda * expected_l2)

        # column entropy encouragement
        col_usage = pi.sum(dim=0)
        p_col = col_usage / (col_usage.sum() + 1e-8)
        entropy_col = -(p_col * (p_col + 1e-8).log()).sum()
        pyro.factor("col_entropy", + self.H_lambda * entropy_col)

        # synergy: HalfCauchy shrinkage prior on per-combo latent vectors
        if self.use_synergy:
            with pyro.plate("combos", self.n_combos):
                syn = pyro.sample(
                    "syn",
                    dist.HalfCauchy(torch.ones(self.latent_dim, device=device)).to_event(1),
                )

        with pyro.plate("cells", x_p.size(0)):
            if self.use_conditions:
                z0_loc = self.c_emb_mu(c)
                z0_scale = (0.5 * self.c_emb_logvar(c)).exp()
                pyro.factor("do_prior_ridge_mu", -1e-4 * (z0_loc ** 2).mean())
                pyro.factor("do_prior_ridge_logvar", -1e-4 * (self.c_emb_logvar.weight ** 2).mean())
            else:
                z0_loc = torch.zeros(x_p.size(0), self.latent_dim, device=device)
                z0_scale = torch.ones(x_p.size(0), self.latent_dim, device=device)

            z0 = pyro.sample("z0", dist.Normal(z0_loc, z0_scale).to_event(1))

            logits_ntc = self.__decode(x_ntc, z0, self.z_decoder, theta, "x_ntc")
            pyro.sample(
                "X_ntc",
                dist.NegativeBinomial(total_count=theta, logits=logits_ntc).to_event(1),
                obs=x_ntc.float(),
                infer={"scale": self.ntc_lambda},
            )

            # gated perturbation shift (no condition branch)
            z0_loc_mod = z0_loc #* (1.0 - W[p])
            
            lin_shift = A[p] * W[p]
            if len(lin_shift.shape) != 2: #p == -1 is no perturbation.
                comb_mask = (p != -1).float().unsqueeze(-1)
                lin_shift = lin_shift * comb_mask
                lin_shift = lin_shift.sum(dim=1)

            # synergy: always 0 for single perturbations; pair embedding for combos
            if self.use_synergy:
                if p.ndim == 1:
                    syn_shift = torch.zeros_like(lin_shift)
                else:
                    p0 = p[:, 0].clamp(min=0)
                    p1 = p[:, 1].clamp(min=0)
                    syn_idx = self.pair_idx[p0, p1]           # (B,)
                    valid = (syn_idx >= 0).float().unsqueeze(-1)
                    syn_shift = syn[syn_idx.clamp(min=0)] * valid

            if self.shift == 'poe':
                z_loc, z_var = self.__poe(z0_loc_mod, z0_scale, lin_shift)
                z_loc = z_loc + syn_shift if self.use_synergy else z_loc
                z = pyro.sample("z", dist.Normal(z_loc, torch.sqrt(z_var)).to_event(1))
            elif self.shift == 'linear':
                z_loc = z0_loc_mod + lin_shift
                z_loc = z_loc + syn_shift if self.use_synergy else z_loc
                z_std = pyro.param("z_var_scale", torch.ones_like(z_loc[0]), constraint=constraints.positive)
                z = pyro.sample("z", dist.Normal(z_loc, torch.sqrt(z_std)).to_event(1))
            else:
                raise ValueError("Invalid shift type")

            cls_logits = self.cls_head(z)
            pyro.deterministic("cls_logits", cls_logits)

            if p.ndim == 1:
                # ----- single-class classification -----
                CE_loss = F.cross_entropy(cls_logits, p, reduction="sum")

            else:
                # ----- multi-label classification (predict K perturbations) -----
                B, P = cls_logits.shape
                target = torch.zeros((B, P), device=cls_logits.device, dtype=cls_logits.dtype)

                mask = (p != -1)                      # (B, K)  only for valid perturbations
                p_safe = p.clamp(min=0)               # replace -1 with 0 for safe scatter

                # Put 1s at the valid perturbation indices
                target.scatter_(1, p_safe, mask.to(target.dtype))

                # BCE over classes (multi-label)
                CE_loss = F.binary_cross_entropy_with_logits(cls_logits, target, reduction="sum")

            pyro.factor("CE_loss", -self.ce_lambda*CE_loss)

            logits_p = self.__decode(x_p, z, self.z_decoder, theta, "x_p")
            pyro.sample(
                "X",
                dist.NegativeBinomial(total_count=theta, logits=logits_p).to_event(1),
                obs=x_p.float(),
                infer={"scale": self.pert_lambda},
            )

    def guide(self, x_p, x_ntc, p, c):
        pyro.module("VAE", self)

        def mednorm(x):
            lib = x.sum(dim=1, keepdim=True)
            med = torch.median(lib).item()
            return torch.log1p(x / lib * med)

        x_p = mednorm(x_p)
        x_ntc = mednorm(x_ntc)

        q_loc = self.rho_enc(self.p_emb.weight)
        with pyro.plate("perturbations", self.perturbs):
            pyro.sample("rho", dist.Normal(q_loc, torch.ones_like(q_loc)).to_event(1))

        if self.use_synergy:
            syn_scale = F.softplus(self.syn_emb.weight)   # (n_combos, d), positive
            with pyro.plate("combos", self.n_combos):
                pyro.sample("syn", dist.HalfNormal(syn_scale).to_event(1))

        with pyro.plate("cells", x_p.size(0)):
            z0_mu, z0_logvar = self.z0_encoder(x_ntc).chunk(2, dim=-1)
            z0_std = (0.5 * z0_logvar).exp()
            pyro.sample("z0", dist.Normal(z0_mu, z0_std).to_event(1))

            z_mu, z_logvar = self.z_encoder(x_p).chunk(2, dim=-1)
            z_std = (0.5 * z_logvar).exp()
            pyro.sample("z", dist.Normal(z_mu, z_std).to_event(1))

# ------------------------------ jaccobian helpers ---------------------------- #
    
    @torch.no_grad()
    def __eval_points(self, x_p, p):
        self.eval()
        z_mu, _ = self.z_encoder(torch.cat([x_p], dim=-1)).chunk(2, dim=-1)
        eval_points = []
        for pp in torch.unique(p):
            mask = (p == pp)
            eval_points.append(z_mu[mask].mean(dim=0, keepdim=True))
        return torch.cat(eval_points, dim=0)
    
    # (1, d) -> (1, G) -> (G,)
    def __decode_one_sample(self, z_single):
        return self.z_decoder(z_single.unsqueeze(0)).squeeze(0)
    
    def __jac(self, eval_points):
        self.eval()
        f = jacrev(self.__decode_one_sample)
        J = vmap(f)(eval_points) # (P, G, d)
        B = J.mean(dim=0).transpose(0,1).contiguous().abs() # (d, G)
        return B
    
    @torch.no_grad()
    def pert_to_target_graph(self, x_p, p, fix_gate=True):
        device = x_p.device
        eval_points = self.__eval_points(x_p, p)
        
        # turn on local gradient for jacobian
        with torch.enable_grad():
            eval_points = eval_points.detach().requires_grad_(True)
            B = self.__jac(eval_points)
        
        W = self.gate(deterministic=fix_gate)
        G = W @ B
        return G / (G.max(dim=1, keepdim=True).values + 1e-8)
    
    
    # ------------------------------ counterfactual helpers ---------------------------- #
    
    @torch.no_grad()
    def _abduct_z0(self, x_ntc):
        """
        q(z0_hat | x_ntc)
        """
        self.eval()
        lib_ntc = x_ntc.sum(dim=1, keepdim=True)
        med_ntc = torch.median(lib_ntc).item()
        x_ntc_n = torch.log1p(x_ntc / lib_ntc * med_ntc)
        z0_mu, _ = self.z0_encoder(x_ntc_n).chunk(2, dim=-1)
        return z0_mu
    
    @torch.no_grad()
    def _do_shift_z0(self, z0, c_from=None, c_to=None):
        """
        p(z0_cf|z0_hat, do(c=c_to), c_from)
        """
        mu_from = self.c_emb_mu(c_from)
        mu_to = self.c_emb_mu(c_to)
        return z0 + (mu_to - mu_from)
    
    @torch.no_grad()
    def latent_counterfactual(self, x_ntc, x_p, p, c_from=None, c_to=None):
        """
        q(z | z0_cf, p)
        """
        device = x_p.device
        # abuduct and shift
        z0_hat = self._abduct_z0(x_ntc)
        z0_cf = self._do_shift_z0(z0_hat, c_from=c_from, c_to=c_to)
        
        # predict with CF
        self.eval()
        L = self.qr()                                           
        sigma = pyro.param("sigma_fac").detach().to(x_ntc.device)
        row_cov = L @ L.T + sigma.pow(2) * torch.eye(self.perturbs, device=device)
        chol_P = torch.linalg.cholesky(row_cov)                 
        q_rho_mean = self.rho_enc(self.p_emb.weight).detach()     # (P, d)
        A = chol_P @ q_rho_mean
        
        W = self.gate(deterministic=True)
        pert_shift = A[p] * W[p]
        z0_cf_masked = z0_cf * (1. - W[p])

        if self.shift == 'poe':
            z_loc_cf, _ = self.__poe(z0_cf_masked, torch.ones_like(z0_cf_masked), pert_shift)
        elif self.shift == 'linear':
            z_loc_cf = z0_cf_masked + pert_shift
        else:
            raise ValueError("Invalid shift type")

        return z_loc_cf

    # ------------------------------ PerturbModelBase interface ----------------------------- #

    @torch.no_grad()
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

    # ------------------------------ Sherlock-specific helpers ----------------------------- #

    @torch.no_grad()
    def _explained_variance(self, x_ntc_cond, cond_idx, device=None):
        """Per-perturbation explained variance of NTC expression relative to total NTC variance."""
        if device is None:
            device = next(self.parameters()).device

        if isinstance(x_ntc_cond, np.ndarray):
            x_ntc = torch.as_tensor(x_ntc_cond, dtype=torch.float32, device=device)
        else:
            x_ntc = x_ntc_cond.to(device=device, dtype=torch.float32)

        n_cells, G = x_ntc.shape
        if n_cells < 2:
            return np.zeros((self.perturbs, G), dtype=np.float32)

        lib = x_ntc.sum(dim=1, keepdim=True) + 1e-8
        x_ntc_norm = torch.log1p(x_ntc / lib * 1e4)
        var_total  = x_ntc_norm.var(dim=0, unbiased=True)

        z0_hat  = self._abduct_z0(x_ntc)
        lib_ntc = x_ntc.sum(dim=1, keepdim=True)

        logits0  = self.z_decoder(z0_hat)
        mu_prob0 = F.softmax(logits0, dim=-1)
        mu0      = lib_ntc * mu_prob0

        ev_pg         = torch.zeros(self.perturbs, G, device=device, dtype=torch.float32)
        cond_idx_long = torch.tensor(int(cond_idx), dtype=torch.long, device=device)

        for p_idx in range(self.perturbs):
            p_vec = torch.full((n_cells,), p_idx, dtype=torch.long, device=device)
            z_cf  = self.latent_counterfactual(
                x_ntc=x_ntc, x_p=x_ntc, p=p_vec,
                c_from=cond_idx_long, c_to=cond_idx_long,
            )
            logits1  = self.z_decoder(z_cf)
            mu1      = lib_ntc * F.softmax(logits1, dim=-1)
            mu0_norm = torch.log1p(mu0 / lib_ntc * 1e4)
            mu1_norm = torch.log1p(mu1 / lib_ntc * 1e4)
            mean_shift  = (mu1_norm - mu0_norm).mean(dim=0)
            ev_pg[p_idx] = mean_shift.pow(2) / (var_total + 1e-8)

        return ev_pg.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def compute_counterfactual(self, x_ntc_mat, x_p, P, cond_list):
        """Counterfactual shift distances across conditions × perturbations."""
        u      = np.unique(P)
        cf_mat = np.zeros((len(u), len(cond_list), len(cond_list)), dtype=np.float32)

        for i in range(len(cond_list)):
            c_from = cond_list[i]
            x_ntc  = x_ntc_mat[i]
            m, n   = x_ntc.shape[0], x_p.shape[0]
            idx    = np.random.permutation(m)[:n] if m >= n else np.random.randint(0, m, size=n)
            x_ntc  = torch.tensor(x_ntc[idx], dtype=torch.float32)

            for j in range(i + 1, len(cond_list)):
                c_to     = cond_list[j]
                cfs_base = self.latent_counterfactual(x_ntc, x_p, P, c_from, c_from).detach().cpu().numpy()
                cfs      = self.latent_counterfactual(x_ntc, x_p, P, c_from, c_to  ).detach().cpu().numpy()
                bulk_base = np.stack([cfs_base[P == g].mean(0) for g in u])
                bulk_cf   = np.stack([cfs     [P == g].mean(0) for g in u])
                cf_mat[:, i, j] = np.linalg.norm(bulk_cf - bulk_base, axis=1).astype(np.float32)

        return cf_mat

    @torch.no_grad()
    def _eval(self, adata, obsm_key: str, device: torch.device) -> dict:
        """
        VAE-specific extras merged into adata.uns by the base eval().

        The base already stores z_corr (all cells) and perts.  This hook
        adds: classifier accuracy, rho_corr from the learned covariance,
        pseudobulk z_corr for non-NTC cells only (z_corr_pert, matching
        rho_corr dimensions), z_rho_corr, gate weights W, counterfactual
        shift matrix, and per-perturbation explained variance.

        z for perturbed cells is read from adata.obsm[obsm_key] (written by
        the base eval loop).  AnnData propagates obsm on subsetting, so
        adata[perturbed_mask].obsm[obsm_key] gives the (n_perturbed, d) rows
        aligned with PerturbMatchingDataset.P_indices.
        """
        import numpy as np
        from scipy.stats import spearmanr, pearsonr
        from ._datasets import PerturbMatchingDataset
        from .._configs import get_config

        p_key         = get_config("pert_key")
        ntc_label     = get_config("ntc_label")
        treatment_key = get_config("treatment_key")

        self.to(device)
        ds = PerturbMatchingDataset(adata)
        result = {}

        # ── classifier accuracy ───────────────────────────────────────
        P_sub    = adata[adata.obs[p_key].values != ntc_label]
        z_np     = P_sub.obsm[obsm_key]
        z        = torch.as_tensor(z_np, dtype=torch.float32, device=device)
        true_idx = np.array(
            [ds.perturbation_dict[n] for n in P_sub.obs[p_key].values], dtype=np.int64
        )
        y_true   = torch.as_tensor(true_idx, dtype=torch.long, device=device)
        result["acc"] = (self.cls_head(z).argmax(dim=-1) == y_true).float().mean().item()

        # ── rho_corr from learned covariance ─────────────────────────
        L        = self.qr().cpu()
        sigma_P  = pyro.param("sigma_fac").cpu().detach()
        Sigma_P  = L @ L.T + sigma_P ** 2 * torch.eye(L.size(0))
        rho_corr = self._corr(Sigma_P).numpy()
        result["rho_corr"] = rho_corr

        # ── pseudobulk z_corr for non-NTC cells (matches rho_corr dims) ─
        num_p       = self.perturbs
        P_idx       = torch.from_numpy(ds.P_indices)
        z_all_pert  = torch.as_tensor(z_np, dtype=torch.float32)
        z_bar       = torch.zeros(num_p, self.latent_dim)
        for pidx in range(num_p):
            mask = P_idx == pidx
            if mask.any():
                z_bar[pidx] = z_all_pert[mask].mean(0)
        z_corr_pert = torch.corrcoef(z_bar)
        result["z_corr_pert"] = z_corr_pert.numpy()

        # ── z_corr_pert vs rho_corr ───────────────────────────────────
        def _flat_triu(M):
            M   = torch.tensor(M) if not torch.is_tensor(M) else M
            idx = torch.triu_indices(M.size(0), M.size(1), offset=1)
            return M[idx[0], idx[1]]

        r_s, p_s = spearmanr(_flat_triu(rho_corr), _flat_triu(z_corr_pert))
        r_p, p_p = pearsonr( _flat_triu(rho_corr), _flat_triu(z_corr_pert))
        result["z_rho_corr"] = {
            "spearman_r": r_s, "spearman_p": p_s,
            "pearson_r":  r_p, "pearson_p":  p_p,
        }

        # ── gate weights W ────────────────────────────────────────────
        result["W"] = self.gate(deterministic=True).cpu().detach().numpy()

        # ── counterfactual shift matrix ───────────────────────────────
        x_p   = torch.tensor(ds.X_pert, dtype=torch.float32)
        P_t   = torch.from_numpy(ds.P_indices)
        conds = torch.tensor(
            [ds.condition_dict[c] for c in adata.obs[treatment_key].unique()],
            dtype=torch.int32,
        )
        result["cfs_mat"] = self.compute_counterfactual(ds.X_ntc, x_p, P_t, conds)

        # ── explained variance (condition × perturbation × gene) ─────
        ev_list = [
            self._explained_variance(ds.X_ntc[int(c)], c, device=device) for c in conds
        ]
        result["explained_variance"] = np.stack(ev_list, axis=0)

        # ── label arrays ─────────────────────────────────────────────
        result["conditions"] = np.array(adata.obs[treatment_key].unique())

        return result