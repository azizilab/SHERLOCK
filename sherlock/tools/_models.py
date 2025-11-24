import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from torch.func import jacrev, vmap


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


class VAE(nn.Module):
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
        gate_row_repulsion_lambda=1e-3,
        gate_init_p=0.5,
        rank=6,
        use_conditions=False,
        shift='poe',
        use_contrastive_jacobian=False,
        use_de_align_loss=False,
        de_align_lambda=1e-3,
        treat_effect_map=None,
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
        self.use_contrastive_jacobian = bool(use_contrastive_jacobian)
        self.use_de_align_loss = bool(use_de_align_loss)
        self.de_align_lambda = float(de_align_lambda)

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

        # optional supervision: target DE per perturbation (perturbs x genes)
        if treat_effect_map is not None:
            self.register_buffer("treat_effect_map", treat_effect_map.float())
        else:
            self.treat_effect_map = None

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
        sigma_P = pyro.param("sigma_fac").detach()
        P_ = L.size(0)
        eyeP = torch.eye(P_, dtype=L.dtype, device=L.device)
        Sigma_P = L @ L.T + (sigma_P**2) * eyeP
        return Sigma_P

    @torch.no_grad()
    def enforce_decoder_basis(self):
        """
        Orthonormalize first decoder layer columns and fix signs for stability.
        """
        if not isinstance(self.z_decoder[0], nn.Linear):
            return
        W = self.z_decoder[0].weight  # (hidden, latent_dim)
        Q, _ = torch.linalg.qr(W, mode="reduced")
        idx = Q.abs().argmax(dim=0)
        signs = torch.sign(Q[idx, torch.arange(Q.size(1), device=Q.device)])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q * signs
        W.copy_(Q)
    
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
        # encourage perturbations to use different latent gates
        if self.perturbs > 1:
            W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
            sim = W_norm @ W_norm.T
            off_diag = sim[~torch.eye(self.perturbs, device=device, dtype=torch.bool)]
            repulsion = (off_diag ** 2).mean()
            pyro.factor("gate_row_repulsion", - self.gate_row_repulsion_lambda * repulsion)

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
            z0_loc_mod = z0_loc * (1.0 - W[p])
            lin_shift = A[p] * W[p]

            if self.shift == 'poe':
                z_loc, z_var = self.__poe(z0_loc_mod, z0_scale, lin_shift)
                z = pyro.sample("z", dist.Normal(z_loc, torch.sqrt(z_var)).to_event(1))
            elif self.shift == 'linear':
                z_loc = z0_loc_mod + lin_shift
                z_std = pyro.param("z_var_scale", torch.ones_like(z_loc[0]), constraint=constraints.positive)
                z = pyro.sample("z", dist.Normal(z_loc, torch.sqrt(z_std)).to_event(1))
            else:
                raise ValueError("Invalid shift type")

            cls_logits = self.cls_head(z)
            pyro.deterministic("cls_logits", cls_logits)
            CE_loss = F.cross_entropy(cls_logits, p, reduction="sum")
            pyro.factor("CE_loss", -self.ce_lambda*CE_loss)

            logits_p = self.__decode(x_p, z, self.z_decoder, theta, "x_p")
            pyro.sample(
                "X",
                dist.NegativeBinomial(total_count=theta, logits=logits_p).to_event(1),
                obs=x_p.float(),
                infer={"scale": self.pert_lambda},
            )

            # optional DE alignment loss against provided targets
            if self.use_de_align_loss and self.treat_effect_map is not None:
                # reuse decoded logits to avoid extra passes
                mu_p = theta * logits_p.exp()          # batch x genes
                mu_ntc = theta * logits_ntc.exp()
                P_max = self.perturbs

                # per-pert sums via scatter for speed
                sum_p = torch.zeros(P_max, self.input_dim, device=device)
                sum_ntc = torch.zeros_like(sum_p)
                cnt = torch.zeros(P_max, device=device)

                expand_idx = p.unsqueeze(1).expand(-1, self.input_dim)
                sum_p.scatter_add_(0, expand_idx, mu_p)
                sum_ntc.scatter_add_(0, expand_idx, mu_ntc)
                cnt.scatter_add_(0, p, torch.ones_like(p, dtype=sum_p.dtype))

                mask = cnt > 0
                if mask.any():
                    mean_p = sum_p[mask] / (cnt[mask].unsqueeze(1) + 1e-8)
                    mean_ntc = sum_ntc[mask] / (cnt[mask].unsqueeze(1) + 1e-8)
                    pred_eff = mean_p - mean_ntc
                    target = self.treat_effect_map.to(pred_eff.device)[:P_max][mask]
                    loss = F.mse_loss(pred_eff, target)
                    pyro.factor("de_align_loss", - self.de_align_lambda * loss)

    def guide(self, x_p, x_ntc, p, c):
        pyro.module("VAE", self)

        x_p = self._mednorm(x_p)
        x_ntc = self._mednorm(x_ntc)

        q_loc = self.rho_enc(self.p_emb.weight)
        with pyro.plate("perturbations", self.perturbs):
            pyro.sample("rho", dist.Normal(q_loc, torch.ones_like(q_loc)).to_event(1))

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
        x_p_n = self._mednorm(x_p)
        z_mu, _ = self.z_encoder(x_p_n).chunk(2, dim=-1)
        eval_points = []
        # gather one eval point per perturbation index; fallback to global mean if absent
        global_mean = z_mu.mean(dim=0, keepdim=True)
        for pp in range(self.perturbs):
            mask = (p == pp)
            if mask.any():
                eval_points.append(z_mu[mask].mean(dim=0, keepdim=True))
            else:
                eval_points.append(global_mean)
        return torch.cat(eval_points, dim=0)

    @torch.no_grad()
    def __control_eval_points(self, p, c, x_ntc_by_cond, device):
        """
        Build per-perturbation control eval points from matching-condition NTC cells.
        """
        if x_ntc_by_cond is None or len(x_ntc_by_cond) == 0:
            return None

        # normalize control list to tensors on device
        ctrl_list = []
        for x in x_ntc_by_cond:
            t = torch.as_tensor(x, device=device, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            ctrl_list.append(t)

        # if no condition labels, fallback to global control mean for all perts
        if c is None:
            pooled = torch.cat(ctrl_list, dim=0)
            z0_mu = self._abduct_z0(pooled)
            ctrl_mean = z0_mu.mean(dim=0, keepdim=True)
            return ctrl_mean.expand(self.perturbs, -1)

        ctrl_points = []
        global_mean = None
        for pp in range(self.perturbs):
            mask = (p == pp)
            if not mask.any():
                ctrl_points.append(None)
                continue
            cond_idx = c[mask][0].item()
            if cond_idx >= len(ctrl_list):
                ctrl_points.append(None)
                continue
            x_ntc = ctrl_list[cond_idx]
            z0_mu = self._abduct_z0(x_ntc)
            ctrl_points.append(z0_mu.mean(dim=0, keepdim=True))
            if global_mean is None:
                global_mean = ctrl_points[-1]
        if global_mean is None:
            return None
        filled = [cp if cp is not None else global_mean for cp in ctrl_points]
        return torch.cat(filled, dim=0)
    
    # (1, d) -> (1, G) -> (G,)
    def __decode_one_sample(self, z_single):
        return self.z_decoder(z_single.unsqueeze(0)).squeeze(0)
    
    def __jac(self, eval_points):
        self.eval()
        f = jacrev(self.__decode_one_sample)
        J = vmap(f)(eval_points) # (P, G, d)
        B = J.transpose(1, 2).contiguous().abs() # (P, d, G)
        return B
    
    @torch.no_grad()
    def pert_to_target_graph(self, x_p, p, c=None, x_ntc_by_cond=None, fix_gate=True):
        device = x_p.device
        x_p = self._mednorm(x_p)
        eval_points = self.__eval_points(x_p, p)
        
        # turn on local gradient for jacobian
        with torch.enable_grad():
            eval_points = eval_points.detach().requires_grad_(True)
            B_treat = self.__jac(eval_points)

            B_ctrl = None
            if self.use_contrastive_jacobian and x_ntc_by_cond is not None:
                ctrl_points = self.__control_eval_points(p, c, x_ntc_by_cond, device)
                if ctrl_points is not None:
                    ctrl_points = ctrl_points.detach().requires_grad_(True)
                    B_ctrl = self.__jac(ctrl_points)
            if B_ctrl is not None:
                B = (B_treat - B_ctrl).abs()
            else:
                B = B_treat
        
        W = self.gate(deterministic=fix_gate)
        G = (W.unsqueeze(-1) * B).sum(dim=1)
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
