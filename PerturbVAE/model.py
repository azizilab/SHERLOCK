import pyro, torch, math, torch.nn.functional as F
import pyro.distributions as dist
from pyro.distributions import constraints
from module import GeneModuleEncoder, SparseModuleTransform
import torch.nn as nn


class QRCov(nn.Module):
    def __init__(self, d, r, device):
        super().__init__()
        self.U_unproj = nn.Parameter(0.01 * torch.randn(d, r, device=device))
        self.log_s    = nn.Parameter(torch.linspace(0., -2., r, device=device))  # unconstrained

    def _canonicalize(self):
        # Reduced QR
        Q, R = torch.linalg.qr(self.U_unproj, mode="reduced")
        
        # Fix column signs via R's diagonal so Q is canonical up to permutation
        # diag = torch.diag(R)
        # signs = torch.sign(diag)
        # signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        # Q = Q * signs  # (d, r)

        # Order columns by descending scale (exp(log_s))
        s = torch.exp(self.log_s)                    
        perm = torch.argsort(s, descending=True)      
        Q_sorted = Q[:, perm]
        s_sorted = s[perm]
        return Q_sorted, s_sorted

    def forward(self):
        # Return the low-rank factor L = Q @ diag(s)
        Q, s = self._canonicalize()
        S = torch.diag(s)
        return Q @ S




class VAE(nn.Module):
    # -----------------------------------------------------------------
    def __init__(self, input_dim, latent_dim, perturbs, conds,
                 beta, module_var, tau, hidden_dims=(16,),
                 center_coeff_init=1e-2, use_gene_modules=False):
        super().__init__()
        self.input_dim   = input_dim
        self.latent_dim  = latent_dim
        self.perturbs    = perturbs
        self.conds       = conds
        self.beta        = beta
        self.center_coeff_init = center_coeff_init
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gene_modules  = use_gene_modules

        # ---- embeddings ---------------------------------------------
        self.p_emb        = nn.Embedding(perturbs, latent_dim)
        self.c_emb_mu     = nn.Embedding(conds,    latent_dim)
        self.c_emb_logvar = nn.Embedding(conds,    latent_dim)

        # ---- encoders / decoders ------------------------------------
        if self.use_gene_modules:
            # self.z_encoder  = SparseModuleTransform(input_dim, latent_dim, module_map=module_var, hidden=4)
            # self.z0_encoder = SparseModuleTransform(input_dim, latent_dim, module_map=module_var, hidden=4)

            self.z_encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.LeakyReLU(), nn.LayerNorm(128),
                nn.Linear(128, 128), nn.LeakyReLU(), nn.LayerNorm(128),
                nn.Linear(128, latent_dim * 2)
            )
            self.z0_encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.LeakyReLU(), nn.LayerNorm(128),
                nn.Linear(128, 128), nn.LeakyReLU(), nn.LayerNorm(128),
                nn.Linear(128, latent_dim * 2)
            )

            self.z_decoder  = SparseModuleTransform(latent_dim, input_dim, module_map=module_var, hidden=4, reverse=True, inference=False)
            
        else:
            self.z_encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.LeakyReLU(), nn.LayerNorm(128),
                nn.Linear(128, 128), nn.LeakyReLU(), nn.LayerNorm(128),
                nn.Linear(128, latent_dim * 2)
            )
            self.z0_encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.LeakyReLU(), nn.LayerNorm(128),
                nn.Linear(128, 128), nn.LeakyReLU(), nn.LayerNorm(128),
                nn.Linear(128, latent_dim * 2)
            )
            self.z_decoder  = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.LeakyReLU(),
                nn.Linear(128, input_dim)
            )
            

        

        self.rho_dec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.rho_enc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, perturbs)
        )
        for m in self.rho_dec.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)

        self.register_buffer("center_coeff",
                             torch.tensor(center_coeff_init, dtype=torch.float32))
        
        self.gate_on   = False    
        self.tau_init = tau
        
        
        # cov params
        self.qr = QRCov(self.perturbs, 6, device=self.device)
    
    # expectation of perturbation simlarity based on pi
    def E_pert_sim(self, min_conf=0.2):
        q_pi_logits = pyro.param("q_pi_logits")
        pi_probs = torch.sigmoid(q_pi_logits).detach()
        conf = (pi_probs - 0.5).abs() * 2.0   
        mask = (conf >= min_conf).float()
        pi_eff = pi_probs * mask            
        S = pi_eff @ pi_eff.T
        m = S.max()
        if m > 0:
            S = S / m
        S.fill_diagonal_(1.0)
        return S
    
    def _corr(self, M, eps=1e-8):
        d = torch.sqrt(torch.clamp(torch.diag(M), min=eps))
        denom = torch.outer(d, d)
        return M / torch.clamp(denom, min=eps)
    
    def _get_cov(self):
        L = self.qr().detach()
        sigma_P = pyro.param("sigma_fac").detach()
        P_ = L.size(0)
        eyeP = torch.eye(P_, dtype=L.dtype, device=L.device)
        Sigma_P = L @ L.T + (sigma_P**2) * eyeP
        return Sigma_P
    
    def _laplacian_loss(self, rho, cov):
        corr = self._corr(cov)
        C_cov = ((corr + 1.0) / 2.0).clamp(0., 1.)
        C_cov.fill_diagonal_(1.0)
        L = torch.diag(C_cov.sum(1)) - C_cov
        return torch.trace(rho.T @ L @ rho) / (self.perturbs)


    def model(self, x_p, x_ntc, p, c):
        pyro.module("VAE", self)

        theta_uncon = pyro.param("theta_uncon",
                                 torch.ones(self.input_dim, device=self.device))
        theta = F.softplus(theta_uncon) + 1e-3

        
        # row covariance across perturbations
        L = self.qr()
        sigma = pyro.param(
            "sigma_fac",
            torch.full((), 0.05, device=self.device),  # () is a scalar tensor
            constraint=constraints.positive,
        )
        row_cov = L @ L.T + sigma.pow(2) * torch.eye(self.perturbs, device=self.device)
        
        # shrinkage on sigma with tight half normal prior(normalisation dosn't matter)
        pyro.factor("half_normal_energy", -0.5*(sigma/0.05)**2)

        # rho noise
        with pyro.plate("perturbations", self.perturbs):
            rho_single = pyro.sample(
                "rho",
                dist.Normal(torch.zeros_like(self.p_emb.weight),
                            torch.ones_like(self.p_emb.weight)).to_event(1)
            )
        # cholesky factorization of row covariance
        chol_P = torch.linalg.cholesky(row_cov)
        
        # left-multiply(covariance reparameterization)
        if self.gate_on:
            rho_single = chol_P @ rho_single
        A = rho_single  # (P,d)
        # A = self.rho_dec(rho_single) / math.sqrt(self.latent_dim)  # (P,d)

        if self.gate_on:
            # Beta–Bernoulli gate  (shape P×d) ----------------------------
            tau   = pyro.param("tau_temp",
                                torch.tensor(self.tau_init, device=self.device),
                                constraint=constraints.positive)
            alpha = torch.tensor(1.0, device=self.device)   # broader prior
            beta  = torch.tensor(10.0, device=self.device)
        
            pi = pyro.sample("pi", dist.Beta(alpha, beta).expand([self.perturbs, self.latent_dim]).to_event(2))          # (P,d)
            s_gate = pyro.sample("s_gate", dist.RelaxedBernoulliStraightThrough(temperature=tau, probs=pi).to_event(2))

            W = s_gate   
                                        
            pyro.deterministic("W", W)

            # C = row_cov.detach()
            # deg = C.sum(dim=1)
            # L  = torch.diag(deg) - C
            # D  = torch.diag(deg).clamp_min(1e-6)
            # #smoothness
            # num = (W.T @ L @ W).diagonal()
            # den = (W.T @ D @ W).diagonal().clamp_min(1e-6)
            # smooth = (num / den).sum()
            # #penalize off-diagonals
            # CW   = C @ W
            # cnrm = (W * CW).sum(dim=0, keepdim=True).clamp_min(1e-6)  # each col: w^T C w
            # Wc   = W / cnrm.sqrt()                                  

            # G = (Wc.T @ C @ Wc)                                     
            # off = G - torch.diag(torch.diag(G))
            # orth = (off**2).sum()

            # #elbo can be too weak early
            # sparsity = W.abs().sum()



            # pyro.factor("graph_smoothness", -1/(max(tau, 1)*1e13) * smooth)    
            # pyro.factor("cov_orthogonality", -1/(max(tau, 1)*1e13) * orth)
            
            # loss_lap = self._laplacian_loss(rho_single, row_cov)
            # pyro.factor("smoothness_reg", -1.0 * loss_lap)
            
            # E_S = self.E_pert_sim()
            # corr = self._corr(row_cov)
            # norm = (corr - E_S).pow(2).mean()
            # pyro.factor("cov_algin", -1.0 * norm)

        # --- cell likelihood -----------------------------------------
        with pyro.plate("cells", x_p.size(0)):
            z0_loc   = self.c_emb_mu(c)
            z0_scale = (0.5 * self.c_emb_logvar(c)).exp()

            z0_loc = torch.zeros_like(z0_loc)
            z0_scale = torch.ones_like(z0_scale)
            pyro.sample("z0", dist.Normal(z0_loc, z0_scale).to_event(1))


            if self.gate_on:
                z0_loc = z0_loc * (1. - W[p])                               # (B,d) gate applied
                lin_shift = A[p] * W[p]                        # (B,d) gate applied
            else:
                lin_shift = A[p]  
                                                         # (B,d)
            # W_scale   = pyro.param("W_scale",
            #                        torch.ones(self.latent_dim, device=self.device),
            #                        constraint=constraints.positive)
            
            z0_var   = z0_scale.pow(2)

            # lin_shift = W[p]                                 # μ₂
            W_scale   = pyro.param("W_scale",
                                torch.ones(self.latent_dim, device=self.device),
                                constraint=constraints.positive)
            W_var = W_scale.pow(2)                           # σ₂²

            # ----- product of experts ------------------------------------
            precision = 1.0 / z0_var + 1.0 / W_var
            z_var     = 1.0 / precision
            z_loc     = z_var * (z0_loc / z0_var + lin_shift / W_var)

            # z_loc = z0 + lin_shift                     # μ₁ + μ₂
            # z_var = torch.ones_like(z_loc)       # σ₁² + σ₂²
            z = pyro.sample("z",
                    dist.Normal(z_loc, torch.sqrt(z_var)).to_event(1))
            
            cls_logits = self.cls_head(z)
            pyro.deterministic("cls_logits", cls_logits)
            # CE_loss = F.cross_entropy(cls_logits, p, reduction="sum")
            # pyro.factor("CE_loss", -1.0*CE_loss)
            
            CE_loss = F.cross_entropy(cls_logits, p, reduction="mean")
            pyro.factor("CE_loss", -1e2 * CE_loss)

            logits_gene = self.z_decoder(z)#.clamp(-8., 8.)
            total_counts = x_p.sum(-1, keepdim=True)#.clamp(min=1.)
            mu_prob = torch.softmax(logits_gene, dim=-1)
            mu = total_counts * mu_prob
            pyro.deterministic("x_mu", mu)
            logits_nb = (mu + 1e-6).log() - theta.log()
            pyro.sample("X",
                        dist.NegativeBinomial(total_count=theta,
                                              logits=logits_nb).to_event(1),
                        # obs=x_p.float())
                        obs=x_p.float(), infer={"scale": 1e-3})

            # centre loss
            if self.gate_on:
                ctr = ((z - rho_single[p].detach())**2).mean()
            else:
                ctr = 0
            pyro.factor("center_loss", -1e-1 * ctr)
            # ctr = (z - rho_single[p]).pow(2).sum()   
            # pyro.factor("center_loss", -1e-2 * ctr)

            # reg = self.z_decoder.regularization_loss(
            #     l1=1.0,         
            #     l2=1.0,         
            #     group_lambda=1e-5  
            # )
            # pyro.factor("elasticnet_group_lasso", -reg)

    def guide(self, x_p, x_ntc, p, c):
        pyro.module("VAE", self)

        x_p = torch.log(x_p + 1.0)  # log-transform input counts
        x_ntc = torch.log(x_ntc + 1.0)

        # ρ posterior --------------------------------------------------
        q_loc = self.rho_enc(self.p_emb.weight)
        with pyro.plate("perturbations", self.perturbs):
            pyro.sample("rho", dist.Normal(q_loc,
                                            torch.ones_like(q_loc)).to_event(1))

        # q_pi  (shape P×d) -------------------------------------------
        if self.gate_on:
            p0 = 0.5 #starting prob
            q_pi_logits = pyro.param(
                "q_pi_logits",
                torch.full((self.perturbs, self.latent_dim),
                        math.log(p0 / (1.0 - p0)),        # prior mean ≈ 0.05
                        device=self.device)
            )                       

            q_pi = torch.sigmoid(q_pi_logits)            # (P, d) probability view

            tau = pyro.param("tau_temp",
                            torch.tensor(self.tau_init, device=self.device),
                            constraint=constraints.positive)

            # MAP point-estimate for π
            pi = pyro.sample("pi", dist.Delta(q_pi).to_event(2))

            # gate, using the SAME logits so gradients flow straight back
            pyro.sample(
                "s_gate",
                dist.RelaxedBernoulliStraightThrough(
                    temperature=tau,
                    logits=q_pi_logits          # important!
                ).to_event(2)
            )

            rho_cov = self._get_cov()
            rho_corr = self._corr(rho_cov).clone().detach()

            # adj  = F.relu(rho_corr)
            # adj.fill_diagonal_(0.)
            adj = F.relu(rho_corr - 0.45).fill_diagonal_(0.)
            deg  = adj.sum(1)
            # L    = torch.diag(deg) - adj
            D    = torch.diag(deg).clamp_min(1e-6)
            # num  = (pi.T @ L @ pi).diagonal()
            # den  = (pi.T @ D @ pi).diagonal().clamp_min(1e-6)
            # smooth = (num / den).sum()

            # # G = pi.T @ rho_corr.clone().detach() @ pi
            # # smooth = torch.trace(G)

            # CW  = rho_corr @ pi
            # cn  = (pi * CW).sum(0, keepdim=True).clamp_min(1e-6)
            # PiC = pi / cn.sqrt()
            # G   = PiC.T @ rho_corr @ PiC
            # off = G - torch.diag(torch.diag(G))
            # orth = (off**2).sum()



             # --- graph smoothness ---
            pi_norm = pi / (pi.norm(dim=0, keepdim=True) + 1e-6)
            # num = (pi_norm.T @ L @ pi_norm).diagonal()
            # den = (pi_norm.T @ D @ pi_norm).diagonal().clamp_min(1e-6)
            # smooth = (num / den).sum()
            # deg_inv_sqrt = D.pow_(-0.5)
            # deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            # A_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt

            # L = I - A_norm.
            L = D - adj
            smooth = torch.trace(pi_norm.T @ L @ pi) / (self.perturbs)

            # --- orthogonality ---
            G = pi_norm.T @ D @ pi_norm
            off = G - torch.diag(torch.diag(G))
            orth = (off**2).sum()

            # off_diag = G - torch.diag(torch.diag(G))
            # orth = (off_diag**2).sum()

            # sparsity = pi.sum()

            pyro.factor("graph_smoothness", -1e-2 * smooth, has_rsample=True)    
            pyro.factor("cov_orthogonality", -1e-2 * orth, has_rsample=True)
            # pyro.factor("mask_sparsity", -1e1 * sparsity, has_rsample=True)

    

        # cell-wise factors -------------------------------------------
        with pyro.plate("cells", x_p.size(0)):
            # z_mu, z_logvar = self.z_encoder(torch.cat([x, self.p_emb(p)], dim=-1)).chunk(2, dim=-1)
            # z_mu, z_logvar = self.z_encoder(torch.cat([x_p], dim=-1)).unbind(dim=-1)
            z_mu, z_logvar = self.z_encoder(torch.cat([x_p], dim=-1)).chunk(2, dim=-1)
            z_std = (0.5 * z_logvar).exp()
            pyro.sample("z", dist.Normal(z_mu, z_std).to_event(1))

            z0_mu, z0_logvar = self.z0_encoder(torch.cat([x_ntc], dim=-1)).chunk(2, dim=-1)
            z0_std = (0.5 * z0_logvar).exp()
            pyro.sample("z0", dist.Normal(z0_mu, z0_std).to_event(1))    


        