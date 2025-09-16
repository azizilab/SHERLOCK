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
        self.p_emb_z      = nn.Embedding(perturbs, latent_dim)
        self.c_emb_mu     = nn.Embedding(conds,    latent_dim)
        self.c_emb_logvar = nn.Embedding(conds,    latent_dim)

        # ---- encoders / decoders ------------------------------------
        if self.use_gene_modules:
            # self.z_encoder  = SparseModuleTransform(input_dim, latent_dim, module_map=module_var, hidden=4)
            # self.z0_encoder = SparseModuleTransform(input_dim, latent_dim, module_map=module_var, hidden=4)

            self.z_encoder = nn.Sequential(
                nn.Linear(input_dim+2*latent_dim, 128), nn.LeakyReLU(),
                nn.Linear(128, 128), nn.LeakyReLU(), 
                nn.Linear(128, latent_dim * 2)
            )
            self.z0_encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.LeakyReLU(),
                nn.Linear(128, 128), nn.LeakyReLU(),
                nn.Linear(128, latent_dim * 2)
            )

            self.z_decoder  = SparseModuleTransform(latent_dim, input_dim, module_map=module_var, hidden=6, reverse=True, inference=False)
            # self.z0_decoder = SparseModuleTransform(latent_dim, input_dim, module_map=module_var, hidden=6, reverse=True, inference=False)
            
        else:
            self.z_encoder = nn.Sequential(
                nn.Linear(input_dim+2*latent_dim, 128), nn.LeakyReLU(), nn.LayerNorm(128),
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
            # self.z0_decoder  = nn.Sequential(
            #     nn.Linear(latent_dim, 128), nn.LeakyReLU(),
            #     nn.Linear(128, input_dim)
            # )
            

        self.pi_dec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim)
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
            nn.Linear(128, 128),
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
    
    def __poe(self, z0_loc, z0_scale, lin_shift):
        
        z0_var   = z0_scale.pow(2)

        W_scale   = pyro.param("W_scale",
                            0.1*torch.ones(self.latent_dim, device=self.device),
                            constraint=constraints.positive)
        W_var = W_scale.pow(2)                           # σ₂²

        # ----- product of experts ------------------------------------
        precision = 1.0 / z0_var + 1.0 / W_var
        z_var     = 1.0 / precision
        z_loc     = z_var * (z0_loc / z0_var + (z0_loc + lin_shift) / W_var) #parameterize N(z0 + shift, σ₂)

        return z_loc, z_var
    
    def __decode(self, x, z, weight, theta, d_key):
        logits_gene = weight(z)#.clamp(-8., 8.)
        total_counts = x.sum(-1, keepdim=True)#.clamp(min=1.)
        mu_prob = torch.softmax(logits_gene, dim=-1)
        mu = total_counts * mu_prob
        pyro.deterministic(d_key, mu)
        logits_nb = (mu + 1e-6).log() - theta.log()
        return logits_nb


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


        # --- cell likelihood -----------------------------------------
        with pyro.plate("cells", x_p.size(0)):
            z0_loc   = self.c_emb_mu(c)
            z0_scale = (0.5 * self.c_emb_logvar(c)).exp()

            z0_loc = torch.zeros_like(z0_loc)
            z0_scale = torch.ones_like(z0_scale)
            z0 = pyro.sample("z0", dist.Normal(z0_loc, z0_scale).to_event(1))

            logits_ntc = self.__decode(x_ntc, z0, self.z_decoder, theta, 'x_ntc')
            pyro.sample("X_ntc",
                        dist.NegativeBinomial(total_count=theta,
                                              logits=logits_ntc).to_event(1),
                        obs=x_ntc.float())
                        # obs=x_ntc.float(), infer={"scale": 1e-5}) #1e-5

            if self.gate_on:
                z0_loc = z0_loc * (1. - W[p])                               # (B,d) gate applied
                lin_shift = A[p] * W[p]                        # (B,d) gate applied
            else:
                lin_shift = A[p]  

            z_loc, z_var = self.__poe(z0_loc, z0_scale, lin_shift)
            z = pyro.sample("z",
                    dist.Normal(z_loc, torch.sqrt(z_var)).to_event(1))
            
            cls_logits = self.cls_head(z)
            pyro.deterministic("cls_logits", cls_logits)
            # CE_loss = F.cross_entropy(cls_logits, p, reduction="sum")
            # pyro.factor("CE_loss", -1.0*CE_loss)
            
            CE_loss = F.cross_entropy(cls_logits, p, reduction="mean")
            # pyro.factor("CE_loss", -1e4 * CE_loss)

            logits_p = self.__decode(x_p, z, self.z_decoder, theta, 'x_p')
            pyro.sample("X",
                        dist.NegativeBinomial(total_count=theta,
                                              logits=logits_p).to_event(1),
                        obs=x_p.float())
                        # obs=x_p.float(), infer={"scale": 1e-2}) #1e-5



    def guide(self, x_p, x_ntc, p, c):
        pyro.module("VAE", self)

        # ----- median normalization for x_p -----
        lib_p = x_p.sum(dim=1, keepdim=True)
        assert torch.all(lib_p > 0), "x_p has a cell with zero library size"
        med_p = torch.median(lib_p).item()
        x_p = torch.log1p(x_p / lib_p * med_p)

        # ----- median normalization for x_ntc -----
        lib_ntc = x_ntc.sum(dim=1, keepdim=True)
        assert torch.all(lib_ntc > 0), "x_ntc has a cell with zero library size"
        med_ntc = torch.median(lib_ntc).item()
        x_ntc = torch.log1p(x_ntc / lib_ntc * med_ntc)

        # ρ posterior --------------------------------------------------
        q_loc = self.rho_enc(self.p_emb.weight)
        with pyro.plate("perturbations", self.perturbs):
            rho = pyro.sample("rho", dist.Normal(q_loc,
                                            torch.ones_like(q_loc)).to_event(1))
            
        if self.gate_on:
            L = self.qr()
            sigma = pyro.param("sigma_fac")
            row_cov = L @ L.T + (sigma**2) * torch.eye(self.perturbs, device=self.device)
            chol_P = torch.linalg.cholesky(row_cov)
            A = chol_P @ rho
        else:
            A = rho

        # q_pi  (shape P×d) -------------------------------------------
        if self.gate_on:

            # q_pi_logits = self.pi_dec(rho)

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
            pyro.sample("pi", dist.Delta(q_pi).to_event(2))

            # gate, using the SAME logits so gradients flow straight back
            W = pyro.sample(
                "s_gate",
                dist.RelaxedBernoulliStraightThrough(
                    temperature=tau,
                    logits=q_pi_logits          # important!
                ).to_event(2)
            )

    

        # cell-wise factors -------------------------------------------
        with pyro.plate("cells", x_p.size(0)):
            # z_mu, z_logvar = self.z_encoder(torch.cat([x, self.p_emb(p)], dim=-1)).chunk(2, dim=-1)
            # z_mu, z_logvar = self.z_encoder(torch.cat([x_p], dim=-1)).unbind(dim=-1)

            z0_mu, z0_logvar = self.z0_encoder(torch.cat([x_ntc], dim=-1)).chunk(2, dim=-1)
            z0_std = (0.5 * z0_logvar).exp()
            z0 = pyro.sample("z0", dist.Normal(z0_mu, z0_std).to_event(1))    

            # z_mu, z_logvar = self.z_encoder(torch.cat([x_p], dim=-1)).chunk(2, dim=-1)
            # z_std = (0.5 * z_logvar).exp()
            # pyro.sample("z", dist.Normal(z_mu, z_std).to_event(1))
            if self.gate_on:
                lin_shift = A[p] * W[p]                        # (B,d) gate applied
            else:
                lin_shift = A[p]  
            z_loc, z_var = self.__poe(z0_mu, z0_std, lin_shift)  # uses inferred z0 + (A[p] * W[p])
            pyro.sample("z", dist.Normal(z_loc, z_var.sqrt()).to_event(1))

        