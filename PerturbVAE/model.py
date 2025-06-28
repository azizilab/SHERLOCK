import pyro, torch, math, torch.nn.functional as F
import pyro.distributions as dist
from pyro.distributions import constraints
from module import GeneModuleEncoder
import torch.nn as nn

class VAE(nn.Module):
    # -----------------------------------------------------------------
    def __init__(self, input_dim, latent_dim, perturbs, conds,
                 beta, module_dict, hidden_dims=(16,),
                 center_coeff_init=1e-2):
        super().__init__()
        self.input_dim   = input_dim
        self.latent_dim  = latent_dim
        self.perturbs    = perturbs
        self.conds       = conds
        self.beta        = beta
        self.center_coeff_init = center_coeff_init
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- embeddings ---------------------------------------------
        self.p_emb        = nn.Embedding(perturbs, latent_dim)
        self.c_emb_mu     = nn.Embedding(conds,    latent_dim)
        self.c_emb_logvar = nn.Embedding(conds,    latent_dim)

        # ---- encoders / decoders ------------------------------------
        # self.z_encoder  = GeneModuleEncoder(input_dim, module_dict, hidden_dims)
        # self.z0_encoder = GeneModuleEncoder(input_dim, module_dict, hidden_dims)

        self.z_encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(), nn.LayerNorm(128),
            nn.Linear(128, 128), nn.LeakyReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim * 2)
        )
        self.z0_encoder = nn.Sequential(
            nn.Linear(input_dim + latent_dim, 128), nn.LeakyReLU(), nn.LayerNorm(128),
            nn.Linear(128, 128), nn.LeakyReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim * 2)
        )

        self.z_decoder  = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, input_dim)
        )

        self.rho_dec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim), nn.Tanh()
        )
        self.rho_enc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim, perturbs)
        )
        for m in self.rho_dec.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)

        self.register_buffer("center_coeff",
                             torch.tensor(center_coeff_init, dtype=torch.float32))
        
        self.gate_on   = False    
        self.tau_hi    = 5.0
        self.tau_lo    = 0.1


    def model(self, x, p, c):
        pyro.module("VAE", self)

        theta_uncon = pyro.param("theta_uncon",
                                 torch.zeros(self.input_dim, device=self.device))
        theta = F.softplus(theta_uncon) + 1e-3

        # ρ prior
        with pyro.plate("perturbations", self.perturbs):
            rho_single = pyro.sample(
                "rho",
                dist.Normal(torch.zeros_like(self.p_emb.weight),
                            torch.ones_like(self.p_emb.weight)).to_event(1)
            )
        A = self.rho_dec(rho_single) / math.sqrt(self.latent_dim)  # (P,d)

        if self.gate_on:
            # Beta–Bernoulli gate  (shape P×d) ----------------------------
            alpha = torch.tensor(1.0, device=self.device)   # broader prior
            beta  = torch.tensor(20.0, device=self.device)
            tau   = pyro.param("tau_temp",
                                torch.tensor(self.tau_hi, device=self.device),
                                constraint=constraints.positive)
        
            pi = pyro.sample("pi", dist.Beta(alpha, beta).expand([self.perturbs, self.latent_dim]).to_event(2))          # (P,d)
            s_gate = pyro.sample("s_gate", dist.RelaxedBernoulliStraightThrough(temperature=tau, probs=pi).to_event(2))

            W = s_gate   
                                        
            pyro.deterministic("W", W)

        # --- cell likelihood -----------------------------------------
        with pyro.plate("cells", x.size(0)):
            z0_loc   = self.c_emb_mu(c)
            z0_scale = (0.5 * self.c_emb_logvar(c)).exp()
            z0 = pyro.sample("z0", dist.Normal(z0_loc, z0_scale).to_event(1))

            if self.gate_on:
                z0 = z0 * W[p]                               # (B,d) gate applied
                lin_shift = A[p] * (1. - W[p])                        # (B,d) gate applied
            else:
                lin_shift = A[p]                                           # (B,d)
            # W_scale   = pyro.param("W_scale",
            #                        torch.ones(self.latent_dim, device=self.device),
            #                        constraint=constraints.positive)
            
            # z0_var   = z0_scale.pow(2)

            # lin_shift = W[p]                                 # μ₂
            # W_scale   = pyro.param("W_scale",
            #                     torch.ones(self.latent_dim, device=self.device),
            #                     constraint=constraints.positive)
            # W_var = W_scale.pow(2)                           # σ₂²

            # # ----- product of experts ------------------------------------
            # precision = 1.0 / z0_var + 1.0 / W_var
            # z_var     = 1.0 / precision
            # z_loc     = z_var * (z0_loc / z0_var + lin_shift / W_var)

            z_loc = z0 + lin_shift                     # μ₁ + μ₂
            z_var = torch.ones_like(z_loc)       # σ₁² + σ₂²

            z = pyro.sample("z",
                    dist.Normal(z_loc, torch.sqrt(z_var)).to_event(1))
            
            cls_logits = self.cls_head(z)
            pyro.deterministic("cls_logits", cls_logits)
            CE_loss = F.cross_entropy(cls_logits, p, reduction="sum")
            pyro.factor("CE_loss", -1*CE_loss)

            logits_gene = self.z_decoder(z).clamp(-8., 8.)
            total_counts = x.sum(-1, keepdim=True).clamp(min=1.)
            mu_prob = torch.softmax(logits_gene, dim=-1)
            mu = total_counts * mu_prob
            pyro.deterministic("x_mu", mu)
            logits_nb = (mu + 1e-6).log() - theta.log()
            pyro.sample("X",
                        dist.NegativeBinomial(total_count=theta,
                                              logits=logits_nb).to_event(1),
                        obs=x.float(), infer={"scale": 1e-3})

            # centre loss
            # ctr = ((z - rho_single[p].detach())**2).sum()
            # pyro.factor("center_loss", -1e0*self.center_coeff * ctr.sum())

    def guide(self, x, p, c):
        pyro.module("VAE", self)

        x = torch.log(x + 1.0)  # log-transform input counts

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
                            torch.tensor(self.tau_hi, device=self.device),
                            constraint=constraints.positive)

            # MAP point-estimate for π
            pyro.sample("pi", dist.Delta(q_pi).to_event(2))

            # gate, using the SAME logits so gradients flow straight back
            pyro.sample(
                "s_gate",
                dist.RelaxedBernoulliStraightThrough(
                    temperature=tau,
                    logits=q_pi_logits          # important!
                ).to_event(2)
            )
        
        # kappa = 2.0
        # a0 = 0.05 * kappa
        # b0 = 0.95 * kappa

        # q_alpha = pyro.param(
        #     "q_alpha",
        #     torch.full((self.perturbs, self.latent_dim), a0, device=self.device),
        #     constraint=constraints.positive
        # )
        # q_beta  = pyro.param(
        #     "q_beta",
        #     torch.full((self.perturbs, self.latent_dim), b0, device=self.device),
        #     constraint=constraints.positive
        # )
        # tau = pyro.param("tau_temp", torch.tensor(1.0, device=self.device),
        #                 constraint=constraints.positive)
        # pi = pyro.sample("pi",   dist.Beta(q_alpha, q_beta).to_event(2))

    

        # cell-wise factors -------------------------------------------
        with pyro.plate("cells", x.size(0)):
            # z_mu, z_logvar = self.z_encoder(torch.cat([x, self.p_emb(p)], dim=-1)).chunk(2, dim=-1)
            z_mu, z_logvar = self.z_encoder(torch.cat([x], dim=-1)).chunk(2, dim=-1)
            z_std = (0.5 * z_logvar).exp()
            pyro.sample("z", dist.Normal(z_mu, z_std).to_event(1))

            z0_mu, z0_logvar = self.z0_encoder(torch.cat([x, self.c_emb_mu(c)], dim=-1)).chunk(2, dim=-1)
            z0_std = (0.5 * z0_logvar).exp()
            pyro.sample("z0", dist.Normal(z0_mu, z0_std).to_event(1))

    @torch.no_grad()
    def canonicalise_columns(self):
        """
        Re-order every tensor that carries a latent dimension so that
        latent‐dim `k` corresponds to the k-th smallest module‐ID in
        `self.z_encoder._module_order`.
        """
        module_ids = torch.tensor(self.z_encoder._module_order,
                                device=self.device)
        _, perm = torch.sort(module_ids)                # ascending

        def _permute_last(t: torch.Tensor):
            return t.index_select(-1, perm)

        # permute every parameter whose *last* axis is latent_dim
        self.p_emb.weight.data        = _permute_last(self.p_emb.weight.data)
        self.c_emb_mu.weight.data     = _permute_last(self.c_emb_mu.weight.data)
        self.c_emb_logvar.weight.data = _permute_last(self.c_emb_logvar.weight.data)

        # first Linear(d→d) inside rho_dec: permute rows AND columns
        w0 = self.rho_dec[0].weight.data
        self.rho_dec[0].weight.data = (
            w0.index_select(0, perm).index_select(1, perm)
        )

        #permute variational probabilities of pi
        q_pi = pyro.param("q_pi")                       # (P,d)
        pyro.get_param_store()["q_pi"] = q_pi.index_select(-1, perm)
        