import pyro
import torch
import pyro.distributions as dist
import torch.nn as nn
from pyro.distributions import constraints
import torch.nn.functional as F
from einops import rearrange
import math
from module import GeneModuleEncoder

        

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, perturbs, conds, 
                 beta, temperature, module_dict, hidden_dims=[512,]):
        super(VAE, self).__init__()

        self.beta = beta
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conds = conds
        self.perturbs = perturbs
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p_emb = nn.Embedding(perturbs, latent_dim)
        self.c_emb_mu = nn.Embedding(conds, latent_dim)
        self.c_emb_logvar = nn.Embedding(conds, latent_dim)
        self.module_dict = module_dict
        self.hidden_dims = hidden_dims

        # Encoder: q(z|x,p)
        self.z_encoder = GeneModuleEncoder(
            num_genes=input_dim,
            module_dict=module_dict,
            hidden_dims=hidden_dims
        )

        # Encoder: q(z0|x,c)
        self.z0_encoder = GeneModuleEncoder(
            num_genes=input_dim,
            module_dict=module_dict,
            hidden_dims=hidden_dims
        )

        # Decoder: p(x|z)
        self.z_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, input_dim),
        )
        
        # classification heads
        self.z_prediction_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, perturbs)
        )
        
        self.z0_prediction_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, perturbs)
        )
        
        
        self.effect_pool = nn.Linear(self.latent_dim, 1)
        self.B = nn.Parameter(torch.empty(perturbs, latent_dim))
        self.phi = nn.LeakyReLU()

        self.rho_dec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), 
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        p0 = torch.tensor(0.05)
        logit_p_prior = torch.log(p0) - torch.log1p(-p0)
        self.logit_p  = pyro.param(
            "logit_p", logit_p_prior *
            torch.ones(self.perturbs, self.latent_dim, device=self.device)
        ) 
        

        # Initialize weights
        self._initialize_weights()

        self.mu = None
        self.total_counts = None
    
    def _initialize_weights(self):        
        nn.init.xavier_uniform_(self.B)
        
    def perturbation_effect(self, rho, p_idx):
        rho_active = rho[torch.arange(rho.shape[0], device=rho.device), p_idx, :]       
        B_active = self.B[p_idx, :]                                        
        return self.phi(rho_active) * B_active   

    def model(self, x, p, c):
        pyro.module("VAE", self)

        theta = pyro.param(
            "theta",
            torch.ones(self.input_dim, device=self.device) * 1.0,
            constraint=constraints.positive,
        ).to(self.device)
        
        z_var = pyro.param(
            "z_var", 
            torch.ones(self.latent_dim, device=self.device)*0.1, 
            constraint=constraints.positive
        ).to(self.device)
        
        loc = pyro.param("rho_loc", torch.zeros(self.perturbs, self.latent_dim, device=self.device))
        factor = pyro.param("rho_factor", 0.01*torch.randn(self.perturbs, self.latent_dim, self.latent_dim//2, device=self.device))
        diag = pyro.param("rho_diag",   0.1*torch.ones(self.perturbs, self.latent_dim, device=self.device), constraint=constraints.positive)    
        
        # rho_single = pyro.sample(
        #             "rho",                       
        #             dist.LowRankMultivariateNormal(loc, factor, diag**2).to_event(1)
        #             )

        with pyro.plate("perturbations", self.perturbs):
            rho_single = pyro.sample(
                        "rho",                       
                        dist.Normal(loc, torch.ones_like(loc)).to_event(1)
                        ).to(self.device)
        A = self.rho_dec(rho_single)/math.sqrt(self.latent_dim)  # (P,d)

        # --------- spike-and-slab gates ----------------------
        with pyro.plate("gate_cols", self.latent_dim), pyro.plate("gate_rows", self.perturbs):
            s_gate = pyro.sample(
                "s_gate",
                dist.RelaxedBernoulliStraightThrough(
                    temperature=torch.tensor(self.temperature),
                    logits=self.logit_p
                )
            ).to(self.device)  # (P,d)                                           

        W = s_gate * A     

        pyro.deterministic("W", W)             
            
        with pyro.plate("cells", x.size(0)):
            # --- expert 1 -----------------------------------------------------------
            z0_loc   = self.c_emb_mu(c)                                 # μ₁
            z0_scale = (0.5 * self.c_emb_logvar(c)).exp()               # σ₁  (shape: [B,d])
            z0_var   = z0_scale.pow(2)

            pyro.sample("z0", dist.Normal(z0_loc, z0_scale).to_event(1))


            # --- expert 2 -----------------------------------------------------------
            lin_shift = W[p]                                            # μ₂  (shape: [B,d])
            W_scale = pyro.param(
                "W_scale",
                torch.ones(self.latent_dim, device=self.device),
                constraint=dist.constraints.positive,
            )                                                           # σ₂  (shape: [d])
            W_var = W_scale.pow(2)
            
     
            # ------------------------------------------------------------------------
            # product-of-experts: element-wise precision addition
            precision = 1.0 / z0_var + 1.0 / W_var                    # [B,d]
            z_var     = 1.0 / precision
            z_loc     = z_var * (z0_loc / z0_var + lin_shift / W_var) # [B,d]
            z_scale   = torch.sqrt(z_var)

            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            logits_gene = self.z_decoder(z)

            mu = torch.softmax(logits_gene, dim=-1)

            total_counts = x.sum(-1, keepdim=True)
            # _dbg("total_counts", total_counts)

            self.mu = mu.clone().detach().to('cpu').numpy()
            self.total_counts = total_counts.clone().detach().to('cpu').numpy()

            x_mu = total_counts * mu
            # _dbg("x_mu", x_mu)

            logits = (x_mu + 1e-6).log() - (theta + 1e-6).log()
            # _dbg("logits", logits)

            nb = dist.NegativeBinomial(total_count=theta, logits=logits)
            pyro.sample("X", nb.to_event(1), obs=x.float())
            
            
            logits_p = self.z_prediction_head(z)       
            pyro.sample("p_label",
                dist.Categorical(logits=logits_p),
                obs=p.long())                  
            
            T = self.temperature
            CE_p  = F.cross_entropy(logits_p / T, p.long(), reduction="sum")
            pyro.factor("cls_p", -CE_p)
            
            center = rho_single.detach()
            loss_ctr = ((z - center[p])**2).sum(1)
            pyro.factor("center_loss", -1e-2 * loss_ctr.sum())  
            
    def guide(self, x, p, c):
        pyro.module("VAE", self)
        
        q_loc = pyro.param("q_rho_loc",   torch.zeros(self.perturbs, self.latent_dim, device=self.device))     # (P,d)
        q_factor = pyro.param("q_rho_factor",0.01*torch.randn(self.perturbs, self.latent_dim, self.latent_dim//2, device=self.device))                                # (P,d,r)
        q_diag = pyro.param("q_rho_diag",  torch.full((self.perturbs, self.latent_dim), 0.1, device=self.device, 
                                                      requires_grad=True))
        # pyro.sample(
        #         "rho",
        #         dist.LowRankMultivariateNormal(q_loc, q_factor, q_diag**2).to_event(1)
        #     )
        with pyro.plate("perturbations", self.perturbs):
            pyro.sample(
                    "rho",
                    dist.Normal(q_loc, torch.ones_like(q_loc)).to_event(1)
                )
            
        with pyro.plate("gate_cols", self.latent_dim), pyro.plate("gate_rows", self.perturbs):
            pyro.sample(
                "s_gate",
                dist.RelaxedBernoulliStraightThrough(
                    temperature=torch.tensor(self.temperature),
                    logits=self.logit_p                 
                )
            )

        with pyro.plate("cells", x.size(0)):
            inp = torch.cat([x, self.p_emb(p)], dim=-1)
            z_mu, z_logvar = self.z_encoder(inp)
            std = (0.5 * z_logvar).exp()
            pyro.sample("z", dist.Normal(z_mu, std+1e-6).to_event(1))

            inp = torch.cat([x, self.c_emb_mu(c)], dim=-1)
            z0_mu, z0_logvar = self.z0_encoder(inp)
            std = (0.5 * z0_logvar).exp()
            pyro.sample("z0", dist.Normal(z0_mu, std+1e-6).to_event(1))