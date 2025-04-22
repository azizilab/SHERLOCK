import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from torch.distributions import constraints
import pyro
from pyro import distributions as dist
import pyro.poutine as poutine
from PerturbVAE.module import *
from PerturbVAE.utils import *

from einops import rearrange


class PerturbVAE(nn.Module):
    def __init__(
        self,
        config = HyperparamConfig(),
        activation=nn.GELU()
    ):
        super().__init__()

        self.config = config
        self.n_genes = config.n_genes
        self.n_conditions = config.n_conditions
        self.n_perturb = config.n_perturb
        self.n_modules = config.n_modules
        self.cond_latent_dim = config.cond_latent_dim
        self.latent_dim = config.latent_dim
        self.latent_rank = config.latent_rank
        self.hidden_dims = config.hidden_dims
        self.expert_dim = config.expert_dim
        self.c_embed_dim = config.c_embed_dim
        self.p_embed_dim = config.p_embed_dim
        self.beta = config.beta
        self.eps = 5.0e-3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # embedding layers
        self.emb_c = nn.Embedding(self.n_conditions, self.latent_dim)
        self.emb_p = nn.Embedding(self.n_perturb+1, self.latent_dim) #+1 for the non-perturbed gene
        
        self.effect_pool = nn.Linear(self.latent_dim, 1)
        
        # design mat networks
        self.f_dec = Dense_NN(self.c_embed_dim, self.hidden_dims, [self.n_modules] * 2)
        self.f_enc = Dense_NN(self.n_genes + self.n_modules + self.c_embed_dim, self.hidden_dims, [self.n_modules] * 2)
        self.g_dec = Dense_NN(self.p_embed_dim, self.hidden_dims, [1] * 2)
        self.g_enc = Dense_NN(self.n_genes + self.n_modules + self.latent_dim, self.hidden_dims, [self.latent_dim] * 2)

        self.module_z_encoder = Dense_NN(self.n_genes+self.latent_dim, self.hidden_dims, [self.n_modules] * 2)
        self.module_z0_encoder = Dense_NN(self.n_genes+self.latent_dim, self.hidden_dims, [self.n_modules] * 2)
        
        # graphs
        self.A = nn.Parameter(torch.empty(self.n_modules, self.latent_rank))
        self.B = nn.Parameter(torch.empty(self.n_perturb, self.n_modules))
        #self.Q = nn.Parameter(torch.empty(self.n_perturb, self.n_perturb))
        
        # z networks
        self.z_enc = MoE_Encoder(self.n_genes, self.n_modules, self.hidden_dims, self.expert_dim, n_params=4)
        self.z_dec = Dense_NN(self.n_modules, self.hidden_dims, [self.n_genes], activation=activation)
        self.phi = nn.GELU()
        self.psi = Dense_NN(self.n_perturb, self.hidden_dims, [self.n_modules], activation=activation)
        
        self.pertubation_head = Dense_NN(self.n_modules, self.hidden_dims, [self.n_conditions], activation=activation)

        # learned embeddings
        self.lin_ = None
        self.nonlin_ = None
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        #nn.init.xavier_uniform_(self.Q)
        
    def normalize_adj(self, G):
        d = G.shape[0]
        G = G + torch.eye(d, device=G.device)
        D = G.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-8))
        G_norm = D_inv_sqrt @ G @ D_inv_sqrt
        return G_norm
    
    def perturbation_effect(self, rho):
        rho = rearrange(rho, 'B P d -> B d P')

        lin_shift = self.phi(rho @ self.B)
        #nonlin_shift = self.psi(rho * (rho @ self.Q))

        effect = self.effect_pool(rearrange(lin_shift, 'B d m -> B m d')).squeeze(-1) # B x m
        return effect, 0
        
    def model(self, X, P, C):
        pyro.module("PerturbVAE", self)

        l = X.sum(axis=-1, keepdim=True)

        theta = pyro.param("theta", torch.ones(self.n_genes, device=self.device), constraint=constraints.positive).to(self.device)

        z_var = pyro.param("z_var", torch.ones(self.n_modules, device=self.device)*0.1, constraint=constraints.positive).to(self.device)

        z0_mean = pyro.param("z0_mean", torch.zeros((self.n_conditions, self.n_modules), device=self.device)).to(self.device)
        z0_var = pyro.param("z0_var", torch.ones((self.n_conditions, self.n_modules), device=self.device)*0.1, constraint=constraints.positive).to(self.device)

        rho_mask = F.one_hot(P, num_classes=self.n_perturb).float().unsqueeze(-1).to(self.device)
        rho_mask_inv = 1 - rho_mask

        # set mean to be 0 for non-perturbed genes and learned for perturbed genes
        # set var to be 1 for non-perturbed genes and learned for perturbed genes

        rho_mean = pyro.param("rho_mean", torch.zeros((self.n_perturb, self.latent_dim), device=self.device)).unsqueeze(0).expand(len(X), -1, -1).to(self.device)
        rho_mean = rho_mask*rho_mean
        rho_var = pyro.param("rho_var", torch.ones((self.n_perturb, self.latent_dim), device=self.device)*0.1, constraint=constraints.positive).unsqueeze(0).expand(len(X), -1, -1).to(self.device)
        rho_var = rho_mask_inv+rho_mask*rho_var

        
        with pyro.plate("cells", len(X), dim=-2):
            # with pyro.plate("pertubations", self.n_perturb, dim=-1):
            #     rho = pyro.sample("rho", dist.Normal(rho_mean, rho_var).to_event(1)) # B x n_perturb x latent_dim
                            
            # control cells
            z_0 = pyro.sample("z0", dist.Normal(torch.zeros([X.size(0), self.n_modules], device=self.device), torch.ones([X.size(0), self.n_modules], device=self.device)).to_event(0)) # B x n_modules
            
            # perturbed cells
            # lin_shift, nonlin_shift = self.perturbation_effect(rho)
            # self.lin_ = lin_shift
            # self.nonlin_ = nonlin_shift
            # z_mean = z_0 + lin_shift + nonlin_shift
            # G_z = self.A @ self.A.T
            #G_z_norm = self.normalize_adj(G_z)
            # z_mean = z_mean #+ F.gelu(z_mean @ G_z) # graph conv with residual 
            # z = pyro.sample("z", dist.Normal(z_mean, z_var).to_event(0))
    
        mu = self.z_dec(z_0)

        mu = torch.softmax(mu, dim=-1)
        x_mu = l * mu
        EPS = 1e-6
        logits = (x_mu+EPS).log() - (theta+EPS).log()

        nb_dist = dist.NegativeBinomial(total_count=theta, logits=logits)
        pyro.sample("X", nb_dist.to_event(), obs=X)

    
    def guide(self, X, P, C):
        pyro.module("PerturbVAE", self)

        with pyro.plate("cells", len(X), dim=-2), poutine.scale(scale=self.beta):
            E_c = self.emb_c(C)
            E_p = self.emb_p(P)

            control_idx = torch.ones([len(X), self.n_perturb], dtype=torch.long).to(self.device) * self.n_perturb
            E_all = self.emb_p(control_idx) # non-perturbed gene
            E_all[torch.arange(len(X)), P] = E_p # replace the entry at index P with the perturb embedding



            x = torch.log1p(X)

            # q(rho, z0, z|x, c, p) = q(z0|x, c)q(z|x)q(rho|x,z,p)

            # q(z0|x, c)
            z0_mean, z0_logvar = self.module_z0_encoder(torch.cat([x, E_c], dim=-1)) # (b, n_modules)
            pyro.sample("z0", dist.Normal(z0_mean, torch.exp(z0_logvar)).to_event(0))

            # q(z|x)
            # z_mean, z_logvar = self.module_z_encoder(torch.cat([x, E_p], dim=-1)) # (b, n_modules)
            # z = pyro.sample("z", dist.Normal(z_mean, torch.exp(z_logvar)).to_event(0)) # (b, n_modules)
        
            # categorial z to perturbation label
            

            # q(rho|x, z, p)
            # with pyro.plate("pertubations", self.n_perturb, dim=-1), poutine.scale(scale=self.beta):

            #     # Expand x and z along a new dimension corresponding to perturbations.
            #     x_expanded = x.unsqueeze(1).expand(len(X), self.n_perturb, x.size(-1))
            #     z_expanded = z.unsqueeze(1).expand(len(X), self.n_perturb, z.size(-1))

            #     # The result will be of shape: [batch, n_perturb, n_genes + n_modules + p_embed_dim]
            #     x_rho = torch.cat([x_expanded, z_expanded, E_all], dim=-1)

            #     rho_mean, rho_logvar = self.g_enc(x_rho) 
            #     pyro.sample("rho", dist.Normal(rho_mean, torch.exp(rho_logvar)).to_event(1)) # (b, n_perturb, latent_dim)
