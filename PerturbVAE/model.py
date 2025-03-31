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
        self.latent_rank = config.latent_rank
        self.hidden_dims = config.hidden_dims
        self.expert_dim = config.expert_dim
        self.c_embed_dim = config.c_embed_dim
        self.p_embed_dim = config.p_embed_dim
        self.beta = config.beta
        self.eps = 5.0e-3
        
        # embedding layers
        self.emb_c = nn.Embedding(self.n_conditions, self.c_embed_dim, padding_idx=0)
        self.emb_p = nn.Embedding(self.n_perturb, self.p_embed_dim, padding_idx=0)
        
        # design mat networks
        self.f_dec = Dense_NN(self.c_embed_dim, self.hidden_dims, [self.n_modules] * 2)
        self.f_enc = Dense_NN(self.n_genes + self.n_modules + self.c_embed_dim, self.hidden_dims, [self.cond_latent_dim] * 2)
        self.g_dec = Dense_NN(self.p_embed_dim, self.hidden_dims, [1] * 2)
        self.g_enc = Dense_NN(self.n_genes + self.n_modules + 2 * self.p_embed_dim, self.hidden_dims, [self.n_perturb] * 2)
        
        # graphs
        self.A = nn.Parameter(torch.empty(self.n_modules, self.latent_rank))
        self.B = nn.Parameter(torch.empty(self.n_perturb, self.n_modules))
        self.Q = nn.Parameter(torch.empty(self.n_perturb, self.n_perturb))
        
        # z networks
        self.z_enc = MoE_Encoder(self.n_genes, self.n_modules, self.hidden_dims, self.expert_dim, n_params=4)
        self.z_dec = Dense_NN(self.n_modules, self.hidden_dims, [self.n_genes], activation=activation)
        self.phi = nn.GELU()
        self.psi = Dense_NN(self.n_perturb, self.hidden_dims, [self.n_modules], activation=activation)
        
        # learned embeddings
        self.lin_ = None
        self.nonlin_ = None
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.Q)
        
    def normalize_adj(self, G):
        d = G.shape[0]
        G = G + torch.eye(d, device=G.device)
        D = G.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-8))
        G_norm = D_inv_sqrt @ G @ D_inv_sqrt
        return G_norm
    
    def perturbation_effect(self, rho):
        lin_shift = self.phi(rho @ self.B)
        nonlin_shift = self.psi(rho * (rho @ self.Q))
        return lin_shift, nonlin_shift
        
    def model(self, X, P, C):
        pyro.module("PerturbVAE", self)
        theta = pyro.param("theta", torch.ones(self.n_genes)*0.3, constraint=constraints.positive)
        E_c = self.emb_c(C).squeeze() #(batch, 1, d_condition)
        E_p = torch.zeros(X.shape[0], self.n_perturb, self.p_embed_dim)
        E_p[:, P] += self.emb_p(P) # (batch, n_perturb, d_pert)
        
        with pyro.plate("cells", len(X)), poutine.scale(scale=1.0):
            with poutine.scale(scale=self.beta):
                z_var = pyro.param("z_var", torch.ones(self.n_modules)*0.1, constraint=constraints.positive)
                z0_var = pyro.param("z0_var", torch.ones(self.n_modules)*0.1, constraint=constraints.positive)
                kappa_mean, kappa_var = self.f_dec(E_c)
                kappa = pyro.sample("kappa", dist.Normal(kappa_mean, F.softplus(kappa_var)))
                rho_mean, rho_var = self.g_dec(E_p) #(batch, n_perturb, 1)
                rho = pyro.sample("rho", dist.Normal(rho_mean.squeeze(), F.softplus(rho_var.squeeze())))
                
                # control cells
                z_0 = pyro.sample("z_0", dist.Normal(kappa, z0_var))
                
                # perturbed cells
                lin_shift, nonlin_shift = self.perturbation_effect(rho)
                self.lin_ = lin_shift
                self.nonlin_ = nonlin_shift
                z_mean = z_0 + lin_shift + nonlin_shift
                G_z = self.A @ self.A.T
                G_z_norm = self.normalize_adj(G_z)
                z_mean = z_mean + F.gelu(z_mean @ G_z_norm) # graph conv with residual 
                z = pyro.sample("z", dist.Normal(z_mean, z_var))
        
            x_mean = self.z_dec(z)
            x_mean = F.gelu(x_mean)
            p = 1/(1 + x_mean * theta)
            r = 1/theta
            pyro.sample("X", dist.NegativeBinomial(total_count=r, probs=1-p))
    
    def guide(self, X, P, C):
        pyro.module("PerturbVAE", self)
        with pyro.plate("cells", len(X)), poutine.scale(scale=1.0):
            x = torch.log1p(X)
            z0_mean, z0_var, z_mean, z_var = self.z_enc(x) # Encoder with shared MoE layer but diff heads
            z0 = pyro.sample("z0", dist.Normal(z0_mean, F.softplus(z0_var)))
            z = pyro.sample("z", dist.Normal(z_mean, F.softplus(z_var)))
            E_c_guide = self.emb_c(C).squeeze() # (b, d_c)
            E_p_guide = self.emb_p(P).view(-1, 2 * self.p_embed_dim) #(b, 2 * d_p)
            
            with poutine.scale(scale=self.beta):
                # x->z0->kappa
                x_z0 = torch.cat([x, z0, E_c_guide], dim=-1) # q(k, z0|x, c) = q(k|z0, x, c)q(z0|x)
                kappa_mean, kappa_var = self.f_enc(x_z0) 
                pyro.sample("kappa", dist.Normal(kappa_mean, F.softplus(kappa_var)))
                
                #x->z->rho
                x_z_p = torch.cat([x, z, E_p_guide], dim=-1) #q(rho, z|x, p) = q(rho|z, x, p)q(z|x)
                rho_mean, rho_var = self.g_enc(x_z_p) 
                pyro.sample("rho", dist.Normal(rho_mean, F.softplus(rho_var))) 