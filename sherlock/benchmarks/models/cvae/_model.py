import torch
import torch.nn as nn
import torch.nn.functional as F
from sherlock.benchmarks.models.base._base import BasePerturbModel, ForwardOut


class cVAE(BasePerturbModel):
    def __init__(
            self, 
            input_dim: int,  
            latent_dim: int, 
            perturbs: int,
            hidden_dim=256,
            dropout: float = 0.0
        ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.perturbs = perturbs
        self.hidden_dim = hidden_dim

        # embeddings 
        self.p_emb = nn.Embedding(self.perturbs, self.latent_dim)
        self.dropout_layer = nn.Dropout(p=float(dropout))
        

        # encoder q(z|x,p)
        self.enc_fc1 = nn.Linear(self.input_dim + self.latent_dim, self.hidden_dim)
        self.enc_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.enc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.enc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # decoder p(x|z,p)
        self.dec_fc1 = nn.Linear(self.latent_dim + self.latent_dim, self.hidden_dim)
        self.dec_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dec_out= nn.Linear(self.hidden_dim, self.input_dim)


        # NB dispersion parameter
        self.log_theta = nn.Parameter(torch.zeros(self.input_dim))


    def cond_emb(self, p):
        return self.p_emb(p)
        
    def encode(self,x,p):
        cond = self.cond_emb(p)
        h = torch.cat((x, cond), dim=1)
        h = F.relu(self.enc_fc1(h))
        h = self.dropout_layer(h)
        h = F.relu(self.enc_fc2(h))
        h = self.dropout_layer(h)

        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)

        logvar = torch.clamp(logvar, -6.0, 6.0 )

        return mu, logvar

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon
    
    def decoder(self, z, p):
        cond = self.cond_emb(p)
        h = torch.cat((z, cond), dim=1)
        h = F.relu(self.dec_fc1(h))
        h = self.dropout_layer(h)
        h = F.relu(self.dec_fc2(h))
        h = self.dropout_layer(h)


        mu = F.softplus(self.dec_out(h)) + 1e-8
        theta = torch.exp(self.log_theta).unsqueeze(0).expand_as(mu) + 1e-8
                                
        return mu, theta 
    
    def forward(self, x: torch.Tensor, p: torch.Tensor) -> ForwardOut:
        mu_q, logvar_q = self.encode(x, p)
        z = self.reparameterize(mu_q, logvar_q)
        mu_nb, theta_nb = self.decoder(z, p)
   
        return ForwardOut(
            x_mu=mu_nb,
            x_theta=theta_nb,
            z=z,
            mu_q=mu_q,
            logvar_q=logvar_q,
            extras={},
        )
    
    @staticmethod
    def _kl_divergence(mu_q: torch.Tensor, logvar_q: torch.Tensor) -> torch.Tensor :
        return -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp(), dim=1).mean()
    
    @staticmethod
    def _nb_loss(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = x.clamp_min(0.0)
        mu = mu.clamp_min(eps)
        theta = theta.clamp_min(eps)

        t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        t2 = theta * (torch.log(theta) - torch.log(mu + theta))
        t3 = x * (torch.log(mu) - torch.log(mu + theta))
        log_prob = t1 + t2 + t3
        return -log_prob.mean()
    
    def loss(self, x: torch.Tensor, p: torch.Tensor, *, epoch: int):
        out = self.forward(x, p)

        kl_loss = self._kl_divergence(out.mu_q, out.logvar_q)
        rec_loss = self._nb_loss(x, out.x_mu, out.x_theta)

        loss = kl_loss + rec_loss

        logs = {
            "loss": float(loss.item()),
            "kl_loss": float(kl_loss.item()),
            "rec_loss": float(rec_loss.item()),
        }
        return loss, logs