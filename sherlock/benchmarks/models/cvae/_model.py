import torch
import torch.nn as nn
import torch.nn.functional as F


class cVAE(nn.Module):
    def __init__(
            self, 
            input_dim,  
            latent_dim, 
            perturbs,
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
        # Decode z and c to reconstruct x
        h = torch.cat((z, cond), dim=1)
        # Hidden layer
        h = F.relu(self.dec_fc1(h))
        h = self.dropout_layer(h)
        h = F.relu(self.dec_fc2(h))
        h = self.dropout_layer(h)


        mu = F.softplus(self.dec_out(h)) + 1e-8

        
        theta = torch.exp(self.log_theta).unsqueeze(0).expand_as(mu) + 1e-8
                                
        return mu, theta 
    
    def forward(self, x, p):
        mu_q, logvar_q = self.encode(x, p)
        z = self.reparameterize(mu_q, logvar_q)
        mu_nb, theta_nb = self.decoder(z, p)
   
        mu_p = torch.zeros_like(mu_q)
        logvar_p = torch.zeros_like(logvar_q)
        return mu_nb, theta_nb, mu_q, logvar_q, mu_p, logvar_p # return mu and sigma for reconstructed loss and KL divergence 