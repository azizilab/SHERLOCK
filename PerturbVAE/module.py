import torch
import torch.nn as nn
import torch.nn.functional as F

class Dense_NN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        out_dims,
        activation=nn.GELU(),
        add_dropout=False,
        dropout_p=0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.param_dims = out_dims
        self.total_output_dim = sum(out_dims)
        self.add_dropout = add_dropout
        self.dropout_p = dropout_p

        layers = [nn.Linear(input_dim, hidden_dims[0])]
        dropouts = [nn.Dropout(p=dropout_p)] if add_dropout else []

        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if add_dropout:
                dropouts.append(nn.Dropout(p=dropout_p))

        layers.append(nn.Linear(hidden_dims[-1], self.total_output_dim))

        self.layers = nn.ModuleList(layers)
        self.dropouts = nn.ModuleList(dropouts) if add_dropout else None
        self.activation = activation

    def forward(self, x):
        H = x
        for i, layer in enumerate(self.layers):
            H = layer(H)
            if i < len(self.layers) - 1:
              H = self.activation(H)
              if self.add_dropout:
                  H = self.dropouts[i](H)
        if len(self.param_dims) == 1:
            return H
        else:
            return torch.split(H, self.param_dims, dim=-1)
        
        
class GeneModuleEncoder(nn.Module):
    def __init__(self,
                 num_genes: int,
                 module_dict: dict[int, list[int]],
                 hidden_dims: list[int],
                 activation=nn.LeakyReLU(),
                 add_dropout: bool = False,
                 dropout_p: float = 0.5,
                 if_conditional: bool = True):
        super().__init__()

        self.module_dict = module_dict           # {module_id: [gene indices]}
        self.num_genes   = num_genes
        self.latent_dim  = len(module_dict)      # one latent dim per module

        # ── build a stable order and a reverse lookup ────────────────────────
        self._module_order        = list(module_dict.keys())   # e.g. [3,7,2]
        self._id2idx: dict[int,int] = {
            mid: idx for idx, mid in enumerate(self._module_order)
        }
        self.networks = nn.ModuleList([
            Dense_NN(
                input_dim=len(module_dict[mid]),
                hidden_dims=hidden_dims,
                out_dims=[1, 1],
                activation=activation,
                add_dropout=add_dropout,
                dropout_p=dropout_p
            )
            for mid in self._module_order
        ])

        # condition network
        self.conditional = if_conditional

    def forward(self, x):
        x_gene, x_cond = x[:, :self.num_genes], x[:, self.num_genes:]

        z_mean, z_log_var = [], []

        # iterate in the same fixed order
        for mid in self._module_order:          
            dims = self.module_dict[mid]        
            sub_in = x_gene[:, dims]            
            idx   = self._id2idx[mid]           
            mu, logvar = self.networks[idx](sub_in)
            z_mean.append(mu)
            z_log_var.append(logvar)

        z_mean = torch.cat(z_mean, dim=-1)   # (B, latent_dim)
        z_log_var = torch.cat(z_log_var, dim=-1)

        # fuse gene & condition information
        if self.conditional:
            mu_c = x_cond
        else:
            mu_c = torch.zeros_like(z_mean)
            
        z_mean = z_mean + mu_c
        z_log_var = z_log_var
        return z_mean, z_log_var

class GeneModuleEncoderRegularized(nn.Module):
    def __init__(self,
                 num_genes: int,
                 module_dict: dict[int, list[int]],
                 if_conditional: bool = False,
                 min_var: float = 1e-3):
        super().__init__()
        self.module_dict = module_dict
        self.num_genes   = num_genes
        self.latent_dim  = len(module_dict)
        self.conditional = if_conditional
        self.min_var     = min_var

        self._module_order = list(module_dict.keys())
        self._id2idx = {mid: i for i, mid in enumerate(self._module_order)}

        # per-module params
        self._w_params   = nn.ParameterList()                   
        self._bias       = nn.Parameter(torch.zeros(self.latent_dim))
        self._scale_raw  = nn.Parameter(torch.zeros(self.latent_dim))      
        self._a_raw      = nn.Parameter(torch.zeros(self.latent_dim))      
        self._shift      = nn.Parameter(torch.zeros(self.latent_dim))      
        self._logvar_raw = nn.Parameter(torch.full((self.latent_dim,), -2.0))

        # optional diagonal condition coupling (mean only)
        if self.conditional:
            self._cond_gain = nn.Parameter(torch.zeros(self.latent_dim))

        for mid in self._module_order:
            gidx = module_dict[mid]
            self._w_params.append(nn.Parameter(torch.zeros(len(gidx))))  # ~uniform start

    def forward(self, x):
        x_gene, x_cond = x[:, :self.num_genes], x[:, self.num_genes:]

        z_mu_list, z_logvar_list = [], []

        for k, mid in enumerate(self._module_order):
            gidx = self.module_dict[mid]
            sub  = x_gene[:, gidx]                                  # (B, |M_k|)

            # convex pooling
            w = F.softmax(self._w_params[k], dim=0)                 # (|M_k|,)
            t = (sub * w.unsqueeze(0)).sum(dim=1, keepdim=True)     # (B,1)

            # per-module GELU with positive slope to keep monotonacity
            a = F.softplus(self._a_raw[k]) + 1e-6                  
            s = F.softplus(self._scale_raw[k]) + 1e-6              
            u = a * (t - self._shift[k])                            
            h = F.gelu(u)                                          
            mu_k = self._bias[k] + s * h
            if self.conditional:
                mu_k = mu_k + self._cond_gain[k] * x_cond[:, k:k+1]

            var_k = F.softplus(self._logvar_raw[k]) + self.min_var
            logvar_k = torch.log(var_k).expand_as(mu_k)

            z_mu_list.append(mu_k)
            z_logvar_list.append(logvar_k)

        z_mean = torch.cat(z_mu_list, dim=1)      # (B, d)
        z_logv = torch.cat(z_logvar_list, dim=1)  # (B, d)
        return z_mean, z_logv
        
        
class GeneModuleOutputLayer(nn.Module):
    def __init__(self,
                 num_genes: int,
                 latent_dim: int,
                 module_dict: dict[int, list[int]],
                 center_zero: bool = True):
        super().__init__()
        self.G, self.d = num_genes, latent_dim
        self.center_zero = center_zero

        # M[g, k] = 1 if gene g belongs to module k
        M = torch.zeros(self.G, self.d)
        order = list(module_dict.keys())
        for j, mid in enumerate(order):
            M[module_dict[mid], j] = 1.0
        self.register_buffer("M", M)

        # per-dimension GELU parameters
        self.alpha_raw = nn.Parameter(torch.zeros(self.d))  
        self.delta     = nn.Parameter(torch.zeros(self.d)) 

        # masked, nonneg, column-normalized projection to genes
        self.W_raw = nn.Parameter(torch.randn(self.G, self.d) * 0.01)
        self.bias  = nn.Parameter(torch.zeros(self.G))

    # per module monotonic function
    def _phi(self, z: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.alpha_raw) + 1e-6         
        u = alpha * (z - self.delta)                     
        h = F.gelu(u)                                     
        if self.center_zero:
            h0 = F.gelu(-alpha * self.delta)              
            h = h - h0
        return h

    def _make_W(self) -> torch.Tensor:
        W = F.softplus(self.W_raw) * self.M               
        W = W / (W.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6))
        return W

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_nl = self._phi(z)               
        W    = self._make_W()             
        logits = z_nl @ W.T + self.bias
        return logits