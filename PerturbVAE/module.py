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

        
        

