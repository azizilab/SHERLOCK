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

        

class SparseModuleTransform(nn.Module):
    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 module_map: list[str],
                 hidden: int = 4,
                 activation=nn.LeakyReLU(),
                 reverse: bool = False,
                 inference: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.activation = activation

        # Compute the intermediate hidden dimension: d × hidden
        d = out_dim
        hidden_dim = d * hidden

        # Create (input_dim x out_dim) binary mask from module_map
        if reverse:
            base_mask = torch.zeros((out_dim, input_dim), dtype=torch.bool)
        else:
            base_mask = torch.zeros((input_dim, out_dim), dtype=torch.bool)
        for i, entry in enumerate(module_map):
            dims = map(int, entry.split(','))
            for dim in dims:
                base_mask[i, dim] = True

        if reverse:
            # Transpose the mask for reverse mapping
            base_mask = base_mask.t()

        # Expand mask to hidden dimension (input_dim x hidden_dim)
        # Each column of base_mask gets repeated `hidden` times
        self.register_buffer("base_mask", base_mask)
        self.register_buffer("mask", base_mask.repeat_interleave(hidden, dim=1))

        # Learnable weights and biases
        self.input_to_hidden = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.input_to_hidden)
        nn.init.zeros_(self.bias)
        
        self.hidden_to_hidden = nn.Linear(hidden, hidden)
        self.hidden_to_out = nn.Linear(hidden, 2 if inference else 1)  # Output is mean and log variance

        group_sizes = base_mask.sum(dim=0).to(torch.float32) * hidden
        self.register_buffer(
            "group_sqrt_sizes",
            torch.sqrt(group_sizes.clamp_min(1.0))  # (O,)
        )

    def forward(self, x):
        # Apply mask to input projection weights
        masked_weight = self.input_to_hidden * self.mask

        # Linear projection: input_dim × hidden_dim
        h = x @ masked_weight + self.bias 
        h = self.activation(h)
        h = h.view(x.shape[0], self.out_dim, self.hidden) # N x latent x hidden
        h = self.hidden_to_hidden(h)
        h = self.activation(h)

        # Final projection to output
        return self.hidden_to_out(h).squeeze(-1) # N x latent x 2 (mean, logvar if inference)
    
    def regularization_loss(self, l1: float = 0.0, l2: float = 0.0,
                            group_lambda: float = 0.0) -> torch.Tensor:
        """
        Vectorized Elastic Net + Group Lasso on masked weights (no Python loops).

        L1:  l1 * ||W∘mask||_1
        L2:  l2 * ||W∘mask||_2^2
        GL:  group_lambda * sum_j sqrt(|G_j|) * ||W_j||_F,
             where W_j collects all masked weights feeding latent j across 'hidden'.
        """
        W = self.input_to_hidden
        Wm = W * self.mask  # (I, O*H)
        loss = Wm.new_zeros(())

        if l1:
            loss = loss + l1 * Wm.abs().sum()

        if l2:
            loss = loss + l2 * (Wm.pow(2).sum())

        if group_lambda:
            # reshape to (I, O, H)
            I, O, H = self.input_dim, self.out_dim, self.hidden
            Wm_IOH = Wm.view(I, O, H)

            # broadcast base mask over hidden: (I, O, 1)
            bm = self.base_mask.to(dtype=Wm.dtype).unsqueeze(-1)

            # zero out rows not in group (vectorized selection)
            Wg = Wm_IOH * bm  # (I, O, H)

            # Frobenius per latent j: ||W_j||_F = sqrt(sum_{i,h} Wg^2)
            fro_per_latent = torch.sqrt((Wg.pow(2).sum(dim=(0, 2))).clamp_min(0.0))  # (O,)

            # weighted sum with sqrt(|G_j|)
            loss = loss + group_lambda * (self.group_sqrt_sizes * fro_per_latent).sum()

        return loss