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
        dropout_p=0.5
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
            return torch.split(H, self.param_dims, dim=1)

# MoE to learn gene grouping. One expert per latent dim, agg over genes
class MoE_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, 
                 hidden_dim_one_expert=16, n_params=1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.group_logits = nn.Parameter(torch.randn(input_dim, latent_dim))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_dim_one_expert),
                    nn.GELU(),
                    nn.Linear(hidden_dim_one_expert, 1)
                ) for _ in range(latent_dim)
            ]
        )
        self.net = Dense_NN(latent_dim, hidden_dims, 
                            [latent_dim]*n_params, add_dropout=False)
    
    def forward(self, x):
        group_prob = F.softmax(self.group_logits, dim=-1) # (input, latent)
        o_experts = torch.cat([expert(x.unsqueeze(-1).view(-1, 1)) for expert in self.experts], dim=-1)
        o_experts = o_experts.view(-1, self.input_dim, self.latent_dim) #(batch, input, latent)
        group_weights = torch.stack([group_prob] * x.shape[0], dim=0)
        h = (group_weights * o_experts).sum(dim=1) # (batch, latent)
        return self.net(h)
        
        

