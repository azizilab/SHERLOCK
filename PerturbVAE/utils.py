class HyperparamConfig:
  beta: float = 0.5 
  n_genes: int = 5000
  n_perturb: int = 500
  n_modules: int = 50
  latent_rank: int= 10
  n_conditions: int = 5
  expert_dim: int = 16
  c_embed_dim: int = 64
  p_embed_dim: int = 64
  hidden_dims: list = [128,]

class LitConfig:
  lr: float = 0.001
  weight_decay: float = 1e-4
  weight_div: float = 1e-3
  weight_hsic: float = 1e-3
  weight_l1: float = 1e-3

class TrainerConfig:
  max_epochs: int = 200
  force_cpu: bool = False
  gradient_clip_val: float = 100.0