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