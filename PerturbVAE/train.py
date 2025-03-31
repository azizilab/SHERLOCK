import lightning.pytorch as pl
import torch
import pyro
from pyro.infer import Predictive, Trace_ELBO
import torch.nn.functional as F
from utils import *
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb


class LitModule(pl.LightningModule):
    def __init__(self, full_model, loss_fn, lit_config=LitConfig()):
        super().__init__()
        self.full_model = full_model
        self.model = full_model.model
        self.guide = full_model.guide
    
        self.lr = lit_config.lr
        self.weight_decay = lit_config.weight_decay
        self.weight_div = lit_config.weight_div
        self.weight_hsic = lit_config.weight_hsic
        self.weight_l1 = lit_config.weight_l1
        
        self.predictive = Predictive(self.model, guide=self.guide, num_samples=1)
        self.loss_fn = loss_fn
    
    def forward(self, *args):
        return self.predictive(*args)
    
    def diversity_reg(self, group_logits):
        avg_probs = F.softmax(group_logits, dim=-1).mean(dim=0)
        latent_dim = avg_probs.size(0)
        uniform = torch.full_like(avg_probs, 1.0 / latent_dim)
        # KL to a uniform distribution
        loss = torch.sum(uniform * torch.log((uniform + 1e-8) / (avg_probs + 1e-8)))
        return loss
    
    # linear kernel HSIC
    def HSIC_reg(self, x, y):
        N = x.shape(0)
        K = x @ x.T
        L = y @ y.T
        H = torch.eye(N, device=x.device) - (1.0/N) * torch.ones(N, N, device=x.device)
        K_c = (H @ K) @ H
        L_c = (H @ L) @ H
        return torch.trace(K_c @ L_c) / ((N - 1) ** 2)
        
    def sparsity_reg(self, x):
        return x.abs().sum()
    
    def training_step(self, batch, batch_idx):
        elbo_loss = self.loss_fn(*batch)
        reg_div = self.weight_div * self.diversity_reg(self.full_model.z_enc.group_logits)
        reg_hsic = self.HSIC_reg(self.full_model.lin_, self.full_model.nonlin_)
        reg_l1 = self.weight_l1 * (self.sparsity_reg(self.full_model.A) + self.sparsity_reg(self.full_model.B) + self.sparsity_reg(self.full_model.Q))
        reg = reg_div + reg_hsic + reg_l1
        loss = elbo_loss + reg 
        self.log("train_loss", loss)
        self.log("train_elbo", elbo_loss)
        self.log("train_diversity_reg", reg_div)
        self.log("train_hsic", reg_hsic)
        self.log("train_total_L1", reg_l1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pass
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.loss_fn.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def train_(full_model, lit_config, train_dataloader, val_loader, seed=1234, project='bruh', trainer_config=TrainerConfig()):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    one_batch = next(iter(train_dataloader))
    loss_fn = Trace_ELBO()(model=full_model.model, guide=full_model.guide)
    lit_obj = LitModule(full_model, loss_fn=loss_fn, lit_config=lit_config)
    # run one batch to initialize params
    loss_fn(*one_batch)
    
    wandb_logger = WandbLogger(project=project, log_model=True)
    wandb.init()
    
    trainer = pl.Trainer(
        accelerator="cpu" if trainer_config.force_cpu else "gpu",
        max_epochs=trainer_config.max_epochs,
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=1,
        gradient_clip_val=trainer_config.gradient_clip_val,
        enable_checkpointing=False,
        accumulate_grad_batches=1.0
    )
    
    wandb_logger.watch(lit_obj, log="all")
    trainer.fit(lit_obj, train_dataloaders=train_dataloader)
    wandb.finish()
    
    
    
        
    
    
        
        
        