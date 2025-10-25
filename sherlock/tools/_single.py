from __future__ import annotations

import multiprocessing as mp
import numpy as np
import pyro
import torch
from torch.utils.data import DataLoader
import pyro.poutine as poutine
from typing import Literal
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import inspect



from ._datasets import PerturbMatchingDataset
from .._configs import get_config
from ._models import VAE
from ._trainers import VAETrainer


def run_single(
    adata,
    batch_size=4096,
    shuffle=True,
    num_workers=0,
    latent_dim=16,
    tau_init=0.67,      
    tau_end=0.1,     
    lr=1e-3,
    num_epochs=200,
    treat_effect_key="treat_effect",
    use_conditions=False,
    validate_every=10,
    device=torch.device('cpu'),
    patience=20,
    **kwargs,
):

    if num_workers > 0:
        ctx = mp.get_context("spawn")

    dataset = PerturbMatchingDataset(adata)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        multiprocessing_context=ctx if num_workers > 0 else None,
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


    pyro.clear_param_store()

    # ---- VAE defaults from run_single args
    default_vae_args = dict(
        input_dim=adata.shape[-1],
        latent_dim=latent_dim,
        perturbs=int(len(np.unique(dataset.P_indices))),
        conds=int(len(np.unique(dataset.C_indices))),
        tau=tau_init,
        use_conditions=use_conditions,
    )

    # ---- Filter user kwargs to ONLY those accepted by VAE.__init__
    try:
        vae_params = set(inspect.signature(VAE.__init__).parameters)
        vae_params.discard("self")
    except Exception:
        # if reflection fails, assume everything is a VAE kw
        vae_params = set(default_vae_args.keys())

    vae_kwargs = {k: kwargs[k] for k in list(kwargs) if k in vae_params}
    # warn on unknown kwargs:
    unknown = [k for k in kwargs.keys() if k not in vae_params]
    if unknown:
        raise TypeError(f"Unknown keyword(s) for VAE/run_single: {unknown}")

    # user kwargs override defaults
    default_vae_args.update(vae_kwargs)

    # construct model
    vae = VAE(**default_vae_args).to(device)

    trainer = VAETrainer(
        vae=vae,
        dataloader=dataloader,
        treat_effect=adata.uns[treat_effect_key],
        lr=lr,
        num_epochs=num_epochs,
        validate_every=validate_every,
        device=device,
        patience=patience,
        tau_init=tau_init,
        tau_end=tau_end,
    )

    best_vae, param_store = trainer.fit()
    return {'model': best_vae, 'param_store': param_store}

@torch.no_grad()
def _gen_uns(model, adata, ds, obsm_key, uns_key):
    uns_data = {}

    #accuracy
    p_key = get_config("pert_key")
    ntc_label = get_config("ntc_label")

    P_sub = adata[adata.obs[p_key].values != ntc_label]
    z_np = P_sub.obsm.get(obsm_key)
    z = torch.as_tensor(z_np, dtype=torch.float32)

    true_idx = np.array([ds.perturbation_dict[name] for name in P_sub.obs[p_key].values], dtype=np.int64)
    y_true = torch.as_tensor(true_idx, dtype=torch.long)

    logits = model.cls_head(z)

    y_pred = logits.argmax(dim=-1)
    acc = (y_pred == y_true).float().mean().item()

    uns_data['acc'] = acc

    #correlation
    L = model.qr()           
    sigma_P = pyro.param("sigma_fac").cpu().detach()

    P_ = L.size(0)
    Sigma_P_posterior = L @ L.T + sigma_P**2 * torch.eye(P_)
    rho_corr = model._corr(Sigma_P_posterior).numpy()
    uns_data['rho_corr'] = rho_corr



    #correlation
    num_p = model.perturbs
    d = model.latent_dim
    P = torch.from_numpy(ds.P_indices)
    Sigma_P_posterior = model._corr(Sigma_P_posterior)
    z_bar = torch.zeros(num_p, d)
    counts = torch.zeros(num_p)
    for p in range(num_p):
        mask = (P == p)
        if mask.any():
            z_bar[p] = z[mask].mean(0)
            counts[p] = mask.sum()

    z_corr = torch.corrcoef(z_bar)     
    uns_data['z_corr'] = z_corr.numpy()    

    #z corr vs rho_corr
    def flat_triu(M):
        M = torch.tensor(M) if not torch.is_tensor(M) else M
        idx = torch.triu_indices(M.size(0), M.size(1), offset=1)
        return M[idx[0], idx[1]]

    rho_flat  = flat_triu(rho_corr)
    z_flat    = flat_triu(z_corr)

    r_s, p_s = spearmanr(rho_flat, z_flat)
    r_p, p_p = pearsonr(rho_flat, z_flat)

    uns_data['z_rho_corr'] = {"spearman_r": r_s, "spearman_p": p_s,
                              "pearson_r": r_p,   "pearson_p": p_p}
    
    # W
    W = model.gate(deterministic=True).cpu().detach().numpy()
    uns_data['W'] = W

    adata.uns[uns_key] = uns_data


@torch.no_grad()
def eval_single(model, adata, obsm_key: str = "z", uns_key: str="results") -> None:
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    p_key = get_config("pert_key")
    ntc_label = get_config("ntc_label")

    ds = PerturbMatchingDataset(adata)

    def mednorm(x_t: torch.Tensor) -> torch.Tensor:
        lib = x_t.sum(dim=1, keepdim=True)
        med = torch.median(lib).item()
        return torch.log1p(x_t / lib * med)

    d_latent = model.latent_dim

    Xp = ds.X_pert
    l = Xp.sum(axis=-1, keepdims=True)
    if hasattr(Xp, "toarray"):
        Xp = Xp.toarray()
    Xp = torch.as_tensor(Xp, dtype=torch.float32, device=device)
    Xp = mednorm(Xp)
    z_mu, _ = model.z_encoder(Xp).chunk(2, dim=-1)

    # latent space
    mask_pert = (adata.obs[p_key].values != ntc_label)
    out = np.full((adata.n_obs, d_latent), np.nan, dtype=np.float32)
    out[mask_pert] = z_mu.detach().cpu().numpy()

    adata.obsm[obsm_key] = out

    # X predictions
    out = np.full((adata.n_obs, adata.n_vars), np.nan, dtype=np.float32)
    out[mask_pert] = model.z_decoder(z_mu).detach().cpu().numpy()
    out[mask_pert] = torch.softmax(torch.from_numpy(out[mask_pert]), dim=-1).numpy()
    out[mask_pert] = l * out[mask_pert]

    adata.layers['x_pred'] = out


    _gen_uns(model, adata, ds, obsm_key, uns_key)