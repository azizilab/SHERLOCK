from __future__ import annotations

import multiprocessing as mp
import numpy as np
import pyro
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pyro.poutine as poutine
from typing import Literal
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import inspect
from tqdm import tqdm



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
    use_contrastive_jacobian=False,
    use_de_align_loss=False,
    de_align_lambda=1e-3,
    device=torch.device('cpu'),
    patience=20,
    combinatorial=False,
    **kwargs,
):

    if num_workers > 0:
        ctx = mp.get_context("spawn")

    dataset = PerturbMatchingDataset(adata, combinatorial=combinatorial)
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

    # ---- optional DE alignment targets
    treat_effect_map = None
    if use_de_align_loss:
        if treat_effect_key not in adata.uns:
            raise ValueError(f"Requested DE alignment but uns['{treat_effect_key}'] is missing.")
        te = adata.uns[treat_effect_key]
        if hasattr(te, "to_df"):
            te_df = te.to_df()
        else:
            te_df = pd.DataFrame(
                getattr(te, "X", None),
                index=getattr(te, "obs_names", None),
                columns=getattr(te, "var_names", None),
            )
        te_df = te_df.reindex(columns=adata.var_names, fill_value=0.0)
        effect_mat = np.zeros((len(dataset.perturbation_dict), adata.n_vars), dtype=np.float32)
        for name, idx in dataset.perturbation_dict.items():
            if name in te_df.index:
                effect_mat[idx] = te_df.loc[name].to_numpy(dtype=np.float32)
        treat_effect_map = torch.tensor(effect_mat, dtype=torch.float32, device=device)

    # ---- VAE defaults from run_single args
    default_vae_args = dict(
        input_dim=adata.shape[-1],
        latent_dim=latent_dim,
        perturbs=np.max(dataset.P_indices)+1,
        conds=int(len(np.unique(dataset.C_indices))),
        tau=tau_init,
        use_conditions=use_conditions,
        use_contrastive_jacobian=use_contrastive_jacobian,
        use_de_align_loss=use_de_align_loss,
        de_align_lambda=de_align_lambda,
        treat_effect_map=treat_effect_map,
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
    treatment_key = get_config("treatment_key")

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

    x_p = torch.tensor(ds.X_pert).float()
    # p2g = model.pert_to_target_graph(x_p, P, fix_gate=True)
    # uns_data['p2g'] = p2g.cpu().detach().numpy()

    cond_unique = adata.obs[treatment_key].unique()
    conds = torch.tensor([ds.condition_dict[c] for c in cond_unique], dtype=torch.int32)

    # Explained Variance (Condition x P x G)
    ev_list = []
    device = next(model.parameters()).device

    for cond_label, cond_idx in zip(cond_unique, conds):
        x_ntc_cond = x_ntc_mat[cond_idx]   # (cells_in_cond, G) NTC for this condition
        ev_pg = _counterfactual_effect_size(
            model,
            x_ntc_cond=x_ntc_cond,
            cond_idx=cond_idx,
            device=device,
        )
        ev_list.append(ev_pg)
    uns_data['counterfactual_effect_size'] = np.stack(ev_list, axis=0)  # (C, P, G)

    # Condition-Perturbation Interaction (P, C, C)
    cpi = condition_perturbation_interaction(model, adata, ds, obsm_key)
    uns_data['condition_perturbation_interaction'] = cpi  # (P, C, C)

    #name saving
    uns_data['conditions'] = np.array(cond_unique)
    idx_to_pert = {idx: name for name, idx in ds.perturbation_dict.items()}
    pert_names = [idx_to_pert[i] for i in range(len(idx_to_pert))]
    uns_data['perts'] = np.array(pert_names)

    adata.uns[uns_key] = uns_data

@torch.no_grad()
def _counterfactual_effect_size(model, x_ntc_cond, cond_idx, device=None):
    if device is None:
        device = next(model.parameters()).device

    if isinstance(x_ntc_cond, np.ndarray):
        x_ntc = torch.as_tensor(x_ntc_cond, dtype=torch.float32, device=device)
    else:
        x_ntc = x_ntc_cond.to(device=device, dtype=torch.float32)

    n_cells, G = x_ntc.shape
    num_p = model.perturbs

    if n_cells < 2:
        return np.zeros((num_p, G), dtype=np.float32)

    # --- work on log-normalized scale for the denominator ---
    lib = x_ntc.sum(dim=1, keepdim=True) + 1e-8
    x_ntc_norm = torch.log1p(x_ntc / lib * 1e4)  # e.g. CPM then log1p
    var_total = x_ntc_norm.var(dim=0, unbiased=True)  # (G,)

    # abduct z0
    z0_hat = model._abduct_z0(x_ntc)          # (n_cells, d)
    lib_ntc = x_ntc.sum(dim=1, keepdim=True)  # (n_cells, 1)

    logits0 = model.z_decoder(z0_hat)
    mu_prob0 = F.softmax(logits0, dim=-1)
    mu0 = lib_ntc * mu_prob0                 # (n_cells, G)

    ev_pg = torch.zeros(num_p, G, device=device, dtype=torch.float32)
    cond_idx_long = torch.tensor(int(cond_idx), dtype=torch.long, device=device)

    for p_idx in range(num_p):
        p_vec = torch.full((n_cells,), p_idx, dtype=torch.long, device=device)

        z_cf = model.latent_counterfactual(
            x_ntc=x_ntc,
            x_p=x_ntc,
            p=p_vec,
            c_from=cond_idx_long,
            c_to=cond_idx_long,
        )

        logits1 = model.z_decoder(z_cf)
        mu_prob1 = F.softmax(logits1, dim=-1)
        mu1 = lib_ntc * mu_prob1

        # log-normalize the predicted means to match denominator scale
        mu0_norm = torch.log1p(mu0 / lib_ntc * 1e4)
        mu1_norm = torch.log1p(mu1 / lib_ntc * 1e4)

        delta = mu1_norm - mu0_norm               # (n_cells, G)

        mean_shift = delta.mean(dim=0)           # (G,)
        # var_shift = delta.var(dim=0, unbiased=True)   # (G,)

        ev = mean_shift.pow(2) / (var_total + 1e-8)
        # ev = var_shift / (var_total + 1e-8)

        ev_pg[p_idx] = ev

    return ev_pg.cpu().numpy().astype(np.float32)

def _silhouette_pair(z_c1: np.ndarray, z_c2: np.ndarray) -> float:
    """
    Mean silhouette score for a 2-condition split in z-space.
    Scale- and rotation-invariant → comparable across model runs.
    Returns float in [-1, 1]: 1=separated, 0=boundary, <0=mixed.
    """
    n1, n2 = len(z_c1), len(z_c2)
    if n1 < 2 or n2 < 2:
        return np.nan
    d11 = np.linalg.norm(z_c1[:, np.newaxis] - z_c1[np.newaxis], axis=-1)
    d22 = np.linalg.norm(z_c2[:, np.newaxis] - z_c2[np.newaxis], axis=-1)
    d12 = np.linalg.norm(z_c1[:, np.newaxis] - z_c2[np.newaxis], axis=-1)
    a1 = d11.sum(axis=1) / (n1 - 1)
    b1 = d12.mean(axis=1)
    s1 = (b1 - a1) / np.maximum(a1, b1)
    a2 = d22.sum(axis=1) / (n2 - 1)
    b2 = d12.mean(axis=0)
    s2 = (b2 - a2) / np.maximum(a2, b2)
    return float(np.concatenate([s1, s2]).mean())


def condition_perturbation_interaction(model, adata, ds, obsm_key):
    """
    Condition-perturbation interaction via silhouette score in z-space.

    score[p, c1, c2] = mean silhouette of cells with perturbation p,
                       labeled by condition (c1 vs c2).

    Range [-1, 1]:
      +1 : conditions perfectly separated for this drug
       0 : conditions on the boundary
      <0 : conditions mixed (what you see as mixing in UMAP)

    Scale- and rotation-invariant: both a_i (intra-condition) and b_i
    (inter-condition) distances are in the same z-space, so their ratio
    is unaffected by uniform scaling or rotation between runs.
    This makes the score comparable across model runs and datasets.

    Returns: (P, C, C) float32 array, NaN where a perturbation has <2 cells
             in one of the two conditions.
    """
    p_key = get_config("pert_key")
    treatment_key = get_config("treatment_key")

    z_all = np.array(adata.obsm[obsm_key], dtype=np.float32)
    pert_col = np.array(adata.obs[p_key].values, dtype=str)
    cond_col = np.array(adata.obs[treatment_key].values, dtype=str)

    P, C = model.perturbs, model.conds

    idx_to_pert = {v: k for k, v in ds.perturbation_dict.items()}
    idx_to_cond = {v: k for k, v in ds.condition_dict.items()}
    pert_names = [idx_to_pert[i] for i in range(P)]
    cond_names = [idx_to_cond[i] for i in range(C)]

    valid = ~np.isnan(z_all).any(axis=1)

    score = np.full((P, C, C), np.nan, dtype=np.float32)
    for p, pname in enumerate(pert_names):
        for c1 in range(C):
            for c2 in range(c1 + 1, C):
                mask_c1 = valid & (pert_col == pname) & (cond_col == cond_names[c1])
                mask_c2 = valid & (pert_col == pname) & (cond_col == cond_names[c2])
                s = _silhouette_pair(z_all[mask_c1], z_all[mask_c2])
                score[p, c1, c2] = s
                score[p, c2, c1] = s

    return score.astype(np.float32)

@torch.no_grad()
def eval_single(model, adata, obsm_key: str = "z", uns_key: str="results", param_store=None) -> None:
    if param_store is not None:
        pyro.clear_param_store()
        pyro.get_param_store().set_state(param_store)

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
