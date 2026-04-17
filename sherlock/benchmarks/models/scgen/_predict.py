from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData


@torch.no_grad()
def _latent_from_model(model, adata: AnnData) -> np.ndarray:
    """
    Extract latent representations directly from the trained scGen module.
    """
    dl = model._make_data_loader(adata=adata, batch_size=1024, shuffle=False)

    zs = []
    for tensors in dl:
        inf_out = model.module.inference(tensors["X"])

        if "qz_m" in inf_out:
            z = inf_out["qz_m"]
        elif "z" in inf_out:
            z = inf_out["z"]
        else:
            raise KeyError(
                f"Could not find latent key in inference output: {list(inf_out.keys())}"
            )

        zs.append(z.detach().cpu().numpy())

    return np.concatenate(zs, axis=0)


@torch.no_grad()
def predict_scgen_manual(
    model,
    *,
    ctrl_key: str,
    stim_key: str,
    condition_key: str = "condition",
    cell_type_key: str = "cell_type",
    celltype_to_predict: str = "all_cells",
):
    if not hasattr(model, "_sherlock_adata"):
        raise AttributeError("Model is missing `_sherlock_adata`. Retrain with updated trainer.")

    adata = model._sherlock_adata

    latent_all = _latent_from_model(model, adata)

    cond = adata.obs[condition_key].astype(str).values
    ctype = adata.obs[cell_type_key].astype(str).values

    ctrl_mask = cond == ctrl_key
    stim_mask = cond == stim_key
    pred_ctrl_mask = ctrl_mask & (ctype == celltype_to_predict)

    if ctrl_mask.sum() == 0:
        raise ValueError(f"No control cells found for ctrl_key={ctrl_key!r}")
    if stim_mask.sum() == 0:
        raise ValueError(f"No stimulated cells found for stim_key={stim_key!r}")
    if pred_ctrl_mask.sum() == 0:
        raise ValueError(
            f"No control cells found for celltype_to_predict={celltype_to_predict!r}"
        )

    latent_ctrl = latent_all[ctrl_mask].mean(axis=0)
    latent_stim = latent_all[stim_mask].mean(axis=0)
    delta = latent_stim - latent_ctrl

    latent_base = latent_all[pred_ctrl_mask]
    latent_pred = latent_base + delta

    latent_pred_t = torch.tensor(
        latent_pred,
        dtype=torch.float32,
        device=next(model.module.parameters()).device,
    )

    gen_out = model.module.generative(latent_pred_t)

    if "px" in gen_out:
        x_pred = gen_out["px"].detach().cpu().numpy()
    elif "px_rate" in gen_out:
        x_pred = gen_out["px_rate"].detach().cpu().numpy()
    else:
        raise KeyError(
            f"Could not find decoder output in generative keys: {list(gen_out.keys())}"
        )

    pred_adata = AnnData(
        X=x_pred,
        obs=adata.obs.loc[pred_ctrl_mask].copy(),
        var=adata.var.copy(),
    )
    pred_adata.obs[condition_key] = stim_key

    return pred_adata, delta