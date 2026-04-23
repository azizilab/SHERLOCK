"""
run() / eval() — model-agnostic entry points for the sherlock package.

run()  : dispatches to the appropriate dataset/model/trainer setup based on
         the 'model' argument, then trains and returns a results dict.

eval() : delegates entirely to model.eval(adata, ...).

Backward-compatible aliases:
    run_single  = run
    eval_single = eval
"""

from __future__ import annotations

import inspect
import multiprocessing as mp

import numpy as np
import pyro
import torch
from torch.utils.data import DataLoader

from ._datasets import PerturbMatchingDataset, PerturbSimpleDataset
from ._models import VAE
from ._cvae import cVAE
from ._svae import sVAE
from ._trainers import VAETrainer, cVAETrainer, sVAETrainer
from .._configs import get_config


# ── run ───────────────────────────────────────────────────────────────────────

def run(
    adata,
    model: str = "vae",
    batch_size: int = 4096,
    shuffle: bool = True,
    num_workers: int = 0,
    lr: float = 1e-3,
    num_epochs: int = 200,
    treat_effect_key: str = "treat_effect",
    validate_every: int = 10,
    device: torch.device = torch.device("cpu"),
    patience: int = 20,
    # ---- VAE-specific ----
    latent_dim: int = 16,
    tau_init: float = 0.67,
    tau_end: float = 0.10,
    **kwargs,
) -> dict:
    """
    Train a perturbation model on *adata*.

    Parameters
    ----------
    adata            : AnnData object.
    model            : 'vae', 'cvae', or 'svae'.
    treat_effect_key : key in adata.uns holding ground-truth ATEs.
    latent_dim       : latent space dimensionality (VAE).
    tau_init         : initial gate temperature (VAE).
    tau_end          : final gate temperature after annealing (VAE).
    **kwargs         : additional model-specific constructor arguments.

    Returns
    -------
    dict with 'model' (best checkpoint) and 'param_store'.
    """
    key = model.lower()

    if key == "vae":
        ctx = mp.get_context("spawn") if num_workers > 0 else None

        combinatorial = kwargs.pop("combinatorial", False)
        dataset = PerturbMatchingDataset(adata, combinatorial=combinatorial)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            multiprocessing_context=ctx,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        pyro.clear_param_store()

        vae_params     = set(inspect.signature(VAE.__init__).parameters) - {"self"}
        trainer_params = set(inspect.signature(VAETrainer.__init__).parameters) - {
            "self", "vae", "dataloader", "treat_effect",
            "lr", "num_epochs", "validate_every", "device", "patience",
            "tau_init", "tau_end",
        }
        vae_kwargs     = {k: kwargs[k] for k in kwargs if k in vae_params}
        trainer_kwargs = {k: kwargs[k] for k in kwargs if k in trainer_params}
        unknown = [k for k in kwargs if k not in vae_params and k not in trainer_params]
        if unknown:
            raise TypeError(f"Unknown keyword(s) for VAE run: {unknown}")

        vae = VAE(
            input_dim=adata.shape[-1],
            latent_dim=latent_dim,
            perturbs=int(np.max(dataset.P_indices)) + 1,
            conds=int(len(np.unique(dataset.C_indices))),
            tau=tau_init,
            **vae_kwargs,
        ).to(device)

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
            **trainer_kwargs,
        )

        best_model, param_store = trainer.fit()

    elif key == "cvae":
        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")

        dataset = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        pyro.clear_param_store()

        cvae_params    = set(inspect.signature(cVAE.__init__).parameters) - {"self"}
        trainer_params = set(inspect.signature(cVAETrainer.__init__).parameters) - {"self"}
        cvae_kwargs    = {k: kwargs[k] for k in kwargs if k in cvae_params}
        trainer_kwargs = {k: kwargs[k] for k in kwargs if k in trainer_params and k not in {
            "model", "dataloader", "treat_effect", "lr", "num_epochs",
            "validate_every", "device", "patience",
        }}
        unknown = [k for k in kwargs if k not in cvae_params and k not in trainer_params]
        if unknown:
            raise TypeError(f"Unknown keyword(s) for cVAE run: {unknown}")

        model_obj = cVAE(
            input_dim=dataset.input_dim,
            latent_dim=latent_dim,
            n_perturbs=dataset.n_perturbs,
            **cvae_kwargs,
        ).to(device)

        trainer = cVAETrainer(
            model=model_obj,
            dataloader=dataloader,
            treat_effect=adata.uns[treat_effect_key],
            lr=lr,
            num_epochs=num_epochs,
            validate_every=validate_every,
            device=device,
            patience=patience,
            **trainer_kwargs,
        )

        best_model, param_store = trainer.fit()

    elif key == "svae":
        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")

        dataset = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        svae_params    = set(inspect.signature(sVAE.__init__).parameters) - {"self"}
        trainer_params = set(inspect.signature(sVAETrainer.__init__).parameters) - {"self"}
        svae_kwargs    = {k: kwargs[k] for k in kwargs if k in svae_params}
        trainer_kwargs = {k: kwargs[k] for k in kwargs if k in trainer_params and k not in {
            "model", "dataloader", "treat_effect", "lr", "num_epochs",
            "validate_every", "device", "patience", "desc_name",
        }}
        unknown = [k for k in kwargs if k not in svae_params and k not in trainer_params]
        if unknown:
            raise TypeError(f"Unknown keyword(s) for sVAE run: {unknown}")

        model_obj = sVAE(
            input_dim=dataset.input_dim,
            n_perturbs=dataset.n_perturbs,
            latent_dim=latent_dim,
            **svae_kwargs,
        ).to(device)

        trainer = sVAETrainer(
            model=model_obj,
            dataloader=dataloader,
            treat_effect=adata.uns[treat_effect_key],
            lr=lr,
            num_epochs=num_epochs,
            validate_every=validate_every,
            device=device,
            patience=patience,
            desc_name="sVAE",
            **trainer_kwargs,
        )

        best_model, param_store = trainer.fit()

    else:
        raise ValueError(
            f"Unknown model '{model}'. Choose from ['vae', 'cvae', 'svae']."
        )

    return {"model": best_model, "param_store": param_store}


# ── eval ──────────────────────────────────────────────────────────────────────

def eval(
    model,
    adata,
    obsm_key: str = "z",
    uns_key: str = "results",
    device: torch.device | None = None,
    batch_size: int = 1024,
    dataset=None,
) -> None:
    """
    Evaluate *model* on *adata*, writing z, x_pred, and metrics in-place.

    Delegates entirely to model.eval(adata, ...).
    """
    model.eval(
        adata=adata,
        obsm_key=obsm_key,
        uns_key=uns_key,
        device=device,
        batch_size=batch_size,
        dataset=dataset,
    )


# ── backward-compatible aliases ───────────────────────────────────────────────

run_single  = run
eval_single = eval
