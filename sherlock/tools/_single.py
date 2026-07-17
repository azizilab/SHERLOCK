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
import pickle
import random
from pathlib import Path

import numpy as np
import pyro
import torch
from torch.utils.data import DataLoader

from ._datasets import PerturbMatchingDataset, PerturbSimpleDataset
from ._models import VAE, build_treat_effect_map, build_go_term_map
from ._cvae import cVAE
from ._svae import sVAE
from ._scgen import SCGENModel
from ._trainers import VAETrainer, cVAETrainer, sVAETrainer, SCGENTrainer
from .._configs import get_config

# TODO: hardcoded to the Norman gene2go cache pending an auto-download fallback
# (gears.utils.get_go_auto) that would work for arbitrary datasets.
_DEFAULT_GENE2GO_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks" / "norman" / "gears_data" / "gene2go_all.pkl"
)


# ── run ───────────────────────────────────────────────────────────────────────

def run(
    adata,
    model: str = "vae",
    batch_size: int = 4096,
    shuffle: bool = True,
    num_workers: int = 0,
    lr: float = 2e-3,
    num_epochs: int = 200,
    treat_effect_key: str = "treat_effect",
    validate_every: int = 10,
    device: torch.device = torch.device("cpu"),
    patience: int = 20,
    seed: int | None = None,
    # ---- VAE-specific ----
    latent_dim: int = 16,
    tau_init: float = 0.67,
    tau_end: float = 0.3,
    debug: bool = False,
    rho_init: bool = False,
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
    seed             : RNG seed for reproducibility (covers weight init + training).
    **kwargs         : additional model-specific constructor arguments. For
                       model='vae' with use_de_align_loss=True, treat_effect_map
                       is auto-built from adata.uns[treat_effect_key] unless
                       passed explicitly; de_align_exclude (e.g. held-out combo
                       labels) and de_align_top_k control that construction.
                       With use_go_prior=True, go_term_indices/go_term_offsets/
                       n_go_terms are auto-built from a hardcoded gene2go pickle
                       (_DEFAULT_GENE2GO_PATH, currently Norman-specific) unless
                       go_term_indices is passed explicitly.

    Returns
    -------
    dict with 'model' (best checkpoint) and 'param_store'.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pyro.set_rng_seed(seed)

    key = model.lower()

    if key == "vae":
        ctx = mp.get_context("spawn") if num_workers > 0 else None

        combinatorial = kwargs.pop("combinatorial", False)
        de_align_exclude = kwargs.pop("de_align_exclude", ())
        de_align_top_k = kwargs.pop("de_align_top_k", None)   # None => all genes
        dataset = PerturbMatchingDataset(adata, combinatorial=combinatorial)

        if kwargs.get("use_de_align_loss", False) and "treat_effect_map" not in kwargs:
            kwargs["treat_effect_map"] = build_treat_effect_map(
                adata.uns[treat_effect_key],
                dataset.perturbation_dict,
                list(adata.var_names),
                exclude=de_align_exclude,
                top_k=de_align_top_k,
            )

        if kwargs.get("use_go_prior", False) and "go_term_indices" not in kwargs:
            with open(_DEFAULT_GENE2GO_PATH, "rb") as f:
                gene2go = pickle.load(f)
            go_term_indices, go_term_offsets, n_go_terms = build_go_term_map(
                gene2go, dataset.perturbation_dict
            )
            kwargs["go_term_indices"] = go_term_indices
            kwargs["go_term_offsets"] = go_term_offsets
            kwargs["n_go_terms"] = n_go_terms

        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.get_collate_fn(),
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
            "self", "vae", "dataloader", "treat_effect", "adata",
            "lr", "num_epochs", "validate_every", "device", "patience",
            "tau_init", "tau_end", "seed",
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
            combinatorial=combinatorial,
            **vae_kwargs,
        ).to(device)

        if rho_init:
            vae.init_p_emb_from_adata(adata)

        trainer = VAETrainer(
            vae=vae,
            dataloader=dataloader,
            treat_effect=adata.uns[treat_effect_key],
            adata=adata,
            lr=lr,
            num_epochs=num_epochs,
            validate_every=validate_every,
            device=device,
            patience=patience,
            tau_init=tau_init,
            tau_end=tau_end,
            seed=seed,
            **trainer_kwargs,
        )

        best_model, param_store = trainer.fit()

        if debug and trainer.history:
            import matplotlib.pyplot as plt
            import pandas as pd
            hist_df = pd.DataFrame(trainer.history)
            epochs = hist_df["epoch"]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

            ax1.plot(epochs, hist_df["ATE"], "b-o", markersize=4, label="ATE")
            ax1.set_ylabel("ATE (Pearson r)")
            kl_end = trainer.n_epochs_kl_warmup
            l0_end = trainer.n_epochs_kl_warmup + trainer.n_epochs_l0_warmup
            ax1.axvline(kl_end, color="gray", linestyle="--", alpha=0.6, label="KL warmup end")
            ax1.axvline(l0_end, color="orange", linestyle="--", alpha=0.6, label="L0 warmup end")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            ax2.plot(epochs, hist_df["pi25"], "g-", label="π p25 (gate activity)")
            ax2.plot(epochs, hist_df["pi99"], "r-", label="π p99 (gate activity)")
            ax2.axvline(kl_end, color="gray", linestyle="--", alpha=0.6)
            ax2.axvline(l0_end, color="orange", linestyle="--", alpha=0.6)
            ax2.set_ylabel("Gate prob. π")
            ax2.set_xlabel("Epoch")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            fig.suptitle("Training trajectory", fontsize=12)
            plt.tight_layout()
            plt.show()

    elif key == "cvae":
        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")
        combinatorial = kwargs.get("combinatorial", False)

        dataset = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label, combinatorial=combinatorial)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        cvae_params    = set(inspect.signature(cVAE.__init__).parameters) - {"self"}
        trainer_params = set(inspect.signature(cVAETrainer.__init__).parameters) - {"self"}
        cvae_kwargs    = {k: kwargs[k] for k in kwargs if k in cvae_params}
        trainer_kwargs = {k: kwargs[k] for k in kwargs if k in trainer_params and k not in {
            "model", "dataloader", "treat_effect", "adata", "lr", "num_epochs",
            "validate_every", "device", "patience", "seed",
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
            adata=adata,
            lr=lr,
            num_epochs=num_epochs,
            validate_every=validate_every,
            device=device,
            patience=patience,
            seed=seed,
            **trainer_kwargs,
        )

        best_model, param_store = trainer.fit()

    elif key == "svae":
        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")
        combinatorial = kwargs.get("combinatorial", False)

        dataset = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label, combinatorial=combinatorial)
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
            "model", "dataloader", "treat_effect", "adata", "lr", "num_epochs",
            "validate_every", "device", "patience", "desc_name", "seed",
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
            adata=adata,
            lr=lr,
            num_epochs=num_epochs,
            validate_every=validate_every,
            device=device,
            patience=patience,
            desc_name="sVAE",
            seed=seed,
            **trainer_kwargs,
        )

        best_model, param_store = trainer.fit()

    elif key == "contrastivevi":
        from scvi.external import ContrastiveVI as _ScviContrastiveVI
        from ._contrastivevi import ContrastiveVI as _SherlockContrastiveVI

        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")

        combinatorial       = kwargs.pop("combinatorial", False)
        n_background_latent = kwargs.pop("n_background_latent", 10)
        n_salient_latent    = kwargs.pop("n_salient_latent", 10)
        n_hidden            = kwargs.pop("n_hidden", 128)
        n_layers            = kwargs.pop("n_layers", 1)
        dropout_rate        = kwargs.pop("dropout_rate", 0.1)
        wasserstein_penalty = kwargs.pop("wasserstein_penalty", 0.0)
        early_stopping      = kwargs.pop("early_stopping", True)
        if kwargs:
            raise TypeError(f"Unknown keyword(s) for ContrastiveVI run: {list(kwargs)}")

        # Build PerturbSimpleDataset so pert indices align with PerturbModelBase eval.
        # Store them as float continuous covariate(s) so scvi passes them through to
        # loss() unmodified (categorical fields re-encode alphabetically, breaking
        # alignment with PerturbSimpleDataset which assigns NTC=0 then alpha order).
        # Combinatorial mode splits "A+B" labels into individual gene indices (N, 2);
        # both columns are registered so loss() can sum the per-component Gaussian priors.
        # These temporary obs columns are removed after training.
        ds = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label,
                                  combinatorial=combinatorial)
        if combinatorial:
            adata.obs["_sherlock_pert_idx_0"] = ds.P_indices[:, 0].astype(np.float32)
            adata.obs["_sherlock_pert_idx_1"] = ds.P_indices[:, 1].astype(np.float32)
            cont_cov_keys = ["_sherlock_pert_idx_0", "_sherlock_pert_idx_1"]
        else:
            adata.obs["_sherlock_pert_idx"] = ds.P_indices.astype(np.float32)
            cont_cov_keys = ["_sherlock_pert_idx"]

        _ScviContrastiveVI.setup_anndata(
            adata, continuous_covariate_keys=cont_cov_keys
        )
        background_indices = np.where(adata.obs[p_key] == ntc_label)[0].tolist()
        target_indices     = np.where(adata.obs[p_key] != ntc_label)[0].tolist()

        # Create the scvi training wrapper to get n_batch / n_input from registration,
        # then replace its standard ContrastiveVAE module with our combined class.
        scvi_model = _ScviContrastiveVI(
            adata,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            wasserstein_penalty=wasserstein_penalty,
            use_observed_lib_size=True,
        )
        old = scvi_model.module
        pert_module_kwargs = dict(
            n_input=old.n_input,
            n_batch=old.n_batch,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=True,
            wasserstein_penalty=wasserstein_penalty,
        )
        if not old.use_observed_lib_size:
            pert_module_kwargs["library_log_means"] = old.library_log_means.cpu().numpy()
            pert_module_kwargs["library_log_vars"]  = old.library_log_vars.cpu().numpy()
        best_model = _SherlockContrastiveVI(
            n_perturbs=ds.n_perturbs, combinatorial=combinatorial, **pert_module_kwargs
        )
        scvi_model.module = best_model   # scvi trains this object in-place

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Trying to infer the `batch_size`.*")
            warnings.filterwarnings("ignore", message=".*does not have many workers.*")
            scvi_model.train(
                background_indices=background_indices,
                target_indices=target_indices,
                max_epochs=num_epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
            )

        # Remove temporary pert-index columns added for scvi's covariate pipeline
        for col in cont_cov_keys:
            adata.obs.drop(columns=[col], inplace=True, errors="ignore")

        # best_model IS scvi_model.module — the trained ContrastiveVI instance
        param_store = None

    elif key == "scgen":
        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")
        combinatorial = kwargs.pop("combinatorial", False)
        if combinatorial:
            raise ValueError(
                "scGen does not support combinatorial=True. "
                "Use combinatorial=False and treat combined labels as atomic."
            )

        scgen_params  = set(inspect.signature(SCGENModel.__init__).parameters) - {"self"}
        trainer_params = set(inspect.signature(SCGENTrainer.__init__).parameters) - {
            "self", "model", "dataloader", "treat_effect", "adata",
            "lr", "num_epochs", "validate_every", "device", "patience", "seed",
        }
        scgen_kwargs   = {k: kwargs[k] for k in kwargs if k in scgen_params}
        trainer_kwargs = {k: kwargs[k] for k in kwargs if k in trainer_params}
        unknown = [k for k in kwargs if k not in scgen_params and k not in trainer_params]
        if unknown:
            raise TypeError(f"Unknown keyword(s) for scGEN run: {unknown}")

        dataset = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model_obj = SCGENModel(
            input_dim  = dataset.input_dim,
            n_perturbs = dataset.n_perturbs,
            latent_dim = latent_dim,
            **scgen_kwargs,
        ).to(device)

        trainer = SCGENTrainer(
            model          = model_obj,
            dataloader     = dataloader,
            treat_effect   = adata.uns[treat_effect_key],
            adata          = adata,
            lr             = lr,
            num_epochs     = num_epochs,
            validate_every = validate_every,
            device         = device,
            patience       = patience,
            seed           = seed,
            **trainer_kwargs,
        )

        best_model, param_store = trainer.fit()

    else:
        raise ValueError(
            f"Unknown model '{model}'. Choose from ['vae', 'cvae', 'svae', 'contrastivevi', 'scgen']."
        )

    best_model.eval()
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
