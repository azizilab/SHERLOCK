"""
auto_sweep_norman.py 
=========================================================
"""

import argparse
import os
import sys
import time
import warnings
import matplotlib.pyplot as plt

os.environ["TQDM_DISABLE"] = "1"
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

import numpy as np
import optuna
import pyro
import torch
import scanpy as sc
from scipy.stats import pearsonr

sys.path.append("../..")
import sherlock as slk
from metrics import extract_final_metrics, print_checklist

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR  = "results/auto_sweep_r3"
NUM_EPOCHS   = 800   # Slightly longer to let low gate_init_p stabilize
BATCH_SIZE   = 4096
HOLDOUT_FRAC = 0.20
HOLDOUT_SEED = 42
MAX_TRIALS   = 20    

TARGET_SCORE       = 0.75   
TARGET_ATE         = 0.58   
TARGET_RHO         = 0.65   
TARGET_MAX_ACTIVE  = 0.25   
MAX_FRAC_DEAD      = 0.45

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/histograms", exist_ok=True)

# ---------------------------------------------------------------------------
# SEARCH SPACE 
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    # Lock in the optimal architecture
    "latent_dim":         (52,   58),     
    "rank":               (16,   21),     
    
    # Dial in the perfect tug-of-war
    "l0_lambda":          (24.0, 26.5),   
    "ce_lambda":          (19.0, 22.0),   
    
    # Keep the offset tight based on Trial 08/14
    "n_epochs_kl_warmup": (160,  190),    
    "l0_warmup_offset":   (120,  150),    
    
    # The sweet spot for initialization
    "tau_end":            (0.175, 0.190),   
    "gate_init_p":        (0.50,  0.65),  
}

# ---------------------------------------------------------------------------
# SHERLOCK SCORE V3
# ---------------------------------------------------------------------------
def sherlock_score_v3(m: dict, gate_w=None) -> float:
    ate         = m.get("ATE", float("nan"))
    frac_dead   = m.get("frac_dead", 1.0)
    frac_active = m.get("frac_active", 1.0)
    pi99        = m.get("pi99", 0.0)
    z_rho       = m.get("z_rho_spearman", 0.0)

    if np.isnan(ate):
        return float("nan")

    polarization = 0.0
    if gate_w is not None:
        w = np.clip(np.array(gate_w).flatten(), 0.0, 1.0)
        polarization = float(np.mean((2 * w - 1) ** 2))
    polarization_reward = polarization

    ate_penalty = max(0.0, 0.58 - ate) * 6.0
    rho_penalty = max(0.0, 0.65 - (z_rho if np.isfinite(z_rho) else 0.0)) * 6.0 # Stricter Rho

    sparsity_penalty = max(0.0, frac_active - 0.25) * 6.0   

    collapse_penalty = max(0.0, frac_dead - 0.45) * 5.0
    blob_penalty = 1.0 if (not np.isnan(pi99) and pi99 < 0.1) else 0.0

    score = (
        polarization_reward
        - ate_penalty
        - rho_penalty
        - sparsity_penalty
        - collapse_penalty
        - blob_penalty
    )
    return float(score)


# ---------------------------------------------------------------------------
# OBJECTIVE
# ---------------------------------------------------------------------------
def objective(trial, train_data, full_data, holdout_combos, single_perts, treat_effect):
    
    # 1. Sample Params
    l0_lambda   = trial.suggest_float("l0_lambda", *SEARCH_SPACE["l0_lambda"])
    ce_lambda   = trial.suggest_float("ce_lambda", *SEARCH_SPACE["ce_lambda"])
    tau_end     = trial.suggest_float("tau_end",   *SEARCH_SPACE["tau_end"])
    
    n_epochs_kl_warmup = trial.suggest_int("n_epochs_kl_warmup", *SEARCH_SPACE["n_epochs_kl_warmup"])
    l0_offset          = trial.suggest_int("l0_warmup_offset",   *SEARCH_SPACE["l0_warmup_offset"])
    n_epochs_l0_warmup = n_epochs_kl_warmup + l0_offset

    # Sample Architectural Params
    latent_dim  = trial.suggest_int("latent_dim", SEARCH_SPACE["latent_dim"][0], SEARCH_SPACE["latent_dim"][1], step=2)
    rank        = trial.suggest_int("rank", *SEARCH_SPACE["rank"])
    gate_init_p = trial.suggest_float("gate_init_p", *SEARCH_SPACE["gate_init_p"])

    trial_name = f"trial_{trial.number:02d}"
    print(f"\n{'='*65}")
    print(f"  {trial_name} | Arch: latent={latent_dim}, rank={rank}, init_p={gate_init_p:.2f}")
    print(f"  Regs: l0={l0_lambda:.2f}, ce={ce_lambda:.2f}, tau={tau_end:.3f}")
    print(f"{'='*65}")

    t0 = time.time()
    pyro.clear_param_store()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = slk.tl.run(
                train_data,
                model="vae",
                batch_size=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                device=DEVICE,
                latent_dim=latent_dim,       # Passed to model
                rank=rank,                   # Passed to model
                gate_init_p=gate_init_p,     # Passed to model
                patience=20,
                l0_lambda=l0_lambda,
                ce_lambda=ce_lambda,
                tau_end=tau_end,
                n_epochs_kl_warmup=n_epochs_kl_warmup,
                n_epochs_l0_warmup=n_epochs_l0_warmup,
                combinatorial=True,
                use_synergy=True,
                debug=False,
            )
        model   = out["model"]
        trainer = out.get("trainer")
    except Exception as e:
        print(f"  [!] Training failed: {e}")
        raise optuna.TrialPruned()

    slk.tl.eval(model, train_data, obsm_key="z", uns_key="results", device=DEVICE)

    # --- CF Metrics ---
    ate_held_r   = float("nan")
    try:
        ate_out  = model.predict_counterfactual_effects(
            full_data, n_centroids=25, observed_effect=treat_effect, device=DEVICE
        )
        pred_df  = ate_out["predicted_df"]
        obs_df   = ate_out["observed_df"]

        def _r(idx):
            idx = [p for p in idx if p in pred_df.index and p in obs_df.index]
            if len(idx) < 2: return float("nan")
            x = pred_df.loc[idx].values.ravel()
            y = obs_df.loc[idx].values.ravel()
            v = np.isfinite(x) & np.isfinite(y)
            return pearsonr(x[v], y[v])[0] if v.sum() > 2 else float("nan")

        ate_held_r   = _r(list(holdout_combos))
    except Exception:
        pass

    # --- Fetch Gates & Score ---
    gate_w = None
    try:
        W = train_data.uns.get("results", {}).get("W", None)
        if W is not None: gate_w = W.flatten()
        elif hasattr(model, "gate_W"): gate_w = model.gate_W
        else: gate_w = model.gate.w.detach().cpu().numpy()
    except Exception:
        pass

    m     = extract_final_metrics(model, trainer, train_data, DEVICE, treat_effect)
    score = sherlock_score_v3(m, gate_w=gate_w)

    # --- Compute Polarization & Plot Histogram ---
    polarization = float("nan")
    if gate_w is not None:
        w = np.clip(np.array(gate_w).flatten(), 0.0, 1.0)
        polarization = float(np.mean((2 * w - 1) ** 2))
        
        plt.figure(figsize=(6, 4))
        plt.hist(w, bins=30, color='royalblue', edgecolor='black')
        plt.title(f"Trial {trial.number:02d} | Polar: {polarization:.3f} | Act: {m.get('frac_active',1):.2f}\n"
                  f"lat={latent_dim}, rank={rank}, init_p={gate_init_p:.2f}, l0={l0_lambda:.1f}")
        plt.xlabel("Gate Weight (W)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        filename = f"{RESULTS_DIR}/histograms/trial_{trial.number:02d}_act-{m.get('frac_active',1):.2f}.png"
        plt.savefig(filename, dpi=150)
        plt.close()

    print_checklist(trial_name, m)
    elapsed = time.time() - t0
    print(f"  Polarization: {polarization:.4f}  |  Score: {score:.4f}  |  CF Held-out R: {ate_held_r:.4f}")

    # Store User Attrs
    for key in ("ATE", "z_rho_spearman", "frac_dead", "frac_active", "pi99"):
        trial.set_user_attr(key, m.get(key, float("nan")))
    trial.set_user_attr("polarization", polarization)
    trial.set_user_attr("ate_held_r", ate_held_r)

    return score


# ---------------------------------------------------------------------------
# EARLY STOPPING
# ---------------------------------------------------------------------------
def early_stopping_callback(study, trial):
    if trial.value is None or trial.state != optuna.trial.TrialState.COMPLETE:
        return

    score  = trial.value
    ate    = trial.user_attrs.get("ATE", 0)
    rho    = trial.user_attrs.get("z_rho_spearman", 0)
    active = trial.user_attrs.get("frac_active", 1.0)
    dead   = trial.user_attrs.get("frac_dead", 1.0)
    polar  = trial.user_attrs.get("polarization", 0)

    if (score >= TARGET_SCORE and ate >= TARGET_ATE and rho >= TARGET_RHO 
        and active <= TARGET_MAX_ACTIVE and dead < MAX_FRAC_DEAD and polar >= 0.90):
        print(f"\n{'*'*60}\n  GOLDEN CONFIG — Trial {trial.number}\n{'*'*60}\n")
        study.stop()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Loading dataset ...")
    data = sc.read_h5ad("./Norman_2019.h5ad")
    data.X = data.layers["counts"].copy()

    slk.pp.slk_prepare_data(data, pert_key="perturbation_name", ntc_label="control", inplace=True)
    p_key = slk.configs.get_config("pert_key")
    NTC   = slk.configs.get_config("ntc_label")

    all_perts    = data.obs[p_key].unique()
    single_perts = [p for p in all_perts if "+" not in str(p) and p != NTC]
    combo_perts  = [p for p in all_perts if "+" in str(p)]

    pert_genes = {g.strip() for p in all_perts if p != NTC for g in str(p).split("+")}

    sc.pp.highly_variable_genes(data, n_top_genes=2000, flavor="seurat_v3", subset=False)
    keep_mask = data.var["highly_variable"] | data.var_names.isin(pert_genes)
    data = data[:, keep_mask].copy()

    slk.pp.treat_effect(data, label_col="pert", control_label="NTC", method="perturbseq", inplace=True, key="treat_effect")
    treat_effect = data.uns["treat_effect"]

    rng = np.random.default_rng(HOLDOUT_SEED)
    combo_arr = np.array(sorted(combo_perts))
    holdout_idx = rng.choice(len(combo_arr), size=max(1, int(np.round(len(combo_arr) * HOLDOUT_FRAC))), replace=False)
    holdout_combos = set(combo_arr[holdout_idx])
    
    train_mask = ~data.obs[p_key].isin(holdout_combos)
    train_data = data[train_mask].copy()

    study = optuna.create_study(
        study_name="Gated_VAE_Norman_R3_Arch",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True)
    )
    
    # Trial 08
    study.enqueue_trial({
        "latent_dim": 54, "rank": 21,        
        "l0_lambda": 24.53, "ce_lambda": 21.57,        
        "tau_end": 0.182, "gate_init_p": 0.61,
        "n_epochs_kl_warmup": 166, "l0_warmup_offset": 143,
    })

    # Trial 14
    study.enqueue_trial({
        "latent_dim": 54, "rank": 16,        
        "l0_lambda": 25.82, "ce_lambda": 20.12,        
        "tau_end": 0.184, "gate_init_p": 0.51,
        "n_epochs_kl_warmup": 184, "l0_warmup_offset": 124,
    })

    print(f"\nStarting sweep without pruner: up to {MAX_TRIALS} trials")
    study.optimize(
        lambda trial: objective(trial, train_data, data, holdout_combos, single_perts, treat_effect),
        n_trials=MAX_TRIALS,
        callbacks=[early_stopping_callback],
        gc_after_trial=True,
    )

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    df = df.sort_values(by="value", ascending=False)
    df.to_csv(f"{RESULTS_DIR}/optuna_leaderboard.csv", index=False)
    
    print("\nTop 3 trials:")
    display_cols = [c for c in df.columns if "user_attrs" in c or "params" in c or c == "value"]
    print(df[["number", "value", "params_latent_dim", "params_rank", "params_gate_init_p", "user_attrs_frac_active", "user_attrs_z_rho_spearman"]].head(3).to_string(index=False))

if __name__ == "__main__":
    main()