"""
Benchmark VAE, sVAE and cVAE on replogle.h5ad.

Metrics
-------
Average Treatment Effect — counterfactual (do-calculus from NTC cells):
  cf_ate_pearson_r   – Pearson r: predict E[x|do(p)] from NTC, vs ground truth
  cf_ate_spearman_r  – Spearman r (same)
  cf_ate_r2          – R² (same)
  (VAE:  encode NTC → z0, apply PoE with masking: z_cf = z0*(1-W[p]) + shift/(W_scale²+1))
  (cVAE: encode NTC with perturbed label → z_cf, decode with NTC embedding)
  (sVAE: encode NTC → z0, apply action_prior_mean[p] * binarized_mask[p], decode)

Latent geometry / clustering (rho_corr on pathway-gene subset):
  cluster_ari        – Adjusted Rand Index vs pathway labels
  cluster_ami        – Adjusted Mutual Information vs pathway labels
  cluster_mean_purity – mean per-cluster purity (fraction majority-pathway)
  cluster_accuracy   – Hungarian-aligned assignment accuracy
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pyro
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    r2_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sherlock as slk
from sherlock.tools._datasets import PerturbSimpleDataset
from sherlock.tools._models import VAE
from sherlock.tools._cvae import cVAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_CLUSTER = 1.5  # hierarchical clustering cut threshold


# ── helpers ───────────────────────────────────────────────────────────────────

def _lognorm(x: np.ndarray) -> np.ndarray:
    """log2(1e4 * x / lib + 1) — matches perturbseq ATE normalization."""
    lib = x.sum(axis=1, keepdims=True)
    return np.log2(1e4 * x / np.clip(lib, 1e-8, None) + 1.0)


# ── metric functions ──────────────────────────────────────────────────────────

def compute_cf_ate_metrics(model, adata, treat_effect_adata) -> dict:
    """
    Counterfactual ATE: predict E[x | do(p)] by intervening on NTC cells.
    VAE  : encode NTC → z0, apply learned A[p]*W[p] shift, decode.
           Shift is cell-independent so using mean(z0) is exact.
    cVAE : feed NTC cells through encoder+decoder with each perturbation label.
    sVAE : encode NTC → z0, apply action_prior_mean[p] * binarized_mask[p], decode.
    """
    _nan = {"cf_ate_pearson_r": np.nan, "cf_ate_spearman_r": np.nan, "cf_ate_r2": np.nan}

    p_key     = slk.configs.get_config("pert_key")
    ntc_label = slk.configs.get_config("ntc_label")
    device    = next(model.parameters()).device

    x_raw    = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    ntc_mask = adata.obs[p_key].values == ntc_label
    x_ntc_np = x_raw[ntc_mask].astype(np.float32)
    x_ntc_t  = torch.from_numpy(x_ntc_np).to(device)

    dataset    = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label)
    ntc_idx    = dataset.ntc_idx
    idx2pert   = dataset.idx_to_pert()
    gene_idx   = {g: i for i, g in enumerate(adata.var_names.tolist())}

    pred_effects: dict[str, np.ndarray] = {}

    if isinstance(model, VAE):
        # ── Encode all NTC cells → z0, take mean (shift is cell-independent) ──
        z0_mu = model._abduct_z0(x_ntc_t)                      # (N_ntc, d)
        z0_mean = z0_mu.mean(dim=0, keepdim=True)              # (1, d)

        # ── Build A (P, d) and W (P, d) ──────────────────────────────────────
        L      = model.qr().detach()
        sigma  = pyro.param("sigma_fac").detach().to(device)
        row_cov = L @ L.T + sigma ** 2 * torch.eye(model.perturbs, device=device)
        chol_P  = torch.linalg.cholesky(row_cov)
        q_rho   = model.rho_enc(model.p_emb.weight).detach()   # (P, d)
        A = chol_P @ q_rho                                      # (P, d)
        W = model.gate(deterministic=True).detach()             # (P, d)

        # ── Remap PerturbSimpleDataset indices → PerturbMatchingDataset indices ──
        # PerturbSimpleDataset: NTC=0, non-NTC=1..P
        # model.p_emb / A / W: indexed 0..P-1 (PerturbMatchingDataset, no NTC)
        vae_perts      = list(adata.uns["results"]["perts"])   # PerturbMatchingDataset order
        name_to_vae    = {name: i for i, name in enumerate(vae_perts)}
        non_ntc_names  = [n for _, n in sorted(idx2pert.items()) if n != ntc_label]
        vae_idx        = torch.tensor(
            [name_to_vae[n] for n in non_ntc_names], dtype=torch.long, device=device
        )

        # ── Batch decode: one z_cf per perturbation ───────────────────────────
        shifts   = A[vae_idx] * W[vae_idx]                     # (P_non, d)
        z_cf_all = z0_mean + shifts                            # (P_non, d)

        lib_med    = float(x_ntc_t.sum(-1).median().item())
        logits_all = model.z_decoder(z_cf_all)                 # (P_non, G)
        mu_cf_all  = lib_med * torch.softmax(logits_all, dim=-1).detach().cpu().numpy()

        # NTC baseline: decode z0_mean with no shift
        logits_ntc = model.z_decoder(z0_mean)
        mu_ntc     = lib_med * torch.softmax(logits_ntc, dim=-1).detach().cpu().numpy()
        ntc_ln     = _lognorm(mu_ntc).squeeze(0)

        for i, name in enumerate(non_ntc_names):
            pred_effects[name] = _lognorm(mu_cf_all[[i]]).squeeze(0) - ntc_ln

    elif isinstance(model, cVAE):
        # Encode NTC with each label to get z-shift, but always decode with the
        # NTC embedding so the decoder shortcut (pert_emb in decoder) cancels out.
        ntc_p   = torch.full((x_ntc_t.shape[0],), ntc_idx, dtype=torch.long, device=device)
        e_ntc   = model.pert_emb(ntc_p)
        library = x_ntc_t.sum(-1, keepdim=True)

        z_ntc  = model.get_z(x_ntc_t, ntc_p)
        mu_ntc = model._decode_mu(z_ntc, e_ntc, library).detach().cpu().numpy()
        ntc_ln = _lognorm(mu_ntc).mean(axis=0)

        for pidx, pname in idx2pert.items():
            if pname == ntc_label:
                continue
            p_t   = torch.full((x_ntc_t.shape[0],), pidx, dtype=torch.long, device=device)
            z_cf  = model.get_z(x_ntc_t, p_t)
            mu_cf = model._decode_mu(z_cf, e_ntc, library).detach().cpu().numpy()
            pred_effects[pname] = _lognorm(mu_cf).mean(axis=0) - ntc_ln

    else:
        # sVAE: encode NTC → z0, apply learned action_prior_mean[p] * binarized mask
        from sherlock.tools._svae import sVAE
        if not isinstance(model, sVAE):
            return _nan

        with torch.no_grad():
            z0 = model.get_z(x_ntc_t)                              # (N_ntc, d)
            z0_mean = z0.mean(dim=0, keepdim=True)                  # (1, d)

            shifts = model.action_prior_mean.detach()               # (P, d)
            mask   = (model.gumbel_action.get_proba() > 0.5).float().detach()    # (P, d)

            lib_med    = float(x_ntc_t.sum(-1).median().item())
            non_ntc_idx = [i for i, n in idx2pert.items() if n != ntc_label]
            pert_shifts = shifts[non_ntc_idx] * mask[non_ntc_idx]   # (P_non, d)
            z_cf_all   = z0_mean + pert_shifts                       # (P_non, d)

            logits_all = model.decoder(z_cf_all)
            mu_cf_all  = lib_med * torch.softmax(logits_all, dim=-1).cpu().numpy()

            logits_ntc = model.decoder(z0_mean)
            mu_ntc     = lib_med * torch.softmax(logits_ntc, dim=-1).cpu().numpy()
            ntc_ln     = _lognorm(mu_ntc).squeeze(0)

        for i, pidx in enumerate(non_ntc_idx):
            pred_effects[idx2pert[pidx]] = _lognorm(mu_cf_all[[i]]).squeeze(0) - ntc_ln

    # ── Align with ground-truth ATE and compute correlations ─────────────────
    common_perts = [p for p in pred_effects if p in treat_effect_adata.obs_names]
    common_genes = [g for g in adata.var_names if g in treat_effect_adata.var_names]
    if len(common_perts) < 2 or not common_genes:
        return _nan

    te_mat = treat_effect_adata[common_perts, common_genes].X
    if hasattr(te_mat, "toarray"):
        te_mat = te_mat.toarray()
    te_flat = np.asarray(te_mat).ravel()

    pred_mat = np.stack(
        [pred_effects[p][[gene_idx[g] for g in common_genes]] for p in common_perts],
        axis=0,
    )
    pred_flat = pred_mat.ravel()

    return {
        "cf_ate_pearson_r":  float(pearsonr(pred_flat, te_flat)[0]),
        "cf_ate_spearman_r": float(spearmanr(pred_flat, te_flat)[0]),
        "cf_ate_r2":         float(r2_score(te_flat, pred_flat)),
    }


def _align_and_summarize(
    clusters_df: pd.DataFrame,
    pathway_df: pd.DataFrame,
    gene_col: str = "gene",
    cluster_col: str = "cluster",
    pathway_col: str = "pathway",
) -> dict:
    """ARI, AMI, mean purity, Hungarian accuracy from cluster vs pathway labels."""
    df = pd.merge(
        clusters_df[[gene_col, cluster_col]],
        pathway_df[[gene_col, pathway_col]],
        on=gene_col,
        how="inner",
        validate="one_to_one",
    ).dropna(subset=[cluster_col, pathway_col]).copy()

    y_true = df[pathway_col].astype(str).values
    y_pred = df[cluster_col].astype(str).values
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    # Hungarian alignment
    C_mat = (
        pd.crosstab(df[pathway_col].astype(str), df[cluster_col].astype(str))
        .reindex(index=true_labels, columns=pred_labels, fill_value=0)
    )
    M = C_mat.values
    row_ind, col_ind = linear_sum_assignment(M.max() - M)
    mapping = {pred_labels[j]: true_labels[i] for i, j in zip(row_ind, col_ind)}

    mapped = np.array([mapping.get(cl) for cl in y_pred], dtype=object)
    accuracy = float((mapped == y_true).mean())

    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)

    purity_vals = [
        float((g[pathway_col] == g[pathway_col].mode().iat[0]).mean())
        for _, g in df.groupby(cluster_col)
    ]

    return {
        "cluster_ari":          float(ari),
        "cluster_ami":          float(ami),
        "cluster_mean_purity":  float(np.mean(purity_vals)),
        "cluster_accuracy":     accuracy,
    }


def compute_clustering_metrics(adata, pathway_df_indexed, all_genes_filtered) -> dict:
    """
    Cluster the perturbation similarity matrix on pathway genes and score vs annotations.
    """
    nan_dict = {
        k: np.nan for k in [
            "cluster_ari", "cluster_ami", "cluster_mean_purity", "cluster_accuracy"
        ]
    }

    results  = adata.uns["results"]
    corr_key = "z_corr"
    C        = np.asarray(results[corr_key], dtype=float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        return nan_dict

    # rho_perts aligns with rho_corr (non-NTC only); perts aligns with z_corr (all)
    perts_key = "rho_perts" if (corr_key == "rho_corr" and "rho_perts" in results) else "perts"
    perts      = results[perts_key]                        # (P,) array of strings
    pert_to_idx = {name: i for i, name in enumerate(perts)}

    # keep only pathway genes present in the matrix
    valid_genes = [g for g in all_genes_filtered if g in pert_to_idx]
    if len(valid_genes) < 4:
        return nan_dict

    gene_ids = [pert_to_idx[g] for g in valid_genes]
    cov_subset = C[np.ix_(gene_ids, gene_ids)]
    cov_subset = np.clip(cov_subset, -1.0, 1.0)

    # Drop rows/cols that contain any non-finite values (NaN/Inf).
    finite_mask = np.isfinite(cov_subset).all(axis=1)
    if int(finite_mask.sum()) < 4:
        return nan_dict

    valid_genes = [g for g, ok in zip(valid_genes, finite_mask) if ok]
    gene_ids = [pert_to_idx[g] for g in valid_genes]
    cov_subset = C[np.ix_(gene_ids, gene_ids)]
    cov_subset = np.clip(cov_subset, -1.0, 1.0)
    cov_subset = 0.5 * (cov_subset + cov_subset.T)
    np.fill_diagonal(cov_subset, 1.0)

    if not np.isfinite(cov_subset).all():
        return nan_dict

    try:
        groups = slk.tl.cluster_rho(adata, t=T_CLUSTER, rho_corr=cov_subset, show=False)
    except ValueError:
        return nan_dict

    clusters_df = pd.DataFrame({"gene": valid_genes, "cluster": groups})
    pw_df = (
        pathway_df_indexed
        .loc[pathway_df_indexed.index.isin(valid_genes)]
        .reset_index()
        .rename(columns={"index": "gene"})
    )
    # ensure 'gene' column exists after reset_index
    if "gene" not in pw_df.columns:
        pw_df = pw_df.rename(columns={pathway_df_indexed.index.name or "index": "gene"})

    return _align_and_summarize(clusters_df, pw_df,
                                gene_col="gene", cluster_col="cluster", pathway_col="pathway")


# ── per-model benchmark ───────────────────────────────────────────────────────

def benchmark_model(
    model_name: str,
    adata,
    treat_effect_adata,
    pathway_df_indexed,
    all_genes_filtered,
    display_name: str | None = None,
) -> dict:
    label = display_name or model_name
    print(f"\n{'='*60}\nBenchmarking: {label.upper()}\n{'='*60}")

    pyro.clear_param_store()

    out = slk.tl.run(
        adata,
        model=model_name,
        batch_size=4096,
        num_epochs=200,
        device=DEVICE,
        latent_dim=16,
        patience=10,
    )
    model_obj = out["model"]

    slk.tl.eval(model_obj, adata, obsm_key="z", uns_key="results", device=DEVICE)

    metrics: dict = {}
    metrics.update(compute_cf_ate_metrics(model_obj, adata, treat_effect_adata))
    metrics.update(compute_clustering_metrics(adata, pathway_df_indexed, all_genes_filtered))

    for k, v in metrics.items():
        print(f"  {k:30s} = {v:.4f}" if not np.isnan(v) else f"  {k:30s} = NaN")

    return metrics


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading replogle dataset …")
    import scanpy as sc
    data = sc.read_h5ad(os.path.join(os.path.dirname(os.path.abspath(__file__)), "replogle.h5ad"))
    slk.pp.slk_prepare_data(data, pert_key="gene", ntc_label="non-targeting",
                            treatment_key=None, inplace=True)

    print("Computing ground-truth ATE …")
    slk.pp.treat_effect(
        data, label_col="pert", control_label="NTC",
        method="perturbseq", inplace=True, key="treat_effect",
    )
    treat_effect_adata = data.uns["treat_effect"]

    # pathway annotations (exclude LSM5 which may be absent as a perturbation)
    pathways = data.uns["pathways"]
    all_genes_filtered = [
        gene
        for genes in pathways.values()
        for gene in genes
        if gene != "LSM5"
    ]
    gene_to_pathway = {
        gene: pathway
        for pathway, genes in pathways.items()
        for gene in genes
        if gene != "LSM5"
    }
    pathway_df_indexed = pd.DataFrame.from_dict(
        gene_to_pathway, orient="index", columns=["pathway"]
    )
    pathway_df_indexed.index.name = "gene"

    all_metrics: dict[str, dict] = {}
    for model_name, display_name in [("svae", "svae"), ("cvae", "cvae")]:
        all_metrics[display_name] = benchmark_model(
            model_name, data, treat_effect_adata,
            pathway_df_indexed, all_genes_filtered,
            display_name=display_name,
        )

    results_df = pd.DataFrame(all_metrics).T
    results_df.index.name = "model"

    print(f"\n{'='*60}\nFINAL BENCHMARK RESULTS\n{'='*60}")
    print(results_df.to_string(float_format="{:.4f}".format))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
    results_df.to_csv(out_path)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
