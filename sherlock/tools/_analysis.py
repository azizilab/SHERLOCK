import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    homogeneity_completeness_v_measure,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
    silhouette_samples
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def ev_sig(
    adata,
    cond_idx: int = 0,
    uns_key: str = "results",
    fdr_alpha: float = 0.05,
    store_key: str = "ev_results",
):
    """
    Compute row-wise significance of explained variance for a single condition.

    For each perturbation (row p) and gene g, test whether the EV[p, g] value
    is unusually large relative to the background EV distribution in row p,
    treating genes as exchangeable within that row.

    Null per (p,g): gene g is not a special target of perturbation p,
    so EV[p, g] comes from the background EV distribution of row p.

    We approximate the row-wise null as Normal:
        z_{p,g} = (EV_{p,g} - center_p) / scale_p
    using robust center/scale (median + MAD), then
        p_{p,g} = P(Z >= z_{p,g}) under N(0,1).

    FDR control is applied *per perturbation* (row-wise BH), yielding q-values.

    Parameters
    ----------
    adata : AnnData
        AnnData object with adata.uns[uns_key]["counterfactual_effect_size"]
        of shape (C, P, G).
    cond_idx : int, default 0
        Index of condition along the first axis of explained_variance.
    uns_key : str, default "results"
        Key in adata.uns where EV and metadata are stored.
    fdr_alpha : float, default 0.05
        FDR threshold to record (does not affect q-values themselves).
    store_key : str, default "ev_results"
        Sub-key inside adata.uns[uns_key] where results are stored.

    Side effects
    ------------
    Stores into:
        adata.uns[uns_key][store_key][cond_idx] = {
            "pvals": (P, G) array,
            "qvals": (P, G) array,
            "alpha": float (fdr_alpha),
        }

    Returns
    -------
    res : dict
        Dictionary with:
            "ev"    : (P, G) explained variance matrix for this condition
            "pvals" : (P, G) raw p-values
            "qvals" : (P, G) BH-FDR q-values (per row)
            "alpha" : fdr_alpha
    """
    uns = adata.uns[uns_key]

    # Explained variance for this condition: shape (P, G)
    ev_all = np.asarray(uns["counterfactual_effect_size"][cond_idx], dtype=float)
    P, G = ev_all.shape

    z_all = np.zeros_like(ev_all)
    pvals = np.ones_like(ev_all)
    qvals = np.ones_like(ev_all)

    for p in range(P):
        x = ev_all[p]  # (G,)

        # Robust center and scale: median + MAD
        med = np.median(x)
        mad = np.median(np.abs(x - med))

        if mad > 0:
            scale = 1.4826 * mad  # MAD -> sd for Normal
        else:
            # Fallback to standard deviation
            scale = x.std(ddof=1)
            if scale == 0 or not np.isfinite(scale):
                # Row has no variation; nothing is special
                z_all[p] = 0.0
                pvals[p] = 1.0
                qvals[p] = 1.0
                continue

        z = (x - med) / scale
        z_all[p] = z

        # One-sided p-value: large EV is more extreme
        row_pvals = 1.0 - norm.cdf(z)
        row_pvals = np.clip(row_pvals, 1e-300, 1.0)

        # BH-FDR per row (per perturbation)
        _, row_qvals, _, _ = multipletests(row_pvals, method="fdr_bh")

        pvals[p] = row_pvals
        qvals[p] = row_qvals

    # --- store in adata.uns[uns_key][store_key] ---
    if store_key not in uns:
        uns[store_key] = {}

    uns[store_key][cond_idx] = {
        "pvals": pvals,
        "qvals": qvals,
        "alpha": float(fdr_alpha),
    }

    # write back
    adata.uns[uns_key] = uns


# ── Benchmark metrics ─────────────────────────────────────────────────────────

def _abs_r_metrics(metrics: dict) -> dict:
    out = dict(metrics)
    for k, v in out.items():
        if k.endswith("_r") and np.isfinite(v):
            out[k] = float(abs(v))
    return out


def compute_cf_ate_metrics(
    model,
    adata,
    treat_effect_adata,
) -> dict:
    """
    Counterfactual ATE: predict E[x | do(p)] by intervening on NTC cells.

    Delegates to model.predict_counterfactual_effects(adata, observed_effect).
    Returns a single Pearson r between predicted and observed ATEs.

    Returns
    -------
    dict with key: cf_ate_pearson_r
    """
    import torch

    _nan = {"cf_ate_pearson_r": np.nan}

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    result = model.predict_counterfactual_effects(
        adata, observed_effect=treat_effect_adata, device=device
    )

    if not result or not result.get("effects"):
        return _nan

    corr = result.get("corr", np.nan)
    pearson_r = float(abs(corr)) if np.isfinite(float(corr)) else np.nan
    return {"cf_ate_pearson_r": pearson_r}


def compute_embedding_metrics(
    embed_arr,
    labels,
    knn_k: int = 5,
    linear_probe_cv: int = 5,
) -> dict:
    """
    Compute silhouette (macro), kNN purity, and linear probe accuracy.

    Silhouette is macro-averaged: mean per-label first, then mean across labels,
    so large clusters do not dominate.

    Parameters
    ----------
    embed_arr       : (N, d) array of embeddings
    labels          : (N,) array of string or int class labels
    knn_k           : neighbours for kNN purity
    linear_probe_cv : StratifiedKFold splits for logistic regression probe

    Returns
    -------
    dict with keys: silhouette, knn_purity, linear_probe (float, NaN on failure)
    """
    sil = knn_pur = lin_probe = np.nan
    embed_arr = np.asarray(embed_arr)
    labels    = np.asarray(labels)
    uniq      = np.unique(labels)

    if len(uniq) < 2 or len(uniq) >= len(labels):
        return {"silhouette": sil, "knn_purity": knn_pur, "linear_probe": lin_probe}

    try:
        s   = silhouette_samples(embed_arr, labels, metric="euclidean")
        sil = float(np.mean([s[labels == lab].mean() for lab in uniq]))
    except Exception:
        pass

    try:
        k = min(knn_k, len(embed_arr) - 1)
        if k >= 1:
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embed_arr)
            _, indices = nbrs.kneighbors(embed_arr)
            knn_pur = float(np.mean([
                (labels[idx[1:]] == labels[i]).mean()
                for i, idx in enumerate(indices)
            ]))
    except Exception:
        pass

    try:
        _, counts = np.unique(labels, return_counts=True)
        n_splits  = min(linear_probe_cv, int(counts.min()))
        if n_splits >= 2:
            cv        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            lr        = LogisticRegression(max_iter=1000, random_state=42)
            X_s       = StandardScaler().fit_transform(embed_arr)
            lin_probe = float(cross_val_score(lr, X_s, labels, cv=cv, scoring="accuracy").mean())
    except Exception:
        pass

    return {"silhouette": sil, "knn_purity": knn_pur, "linear_probe": lin_probe}


def compute_clustering_metrics(
    adata,
    pathway_df_indexed: pd.DataFrame,
    all_genes_filtered: list,
    n_clusters: int = 10,
    uns_key: str = "results",
    knn_k: int = 5,
    linear_probe_cv: int = 5,
) -> dict:
    """
    Cluster perturbations by latent similarity and score against pathway annotations.

    Hierarchically clusters on the pathway-gene subset of rho_corr, then
    evaluates the full suite of clustering metrics vs ground-truth pathway labels.
    Embedding-based metrics (silhouette, kNN purity, linear probe) use rho_embed
    stored by eval().

    Parameters
    ----------
    adata              : AnnData with adata.uns[uns_key] containing the results dict
    pathway_df_indexed : DataFrame with index=gene and column 'pathway'
    all_genes_filtered : genes that belong to any pathway (used for subsetting)
    n_clusters         : number of clusters to form (maxclust criterion)
    uns_key            : key in adata.uns to read results from
    knn_k              : number of nearest neighbours for knn_purity
    linear_probe_cv    : number of StratifiedKFold splits for linear probe

    Returns
    -------
    dict with keys:
        cluster_accuracy, cluster_precision, cluster_recall, cluster_f1,
        cluster_ari, cluster_ami,
        cluster_homogeneity, cluster_completeness, cluster_v_measure,
        cluster_silhouette, cluster_purity, cluster_knn_purity, cluster_linear_probe
    """
    from ._clustering import cluster_rho

    _nan_keys = [
        "cluster_accuracy", "cluster_precision", "cluster_recall", "cluster_f1",
        "cluster_ari", "cluster_ami",
        "cluster_homogeneity", "cluster_completeness", "cluster_v_measure",
        "cluster_silhouette", "cluster_purity", "cluster_knn_purity", "cluster_linear_probe",
    ]
    nan_dict = {k: np.nan for k in _nan_keys}

    results = adata.uns.get(uns_key, {})
    if "rho_corr" not in results or "rho_perts" not in results:
        print("Warning: Missing 'rho_corr' or 'rho_perts' in adata.uns[uns_key]. Cannot compute clustering metrics.")
        return nan_dict

    C          = np.asarray(results["rho_corr"], dtype=float)
    perts      = np.asarray(results["rho_perts"])
    rho_embed  = results.get("rho_embed")

    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 4:
        return nan_dict

    pert_to_idx = {name: i for i, name in enumerate(perts)}
    valid_genes = [g for g in all_genes_filtered if g in pert_to_idx]
    if len(valid_genes) < 4:
        print("Warning: Not enough valid genes found in both pathway_df_indexed and rho_corr. Cannot compute clustering metrics.")
        return nan_dict

    gene_ids   = [pert_to_idx[g] for g in valid_genes]
    cov_subset = C[np.ix_(gene_ids, gene_ids)]
    cov_subset = np.clip(cov_subset, -1.0, 1.0)

    finite_mask = np.isfinite(cov_subset).all(axis=1)
    if int(finite_mask.sum()) < 4:
        print("Warning: Not enough genes with finite covariance values after filtering. Cannot compute clustering metrics.")
        return nan_dict

    valid_genes = [g for g, ok in zip(valid_genes, finite_mask) if ok]
    gene_ids    = [pert_to_idx[g] for g in valid_genes]
    cov_subset  = C[np.ix_(gene_ids, gene_ids)]
    cov_subset  = np.clip(cov_subset, -1.0, 1.0)
    cov_subset  = 0.5 * (cov_subset + cov_subset.T)
    np.fill_diagonal(cov_subset, 1.0)

    if not np.isfinite(cov_subset).all():
        print("Warning: Non-finite values found in covariance subset after filtering. Cannot compute clustering metrics.")
        return nan_dict

    try:
        groups = cluster_rho(adata, t=n_clusters, rho_corr=cov_subset, criterion='maxclust', show=False)
    except Exception:
        print("Warning: Exception occurred during clustering. Cannot compute clustering metrics.")
        return nan_dict

    clusters_df = pd.DataFrame({"gene": valid_genes, "cluster": groups})
    pw_df = (
        pathway_df_indexed
        .loc[pathway_df_indexed.index.isin(valid_genes)]
        .reset_index()
    )
    if "gene" not in pw_df.columns:
        pw_df = pw_df.rename(columns={pathway_df_indexed.index.name or "index": "gene"})

    df = pd.merge(
        clusters_df[["gene", "cluster"]],
        pw_df[["gene", "pathway"]],
        on="gene", how="inner", validate="one_to_one",
    ).dropna(subset=["cluster", "pathway"]).copy()
    
    if len(df) < 4:
        print("Warning: Not enough genes with both cluster and pathway annotations after merging. Cannot compute clustering metrics.")
        return nan_dict

    y_true      = df["pathway"].astype(str).values
    y_pred      = df["cluster"].astype(str).values
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    # Hungarian alignment for accuracy / precision / recall / F1
    C_mat = (
        pd.crosstab(df["pathway"].astype(str), df["cluster"].astype(str))
        .reindex(index=true_labels, columns=pred_labels, fill_value=0)
    )
    M = C_mat.values
    row_ind, col_ind = linear_sum_assignment(M.max() - M)
    mapping = {pred_labels[j]: true_labels[i] for i, j in zip(row_ind, col_ind)}
    for cl in pred_labels:
        if cl not in mapping:
            cl_mask = y_pred == cl
            if cl_mask.any():
                vals, counts = np.unique(y_true[cl_mask], return_counts=True)
                mapping[cl] = vals[np.argmax(counts)]

    mapped = np.array([mapping[cl] for cl in y_pred], dtype=object)
    accuracy = float((mapped == y_true).mean())

    try:
        y_mapped = mapped.astype(str)
        prec  = float(precision_score(y_true, y_mapped, average="macro", zero_division=0))
        rec   = float(recall_score(y_true, y_mapped, average="macro", zero_division=0))
        f1    = float(f1_score(y_true, y_mapped, average="macro", zero_division=0))
    except Exception:
        prec = rec = f1 = np.nan

    ari = float(adjusted_rand_score(y_true, y_pred))
    ami = float(adjusted_mutual_info_score(y_true, y_pred))

    try:
        hom, com, vmeas = homogeneity_completeness_v_measure(y_true, y_pred)
        hom = float(hom); com = float(com); vmeas = float(vmeas)
    except Exception:
        hom = com = vmeas = np.nan

    # Purity: for each predicted cluster, fraction of majority true label
    try:
        contingency = pd.crosstab(y_pred, y_true).values
        purity = float(contingency.max(axis=1).sum() / len(y_true))
    except Exception:
        purity = np.nan

    # Embedding-based metrics
    sil = knn_pur = lin_probe = np.nan

    if rho_embed is not None:
        try:
            embed_arr   = np.asarray(rho_embed)[[pert_to_idx[g] for g in df["gene"].values]]
            emb_metrics = compute_embedding_metrics(
                embed_arr, y_true, knn_k=knn_k, linear_probe_cv=linear_probe_cv
            )
            sil       = emb_metrics["silhouette"]
            knn_pur   = emb_metrics["knn_purity"]
            lin_probe = emb_metrics["linear_probe"]
        except Exception:
            pass

    return {
        "cluster_accuracy":    accuracy,
        "cluster_precision":   prec,
        "cluster_recall":      rec,
        "cluster_f1":          f1,
        "cluster_ari":         ari,
        "cluster_ami":         ami,
        "cluster_homogeneity": hom,
        "cluster_completeness": com,
        "cluster_v_measure":   vmeas,
        "cluster_silhouette":  sil,
        "cluster_purity":      purity,
        "cluster_knn_purity":  knn_pur,
        "cluster_linear_probe": lin_probe,
    }


def evaluate_model(
    model,
    adata,
    treat_effect_adata,
    pathway_df_indexed: pd.DataFrame,
    all_genes_filtered: list,
    n_clusters: int = 10,
    device=None,
    batch_size: int = 1024,
    uns_key: str = "results",
) -> dict:
    """
    Run inference on adata, then compute all benchmark metrics.

    Calls model.eval(adata, ...) to store z, x_pred, and correlation matrix,
    then computes counterfactual ATE and clustering metrics.

    Parameters
    ----------
    model              : trained model (VAE, cVAE, or sVAE)
    adata              : AnnData with raw counts
    treat_effect_adata : AnnData with ground-truth ATEs (obs=perts, var=genes)
    pathway_df_indexed : DataFrame with index=gene and column 'pathway'
    all_genes_filtered : pathway genes to use for clustering evaluation
    n_clusters         : number of clusters to form (maxclust criterion)
    device             : torch device (defaults to model's current device)
    batch_size         : inference batch size
    uns_key            : key in adata.uns to store and read results

    Returns
    -------
    dict of scalar metric values
    """
    import torch
    from ._single import eval as _eval_fn

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    _eval_fn(model, adata, obsm_key="z", uns_key=uns_key, device=device, batch_size=batch_size)

    metrics: dict = {}
    metrics.update(compute_cf_ate_metrics(model, adata, treat_effect_adata))
    metrics.update(compute_clustering_metrics(adata, pathway_df_indexed, all_genes_filtered, n_clusters=n_clusters, uns_key=uns_key))

    # Pearson r between upper-triangle of rho_corr and z_corr
    results_uns = adata.uns.get(uns_key, {})
    rho_mat = results_uns.get("rho_corr")
    z_mat   = results_uns.get("z_corr")
    rho_z_r = np.nan
    if rho_mat is not None and z_mat is not None:
        rho = np.asarray(rho_mat, dtype=float)
        z   = np.asarray(z_mat,   dtype=float)
        if rho.ndim == 2 and rho.shape == z.shape and rho.shape[0] >= 2:
            idx   = np.triu_indices(rho.shape[0], k=1)
            r_flat, z_flat = rho[idx], z[idx]
            valid = np.isfinite(r_flat) & np.isfinite(z_flat)
            if valid.sum() > 2:
                rho_z_r = float(pearsonr(r_flat[valid], z_flat[valid])[0])
    metrics["rho_z_pearson_r"] = rho_z_r

    return metrics
