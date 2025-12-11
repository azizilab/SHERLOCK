from __future__ import annotations

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib import gridspec
import torch
import pandas as pd

from .._configs import get_config


def plot_r2(adata):
    
    p_key = get_config('pert_key')
    NTC = get_config('ntc_label')

    X_p = adata[adata.obs[p_key] != NTC].X
    
    prediction = adata[adata.obs[p_key] != NTC].layers['x_pred']

    r2 = r2_score(X_p.flatten(), prediction.flatten())

    idx = np.arange(0, X_p.shape[0])
    np.random.shuffle(idx)
    idx = idx[:1000]

    true_vals = X_p[idx].flatten()
    pred_vals = prediction[idx].flatten()

    # Scatterplot
    plt.figure(figsize=(8, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.5)

    # y = x line
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

    # Equal axis scaling
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"R² = {r2:.2f}")
    plt.grid(True)
    plt.legend()
    plt.show()
               

def plot_corr(adata, uns_key='results'):
    corr = adata.uns[uns_key]['rho_corr']
    sns.clustermap(corr, cmap="coolwarm", annot=False)

def plot_zcorr(adata, uns_key='results'):
    
    Sigma_P_posterior = adata.uns[uns_key]['rho_corr']
    Sigma_z = adata.uns[uns_key]['z_corr']

    link = linkage(Sigma_P_posterior, method="average", metric="euclidean")
    order = leaves_list(link)              # permutation indices, shape (P,)

    # ── 2.  apply the same permutation to both matrices ─────────────────
    SigmaP_ord = Sigma_P_posterior[order][:, order]

    # Sigma_z : correlation between perturbation-mean z’s  (P × P)
    Sigmaz_ord = Sigma_z[order][:, order]


    # ── 3.  plot side-by-side heat-maps ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    sns.heatmap(SigmaP_ord, cmap="coolwarm", square=True, ax=axes[0])
    axes[0].set_title(r"$\Sigma_P$  (posterior row covariance)")
    axes[0].set_xlabel("perturbation"); axes[0].set_ylabel("perturbation")

    sns.heatmap(Sigmaz_ord, cmap="coolwarm", square=True, ax=axes[1])
    axes[1].set_title("Correlation of mean $z$")
    axes[1].set_xlabel("perturbation")

    # plt.title(f"Spearman ρ = {r_s:.3f}  Pearson r = {r_p:.3f}")
    plt.tight_layout()
    plt.show()

def plot_ev(adata, uns_key='results', cond_idx=0, cmin=None, cmax=None, top_n=None):
    """
    Plot explained variance (perturbations × genes) for a single condition.

    Parameters
    ----------
    adata : AnnData
        AnnData with `adata.uns[uns_key]['explained_variance']`.
    uns_key : str, default 'results'
        Key in `adata.uns` where results are stored.
    cond_idx : int, default 0
        Index of condition along the first axis of `explained_variance`.
    cmin : float or None, default None
        Minimum value for colormap (vmin).
    cmax : float or None, default None
        Maximum value for colormap (vmax).
    top_n : int or None, default None
        If set, plot only the top_n genes by max explained variance
        (within this condition). If None, plot all genes.
    """
    uns = adata.uns[uns_key]

    # (C, P, G) → (P, G) for chosen condition
    ev_pg = uns["explained_variance"][cond_idx]  # shape (P, G)

    pert_names = np.array(uns["perts"])
    gene_names = np.array(adata.var_names)
    P, G = ev_pg.shape

    # --- select top genes if requested ---
    if top_n is not None and top_n > 0 and top_n < G:
        # rank genes by max EV across perts (for this condition)
        gene_scores = ev_pg.max(axis=0)                  # (G,)
        top_idx = np.argsort(gene_scores)[::-1][:top_n]  # indices of top genes
        ev_pg = ev_pg[:, top_idx]
        gene_names = gene_names[top_idx]

    # Put into a DataFrame for nicer labeling
    df = pd.DataFrame(ev_pg, index=pert_names, columns=gene_names)

    # Show x labels only if reasonably small
    show_xticks = (df.shape[1] <= 40)

    g = sns.clustermap(
        df,
        cmap="viridis",
        vmin=cmin,
        vmax=cmax,
        metric="euclidean",
        method="average",
        figsize=(8, 6),
        xticklabels=show_xticks,
        yticklabels=True,
        row_cluster=True,
        col_cluster=False,  # cluster perts only; keep gene order (by EV if top_n used)
    )

    g.ax_heatmap.set_xlabel("Genes")
    g.ax_heatmap.set_ylabel("Perturbations")
    g.ax_heatmap.set_title(f"Explained variance – condition {cond_idx}")

    plt.show()


def _ev_results_from_uns(adata, cond_idx=0, uns_key="results", store_key="ev_results"):
    """Internal helper to fetch EV, pvals, qvals, perts, genes for a condition."""
    uns = adata.uns[uns_key]

    if "explained_variance" not in uns:
        raise KeyError(f"adata.uns['{uns_key}']['explained_variance'] not found.")

    if store_key not in uns or cond_idx not in uns[store_key]:
        raise KeyError(
            f"EV significance results not found. "
            f"Run ev_sig(...) first for cond_idx={cond_idx}."
        )

    ev = np.asarray(uns["explained_variance"][cond_idx], dtype=float)  # (P, G)
    pvals = np.asarray(uns[store_key][cond_idx]["pvals"], dtype=float)
    qvals = np.asarray(uns[store_key][cond_idx]["qvals"], dtype=float)

    # Perturbation names and gene names
    perts = np.asarray(uns.get("perts", np.arange(ev.shape[0])))
    genes = np.asarray(adata.var_names)

    return ev, pvals, qvals, perts, genes


def make_ev_long_df(
    adata,
    cond_idx: int = 0,
    uns_key: str = "results",
    store_key: str = "ev_results",
) -> pd.DataFrame:
    """
    Construct a long-form DataFrame with (pert, gene, EV, pval, qval)
    for a given condition.

    Requires that ev_rowwise_significance(...) has already been run.
    """
    ev, pvals, qvals, perts, genes = _ev_results_from_uns(
        adata, cond_idx=cond_idx, uns_key=uns_key, store_key=store_key
    )

    rows = []
    for p_idx, pert in enumerate(perts):
        for g_idx, gene in enumerate(genes):
            rows.append(
                (
                    str(pert),
                    str(gene),
                    float(ev[p_idx, g_idx]),
                    float(pvals[p_idx, g_idx]),
                    float(qvals[p_idx, g_idx]),
                )
            )

    df = pd.DataFrame(rows, columns=["pert", "gene", "ev", "pval", "qval"])
    return df


def plot_ev_sig(
    adata,
    cond_idx: int = 0,
    uns_key: str = "results",
    store_key: str = "ev_results",
    alpha: float = 0.05,
    top_n_genes: int = 10,
    min_sig_genes: int = 3,
    cluster_rows: bool = True,
    cluster_cols: bool = False,
    use_clustermap: bool = True,
    cmap: str = "viridis",
):
    """
    Plot a heatmap of -log10(q-value) for top significant perturbation→gene
    pairs for a given condition.

    Steps:
    1. Read EV + pvals/qvals from adata.uns[uns_key][store_key][cond_idx].
    2. Build long DataFrame (pert, gene, EV, pval, qval).
    3. Filter to qval < alpha.
    4. For each perturbation, rank genes by EV and keep top_n_genes.
    5. Retain only perts that have at least min_sig_genes genes passing.
    6. Pivot to pert × gene matrix of q-values.
    7. Plot -log10(q) as either:
        - seaborn.clustermap with optional row/col clustering, or
        - plain imshow heatmap with rows sorted by max signal.

    Parameters
    ----------
    adata : AnnData
    cond_idx : int, default 0
    uns_key : str, default "results"
    store_key : str, default "ev_results"
    alpha : float, default 0.05
        q-value threshold for significance.
    top_n_genes : int, default 10
        Maximum number of top genes per perturbation (by EV).
    min_sig_genes : int, default 3
        Only keep perturbations with at least this many significant genes.
    cluster_rows : bool, default True
        If True and use_clustermap=True, cluster perturbations.
    cluster_cols : bool, default False
        If True and use_clustermap=True, cluster genes.
    use_clustermap : bool, default True
        If True, use seaborn.clustermap; otherwise use matplotlib.imshow.
    cmap : str, default "viridis"
        Colormap for the heatmap.
    """
    df = make_ev_long_df(
        adata, cond_idx=cond_idx, uns_key=uns_key, store_key=store_key
    )

    # --- filter by q-value ---
    df_sig = df[df["qval"] < alpha].copy()
    if df_sig.empty:
        print(f"No significant (pert, gene) pairs at q < {alpha} for cond_idx={cond_idx}.")
        return

    # Rank genes within each perturbation by EV (largest = rank 1)
    df_sig["rank_in_pert"] = df_sig.groupby("pert")["ev"].rank(
        method="first", ascending=False
    )
    df_top = df_sig[df_sig["rank_in_pert"] <= top_n_genes].copy()

    # Keep only perturbations with >= min_sig_genes unique genes
    good_perts = (
        df_top.groupby("pert")["gene"]
        .nunique()
        .loc[lambda s: s >= min_sig_genes]
        .index
    )
    df_top = df_top[df_top["pert"].isin(good_perts)]

    if df_top.empty:
        print(
            f"No perturbations with at least {min_sig_genes} significant genes "
            f"after filtering at q < {alpha} for cond_idx={cond_idx}."
        )
        return

    # --- build matrix of q-values for plotting ---
    qmat = df_top.pivot(index="pert", columns="gene", values="qval")
    qmat = qmat.fillna(1.0)                   # missing = non-significant
    qmat = qmat.clip(lower=1e-16, upper=1.0)  # avoid log10(0)
    heat = -np.log10(qmat)

    # sanity check
    if not np.isfinite(heat.values).all():
        raise ValueError("Non-finite values in heat matrix after processing.")

    # --- plotting ---
    if use_clustermap:
        # seaborn.clustermap handles clustering for us
        g = sns.clustermap(
            heat,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            cmap=cmap,
            figsize=(0.3 * heat.shape[1] + 4, 0.3 * heat.shape[0] + 4),
            cbar_kws={"label": "-log10(q-value)"},
        )

        ax = g.ax_heatmap
        ax.set_xlabel("Gene")
        ax.set_ylabel("Perturbation")
        ax.set_title("Top significant perturbation→gene effects", pad=20)

        # Rotate gene labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.show()

    else:
        # plain matplotlib imshow with manual row sorting by max signal
        heat_sorted = heat.loc[heat.max(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(
            figsize=(0.3 * heat_sorted.shape[1] + 4, 0.3 * heat_sorted.shape[0] + 4)
        )

        im = ax.imshow(heat_sorted.values, aspect="auto", cmap=cmap)

        ax.set_xticks(np.arange(heat_sorted.shape[1]))
        ax.set_xticklabels(heat_sorted.columns, rotation=90)
        ax.set_yticks(np.arange(heat_sorted.shape[0]))
        ax.set_yticklabels(heat_sorted.index)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("-log10(q-value)")

        ax.set_xlabel("Gene")
        ax.set_ylabel("Perturbation")
        ax.set_title("Top significant perturbation→gene effects")

        plt.tight_layout()
        plt.show()
