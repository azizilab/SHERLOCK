from __future__ import annotations

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    Plot perturbation-target interaction (perturbations × genes) for a single condition.

    Parameters
    ----------
    adata : AnnData
        AnnData with `adata.uns[uns_key]['counterfactual_effect_size']`.
    uns_key : str, default 'results'
        Key in `adata.uns` where results are stored.
    cond_idx : int, default 0
        Index of condition along the first axis of `counterfactual_effect_size`.
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
    ev_pg = uns["counterfactual_effect_size"][cond_idx]  # shape (P, G)

    pert_names = np.array(uns.get("rho_perts", uns["perts"]))
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
    g.ax_heatmap.set_title(f"Condition-perturbation interaction – condition {cond_idx}")

    plt.show()


def _ev_results_from_uns(adata, cond_idx=0, uns_key="results", store_key="ev_results"):
    """Internal helper to fetch EV, pvals, qvals, perts, genes for a condition."""
    uns = adata.uns[uns_key]

    if "counterfactual_effect_size" not in uns:
        raise KeyError(f"adata.uns['{uns_key}']['counterfactual_effect_size'] not found.")

    if store_key not in uns or cond_idx not in uns[store_key]:
        raise KeyError(
            f"EV significance results not found. "
            f"Run ev_sig(...) first for cond_idx={cond_idx}."
        )

    ev = np.asarray(uns["counterfactual_effect_size"][cond_idx], dtype=float)  # (P, G)
    pvals = np.asarray(uns[store_key][cond_idx]["pvals"], dtype=float)
    qvals = np.asarray(uns[store_key][cond_idx]["qvals"], dtype=float)

    # Perturbation names and gene names
    perts = np.asarray(uns.get("rho_perts", uns.get("perts", np.arange(ev.shape[0]))))
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


# ---------------------------------------------------------------------------
# Bipartite graph helpers – counterfactual effect size
# ---------------------------------------------------------------------------

def _get_cf_data(adata, uns_key="results", cf_key="counterfactual_effect_size"):
    """Extract counterfactual effect size array (C, P, G), pert names, gene names."""
    uns = adata.uns[uns_key]
    if cf_key not in uns:
        raise KeyError(
            f"adata.uns['{uns_key}']['{cf_key}'] not found. "
            "Run compute_counterfactual_effect_size first."
        )
    cf = np.asarray(uns[cf_key], dtype=float)          # (C, P, G)
    perts = np.asarray(uns.get("perts", np.arange(cf.shape[1])), dtype=str)
    genes = np.asarray(adata.var_names)
    return cf, perts, genes


def _draw_bipartite_nx(
    pert_nodes,
    gene_nodes,
    edges,
    ax,
    node_size=800,
    pert_color="#4C72B0",
    gene_color="#DD8452",
    gene_overlap_color="#2CA02C",
    edge_color="#999999",
    label_fontsize=8,
):
    """
    Draw a bipartite graph on *ax* using networkx.

    Perturbation nodes sit on the top row; gene nodes on the bottom row.
    When there is a single perturbation it is centred above the gene nodes.

    Parameters
    ----------
    pert_nodes : list of str
    gene_nodes  : list of str  (pre-ordered)
    edges       : list of (pert_name, gene_name, weight)
    ax          : matplotlib Axes
    """
    import networkx as nx

    G = nx.Graph()

    # Prefix to avoid name collisions between perts and genes
    pert_ids = {p: f"__pert__{p}" for p in pert_nodes}
    gene_ids = {g: f"__gene__{g}" for g in gene_nodes}

    G.add_nodes_from(pert_ids.values(), bipartite=0)
    G.add_nodes_from(gene_ids.values(), bipartite=1)

    for p_name, g_name, w in edges:
        G.add_edge(pert_ids[p_name], gene_ids[g_name], weight=w)

    # --- layout ---
    n_perts = len(pert_nodes)
    n_genes = len(gene_nodes)

    def _xcoords(n, center=None):
        """Evenly spaced x coordinates in [0, 1].  If center is given, shift so
        that the single node sits at that x position."""
        if n == 1:
            return [0.5 if center is None else center]
        return [i / (n - 1) for i in range(n)]

    gene_xs = _xcoords(n_genes)
    # For a single perturbation: centre it over the gene row
    single_pert = n_perts == 1
    pert_xs = _xcoords(n_perts, center=0.5 if single_pert else None)

    pos = {}
    for x, p in zip(pert_xs, pert_nodes):
        pos[pert_ids[p]] = (x, 1.0)
    for x, g in zip(gene_xs, gene_nodes):
        pos[gene_ids[g]] = (x, 0.0)

    # --- edges (uniform style) ---
    edge_list = [(pert_ids[p], gene_ids[g]) for p, g, _ in edges]
    if edge_list:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edge_list,
            width=1.2,
            edge_color=edge_color,
            alpha=0.5,
        )

    # --- pert nodes + labels ---
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=list(pert_ids.values()),
        node_color=pert_color,
        node_size=node_size,
        edgecolors="white", linewidths=1.0,
    )
    for p in pert_nodes:
        x, y = pos[pert_ids[p]]
        ax.text(x, y + 0.07, p, ha="center", va="bottom",
                fontsize=label_fontsize, color=pert_color,
                rotation=45, rotation_mode="anchor")

    # --- gene nodes + labels ---
    # count incoming edges per gene to identify overlaps
    gene_degree = {g: sum(1 for p, gg, _ in edges if gg == g) for g in gene_nodes}

    unique_genes = [g for g in gene_nodes if gene_degree[g] == 1]
    overlap_genes = [g for g in gene_nodes if gene_degree[g] > 1]

    if unique_genes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=[gene_ids[g] for g in unique_genes],
            node_color=gene_color,
            node_size=node_size,
            edgecolors="white", linewidths=1.0,
        )
    if overlap_genes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=[gene_ids[g] for g in overlap_genes],
            node_color=gene_overlap_color,
            node_size=node_size,
            edgecolors="white", linewidths=1.0,
        )

    for g in gene_nodes:
        color = gene_overlap_color if gene_degree[g] > 1 else gene_color
        x, y = pos[gene_ids[g]]
        ax.text(x, y - 0.07, g, ha="center", va="top",
                fontsize=label_fontsize, color=color,
                rotation=45, rotation_mode="anchor")

    # --- legend ---
    legend_handles = [
        plt.scatter([], [], s=40, color=pert_color,
                    edgecolors="white", linewidths=0.5, label="Perturbation / drug"),
        plt.scatter([], [], s=40, color=gene_color,
                    edgecolors="white", linewidths=0.5, label="Gene target (unique)"),
        plt.scatter([], [], s=40, color=gene_overlap_color,
                    edgecolors="white", linewidths=0.5,
                    label="Gene target (shared across perturbations)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              fontsize=label_fontsize, scatterpoints=1)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.35, 1.35)
    ax.axis("off")


def plot_cf_bipartite(
    adata,
    cond_idx=0,
    pert_name=None,
    top_n=20,
    uns_key="results",
    cf_key="counterfactual_effect_size",
    min_effect=0.0,
    figsize=None,
    title=None,
):
    """
    Visualize counterfactual effect size as a bipartite graph.

    Perturbation nodes sit on the top row; gene nodes on the bottom row.
    When a single perturbation is selected it is centred above the gene nodes.
    All edges are drawn with uniform style (no width/colour encoding).

    Parameters
    ----------
    adata : AnnData
    cond_idx : int, default 0
        Index into the condition axis of ``counterfactual_effect_size``.
    pert_name : str or list of str or None, default None
        * ``str``       – single perturbation (centred when alone).
        * ``list[str]`` – subset of perturbations.
        * ``None``      – all perturbations.
    top_n : int, default 20
        Number of top target genes to show per perturbation (ranked by effect size).
    uns_key : str, default 'results'
    cf_key : str, default 'counterfactual_effect_size'
    min_effect : float, default 0.0
        Drop edges below this effect size threshold.
    figsize : tuple or None
        Auto-scaled if None.
    title : str or None
    """
    cf, perts, genes = _get_cf_data(adata, uns_key=uns_key, cf_key=cf_key)
    cf_cond = cf[cond_idx]  # (P, G)

    if pert_name is None:
        pert_subset = list(perts)
        p_indices = list(range(len(perts)))
    else:
        names = [pert_name] if isinstance(pert_name, str) else list(pert_name)
        pert_subset, p_indices = [], []
        missing = [n for n in names if n not in perts]
        if missing:
            raise ValueError(
                f"Perturbation(s) not found: {missing}. Available: {list(perts)}"
            )
        for n in names:
            idx = int(np.where(perts == n)[0][0])
            pert_subset.append(n)
            p_indices.append(idx)

    # build edge list
    edges = []
    gene_set = set()
    for p_idx, p_name in zip(p_indices, pert_subset):
        ev = cf_cond[p_idx]                         # (G,)
        top_idx = np.argsort(ev)[::-1][:top_n]
        for g_idx in top_idx:
            w = float(ev[g_idx])
            if w > min_effect:
                gene_name = str(genes[g_idx])
                edges.append((p_name, gene_name, w))
                gene_set.add(gene_name)

    # order genes by total effect across perts so layout is stable
    gene_totals = {}
    for _, g, w in edges:
        gene_totals[g] = gene_totals.get(g, 0.0) + w
    gene_nodes = sorted(gene_set, key=lambda g: -gene_totals[g])

    n_perts = len(pert_subset)
    n_genes = len(gene_nodes)
    if figsize is None:
        w = max(8, max(n_perts, n_genes) * 0.55 + 2)
        figsize = (w, 6)

    fig, ax = plt.subplots(figsize=figsize)

    _draw_bipartite_nx(pert_subset, gene_nodes, edges, ax=ax)

    cond_names = np.asarray(
        adata.uns[uns_key].get("conditions", [f"cond_{cond_idx}"])
    )
    cond_label = (
        str(cond_names[cond_idx]) if cond_idx < len(cond_names) else f"cond_{cond_idx}"
    )

    if title is None:
        if pert_name is not None:
            title = (
                f"Counterfactual targets: {pert_name}  |  "
                f"condition: {cond_label}  |  top {top_n}"
            )
        else:
            title = (
                f"Counterfactual bipartite graph  |  "
                f"condition: {cond_label}  |  top {top_n} per perturbation"
            )

    ax.set_title(title, fontsize=11, pad=12)
    plt.tight_layout()
    plt.show()


def plot_cf_target_overlap(
    adata,
    cond_idx=0,
    top_n=20,
    pert_names=None,
    uns_key="results",
    cf_key="counterfactual_effect_size",
    figsize=None,
    show_jaccard=True,
    show_membership=True,
    jaccard_cmap="Blues",
    membership_cmap="Blues",
    label_fontsize=8,
):
    """
    Visualize the overlap between top-N target gene sets across perturbations.

    Produces up to two panels:

    * **Jaccard heatmap** – pairwise Jaccard similarity of each pair of
      perturbations' top-N target sets.  High values mean the two
      perturbations share many top targets.
    * **Binary membership matrix** – genes (rows) × perturbations (columns).
      A filled cell means the gene is in that perturbation's top-N set.
      Genes are sorted by the number of perturbations they appear in
      (most shared genes first).

    Parameters
    ----------
    adata : AnnData
    cond_idx : int, default 0
    top_n : int, default 20
        Number of top-ranked target genes per perturbation.
    pert_names : list of str or None, default None
        Subset of perturbations to include.  None = all.
    uns_key : str, default 'results'
    cf_key : str, default 'counterfactual_effect_size'
    figsize : tuple or None
        Auto-scaled if None.
    show_jaccard : bool, default True
        Whether to draw the Jaccard heatmap.
    show_membership : bool, default True
        Whether to draw the binary membership matrix.
    jaccard_cmap : str, default 'Blues'
    membership_cmap : str, default 'Blues'
    label_fontsize : int, default 8
    """
    cf, perts, genes = _get_cf_data(adata, uns_key=uns_key, cf_key=cf_key)
    cf_cond = cf[cond_idx]  # (P, G)

    if pert_names is not None:
        mask = np.isin(perts, pert_names)
        cf_cond = cf_cond[mask]
        perts = perts[mask]

    P = len(perts)
    if P == 0:
        print("No perturbations to plot.")
        return

    # top-N gene sets per perturbation
    target_sets = {}
    for p_idx, p_name in enumerate(perts):
        ev = cf_cond[p_idx]
        top_idx = np.argsort(ev)[::-1][:top_n]
        target_sets[p_name] = set(genes[top_idx])

    # --- Jaccard matrix ---
    jaccard = np.zeros((P, P))
    for i, pi in enumerate(perts):
        for j, pj in enumerate(perts):
            si, sj = target_sets[pi], target_sets[pj]
            union = len(si | sj)
            jaccard[i, j] = len(si & sj) / union if union > 0 else 0.0

    # --- binary membership matrix ---
    all_genes = sorted(
        set.union(*target_sets.values()),
        # sort: genes shared by most perts first, then alphabetical
        key=lambda g: (
            -sum(g in target_sets[p] for p in perts),
            g,
        ),
    )
    G_all = len(all_genes)
    membership = np.zeros((G_all, P), dtype=float)
    for j, p_name in enumerate(perts):
        for i, gene in enumerate(all_genes):
            membership[i, j] = 1.0 if gene in target_sets[p_name] else 0.0

    n_panels = int(show_jaccard) + int(show_membership)
    if n_panels == 0:
        return

    if figsize is None:
        w = 5.5 * n_panels + max(0, (P - 8) * 0.3 * n_panels)
        h = max(5, G_all * 0.22 + 3) if show_membership else max(5, P * 0.4 + 2)
        figsize = (w, h)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    ax_iter = iter(axes)

    cond_names = np.asarray(
        adata.uns[uns_key].get("conditions", [f"cond_{cond_idx}"])
    )
    cond_label = (
        str(cond_names[cond_idx]) if cond_idx < len(cond_names) else f"cond_{cond_idx}"
    )

    if show_jaccard:
        ax = next(ax_iter)
        im = ax.imshow(jaccard, cmap=jaccard_cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(P))
        ax.set_xticklabels(perts, rotation=45, ha="right", fontsize=label_fontsize)
        ax.set_yticks(np.arange(P))
        ax.set_yticklabels(perts, fontsize=label_fontsize)
        # annotate cells
        for i in range(P):
            for j in range(P):
                ax.text(
                    j, i, f"{jaccard[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=max(5, label_fontsize - 1),
                    color="black" if jaccard[i, j] < 0.6 else "white",
                )
        plt.colorbar(im, ax=ax, label="Jaccard similarity", fraction=0.046, pad=0.04)
        ax.set_title(
            f"Top-{top_n} target set overlap\n(Jaccard)  |  cond: {cond_label}",
            fontsize=10,
        )

    if show_membership:
        ax = next(ax_iter)
        im = ax.imshow(membership, cmap=membership_cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(P))
        ax.set_xticklabels(perts, rotation=45, ha="right", fontsize=label_fontsize)
        if G_all <= 80:
            ax.set_yticks(np.arange(G_all))
            ax.set_yticklabels(all_genes, fontsize=label_fontsize)
        else:
            ax.set_yticks([])
            ax.set_ylabel(f"Genes (n={G_all})", fontsize=label_fontsize)
        ax.set_title(
            f"Target gene membership  |  top {top_n}  |  cond: {cond_label}",
            fontsize=10,
        )
        ax.set_xlabel("Perturbation", fontsize=label_fontsize)

        # horizontal line separating "shared" from "unique" genes
        n_shared = int((membership.sum(axis=1) > 1).sum())
        if 0 < n_shared < G_all:
            ax.axhline(n_shared - 0.5, color="red", lw=1, linestyle="--", alpha=0.7)
            ax.text(
                P - 0.5, n_shared - 0.5, " shared ↑",
                ha="right", va="bottom", fontsize=7, color="red", alpha=0.8,
            )

    plt.tight_layout()
    plt.show()
