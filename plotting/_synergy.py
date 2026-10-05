from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


_CLASS_COLORS = {
    "neomorphic":                   "#762a83",
    "epistasis":                    "#d73027",
    "approximately additive":       "#92c5de",
    "redundant":                    "#4dac26",
    "suppression":                  "#f4a582",
    "synergy":                      "#2166ac",
    "synergy (similar phenotype)":  "#2166ac",
    "synergy (dissimilar phenotype)": "#1a9850",
    "potentiation":                 "#e08214",
    "weak / undefined":             "#aaaaaa",
    "too few cells":                "#dddddd",
    "mixed / ambiguous":            "#777777",
}

# Per-class config: which two metrics are the primary delimiters for that gate.
# x_col / y_col starting with "_" are derived inside the function
# (e.g. "_obs_cos_max" = max(obs_cos_a, obs_cos_b)).
_CLASS_GATE_CONFIG = {
    "potentiation": {
        "x_col":  "latent_c_sum",
        "y_col":  "dominance",
        "xlabel": r"latent c-sum  ($c_1 + c_2$)",
        "ylabel": r"c magnitude imbalance  ($|\log_{10}(c_1/c_2)|$)",
    },
    "neomorphic": {
        "x_col":  "_obs_cos_max",
        "y_col":  "_obs_cos_min",
        "xlabel": r"obs cos max  ($\max\cos(\Delta_{ab},\,\Delta_a^{obs})$)",
        "ylabel": r"obs cos min  ($\min\cos(\Delta_{ab},\,\Delta_b^{obs})$)",
    },
    "epistasis": {
        "x_col":  "dom_ratio",
        "y_col":  "dominance",
        "xlabel": r"direction ratio  ($\cos_{min}/\cos_{max}$)",
        "ylabel": r"c magnitude imbalance",
    },
    "suppression": {
        "x_col":  "off_axis_norm",
        "y_col":  "model_mpr_decoded",
        "xlabel": r"off-axis activity  ($\|\Delta_{ab}^{\perp}\|/\|\Delta_{pred}\|$)",
        "ylabel": r"decoded mag. ratio  (model $z_{ab}^{pred}$)",
    },
    "synergy": {
        "x_col":  "latent_c_sum",
        "y_col":  "signed_excess",
        "xlabel": r"latent c-sum  ($c_1 + c_2$)",
        "ylabel": r"latent excess  (super-additivity along $\hat\Delta_{pred}$)",
    },
    "redundant": {
        "x_col":  "_obs_cos_min",
        "y_col":  "rho_ab",
        "xlabel": r"obs cos min  ($\min\cos$)",
        "ylabel": r"model genetic co-variation  ($\rho_{ab}$)",
    },
    "approximately additive": {
        "x_col":  "latent_c_sum",
        "y_col":  "model_mpr_decoded",
        "xlabel": r"latent c-sum  ($c_1 + c_2$)",
        "ylabel": r"decoded mag. ratio",
    },
}

_SYN_MAP = {
    "synergy (similar phenotype)":   "synergy",
    "synergy (dissimilar phenotype)": "synergy",
}


def plot_synergy_gate_boundaries(
    df: pd.DataFrame,
    gt_col: str | None = None,
    figsize: tuple | None = None,
    point_size: int = 35,
    alpha_in: float = 0.85,
    alpha_out: float = 0.12,
    ncols: int = 4,
) -> tuple:
    """
    One 2-D scatter per interaction class with axes chosen as the prime gate
    delimiters for that class.

    Parameters
    ----------
    df :
        DataFrame returned by ``get_synergy()``.  Must contain the
        ``classification`` column plus the metric columns referenced in
        ``_CLASS_GATE_CONFIG``.
    gt_col :
        Optional column of ground-truth interaction labels.  When provided a
        third marker category is drawn: open rings mark GT members that the
        model classified differently (false negatives for each class).
        ``"synergy (similar phenotype)"`` and ``"synergy (dissimilar phenotype)"``
        are both normalised to ``"synergy"`` automatically.
    figsize :
        Override auto-computed figure size.
    point_size :
        Scatter marker area.
    alpha_in :
        Opacity of in-class (model-predicted) points.
    alpha_out :
        Opacity of background (out-of-class) points.
    ncols :
        Number of subplot columns.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of Axes
    """
    df = df.copy()

    # Derive combined obs cosines if not present
    if "_obs_cos_max" not in df.columns:
        df["_obs_cos_max"] = df[["obs_cos_a", "obs_cos_b"]].max(axis=1)
    if "_obs_cos_min" not in df.columns:
        df["_obs_cos_min"] = df[["obs_cos_a", "obs_cos_b"]].min(axis=1)

    df["_cls"] = df["classification"].replace(_SYN_MAP)

    has_gt = gt_col is not None and gt_col in df.columns
    if has_gt:
        df["_gt"] = df[gt_col].replace(_SYN_MAP)

    classes = list(_CLASS_GATE_CONFIG.keys())
    nrows   = int(np.ceil(len(classes) / ncols))

    if figsize is None:
        figsize = (ncols * 4.0, nrows * 3.6)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.asarray(axes).flatten()

    for ax_idx, cls in enumerate(classes):
        ax        = axes_flat[ax_idx]
        cfg       = _CLASS_GATE_CONFIG[cls]
        cls_color = _CLASS_COLORS.get(cls, "#333333")
        x_col, y_col = cfg["x_col"], cfg["y_col"]

        if x_col not in df.columns or y_col not in df.columns:
            ax.set_visible(False)
            continue

        mask_in  = df["_cls"] == cls
        mask_out = ~mask_in
        x = df[x_col].values.astype(float)
        y = df[y_col].values.astype(float)

        # Background: all other classes
        ax.scatter(
            x[mask_out], y[mask_out],
            c="#aaaaaa", s=point_size * 0.65,
            alpha=alpha_out, linewidths=0, zorder=2, rasterized=True,
        )

        # Foreground: model-predicted members of this class
        ax.scatter(
            x[mask_in], y[mask_in],
            c=cls_color, s=point_size,
            alpha=alpha_in, linewidths=0.3, edgecolors="white",
            zorder=4, rasterized=True,
        )

        # Open rings: GT members missed by the model (false negatives)
        if has_gt:
            mask_fn = (~mask_in) & (df["_gt"] == cls)
            if mask_fn.any():
                ax.scatter(
                    x[mask_fn], y[mask_fn],
                    s=point_size * 1.9,
                    facecolors="none", edgecolors=cls_color,
                    linewidths=1.1, zorder=5, alpha=0.88,
                )

        # Title
        n_model = int(mask_in.sum())
        title   = f"{cls}  (n={n_model})"
        if has_gt:
            n_gt      = int((df["_gt"] == cls).sum())
            n_correct = int((mask_in & (df["_gt"] == cls)).sum())
            title += f"\nGT: {n_correct}/{n_gt} correct"

        ax.set_title(title, fontsize=8.5, color=cls_color, fontweight="bold", pad=4)
        ax.set_xlabel(cfg["xlabel"], fontsize=7.5)
        ax.set_ylabel(cfg["ylabel"], fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    # Extra axes: hide or use for legend
    extra_axes = axes_flat[len(classes):]
    for ax in extra_axes:
        ax.set_visible(False)

    if len(extra_axes) >= 1:
        leg_ax = extra_axes[0]
        leg_ax.set_visible(True)
        leg_ax.set_axis_off()

        handles = [
            mpatches.Patch(color=_CLASS_COLORS.get(c, "#333333"), label=c, alpha=0.85)
            for c in classes
        ]
        handles.append(mpatches.Patch(color="#aaaaaa", label="other class", alpha=0.35))
        if has_gt:
            handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="none", markeredgecolor="#444444",
                       markeredgewidth=1.1, markersize=8,
                       label="GT member (missed by model)")
            )
        leg_ax.legend(handles=handles, loc="center", fontsize=8,
                      frameon=False, title="class", title_fontsize=8.5)

    fig.suptitle("Synergy classification — gate delimiter scatterplots", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig, axes


# ── Interaction residual / metric UMAP ───────────────────────────────────────

# Scalar features that characterise the interaction geometry.
# These are direction-agnostic and comparable across pairs — unlike raw residual
# vectors whose direction is pair-specific and therefore does not cluster by class.
_METRIC_FEATURES = [
    "latent_c_sum",   # total additive strength (used in 7/10 gates)
    "lcs_mpr_gap",    # excess beyond max parent — top RF importance
    "off_axis_norm",  # emergent orthogonal activity (neomorphic / suppression)
    "rho_ab",         # obs–model correlation (redundancy)
    "_obs_cos_max",   # max cosine to either parent in obs space (neomorphic vs redundant)
]


def plot_residual_umap(
    residuals_dict: dict,
    syn_df: "pd.DataFrame | None" = None,
    gt_col: str | None = None,
    gt_labels: "pd.Series | dict | None" = None,
    features: "list[str] | None" = None,
    n_neighbors: int = 10,
    min_dist: float = 0.3,
    metric: str = "euclidean",
    random_state: int = 42,
    point_size: int = 80,
    alpha: float = 0.85,
    figsize: tuple | None = None,
    label_pairs: bool = False,
    label_fontsize: float = 6.5,
) -> tuple:
    """
    UMAP of genetic interaction pairs coloured by class.

    By default uses the **scalar metric features** from *syn_df* (interaction
    geometry described as ~10 direction-agnostic numbers per pair).  Raw residual
    vectors are available via ``syn_df=None`` but rarely cluster by class because
    their directions are pair-specific rather than class-specific.

    When *gt_col* is provided the figure has two side-by-side panels sharing the
    same embedding: left = ground-truth labels, right = model predictions.

    Parameters
    ----------
    residuals_dict :
        Dict returned by ``model.get_interaction_residuals()``.  Always required
        for pair ordering and classification labels; used as the feature matrix
        only when *syn_df* is ``None``.
    syn_df :
        DataFrame returned by ``get_synergy()``.  When provided (recommended),
        scalar metric features are used instead of raw residual vectors.
    gt_col :
        Column in *syn_df* holding ground-truth labels.  Triggers a two-panel
        figure (GT left, model prediction right).
    gt_labels :
        External ground-truth labels as a ``pd.Series`` (indexed by pair names)
        or ``dict`` mapping pair names to labels.  Use when GT lives outside
        *syn_df*.  Takes precedence over *gt_col*.
    features :
        Override the default metric columns.  Ignored when *syn_df* is ``None``.
    n_neighbors, min_dist, metric :
        UMAP hyperparameters.
    random_state :
        Reproducibility seed.
    point_size, alpha :
        Scatter aesthetics.
    figsize :
        Auto-computed when ``None``; ``(7, 6)`` for single panel, ``(13, 6)``
        for two-panel GT vs predicted.
    label_pairs :
        Annotate each point with the pair name.
    label_fontsize :
        Font size for pair annotations.

    Returns
    -------
    fig        : matplotlib.figure.Figure
    axes       : Axes or (ax_gt, ax_pred) tuple
    embedding  : np.ndarray (n, 2)
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError("umap-learn is required:  pip install umap-learn") from e

    from sklearn.preprocessing import StandardScaler

    _syn_map = {
        "synergy (similar phenotype)":   "synergy",
        "synergy (dissimilar phenotype)": "synergy",
    }

    pairs       = residuals_dict["pairs"]
    pred_labels = [_syn_map.get(c, c) for c in residuals_dict["classification"]]

    # ── Build feature matrix ──────────────────────────────────────────────────
    if syn_df is not None:
        df = syn_df.copy()
        if "_obs_cos_max" not in df.columns:
            df["_obs_cos_max"] = df[["obs_cos_a", "obs_cos_b"]].max(axis=1)
        if "_obs_cos_min" not in df.columns:
            df["_obs_cos_min"] = df[["obs_cos_a", "obs_cos_b"]].min(axis=1)
        if "lcs_mpr_gap" not in df.columns and "latent_c_sum" in df.columns and "model_mpr_decoded" in df.columns:
            df["lcs_mpr_gap"] = df["latent_c_sum"] - df["model_mpr_decoded"]

        cols = features if features is not None else _METRIC_FEATURES
        cols = [c for c in cols if c in df.columns]

        df_sub = df.reindex(pairs)[cols].astype(float)
        df_sub = df_sub.fillna(df_sub.median())
        X = StandardScaler().fit_transform(df_sub.values)
        feature_label = "metric features"
    else:
        X = StandardScaler().fit_transform(residuals_dict["residuals"])
        feature_label = "residual vectors"

    # ── UMAP ─────────────────────────────────────────────────────────────────
    embedding = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2,
    ).fit_transform(X)

    # ── GT labels — accept column name, external Series/dict, or None ─────────
    if gt_labels is not None:
        # external Series or dict passed directly
        if isinstance(gt_labels, dict):
            gt_series = pd.Series(gt_labels)
        else:
            gt_series = pd.Series(gt_labels)
        resolved_gt = [_syn_map.get(str(gt_series.get(p, "unknown")), str(gt_series.get(p, "unknown"))) for p in pairs]
        has_gt = True
    elif gt_col is not None and syn_df is not None and gt_col in syn_df.columns:
        gt_ser     = syn_df.reindex(pairs)[gt_col].fillna("unknown")
        resolved_gt = [_syn_map.get(g, g) for g in gt_ser]
        has_gt = True
    else:
        resolved_gt = None
        has_gt = False

    two_panel = has_gt
    if figsize is None:
        figsize = (13, 6) if two_panel else (7, 6)

    fig, axes = plt.subplots(1, 2 if two_panel else 1, figsize=figsize)
    panel_list = list(axes) if two_panel else [axes]

    def _draw(ax, labels_list, title):
        labels_list = [_syn_map.get(str(l), str(l)) for l in labels_list]
        unique_cls = sorted(set(labels_list))
        for cls in unique_cls:
            mask  = np.array([l == cls for l in labels_list])
            color = _CLASS_COLORS.get(cls, "#bbbbbb")
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=color, s=point_size, alpha=alpha,
                edgecolors="white", linewidths=0.3,
                zorder=3, rasterized=True,
            )
        if label_pairs:
            for idx, (x, y) in enumerate(embedding):
                ax.text(x, y, pairs[idx], fontsize=label_fontsize,
                        ha="left", va="bottom", zorder=4, alpha=0.72)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        handles = [
            mpatches.Patch(color=_CLASS_COLORS.get(c, "#bbbbbb"), label=c, alpha=0.85)
            for c in unique_cls
        ]
        ax.legend(handles=handles, fontsize=7.5, frameon=False,
                  bbox_to_anchor=(1.01, 1), loc="upper left")

    if two_panel:
        _draw(panel_list[0], resolved_gt,  "Ground truth")
        _draw(panel_list[1], pred_labels,  "Model prediction")
        fig.suptitle(f"Genetic interaction UMAP  ({feature_label})", fontsize=11, y=1.02)
    else:
        _draw(panel_list[0], pred_labels, f"Model prediction  ({feature_label})")

    fig.tight_layout()
    return fig, axes, embedding


# ── Legacy 3D plot ────────────────────────────────────────────────────────────

def _scatter_3d(ax, x, y, z, c, cmap=None, norm=None, colors=None, point_size=30, alpha=0.8):
    if colors is not None:
        return ax.scatter(x, y, z, c=colors, s=point_size, alpha=alpha, depthshade=True)
    return ax.scatter(x, y, z, c=c, cmap=cmap, norm=norm, s=point_size, alpha=alpha, depthshade=True)


def _label_axes(ax, elev, azim):
    ax.set_xlabel("cos(AB, A)", labelpad=8)
    ax.set_ylabel("cos(AB, B)", labelpad=8)
    ax.set_zlabel("cos(A, B)", labelpad=8)
    ax.view_init(elev=elev, azim=azim)


def plot_synergy_classes(
    adata,
    uns_key: str = "results",
    cmap: str = "viridis",
    figsize: tuple = (16, 7),
    point_size: int = 30,
    alpha: float = 0.8,
    elev: float = 25.0,
    azim: float = 45.0,
):
    """
    Side-by-side 3D scatter of synergy pairs.
      Left:  colored by interaction_score (continuous)
      Right: colored by classification (categorical)

    Axes:
      x = cos_ab_a  (how aligned AB is with A)
      y = cos_ab_b  (how aligned AB is with B)
      z = cos_a_b   (how similar A and B are)

    Synergy data is read from adata.uns[uns_key]['synergy'].
    """
    syn = adata.uns[uns_key].get("synergy")
    if syn is None:
        raise ValueError(f"No 'synergy' key in adata.uns['{uns_key}']. Run eval() with combinatorial=True.")
    if not isinstance(syn, pd.DataFrame):
        syn = pd.DataFrame(syn)

    df = syn.dropna(subset=["cos_ab_a", "cos_ab_b", "cos_a_b", "interaction_score", "classification"])

    x = df["cos_ab_a"].values
    y = df["cos_ab_b"].values
    z = df["cos_a_b"].values

    scores = df["interaction_score"].values
    norm = mcolors.Normalize(vmin=np.nanpercentile(scores, 2), vmax=np.nanpercentile(scores, 98))

    cat_colors = [_CLASS_COLORS.get(cls, "#333333") for cls in df["classification"]]

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121, projection="3d")
    sc = ax1.scatter(x, y, z, c=scores, cmap=cmap, norm=norm, s=point_size, alpha=alpha, depthshade=True)
    _label_axes(ax1, elev, azim)
    ax1.set_title("Interaction score", fontsize=12)
    cbar = fig.colorbar(sc, ax=ax1, pad=0.1, shrink=0.6)
    cbar.set_label("interaction_score", fontsize=9)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(x, y, z, c=cat_colors, s=point_size, alpha=alpha, depthshade=True)
    _label_axes(ax2, elev, azim)
    ax2.set_title("Synergy class", fontsize=12)

    present = df["classification"].unique()
    handles = [
        mpatches.Patch(color=_CLASS_COLORS.get(cls, "#333333"), label=cls)
        for cls in present
    ]
    ax2.legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.0))

    plt.tight_layout()
    return fig, (ax1, ax2)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _prep_syn_features(syn_df, features=None, default=None):
    """Derive _obs_cos_max/_obs_cos_min/lcs_mpr_gap if absent; return (df, cols)."""
    df = syn_df.copy()
    if {"obs_cos_a", "obs_cos_b"}.issubset(df.columns):
        if "_obs_cos_max" not in df.columns:
            df["_obs_cos_max"] = df[["obs_cos_a", "obs_cos_b"]].max(axis=1)
        if "_obs_cos_min" not in df.columns:
            df["_obs_cos_min"] = df[["obs_cos_a", "obs_cos_b"]].min(axis=1)
    if "lcs_mpr_gap" not in df.columns and {"latent_c_sum", "model_mpr_decoded"}.issubset(df.columns):
        df["lcs_mpr_gap"] = df["latent_c_sum"] - df["model_mpr_decoded"]
    cols = [c for c in (features or default or _METRIC_FEATURES) if c in df.columns]
    return df, cols


# All metrics that appear in at least one gate condition.
_GATE_FEATURES = [
    "latent_c_sum",    # gates 1,2,6.7,6.8,7,8,9,10,11,12 — primary strength axis
    "lcs_mpr_gap",     # gate 6.8 (lc_sum − mpr_decoded); top RF importance
    "off_axis_norm",   # gates 4,5,6.5,6.7,6.8,9,10 — emergent orthogonal activity
    "cos_emb_a_b",     # gates 1,5,6.5,6.7,6.8,9,10 — parent program similarity
    "model_mpr_decoded", # gates 6.5,6.7,6.8,8,10,11 — decoded expression ratio
    "dom_ratio",       # gates 1,6 — c1/(c1+c2) directional balance
    "dominance",       # gates 1,6 — |log10(c1/c2)| magnitude balance
    "signed_excess",   # gates 7,11 — super/sub-additivity direction
    "rho_ab",          # gate 8 — obs–model co-variation (redundancy)
    "_obs_cos_max",    # gates 2,4 — max cosine to either parent (obs space)
    "_obs_cos_min",    # gates 2,8 — min cosine to either parent (obs space)
]


_FMT_FEAT = {
    "latent_c_sum":    "c_sum",
    "lcs_mpr_gap":     "lcs_gap",
    "off_axis_norm":   "off-axis",
    "rho_ab":          "ρ_ab",
    "_obs_cos_max":    "cos_max",
    "signed_excess":   "signed\nexcess",
    "dom_ratio":       "dom_ratio",
    "model_mpr_decoded": "mpr_dec",
}


def _fmt_feat(name: str) -> str:
    return _FMT_FEAT.get(name, name.lstrip("_").replace("_", "\n"))


# ── Parallel coordinates ──────────────────────────────────────────────────────

def plot_parallel_coordinates(
    syn_df: "pd.DataFrame",
    gt_labels: "pd.Series | dict | None" = None,
    features=None,
    figsize=None,
    alpha: float = 0.30,
    lw: float = 0.9,
    show_median: bool = True,
    median_lw: float = 2.5,
) -> tuple:
    """
    Parallel coordinates: one polyline per pair, each feature normalised to [0,1].
    Thick lines show the per-class median profile.

    Parameters
    ----------
    syn_df      : DataFrame from ``get_synergy()``.
    gt_labels   : optional Series / dict (index = pair name) of GT labels.
    features    : column list; defaults to ``_METRIC_FEATURES``.
    show_median : overlay thick class-median lines.

    Returns
    -------
    fig, axes
    """
    df, cols = _prep_syn_features(syn_df, features)
    pairs       = list(df.index)
    pred_labels = [_SYN_MAP.get(c, c) for c in df["classification"]]

    has_gt = False
    resolved_gt = None
    if gt_labels is not None:
        gt_ser      = pd.Series(gt_labels)
        resolved_gt = [_SYN_MAP.get(str(gt_ser.get(p, "unknown")), str(gt_ser.get(p, "unknown")))
                       for p in pairs]
        has_gt = True

    panels = ([("Ground truth", resolved_gt), ("Predicted", pred_labels)]
              if has_gt else [("Predicted", pred_labels)])

    feat_df  = df[cols].astype(float).copy()
    col_min  = feat_df.min()
    col_rng  = feat_df.max() - col_min + 1e-12
    feat_norm = (feat_df - col_min) / col_rng

    n_panels = len(panels)
    figsize  = figsize or (7 * n_panels, 5)
    fig, axs = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
    axs      = list(axs) if n_panels > 1 else [axs]

    x = np.arange(len(cols))

    for ax, (title, labels) in zip(axs, panels):
        for idx in range(len(pairs)):
            color = _CLASS_COLORS.get(labels[idx], "#aaaaaa")
            ax.plot(x, feat_norm.iloc[idx].values, color=color, alpha=alpha, lw=lw)

        for xi in x:
            ax.axvline(xi, color="gray", lw=0.4, alpha=0.5, zorder=0)

        if show_median:
            for cls_name, color in _CLASS_COLORS.items():
                idxs = [i for i, l in enumerate(labels) if l == cls_name]
                if not idxs:
                    continue
                med = feat_norm.iloc[idxs].median(axis=0).values
                ax.plot(x, med, color=color, lw=median_lw, zorder=5,
                        solid_capstyle="round")

        ax.set_xticks(x)
        ax.set_xticklabels([_fmt_feat(c) for c in cols], fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.3, len(cols) - 0.7)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["min", "0.5", "max"], fontsize=8)
        ax.set_ylabel("Normalised value", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        present = sorted({l for l in labels if l not in ("unknown", "nan", "NaN", "None")})
        handles = [mpatches.Patch(color=_CLASS_COLORS.get(cls, "#aaaaaa"), label=cls)
                   for cls in present]
        ax.legend(handles=handles, fontsize=7, loc="upper right",
                  framealpha=0.85, edgecolor="none")

    plt.tight_layout()
    return fig, axs


# ── PCA biplot ────────────────────────────────────────────────────────────────

def plot_pca_biplot(
    syn_df: "pd.DataFrame",
    gt_labels: "pd.Series | dict | None" = None,
    features=None,
    figsize=None,
    point_size: float = 80,
    alpha: float = 0.85,
    loading_scale: float = 1.0,
    label_pairs: bool = False,
    label_fontsize: float = 6.5,
) -> tuple:
    """
    PCA biplot of synergy pairs with feature loading vectors.

    Parameters
    ----------
    syn_df         : DataFrame from ``get_synergy()``.
    gt_labels      : optional Series / dict (index = pair name) of GT labels.
    features       : column list; defaults to ``_METRIC_FEATURES``.
    loading_scale  : scalar multiplier for arrow length (default 1.0).
    label_pairs    : annotate each point with its pair name.

    Returns
    -------
    fig, axes, embedding   (embedding is (n, 2) PCA scores)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df, cols = _prep_syn_features(syn_df, features)
    pairs       = list(df.index)
    pred_labels = [_SYN_MAP.get(c, c) for c in df["classification"]]

    X_raw = df[cols].astype(float)
    X_raw = X_raw.fillna(X_raw.median())
    X     = StandardScaler().fit_transform(X_raw.values)
    pca   = PCA(n_components=2, random_state=42)
    emb   = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100

    has_gt = False
    resolved_gt = None
    if gt_labels is not None:
        gt_ser      = pd.Series(gt_labels)
        resolved_gt = [_SYN_MAP.get(str(gt_ser.get(p, "unknown")), str(gt_ser.get(p, "unknown")))
                       for p in pairs]
        has_gt = True

    panels = ([("Ground truth", resolved_gt), ("Predicted", pred_labels)]
              if has_gt else [("Predicted", pred_labels)])

    n_panels = len(panels)
    figsize  = figsize or (6.5 * n_panels, 6)
    fig, axs = plt.subplots(1, n_panels, figsize=figsize, sharex=True, sharey=True)
    axs      = list(axs) if n_panels > 1 else [axs]

    loadings   = pca.components_.T                      # (n_feats, 2)
    emb_scale  = np.abs(emb).max(axis=0).mean()
    arrow_scale = emb_scale * loading_scale * 0.8 / (np.abs(loadings).max() + 1e-9)

    for ax, (title, labels) in zip(axs, panels):
        colors = [_CLASS_COLORS.get(l, "#aaaaaa") for l in labels]
        ax.scatter(emb[:, 0], emb[:, 1],
                   c=colors, s=point_size, alpha=alpha,
                   edgecolors="white", linewidths=0.4, zorder=3)

        if label_pairs:
            for i, (xi, yi) in enumerate(emb):
                ax.text(xi, yi, pairs[i], fontsize=label_fontsize,
                        ha="center", va="bottom", alpha=0.75)

        for j, feat in enumerate(cols):
            lx = loadings[j, 0] * arrow_scale
            ly = loadings[j, 1] * arrow_scale
            ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.0),
                        zorder=6)
            ax.text(lx * 1.18, ly * 1.18, _fmt_feat(feat),
                    fontsize=8, ha="center", va="center", color="#333333", zorder=7)

        ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.4, alpha=0.5)
        ax.set_xlabel(f"PC1  ({var_exp[0]:.1f}% var)", fontsize=10)
        ax.set_ylabel(f"PC2  ({var_exp[1]:.1f}% var)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        present = sorted({l for l in labels if l not in ("unknown", "nan", "NaN", "None")})
        handles = [mpatches.Patch(color=_CLASS_COLORS.get(cls, "#aaaaaa"), label=cls)
                   for cls in present]
        ax.legend(handles=handles, fontsize=7, loc="upper left",
                  framealpha=0.85, edgecolor="none")

    plt.tight_layout()
    return fig, axs, emb


# ── Feature stripplot ─────────────────────────────────────────────────────────

_CLASS_ORDER = [
    "approximately additive", "redundant", "potentiation",
    "synergy", "epistasis", "neomorphic", "suppression",
    "weak / undefined",
]


def plot_feature_stripplot(
    syn_df: "pd.DataFrame",
    gt_labels: "pd.Series | dict | None" = None,
    features=None,
    figsize=None,
    point_size: float = 28,
    alpha: float = 0.75,
    jitter: float = 0.18,
    show_median: bool = True,
) -> tuple:
    """
    Per-feature strip plot: one column per feature, x = class, y = feature value.
    Thick horizontal bars show the per-class median.

    Parameters
    ----------
    syn_df      : DataFrame from ``get_synergy()``.
    gt_labels   : optional Series / dict of GT labels (index = pair name).
    features    : column list; defaults to ``_METRIC_FEATURES``.
    show_median : draw a horizontal bar at the class median.

    Returns
    -------
    fig, axes   (axes is a 2-D array of shape (n_rows, n_features))
    """
    df, cols    = _prep_syn_features(syn_df, features, default=_GATE_FEATURES)
    pairs       = list(df.index)
    pred_labels = [_SYN_MAP.get(c, c) for c in df["classification"]]

    has_gt = False
    resolved_gt = None
    if gt_labels is not None:
        gt_ser      = pd.Series(gt_labels)
        resolved_gt = [_SYN_MAP.get(str(gt_ser.get(p, "unknown")), str(gt_ser.get(p, "unknown")))
                       for p in pairs]
        has_gt = True

    panels  = ([("Ground truth", resolved_gt), ("Predicted", pred_labels)]
               if has_gt else [("Predicted", pred_labels)])
    n_feats = len(cols)
    n_rows  = len(panels)
    figsize = figsize or (max(2.2 * n_feats, 14), 3.8 * n_rows)
    fig, axs = plt.subplots(n_rows, n_feats, figsize=figsize, squeeze=False)

    rng = np.random.default_rng(42)

    for row_idx, (row_title, labels) in enumerate(panels):
        present = sorted(
            {l for l in labels if l not in ("unknown", "nan", "NaN", "None")},
            key=lambda c: _CLASS_ORDER.index(c) if c in _CLASS_ORDER else len(_CLASS_ORDER),
        )
        x_pos = {cls: i for i, cls in enumerate(present)}

        for col_idx, feat in enumerate(cols):
            ax        = axs[row_idx, col_idx]
            feat_vals = df[feat].astype(float).values

            for cls in present:
                xi    = x_pos[cls]
                color = _CLASS_COLORS.get(cls, "#aaaaaa")
                idxs  = [i for i, l in enumerate(labels) if l == cls]
                vals  = feat_vals[idxs]
                xs    = xi + rng.uniform(-jitter, jitter, size=len(idxs))
                ax.scatter(xs, vals, c=color, s=point_size, alpha=alpha,
                           edgecolors="none", zorder=3)
                if show_median and len(vals):
                    ax.plot([xi - jitter * 1.6, xi + jitter * 1.6],
                            [np.nanmedian(vals)] * 2,
                            color=color, lw=2.0, solid_capstyle="butt", zorder=4)

            ax.set_xticks(range(len(present)))
            ax.set_xticklabels(present, rotation=40, ha="right", fontsize=8)
            ax.set_xlim(-0.6, len(present) - 0.4)
            ax.spines[["top", "right"]].set_visible(False)

            if row_idx == 0:
                ax.set_title(_fmt_feat(feat), fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_title, fontsize=9)

    plt.tight_layout()
    return fig, axs
