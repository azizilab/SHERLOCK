import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm


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
        AnnData object with adata.uns[uns_key]['explained_variance']
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
    ev_all = np.asarray(uns["explained_variance"][cond_idx], dtype=float)
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
