from __future__ import annotations

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def cluster_rho(adata, t, uns_key='results', rho_corr=None, method='ward', criterion='distance', show=True):
    """
    Hierarchically cluster perturbations by their rho correlation matrix.

    Parameters
    ----------
    adata     : AnnData; used to read rho_corr when rho_corr is None
    t         : cut parameter (required). Meaning depends on criterion:
                  'distance'  — distance threshold on the dendrogram
                  'maxclust'  — exact number of clusters to form
    uns_key   : key in adata.uns to read rho_corr from
    rho_corr  : optional precomputed (P, P) correlation matrix
    method    : linkage method passed to scipy.cluster.hierarchy.linkage
    criterion : fcluster criterion ('distance' or 'maxclust')
    show      : if True and criterion=='distance', plot the dendrogram

    Returns
    -------
    groups : (P,) int array of cluster labels
    """
    if rho_corr is None:
        C = adata.uns[uns_key]['rho_corr']
    else:
        C = rho_corr

    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("rho_corr must be a square 2D matrix.")

    C = np.clip(C, -1.0, 1.0)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)

    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    condensed = D[np.triu_indices_from(D, 1)]
    if condensed.size == 0:
        raise ValueError("Need at least 2 items to cluster.")
    if not np.isfinite(condensed).all():
        raise ValueError("The condensed distance matrix must contain only finite values.")

    Z_link = linkage(condensed, method=method)

    if show and criterion == 'distance':
        plt.figure(figsize=(10, 6))
        dendrogram(
            Z_link,
            color_threshold=t,
            above_threshold_color="gray",
            labels=np.arange(len(C)),
            leaf_rotation=90,
            leaf_font_size=8,
        )
        plt.axhline(y=t, color="red", linestyle="--", linewidth=1)
        plt.title(f"Covariance hierarchical clustering t={t}")
        plt.xlabel("Item")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()
    elif show and criterion != 'distance':
        print("show=True is only supported for criterion='distance'. Ignoring.")

    groups = fcluster(Z_link, t=t, criterion=criterion)
    return groups

def group2name(groups, all_genes):
    return pd.DataFrame(groups, index=all_genes, columns=["group"])
