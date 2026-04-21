from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def cluster_rho(adata, t=1.5, uns_key='results', rho_corr=None, show=True):
    if rho_corr is None:
        C = adata.uns[uns_key]['rho_corr']
    else:
        C = rho_corr

    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("rho_corr must be a square 2D matrix.")

    # Numerical guardrails for downstream clustering.
    C = np.clip(C, -1.0, 1.0)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)

    D = 1.0 - C  # distance
    np.fill_diagonal(D, 0.0)
    condensed = D[np.triu_indices_from(D, 1)]
    if condensed.size == 0:
        raise ValueError("Need at least 2 items to cluster.")
    if not np.isfinite(condensed).all():
        raise ValueError("The condensed distance matrix must contain only finite values.")

    Z_link = linkage(condensed, method='ward')

    if show:
        plt.figure(figsize=(10, 6))
        dendrogram(
            Z_link,
            color_threshold=t,            # <-- color branches by your cut
            above_threshold_color="gray", # color for branches above the cut
            labels=np.arange(len(C)),     # optional
            leaf_rotation=90,
            leaf_font_size=8,
        )
        plt.axhline(y=t, color="red", linestyle="--", linewidth=1)
        plt.title(f"Covariance hierarchal clustering t={t}")
        plt.xlabel("Item")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()

    # Cluster labels at the same cut
    groups = fcluster(Z_link, t=t, criterion='distance')

    return groups

def group2name(groups, all_genes):
    return pd.DataFrame(groups, index=all_genes, columns=["group"])
