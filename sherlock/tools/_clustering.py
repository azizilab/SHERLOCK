from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def cluster_rho(adata, t=1.5, uns_key='results', rho_corr=None, show=True):
    if rho_corr is None:
        C = adata.uns[uns_key]['rho_corr']
    else:
        C = rho_corr
    D = 1.0 - C  # distance
    Z_link = linkage(D[np.triu_indices_from(D, 1)], method='ward')

    t = 1.5  # your cut in the same distance scale as Z_link

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