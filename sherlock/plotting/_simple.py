from __future__ import annotations

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib import gridspec
import torch


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
    corr = adata.uns[uns_key]['corr']
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