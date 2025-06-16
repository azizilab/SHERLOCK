import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from typing import List, Union, Literal, Sequence
import scanpy as sc
from typing import Sequence, Union, Literal, Tuple, List
from numpy.linalg import svd
from statsmodels.stats.multitest import multipletests
import pyro

Array = Union[np.ndarray, torch.Tensor]

def _to_numpy(t: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t

def _align_columns_binary(ref: np.ndarray,
                   mat: np.ndarray,
                   maximise: Literal["intersection", "dot"] = "intersection"):
    """
    Permute columns of `mat` to best match `ref` (Hungarian assignment).
    For binary structure we want to maximise column-wise edge intersection.
    """
    # similarity matrix S[col_ref, col_mat]
    if maximise == "intersection":
        S = (ref.T @ mat)          # counts of shared 1’s
    else:                          # fallback: dot product (same here because binary)
        S = ref.T @ mat

    from scipy.optimize import linear_sum_assignment
    row, col = linear_sum_assignment(-S)      # maximise similarity
    return mat[:, col]

# ------------------------------------------------------------------------
def compare_W_runs_binary(
    W_list : Sequence[Union[np.ndarray, torch.Tensor]],
    labels : Sequence[str] | None = None,
    metric : Literal["jaccard", "f1"] = "f1",
    align  : bool = True,
    figsize: tuple[int,int] = (6,5),
):
    W_np = [(_to_numpy(W) > 0).astype(np.int8) for W in W_list]  # ensure {0,1}
    n    = len(W_np)
    if labels is None:
        labels = [f"run-{i}" for i in range(n)]

    # optional column-wise alignment to the first run
    if align and n > 1:
        ref = W_np[0]
        W_np = [ref] + [_align_columns_binary(ref, W) for W in W_np[1:]]

    # similarity matrix
    S = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            A, B = W_np[i], W_np[j]
            intersect = (A & B).sum()
            union     = (A | B).sum()
            if metric == "jaccard":
                sim = intersect / union if union else 1.0
            else:  # F1
                sim = (2 * intersect) / (A.sum() + B.sum()) if intersect else 0.0
            S[i,j] = S[j,i] = sim

    # --- plot -------------------------------------------------------------
    plt.figure(figsize=figsize)
    ax = sns.heatmap(S, annot=True, vmin=0, vmax=1,
                     cmap="viridis", square=True,
                     xticklabels=labels, yticklabels=labels)
    ax.set_title(f"Pairwise {metric.upper()} similarity of binary W")
    plt.tight_layout()
    plt.show()

    # summary
    tri = S[np.triu_indices(n, k=1)]
    print(f"Mean {metric} = {tri.mean():.3f} ± {tri.std():.3f}")

  
  
def plot_aligned_Ws(W_list : Sequence[Union[np.ndarray, torch.Tensor]],
                    labels  : Sequence[str] | None = None,
                    cmap    : str = "bwr",
                    figsize : tuple[int,int] | None = None,
                    vlim    : float | None = None):
    W_np  = [_to_numpy(W) for W in W_list]
    n_run = len(W_np)
    if labels is None:
        labels = [f"run-{i}" for i in range(n_run)]

    # align every run (except the reference) to run-0
    ref = W_np[0]
    W_aligned = [ref] + [_align_columns_binary(ref, W, maximise='intersection') for W in W_np[1:]]
    # shared colour scale
    vmax = vlim if vlim is not None else max(np.abs(W).max() for W in W_aligned)

    # create figure
    if figsize is None:
        figsize = (4 * n_run, 4)
    fig, axes = plt.subplots(1, n_run, figsize=figsize, squeeze=False)
    for ax, W, lbl in zip(axes[0], W_aligned, labels):
        sns.heatmap(W,
                    ax=ax,
                    cmap=cmap, vmin=-vmax, vmax=vmax,
                    cbar=False, square=False)
        ax.set_title(lbl)
        ax.set_xlabel("latent dim")
        ax.set_ylabel("perturbation")

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()


def plot_binary_Ws(
        W_list : Sequence[Union[np.ndarray, torch.Tensor]],
        labels : Sequence[str] | None = None,
        cmap   : str = "Greys",                    # single–hue palette
        figsize: tuple[int, int] | None = None):
    """
    Plot a set of binary gate matrices (0/1) side-by-side.

    Parameters
    ----------
    W_list   : list of (P × d) tensors / arrays, values 0/1 or boolean.
    labels   : optional list of titles, one per matrix.
    cmap     : matplotlib colormap; default 'Greys' (white-black gradient).
    figsize  : overall figure size; defaults to (4 * n_runs, 4).
    """
    # -- helper ------------------------------------------------------
    def _to_numpy(w):
        return w.detach().cpu().numpy() if torch.is_tensor(w) else w

    # -- stack + align ----------------------------------------------
    W_np  = [_to_numpy(w).astype(int) for w in W_list]
    n_run = len(W_np)
    labels = labels or [f"run-{i}" for i in range(n_run)]

    ref = W_np[0]
    W_aligned = [ref] + [_align_columns_binary(ref, w, maximise="intersection")
                         for w in W_np[1:]]

    # -- binary colour limits ---------------------------------------
    vmin, vmax = 0.0, 1.0

    # -- figure ------------------------------------------------------
    if figsize is None:
        figsize = (4 * n_run, 4)

    fig, axes = plt.subplots(1, n_run, figsize=figsize, squeeze=False)

    for ax, W, lbl in zip(axes[0], W_aligned, labels):
        sns.heatmap(
            W,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=False,
            linewidths=0.2,
            linecolor="lightgrey"
        )
        ax.set_title(lbl)
        ax.set_xlabel("latent dim")
        ax.set_ylabel("perturbation")

    plt.tight_layout()
    plt.show()
  
def plot_umaps_for_z_list(
    adata,                                
    z_list: Sequence[Union[np.ndarray]],  
    labels: Sequence[str] | None = None,  # panel titles; defaults to “z-0”, “z-1”, …
    color: str | list[str] = "top_sg",    # obs key(s) used for colouring
    n_neighbors: int = 15,
    figsize: tuple[int, int] | None = None,
):
    # convert all tensors to numpy once
    z_np = [np.asarray(z.detach().cpu() if hasattr(z, "detach") else z) for z in z_list]
    n_z  = len(z_np)
    if labels is None:
        labels = [f"z-{i}" for i in range(n_z)]
    if figsize is None:
        figsize = (4 * n_z, 4)

    fig, axes = plt.subplots(1, n_z, figsize=figsize, squeeze=False)

    for ax, z, label in zip(axes[0], z_np, labels):
        z_data = adata.copy()
        z_data.obsm["z"] = z

        sc.pp.neighbors(z_data, use_rep="z", n_neighbors=n_neighbors)
        sc.tl.umap(z_data, random_state=0)                 # deterministic layout
        sc.pl.umap(z_data,
                   color=color,
                   frameon=False,
                   ax=ax,
                   show=False)                              # draw on given axis
        ax.set_title(label)

    plt.tight_layout()
    plt.show()



def _procrustes_align(
    X: np.ndarray,                       # reference  (n × d)
    Y: np.ndarray,                       # target     (n × d)
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    # centre (important for Procrustes)
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)

    # orthogonal part
    C          = Yc.T @ Xc
    U, _, Vt   = svd(C, full_matrices=False)
    Q          = U @ Vt                            # rotation / reflection

    # optional isotropic scaling
    if scale:
        s = np.trace((Xc).T @ Yc @ Q) / np.trace((Yc @ Q).T @ (Yc @ Q))
    else:
        s = 1.0

    Y_aligned = s * Y @ Q          # NOTE: use *original* Y to keep means intact
    return Y_aligned, s * Q        # store combined transform

def _permute_columns(
    X: np.ndarray,    # reference (n × d)
    Y: np.ndarray,    # target    (n × d)
    metric: Literal["cosine", "pearson"] = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (Y_perm, P) where Y_perm = Y[:, perm] and P is the permutation matrix.
    """
    if metric == "cosine":
        S = np.abs(X.T @ Y) / (
            np.linalg.norm(X, axis=0)[:, None] * np.linalg.norm(Y, axis=0)[None, :]
        )
    else:  # pearson
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        S = np.abs(Xc.T @ Yc) / (
            np.linalg.norm(Xc, axis=0)[:, None] * np.linalg.norm(Yc, axis=0)[None, :]
        )

    row, col = linear_sum_assignment(-S)           # maximise similarity
    P        = np.eye(X.shape[1])[:, col]          # permutation matrix (d × d)
    return Y @ P, P

def align_z_list(
    z_list : Sequence[Array],
    ref_idx: int = 0,
    mode   : Literal["procrustes", "permute", "permute+procrustes"] = "procrustes",
    scale  : bool = True,
    perm_metric: Literal["cosine","pearson"] = "cosine",
    return_transforms: bool = False,
) -> Union[
    List[np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray]]
]:
    Z       = [_to_numpy(z) for z in z_list]
    ref     = Z[ref_idx]
    n_runs  = len(Z)
    aligned = []
    Ts      = []

    for i, Y in enumerate(Z):
        if i == ref_idx:
            aligned.append(Y.copy())
            Ts.append(np.eye(Y.shape[1]))
            continue

        # 1. optional permutation
        if "permute" in mode:
            Y_perm, P = _permute_columns(ref, Y, metric=perm_metric)
        else:
            Y_perm, P = Y, np.eye(Y.shape[1])

        # 2. optional Procrustes
        if "procrustes" in mode:
            Y_aln, Q = _procrustes_align(ref, Y_perm, scale=scale)
        else:
            Y_aln, Q = Y_perm, np.eye(Y_perm.shape[1])

        aligned.append(Y_aln)
        Ts.append(P @ Q)                         # note: apply permutation then rotation

    return (aligned, Ts) if return_transforms else aligned


def get_bipartite_graph(mode : Literal["weighted", "threshold", "fdr"] = "weighted",
                        threshold : float = 0.5,
                        fdr_alpha : float = 0.05) -> torch.Tensor:
    q_pi = pyro.param("q_pi").detach().cpu()      # (P, d)
    if mode == "weighted":
        return q_pi

    if mode == "threshold":
        return (q_pi >= threshold).byte()

    if mode == "fdr":
        p_vals = (1.0 - q_pi).flatten().numpy() 
        _, rejected, _, _ = multipletests(
            p_vals,
            alpha=fdr_alpha,
            method="fdr_bh"
        )
        mask = torch.as_tensor(rejected, dtype=torch.uint8).reshape_as(q_pi)
        return mask