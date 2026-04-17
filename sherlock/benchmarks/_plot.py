from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def plot_metric_summary(results: list[dict], metric: str = "r2", title: str | None = None):
    df = pd.DataFrame(results)

    plt.figure(figsize=(6, 4))
    plt.bar(df["model"], df[metric])
    plt.ylabel(metric.upper())
    plt.title(title or f"Benchmark comparison ({metric})")
    plt.tight_layout()
    plt.show()


def plot_per_perturbation_bar(
    results_df: pd.DataFrame,
    metric: str = "r2",
    title: str | None = None,
    sort_desc: bool = True,
):
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results DataFrame")

    df = results_df.copy()
    df = df.sort_values(metric, ascending=not sort_desc)

    plt.figure(figsize=(10, 4))
    plt.bar(df["perturbation"], df[metric])
    plt.xticks(rotation=90)
    plt.ylabel(metric.upper())
    plt.title(title or f"Per-perturbation {metric}")
    plt.tight_layout()
    plt.show()


def plot_r2_scatter(
    true_x,
    pred_x,
    title: str = "Prediction vs truth",
    max_points: int = 1000,
    seed: int = 0,
):
    true_x = np.asarray(true_x)
    pred_x = np.asarray(pred_x)

    if true_x.shape != pred_x.shape:
        raise ValueError(f"Shape mismatch: true {true_x.shape}, pred {pred_x.shape}")

    rng = np.random.default_rng(seed)

    n = true_x.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        true_x = true_x[idx]
        pred_x = pred_x[idx]

    true_vals = true_x.reshape(-1)
    pred_vals = pred_x.reshape(-1)

    r2 = r2_score(true_vals, pred_vals)

    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())

    plt.figure(figsize=(8, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.5)
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{title}\nR² = {r2:.2f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_curves(
    train_loss,
    val_loss=None,
    title: str = "Training / validation loss",
):
    """
    Plot training loss and optional validation loss across epochs.
    """
    train = np.asarray(train_loss, dtype=float)

    plt.figure(figsize=(6, 4))
    plt.plot(train, label="train_loss")

    if val_loss is not None:
        val = np.asarray(val_loss, dtype=float)
        plt.plot(val, label="val_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_umap_three_way(
    ctrl_adata,
    real_adata,
    pred_adata,
    title: str = "UMAP: control vs real vs predicted",
):
    """
    Build one combined AnnData and plot a UMAP comparing:
    - control cells
    - real perturbed cells
    - predicted perturbed cells
    """
    import anndata as ad
    import scanpy as sc

    ctrl = ctrl_adata.copy()
    real = real_adata.copy()
    pred = pred_adata.copy()

    ctrl.obs["source"] = "control"
    real.obs["source"] = "real_perturbed"
    pred.obs["source"] = "predicted_perturbed"

    combined = ad.concat([ctrl, real, pred], join="inner", merge="same")
    combined.obs_names_make_unique()

    sc.pp.pca(combined)
    sc.pp.neighbors(combined)
    sc.tl.umap(combined)

    sc.pl.umap(combined, color="source", title=title)
    return combined