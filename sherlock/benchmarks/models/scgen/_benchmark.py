from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable

from sherlock.benchmarks._types import BenchmarkResult
from ._data_preprocessing import preprocess_for_scgen
from ._trainer import train_scgen
from ._predict import predict_scgen_manual


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch in Pearson calc: {x.shape} vs {y.shape}")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def _safe_r2(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    x_true = np.asarray(x_true).ravel()
    x_pred = np.asarray(x_pred).ravel()

    if x_true.size == 0 or x_pred.size == 0:
        return float("nan")
    if x_true.shape != x_pred.shape:
        raise ValueError(f"Shape mismatch in R2 calc: {x_true.shape} vs {x_pred.shape}")

    ss_res = np.sum((x_true - x_pred) ** 2)
    ss_tot = np.sum((x_true - np.mean(x_true)) ** 2)

    if ss_tot < 1e-12:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


def _get_gt_effect_for_pert(
    treat_effect,
    pert: str,
    var_names: Iterable[str],
) -> np.ndarray | None:
    if treat_effect is None or pert not in treat_effect.obs_names:
        return None

    gt_row = treat_effect[pert].copy()
    shared_genes = [g for g in var_names if g in gt_row.var_names]

    if len(shared_genes) == 0:
        return None

    gt_row = gt_row[:, shared_genes]
    return np.asarray(gt_row.X).ravel()


def run_single_holdout(
    data_scgen,
    test_pert: str,
    *,
    pert_col: str = "pert",
    control_label: str = "NTC",
    train_frac: float = 0.8,
    seed: int = 0,
    n_epochs: int = 20,
    treat_effect=None,
):
    rng = np.random.default_rng(seed)

    pert_vals = data_scgen.obs[pert_col].astype(str).values
    is_ctrl = pert_vals == control_label
    is_test_pert = pert_vals == test_pert

    if is_ctrl.sum() == 0:
        raise ValueError("No control cells found.")
    if is_test_pert.sum() == 0:
        raise ValueError(f"No cells found for held-out perturbation {test_pert!r}")

    ctrl_idx = np.where(is_ctrl)[0]
    pert_idx = np.where(is_test_pert)[0]

    rng.shuffle(ctrl_idx)
    rng.shuffle(pert_idx)

    n_ctrl_train = max(1, int(round(train_frac * len(ctrl_idx))))
    n_pert_train = max(1, int(round(train_frac * len(pert_idx))))

    ctrl_train_idx = ctrl_idx[:n_ctrl_train]
    ctrl_test_idx = ctrl_idx[n_ctrl_train:]
    pert_train_idx = pert_idx[:n_pert_train]
    pert_test_idx = pert_idx[n_pert_train:]

    if len(ctrl_test_idx) == 0:
        raise ValueError(f"No held-out control cells left for {test_pert!r}")
    if len(pert_test_idx) == 0:
        raise ValueError(f"No held-out perturbation cells left for {test_pert!r}")

    # train on held-in controls + held-in cells for this perturbation
    train_mask = np.zeros(data_scgen.n_obs, dtype=bool)
    train_mask[ctrl_train_idx] = True
    train_mask[pert_train_idx] = True

    train_data = data_scgen[train_mask].copy()
    real_test = data_scgen[pert_test_idx].copy()
    ctrl_test = data_scgen[ctrl_test_idx].copy()

    model = train_scgen(
        train_data,
        pert_col=pert_col,
        control_label=control_label,
        n_epochs=n_epochs,
    )

    pred_adata, delta = predict_scgen_manual(
        model,
        ctrl_key=control_label,
        stim_key=test_pert,
        condition_key="condition",
        cell_type_key="cell_type",
        celltype_to_predict="all_cells",
    )

    pred_mean = np.asarray(pred_adata.X.mean(axis=0)).ravel()
    real_mean = np.asarray(real_test.X.mean(axis=0)).ravel()
    ctrl_mean = np.asarray(ctrl_test.X.mean(axis=0)).ravel()

    r2 = _safe_r2(real_mean, pred_mean)
    pearson = _safe_pearson(real_mean, pred_mean)

    ate = float("nan")
    if treat_effect is not None:
        pred_effect = pred_mean - ctrl_mean
        pred_gene_names = list(pred_adata.var_names)

        gt_effect = _get_gt_effect_for_pert(
            treat_effect=treat_effect,
            pert=test_pert,
            var_names=pred_gene_names,
        )

        if gt_effect is not None:
            shared_genes = [g for g in pred_gene_names if g in treat_effect.var_names]
            gene_to_idx = {g: i for i, g in enumerate(pred_gene_names)}
            pred_effect_aligned = np.asarray([pred_effect[gene_to_idx[g]] for g in shared_genes])

            if pred_effect_aligned.shape != gt_effect.shape:
                raise ValueError(
                    f"ATE alignment mismatch for {test_pert}: "
                    f"{pred_effect_aligned.shape} vs {gt_effect.shape}"
                )

            ate = _safe_pearson(gt_effect, pred_effect_aligned)

    return {
        "perturbation": test_pert,
        "r2": r2,
        "pearson": pearson,
        "ate": ate,
        "n_train_pert": int(len(pert_train_idx)),
        "n_test_pert": int(len(pert_test_idx)),
        "n_train_ntc": int(len(ctrl_train_idx)),
        "n_test_ntc": int(len(ctrl_test_idx)),
        "delta_norm": float(np.linalg.norm(delta)),
    }


def run_scgen_benchmark(
    adata,
    *,
    pert_col: str,
    control_label: str,
    holdout_perts: list[str] | None = None,
    min_cells: int = 30,
    target_sum: float = 1e4,
    train_frac: float = 0.8,
    seed: int = 0,
    n_epochs: int = 20,
    treat_effect=None,
) -> BenchmarkResult:
    data_scgen = preprocess_for_scgen(
        adata,
        pert_col=pert_col,
        control_label=control_label,
        min_cells=min_cells,
        target_sum=target_sum,
        inplace=False,
    )

    counts = data_scgen.obs[pert_col].astype(str).value_counts()

    if holdout_perts is None:
        selected_perts = [
            p for p, n in counts.items()
            if p != control_label and n >= min_cells
        ]
    else:
        selected_perts = [
            p for p in holdout_perts
            if p != control_label and int(counts.get(p, 0)) >= min_cells
        ]

    rows = []
    failures = []

    for pert in selected_perts:
        try:
            rows.append(
                run_single_holdout(
                    data_scgen,
                    pert,
                    pert_col=pert_col,
                    control_label=control_label,
                    train_frac=train_frac,
                    seed=seed,
                    n_epochs=n_epochs,
                    treat_effect=treat_effect,
                )
            )
        except Exception as e:
            failures.append({"perturbation": pert, "error": repr(e)})

    if len(rows) == 0:
        raise RuntimeError(
            "scGen benchmark produced no successful holdout runs. "
            f"Failures: {failures}"
        )

    results_df = pd.DataFrame(rows)

    return BenchmarkResult(
        model_name="scgen",
        model=None,
        metrics={
            "r2": float(results_df["r2"].mean()),
            "ate": float(results_df["ate"].mean()),
            "val_loss": float("nan"),
        },
        extras={
            "per_perturbation": results_df,
            "failures": (
                pd.DataFrame(failures)
                if len(failures) > 0
                else pd.DataFrame(columns=["perturbation", "error"])
            ),
            "selected_holdouts": selected_perts,
            "n_successful": int(len(results_df)),
            "n_failed": int(len(failures)),
            "pearson_mean": float(results_df["pearson"].mean()),
        },
    )