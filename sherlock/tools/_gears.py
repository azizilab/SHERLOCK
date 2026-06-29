from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .._configs import get_config


@dataclass
class GEARSRun:
    """Container returned by GEARS training helpers."""

    pert_data: Any
    model: Any
    adata: Any
    predictions: pd.DataFrame | None = None
    gi: pd.DataFrame | None = None


def _require_gears():
    try:
        from gears import GEARS, PertData
    except ImportError as exc:
        raise ImportError(
            "GEARS is an optional dependency. Install the official package with "
            "`pip install cell-gears` after installing PyTorch Geometric for your "
            "PyTorch/CUDA stack."
        ) from exc
    return PertData, GEARS


def _patch_gears_filter_pert_in_go(PertData=None) -> bool:
    """
    Patch a GEARS helper that assumes every non-control condition contains '+'.

    Some custom datasets store singles as "GENE" rather than "GENE+ctrl".
    Official GEARS then crashes while checking GO-graph membership for singles.
    This patch preserves the intended behavior:
      - ctrl is valid
      - a single is valid iff it is in pert_names
      - a combo is valid iff all non-ctrl parts are in pert_names
    """

    def safe_filter_pert_in_go(condition, pert_names):
        condition = str(condition)
        if condition == "ctrl":
            return True
        parts = [part.strip() for part in condition.split("+") if part.strip()]
        if not parts:
            return False
        return all(part == "ctrl" or part in pert_names for part in parts)

    patched = False
    try:
        import sys

        pertdata_module = sys.modules.get("gears.pertdata")
        if pertdata_module is None:
            import gears.pertdata as pertdata_module
        pertdata_module.filter_pert_in_go = safe_filter_pert_in_go
        patched = True
    except Exception:
        pass

    if PertData is not None:
        try:
            PertData.load.__globals__["filter_pert_in_go"] = safe_filter_pert_in_go
            patched = True
        except Exception:
            pass

    return patched


def _filter_loaded_gears_cache(pert_data, *, verbose: bool = True) -> None:
    """Prune cached GEARS graph objects that are no longer present in adata."""

    if not hasattr(pert_data, "adata") or pert_data.adata is None:
        return

    valid_conditions = set(pert_data.adata.obs["condition"].astype(str).unique())
    dataset_processed = getattr(pert_data, "dataset_processed", None)
    if isinstance(dataset_processed, dict):
        before = len(dataset_processed)
        pert_data.dataset_processed = {
            str(cond): graphs
            for cond, graphs in dataset_processed.items()
            if str(cond) in valid_conditions
        }
        removed = before - len(pert_data.dataset_processed)
        if removed and verbose:
            print(f"[GEARS] removed {removed} cached graph conditions absent from the GO-filtered AnnData")


def _patch_pandas_series_nonzero() -> bool:
    """
    Patch pandas/scipy sparse boolean indexing compatibility used by GEARS.

    Some GEARS versions index sparse AnnData matrices with a pandas Series:
    `adata.X[adata.obs.condition == "ctrl"]`. Newer scipy sparse indexers call
    `.nonzero()` on the boolean mask, but pandas Series does not expose that
    method. Adding this tiny compatibility method avoids densifying `adata.X`.
    """

    try:
        import pandas as pd

        if not hasattr(pd.Series, "nonzero"):
            pd.Series.nonzero = lambda self: self.to_numpy().nonzero()
        return True
    except Exception:
        return False


def _ensure_gears_legacy_cache_dirs() -> None:
    """Create legacy relative cache directories used by upstream GEARS."""

    Path("./data").mkdir(parents=True, exist_ok=True)


@contextmanager
def _gears_epoch_only_logging(enabled: bool = True):
    """Suppress GEARS batch-step logs while keeping epoch-level messages."""

    if not enabled:
        yield
        return

    try:
        import gears.gears as gears_module
        import gears.utils as utils_module
    except Exception:
        yield
        return

    original_gears_print = getattr(gears_module, "print_sys", None)
    original_utils_print = getattr(utils_module, "print_sys", None)

    def filtered_print_sys(message):
        text = str(message)
        if text.startswith("Epoch ") and " Step " in text:
            return
        if original_utils_print is not None:
            original_utils_print(message)
        elif original_gears_print is not None:
            original_gears_print(message)

    gears_module.print_sys = filtered_print_sys
    try:
        yield
    finally:
        if original_gears_print is not None:
            gears_module.print_sys = original_gears_print
        if original_utils_print is not None:
            utils_module.print_sys = original_utils_print


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def canonical_combo(combo: str, sep: str = "+") -> str:
    """Canonicalize a combo label by sorting its two perturbation names."""

    parts = [part.strip() for part in str(combo).split(sep) if part.strip()]
    if len(parts) <= 1:
        return str(combo).strip()
    return sep.join(sorted(parts))


def split_combo(combo: str, sep: str = "+") -> list[str]:
    """Split a perturbation label into GEARS perturbation-set format."""

    combo = str(combo).strip()
    if combo in {"", "nan", "None"}:
        return []
    return [part.strip() for part in combo.split(sep) if part.strip()]


def prepare_gears_adata(
    adata,
    *,
    pert_key: str | None = None,
    ntc_label: str | None = None,
    condition_key: str = "condition",
    cell_type_key: str = "cell_type",
    gene_name_key: str = "gene_name",
    gears_control_label: str = "ctrl",
    default_cell_type: str = "cell",
    copy: bool = True,
):
    """
    Return a GEARS-compatible AnnData object.

    GEARS expects:
      - `adata.obs["condition"]`: perturbation label, with control as "ctrl"
      - `adata.obs["cell_type"]`: a single cell-type label for one-context runs
      - `adata.var["gene_name"]`: gene symbols matching perturbation names
    """

    pert_key = pert_key or get_config("pert_key")
    ntc_label = ntc_label or get_config("ntc_label")
    out = adata.copy() if copy else adata

    if pert_key not in out.obs:
        raise KeyError(f"Expected perturbation column {pert_key!r} in adata.obs.")

    condition = out.obs[pert_key].astype(str).copy()
    condition.loc[condition == str(ntc_label)] = gears_control_label
    out.obs[condition_key] = condition

    if cell_type_key not in out.obs:
        out.obs[cell_type_key] = default_cell_type

    if gene_name_key not in out.var:
        out.var[gene_name_key] = out.var_names.astype(str)

    return out


def get_single_perturbations(
    adata,
    *,
    pert_key: str | None = None,
    ntc_label: str | None = None,
    sep: str = "+",
) -> list[str]:
    """Return sorted single perturbation names from an AnnData obs perturb column."""

    pert_key = pert_key or get_config("pert_key")
    ntc_label = ntc_label or get_config("ntc_label")
    perts = pd.Series(adata.obs[pert_key].astype(str).unique())
    singles = [
        pert
        for pert in perts
        if sep not in pert and pert != str(ntc_label) and pert not in {"ctrl", "control"}
    ]
    return sorted(singles)


def get_observed_combinations(
    adata,
    *,
    pert_key: str | None = None,
    ntc_label: str | None = None,
    sep: str = "+",
) -> list[str]:
    """Return sorted observed combinatorial perturbation labels."""

    pert_key = pert_key or get_config("pert_key")
    ntc_label = ntc_label or get_config("ntc_label")
    perts = pd.Series(adata.obs[pert_key].astype(str).unique())
    combos = [
        canonical_combo(pert, sep=sep)
        for pert in perts
        if sep in pert and pert != str(ntc_label)
    ]
    return sorted(set(combos))


def make_all_pair_combinations(
    genes: Sequence[str],
    *,
    sep: str = "+",
    include_self: bool = False,
) -> list[str]:
    """Build canonical pair labels for all requested perturbation genes."""

    genes = sorted({str(g).strip() for g in genes if str(g).strip()})
    if include_self:
        pairs = [(a, b) for i, a in enumerate(genes) for b in genes[i:]]
    else:
        pairs = combinations(genes, 2)
    return [sep.join(pair) for pair in pairs]


def _filter_valid_prediction_combos(
    combos: Sequence[str],
    *,
    valid_perturbs: set[str] | None = None,
    sep: str = "+",
) -> tuple[list[str], list[str], set[str], list[str]]:
    """Canonicalize requested pairs and drop combos GEARS cannot predict."""

    valid_combos: list[str] = []
    skipped_combos: list[str] = []
    malformed_combos: list[str] = []
    missing_genes: set[str] = set()
    seen: set[str] = set()
    valid_perturbs = valid_perturbs or set()

    for requested in combos:
        combo = canonical_combo(requested, sep=sep)
        parts = split_combo(combo, sep=sep)
        if len(parts) != 2:
            skipped_combos.append(combo)
            malformed_combos.append(combo)
            continue

        missing = [part for part in parts if valid_perturbs and part not in valid_perturbs]
        if missing:
            skipped_combos.append(combo)
            missing_genes.update(missing)
            continue

        if combo not in seen:
            valid_combos.append(combo)
            seen.add(combo)

    return valid_combos, skipped_combos, missing_genes, malformed_combos


def _as_gears_perturbation_sets(combos: Iterable[str], sep: str = "+") -> list[list[str]]:
    return [split_combo(combo, sep=sep) for combo in combos]


def _prediction_to_frame(prediction: Any, combos: Sequence[str]) -> pd.DataFrame:
    """
    Normalize GEARS prediction output into a dataframe when possible.

    GEARS versions differ in return types. This keeps the raw object accessible
    while still giving notebook-friendly metadata.
    """

    rows = [{"combination": canonical_combo(combo), "prediction": None} for combo in combos]
    pred_df = pd.DataFrame(rows)
    pred_df.attrs["raw_prediction"] = prediction

    if isinstance(prediction, pd.DataFrame):
        return prediction.copy()

    if isinstance(prediction, dict):
        recs: list[dict[str, Any]] = []
        for key, value in prediction.items():
            recs.append({"combination": canonical_combo(str(key)), "prediction": value})
        out = pd.DataFrame(recs)
        out.attrs["raw_prediction"] = prediction
        return out

    if isinstance(prediction, (list, tuple)) and len(prediction) == len(combos):
        pred_df["prediction"] = list(prediction)

    return pred_df


def _gi_to_frame(gi_result: Any, combo: str) -> pd.DataFrame:
    """Normalize one GEARS GI_predict return value into one or more rows."""

    combo = canonical_combo(combo)
    if isinstance(gi_result, pd.DataFrame):
        out = gi_result.copy()
        if "combination" not in out:
            out.insert(0, "combination", combo)
        return out

    if isinstance(gi_result, dict):
        row = {"combination": combo}
        row.update(gi_result)
        return pd.DataFrame([row])

    if isinstance(gi_result, (list, tuple, np.ndarray)):
        return pd.DataFrame(
            [{"combination": combo, "gi_result": gi_result}]
        )

    return pd.DataFrame([{"combination": combo, "gi_result": gi_result}])


def _prediction_key(pert_set: Sequence[str]) -> str:
    return "_".join(pert_set)


def _compute_gears_gi_from_predictions(
    model,
    predictions: dict[str, Any],
    combos: Sequence[str],
    *,
    sep: str = "+",
    gi_genes_file: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute GEARS GI metrics from already cached/predicted expression."""

    from gears.utils import get_GI_genes_idx, get_mean_control
    from ._models import regress_gi_params

    mean_control = get_mean_control(model.adata).values
    if gi_genes_file is not None:
        gi_genes_idx = get_GI_genes_idx(model.adata, str(gi_genes_file))
    else:
        gi_genes_idx = np.arange(len(model.adata.var.gene_name.values))

    effects = {}
    for combo in combos:
        parts = split_combo(combo, sep=sep)
        if len(parts) != 2:
            continue
        combo_key = _prediction_key(parts)
        effects[parts[0]] = (np.asarray(predictions[parts[0]]) - mean_control)[gi_genes_idx]
        effects[parts[1]] = (np.asarray(predictions[parts[1]]) - mean_control)[gi_genes_idx]
        effects[canonical_combo(combo, sep=sep)] = (
            np.asarray(predictions[combo_key]) - mean_control
        )[gi_genes_idx]

    return regress_gi_params(
        effects,
        combos=list(combos),
        sep=sep,
        min_cells=0,
        progress=verbose,
    )


def train_gears(
    adata,
    *,
    data_dir: str | Path = "./gears_data",
    dataset_name: str = "sherlock_gears",
    pert_key: str | None = None,
    ntc_label: str | None = None,
    split: str = "all",
    seed: int = 1,
    batch_size: int = 32,
    test_batch_size: int = 128,
    hidden_size: int = 64,
    epochs: int = 20,
    device: str | None = None,
    gears_control_label: str = "ctrl",
    default_cell_type: str = "cell",
    model_path: str | Path | None = None,
    load_path: str | Path | None = None,
    force_reprocess: bool = False,
    pert_data_kwargs: dict[str, Any] | None = None,
    split_kwargs: dict[str, Any] | None = None,
    dataloader_kwargs: dict[str, Any] | None = None,
    model_initialize_kwargs: dict[str, Any] | None = None,
    train_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
    quiet_steps: bool = True,
) -> GEARSRun:
    """
    Process an AnnData object, train GEARS, and return the fitted model.

    This is intentionally thin over the official GEARS API so notebook code can
    still pass through advanced options without this wrapper chasing every
    upstream release.
    """

    PertData, GEARS = _require_gears()
    data_dir = Path(data_dir)
    device = device or _default_device()

    gears_adata = prepare_gears_adata(
        adata,
        pert_key=pert_key,
        ntc_label=ntc_label,
        gears_control_label=gears_control_label,
        default_cell_type=default_cell_type,
    )

    # For custom AnnData, avoid GEARS' large default perturbation graph pickle.
    # That external pickle can be brittle across numpy versions, and the local
    # graph is the more direct choice for dataset-specific Norman runs.
    pert_data_options = {"default_pert_graph": False}
    pert_data_options.update(pert_data_kwargs or {})
    pert_data = PertData(str(data_dir), **pert_data_options)
    dataset_dir = data_dir / dataset_name
    processed_h5ad = dataset_dir / "perturb_processed.h5ad"

    if processed_h5ad.exists() and not force_reprocess:
        if verbose:
            print(f"[GEARS] using cached processed dataset at {dataset_dir}")
    else:
        if verbose:
            action = "reprocessing" if force_reprocess else "processing"
            print(f"[GEARS] {action} AnnData into {dataset_dir}")
        pert_data.new_data_process(dataset_name=dataset_name, adata=gears_adata)

    patched_go_filter = _patch_gears_filter_pert_in_go(PertData)
    if verbose:
        status = "patched" if patched_go_filter else "not patched"
        print(f"[GEARS] GO membership filter for single perturbations: {status}")
    pert_data.load(data_path=str(dataset_dir))
    _filter_loaded_gears_cache(pert_data, verbose=verbose)

    if split in {"all", "all_data", "train_all"}:
        conditions = sorted(pert_data.adata.obs["condition"].astype(str).unique())
        train_conditions = conditions
        val_conditions = [cond for cond in conditions if cond != gears_control_label]
        if not val_conditions:
            val_conditions = train_conditions
        pert_data.split = "no_test"
        pert_data.seed = seed
        pert_data.subgroup = None
        pert_data.train_gene_set_size = 1.0
        pert_data.set2conditions = {
            "train": train_conditions,
            "val": val_conditions,
        }
    else:
        pert_data.prepare_split(split=split, seed=seed, **(split_kwargs or {}))

    pert_data.get_dataloader(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        **(dataloader_kwargs or {}),
    )

    patched_series_nonzero = _patch_pandas_series_nonzero()
    if verbose:
        status = "patched" if patched_series_nonzero else "not patched"
        print(f"[GEARS] pandas/scipy sparse boolean indexing compatibility: {status}")

    model = GEARS(pert_data, device=device)
    if load_path is not None:
        if verbose:
            print(f"[GEARS] loading pretrained model from {load_path}")
        model.load_pretrained(str(load_path))
    else:
        _ensure_gears_legacy_cache_dirs()
        if verbose:
            print(f"[GEARS] initializing model with hidden_size={hidden_size}")
        model.model_initialize(
            hidden_size=hidden_size,
            **(model_initialize_kwargs or {}),
        )
        if verbose:
            print(f"[GEARS] training for epochs={epochs}")
        with _gears_epoch_only_logging(enabled=quiet_steps):
            model.train(epochs=epochs, **(train_kwargs or {}))

    if model_path is not None:
        if verbose:
            print(f"[GEARS] saving model to {model_path}")
        model.save_model(str(model_path))

    return GEARSRun(pert_data=pert_data, model=model, adata=gears_adata)


def predict_gears_combinations(
    model,
    combos: Sequence[str],
    *,
    sep: str = "+",
    compute_gi: bool = True,
    gi_genes_file: str | Path | None = None,
    filter_invalid: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Predict expression and, optionally, GEARS GI summaries for combinations.

    Returns
    -------
    prediction_df
        Metadata-normalized GEARS `predict` output for requested combinations.
        The raw object includes both singles and combinations and is stored in
        `prediction_df.attrs["raw_prediction"]`.
    gi_df
        GEARS GI scores computed from the batched prediction output.
    """

    requested_combos = [canonical_combo(combo, sep=sep) for combo in combos]
    valid_perturbs = set(getattr(model, "pert_list", []) or [])
    if filter_invalid and valid_perturbs:
        combos, skipped_combos, missing_genes, malformed_combos = _filter_valid_prediction_combos(
            requested_combos,
            valid_perturbs=valid_perturbs,
            sep=sep,
        )
    else:
        combos, skipped_combos, missing_genes, malformed_combos = _filter_valid_prediction_combos(
            requested_combos,
            sep=sep,
        )

    if skipped_combos and verbose:
        preview = ", ".join(sorted(missing_genes)[:20])
        suffix = "" if len(missing_genes) <= 20 else f", ... ({len(missing_genes)} genes total)"
        if missing_genes:
            print(
                f"[GEARS] skipping {len(skipped_combos)} requested combos containing "
                f"perturbations absent from the GO graph: {preview}{suffix}"
            )
        if malformed_combos:
            print(
                f"[GEARS] skipping {len(malformed_combos)} malformed requested combos "
                f"(expected exactly two perturbations separated by {sep!r})"
            )

    if not combos:
        prediction_df = pd.DataFrame(
            {
                "combination": requested_combos,
                "skipped": [combo in set(skipped_combos) for combo in requested_combos],
            }
        )
        prediction_df.attrs["skipped_combos"] = skipped_combos
        prediction_df.attrs["missing_genes"] = sorted(missing_genes)
        prediction_df.attrs["malformed_combos"] = malformed_combos
        return prediction_df, None

    combo_sets = _as_gears_perturbation_sets(combos, sep=sep)
    single_genes = sorted({gene for pert_set in combo_sets for gene in pert_set})
    single_sets = [[gene] for gene in single_genes]
    perturbation_sets = single_sets + combo_sets

    if verbose:
        print(
            f"[GEARS] predicting {len(single_sets)} singles and "
            f"{len(combo_sets)} combinations once for GI scoring"
        )

    prediction = model.predict(perturbation_sets)
    combo_prediction = {
        _prediction_key(pert_set): prediction[_prediction_key(pert_set)]
        for pert_set in combo_sets
        if _prediction_key(pert_set) in prediction
    }
    prediction_df = _prediction_to_frame(combo_prediction, combos)
    prediction_df.attrs["skipped_combos"] = skipped_combos
    prediction_df.attrs["missing_genes"] = sorted(missing_genes)
    prediction_df.attrs["malformed_combos"] = malformed_combos
    prediction_df.attrs["requested_combos"] = requested_combos
    prediction_df.attrs["valid_combos"] = combos
    prediction_df.attrs["raw_prediction"] = prediction

    gi_df = None
    if compute_gi:
        predicted_combos = [
            combo
            for combo in combos
            if _prediction_key(split_combo(combo, sep=sep)) in prediction
        ]
        missing_prediction_combos = sorted(set(combos) - set(predicted_combos))
        if missing_prediction_combos and verbose:
            print(
                f"[GEARS] skipping {len(missing_prediction_combos)} combos missing "
                "from GEARS prediction output"
            )
        gi_df = _compute_gears_gi_from_predictions(
            model,
            prediction,
            predicted_combos,
            sep=sep,
            gi_genes_file=gi_genes_file,
            verbose=verbose,
        )
        gi_df.attrs["skipped_combos"] = skipped_combos + missing_prediction_combos
        gi_df.attrs["missing_genes"] = sorted(missing_genes)
        gi_df.attrs["malformed_combos"] = malformed_combos

    return prediction_df, gi_df


def run_gears_gi(
    adata,
    *,
    genes: Sequence[str] | None = None,
    combos: Sequence[str] | None = None,
    observed_combos: bool = False,
    include_self: bool = False,
    sep: str = "+",
    **train_kwargs,
) -> GEARSRun:
    """
    Notebook-friendly one-call GEARS training plus all-combination GI prediction.

    By default, this predicts all pairwise combinations among observed single
    perturbations. Pass `combos=` to use a custom list or `observed_combos=True`
    to restrict to combinations already present in the AnnData.
    """

    pert_key = train_kwargs.get("pert_key")
    ntc_label = train_kwargs.get("ntc_label")

    run = train_gears(adata, **train_kwargs)

    if combos is None:
        if observed_combos:
            combos = get_observed_combinations(
                adata,
                pert_key=pert_key,
                ntc_label=ntc_label,
                sep=sep,
            )
        else:
            if genes is None:
                genes = get_single_perturbations(
                    adata,
                    pert_key=pert_key,
                    ntc_label=ntc_label,
                    sep=sep,
                )
            combos = make_all_pair_combinations(genes, sep=sep, include_self=include_self)

    predictions, gi = predict_gears_combinations(
        run.model,
        combos,
        sep=sep,
        verbose=train_kwargs.get("verbose", True),
    )
    run.predictions = predictions
    run.gi = gi
    return run
