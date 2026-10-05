from __future__ import annotations

from ._single import run, eval, run_single, eval_single
from ._IO import load_results, save_results
from ._clustering import cluster_rho, group2name
from ._datasets import PerturbMatchingDataset, PerturbSimpleDataset
from ._models import VAE, regress_gi_params, classify_gi, build_treat_effect_map, build_go_term_map
from ._trainers import VAETrainer, cVAETrainer, sVAETrainer, SCGENTrainer
from ._analysis import ev_sig, compute_cf_ate_metrics, compute_embedding_metrics, compute_clustering_metrics, evaluate_model
from ._gears import (
    GEARSRun,
    get_observed_combinations,
    prepare_gears_adata,
    train_gears,
    predict_gears_combinations,
    run_gears_gi,
)

# Baseline models (cVAE, sVAE, contrastiveVI, scGen) depend on scvi-tools, which is an
# optional dependency (`pip install azizilab-sherlock[benchmarks]`). They are imported on
# first use so that `import sherlock` works without it.
_BASELINES = {
    "cVAE": ("._cvae", "cVAE"),
    "sVAE": ("._svae", "sVAE"),
    "ContrastiveVIModel": ("._contrastivevi", "ContrastiveVI"),
    "SCGENModel": ("._scgen", "SCGENModel"),
}


def __getattr__(name):
    if name in _BASELINES:
        import importlib

        module, attr = _BASELINES[name]
        try:
            value = getattr(importlib.import_module(module, __name__), attr)
        except ImportError as err:
            raise ImportError(
                f"sherlock.tl.{name} is a benchmarking baseline and needs scvi-tools. "
                "Install it with `pip install azizilab-sherlock[benchmarks]`."
            ) from err
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
