from __future__ import annotations

from ._single import run, eval, run_single, eval_single
from ._IO import load_results, save_results
from ._clustering import cluster_rho, group2name
from ._datasets import PerturbMatchingDataset, PerturbSimpleDataset
from ._models import VAE, regress_gi_params, classify_gi, build_treat_effect_map, build_go_term_map
from ._cvae import cVAE
from ._svae import sVAE
from ._contrastivevi import ContrastiveVI as ContrastiveVIModel
from ._scgen import SCGENModel
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
