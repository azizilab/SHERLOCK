"""Label preparation and treatment-effect computation on a tiny synthetic dataset."""

import anndata as ad
import numpy as np
import pandas as pd

import sherlock as slk


def _toy(n_per=20, n_genes=15, seed=0):
    rng = np.random.default_rng(seed)
    labels = ["ctrl"] * n_per + ["A"] * n_per + ["B"] * n_per
    X = rng.poisson(5, size=(len(labels), n_genes)).astype(np.float32)
    X[n_per:2 * n_per, 0] += 20  # perturbation A raises gene 0
    obs = pd.DataFrame({"guide": labels}, index=[f"c{i}" for i in range(len(labels))])
    return ad.AnnData(X=X, obs=obs)


def test_prepare_and_treat_effect():
    slk.configs.reset_config()
    data = _toy()
    slk.pp.slk_prepare_data(data, pert_key="guide", ntc_label="ctrl", inplace=True)
    pk, ntc = slk.configs.get_config("pert_key"), slk.configs.get_config("ntc_label")
    assert set(data.obs[pk]) == {ntc, "A", "B"}

    slk.pp.treat_effect(data, label_col=pk, control_label=ntc, method="mean", inplace=True)
    assert "treat_effect" in data.uns
