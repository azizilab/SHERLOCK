from anndata import AnnData
from .._configs import get_config
from typing import Any, Dict, List, Literal, Optional
import scipy as sp
import numpy as np
from tqdm import tqdm
import pandas as pd

def slk_prepare_data(data: AnnData, pert_key: str, ntc_label: str, treatment_key: str = None, untreated_label: str = None, inplace: bool = True, force: bool = False) -> AnnData:

    if not inplace:
        data = data.copy()

    model_treatment_key = get_config('treatment_key')
    model_untreated_label = get_config('untreated_label')
    model_pert_key = get_config('pert_key')

    if not force:
        if model_treatment_key in data.obs.columns and model_treatment_key != treatment_key:
            raise ValueError(f"Column '{model_treatment_key}' already exists in data.obs. Rename the existing column or use force=True to overwrite.")
        if model_pert_key in data.obs.columns and model_pert_key != pert_key:
            raise ValueError(f"Column '{model_pert_key}' already exists in data.obs. Rename the existing column or use force=True to overwrite.")

    if treatment_key is None:
        data.obs[model_treatment_key] = model_untreated_label
    else:
        data.obs[model_treatment_key] = data.obs[treatment_key].astype(str)

        untreated_idx = data.obs[treatment_key] == untreated_label
        assert sum(untreated_idx) > 0, f"No cells with {treatment_key} == {untreated_label} found."
        data.obs.loc[untreated_idx, model_treatment_key] = model_untreated_label

    model_ntc_label = get_config('ntc_label')

    data.obs[model_pert_key] = data.obs[pert_key].astype(str)

    ntc_idx = data.obs[pert_key] == ntc_label
    assert sum(ntc_idx) > 0, f"No cells with {pert_key} == {ntc_label} found."
    data.obs.loc[ntc_idx, model_pert_key] = model_ntc_label

    return data

def treat_effect(
    adata: AnnData,
    label_col: str,
    control_label: Any,
    method: Literal["mean", "perturbseq"],
    compute_fdr: bool = False,
    inplace: bool = True,
    key: str = 'treat_effect'
) -> AnnData:
    """

    Parameters
    ----------
    adata: AnnData containing observations and annotated perturbations. Observations
            should be in X
    label_col: column in adata obs dataframe with labels of unique perturbations to
            compute average treatment effects
    control_label: value in label_col to compute effects relative to
    method: method key for computing average treatment effect. "mean" is standard average
            effect, "perturbseq" is effect after normalizing for library size and applying log
    compute_fdr: whether to compute false discovery rate


    Returns
    -------
    AnnData with average treatment effects in X, obs index with perturbation annotations,
    and control perturbation in uns
    """
    if compute_fdr:
        raise NotImplementedError

    valid_methods = ["mean", "perturbseq"]
    assert method in valid_methods, f"Method must be one of {valid_methods}"

    perturbations = adata.obs[label_col].unique()
    assert control_label in perturbations
    alt_labels = [x for x in perturbations if x != control_label]

    X_control = adata[adata.obs[label_col] == control_label].X
    if sp.sparse.issparse(X_control):
        X_control = X_control.toarray()

    if method == "perturbseq":
        X_control = 1e4 * X_control / np.sum(X_control, axis=1, keepdims=True)
        X_control = np.log2(X_control + 1)
    X_control_mean = X_control.mean(0)

    average_effects = []

    for alt_label in tqdm(alt_labels):
        X_alt = adata[adata.obs[label_col] == alt_label].X
        if sp.sparse.issparse(X_alt):
            X_alt = X_alt.toarray()
        if method == "perturbseq":
            X_alt = 1e4 * X_alt / np.sum(X_alt, axis=1, keepdims=True)
            X_alt = np.log2(X_alt + 1)
        X_alt_mean = X_alt.mean(0)
        average_effects.append(X_alt_mean - X_control_mean)

    average_effects = np.stack(average_effects)
    results = AnnData(
        obs=pd.DataFrame(index=alt_labels),
        X=average_effects,
        var=adata.var.copy(),
        uns=dict(control=control_label),
    )

    if inplace:
        adata.uns[key] = results
    else:
        return results