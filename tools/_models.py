from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
import numpy as np

from ._base import PerturbModelBase
from .._configs import get_config


def regress_gi_params(
    effects: "dict[str, np.ndarray]",
    combos: "list[str] | None" = None,
    *,
    sep: str = "+",
    n_obs: "dict[str, int] | None" = None,
    min_cells: int = 0,
    feature_mask: "np.ndarray | None" = None,
    progress: bool = True,
) -> "pd.DataFrame":
    """Regress Norman-style GI parameters from perturbation effect vectors.

    Parameters
    ----------
    effects
        Mapping from perturbation label to control-normalized effect vector.
        Singles should be keyed by gene name. Doubles can be keyed as
        ``A+B`` or ``B+A``; GEARS-style ``A_B`` is also accepted.
    combos
        Combination labels to score. If omitted, all keys containing ``sep``
        are used.
    sep
        Separator used in output combination labels.
    n_obs
        Optional observed cell counts by perturbation label. Missing counts are
        treated as 1 so externally predicted effects can still be scored.
    min_cells
        Minimum count threshold, matching ``get_gi`` behavior.
    feature_mask
        Optional boolean/index mask selecting features before regression.
    progress
        Show a tqdm progress bar.

    Returns
    -------
    pandas.DataFrame
        Same GI metric schema emitted by ``VAE.get_gi``.
    """
    import numpy as np
    import pandas as pd
    import dcor as _dcor
    from sklearn.linear_model import TheilSenRegressor
    from tqdm.auto import tqdm

    eps_ = 1e-8
    n_obs = n_obs or {}

    def _norm(x):
        return float(np.linalg.norm(x))

    def _cos(x, y):
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        if denom < eps_:
            return np.nan
        return float(np.dot(x, y) / denom)

    def _canon_pair(a, b):
        return sep.join(sorted([str(a).strip(), str(b).strip()]))

    def _effect_key(label):
        label = str(label)
        if label in effects:
            return label
        if sep in label:
            a, b = [part.strip() for part in label.split(sep, 1)]
            candidates = [
                _canon_pair(a, b),
                sep.join([a, b]),
                sep.join([b, a]),
                "_".join([a, b]),
                "_".join([b, a]),
            ]
            for candidate in candidates:
                if candidate in effects:
                    return candidate
        return label

    def _as_vec(label):
        key = _effect_key(label)
        if key not in effects:
            raise KeyError(f"Missing effect vector for {label!r}.")
        x = np.asarray(effects[key], dtype=np.float64).ravel()
        if feature_mask is not None:
            x = x[feature_mask]
        return x

    def _count(label):
        key = _effect_key(label)
        return int(n_obs.get(label, n_obs.get(key, 1)))

    if combos is None:
        combos = [key for key in effects if sep in str(key)]

    records = []
    iterator = tqdm(sorted(combos), desc="regress_gi_params", unit="pair") if progress else sorted(combos)

    for combo in iterator:
        combo = str(combo)
        if sep in combo:
            name_a, name_b = [part.strip() for part in combo.split(sep, 1)]
        elif "_" in combo:
            name_a, name_b = [part.strip() for part in combo.split("_", 1)]
        else:
            continue

        combo_name = _canon_pair(name_a, name_b)
        n_a, n_b, n_ab = _count(name_a), _count(name_b), _count(combo_name)

        if n_a < min_cells or n_b < min_cells or n_ab < min_cells:
            records.append({
                "pert_a": name_a,
                "pert_b": name_b,
                "n_a": n_a,
                "n_b": n_b,
                "n_ab": n_ab,
            })
            continue

        da_dec = _as_vec(name_a)
        db_dec = _as_vec(name_b)
        dab = _as_vec(combo_name)
        additive = da_dec + db_dec

        X = np.column_stack([da_dec, db_dec])
        ts = TheilSenRegressor(
            fit_intercept=False,
            max_subpopulation=5000,
            max_iter=300,
            random_state=1000,
        )
        ts.fit(X, dab)
        coef = np.asarray(ts.coef_, dtype=np.float64)
        c1, c2 = float(coef[0]), float(coef[1])
        y_pred = ts.predict(X).astype(np.float64)

        magnitude = float(np.sqrt(c1 ** 2 + c2 ** 2))
        dominance = float(abs(np.log10((abs(c1) + eps_) / (abs(c2) + eps_))))
        ss_res = float(np.sum((dab - y_pred) ** 2))
        ss_tot = float(np.sum(dab ** 2))
        model_fit = float(1.0 - ss_res / (ss_tot + eps_))
        ss_tot_centered = float(np.sum((dab - dab.mean()) ** 2))
        model_fit_r2_centered = float(1.0 - ss_res / (ss_tot_centered + eps_))
        model_fit_dcor = float(_dcor.distance_correlation(
            dab.reshape(-1, 1),
            y_pred.reshape(-1, 1),
        ))
        model_corr = _cos(y_pred, dab)

        singles_sim = float(_dcor.distance_correlation(da_dec, db_dec))
        singles_to_dbl = float(_dcor.distance_correlation(
            np.column_stack([da_dec, db_dec]), dab
        ))

        dc_a = float(_dcor.distance_correlation(da_dec, dab))
        dc_b = float(_dcor.distance_correlation(db_dec, dab))

        eq_contrib = (
            min(dc_a, dc_b) / (max(dc_a, dc_b) + eps_)
            if np.isfinite(dc_a) and np.isfinite(dc_b) else np.nan
        )

        max_parent_dcor = max(dc_a, dc_b)
        min_parent_dcor = min(dc_a, dc_b)
        parent_dcor_diff = abs(dc_a - dc_b)

        c1_abs = abs(c1)
        c2_abs = abs(c2)
        c1_plus_c2 = c1 + c2
        c1_minus_c2 = c1 - c2
        c1_times_c2 = c1 * c2
        same_sign_coeffs = float(np.sign(c1) == np.sign(c2))

        norm_a = _norm(da_dec)
        norm_b = _norm(db_dec)
        norm_ab = _norm(dab)
        norm_additive = _norm(additive)
        norm_fit = _norm(y_pred)

        norm_ab_over_additive = norm_ab / (norm_additive + eps_)
        norm_fit_over_ab = norm_fit / (norm_ab + eps_)

        contrib_a = c1_abs * norm_a
        contrib_b = c2_abs * norm_b

        effective_dominance = float(
            abs(np.log10((contrib_a + eps_) / (contrib_b + eps_)))
        )
        contribution_balance = float(
            min(contrib_a, contrib_b) / (max(contrib_a, contrib_b) + eps_)
        )

        cos_a_b = _cos(da_dec, db_dec)
        cos_ab_a = _cos(dab, da_dec)
        cos_ab_b = _cos(dab, db_dec)
        cos_ab_additive = _cos(dab, additive)
        cos_ab_fit = _cos(dab, y_pred)

        fit_resid = dab - y_pred
        add_resid = dab - additive
        span_coef, *_ = np.linalg.lstsq(X, dab, rcond=None)
        span_pred = X @ span_coef
        span_resid = dab - span_pred

        fit_residual_norm = _norm(fit_resid) / (norm_ab + eps_)
        additive_residual_norm = _norm(add_resid) / (norm_additive + eps_)
        orthogonal_residual_norm = _norm(span_resid) / (norm_ab + eps_)
        orthogonal_residual_fraction = _norm(span_resid) / (_norm(fit_resid) + eps_)

        cos_fit_resid_a = _cos(fit_resid, da_dec)
        cos_fit_resid_b = _cos(fit_resid, db_dec)
        cos_fit_resid_ab = _cos(fit_resid, dab)

        cos_add_resid_a = _cos(add_resid, da_dec)
        cos_add_resid_b = _cos(add_resid, db_dec)
        cos_add_resid_ab = _cos(add_resid, dab)
        signed_residual_alignment = cos_add_resid_ab

        records.append({
            "pert_a": name_a,
            "pert_b": name_b,
            "n_a": n_a,
            "n_b": n_b,
            "n_ab": n_ab,
            "magnitude": magnitude,
            "dominance": dominance,
            "model_fit": model_fit,
            "singles_similarity": singles_sim,
            "singles_to_doubles": singles_to_dbl,
            "equality_contribution": eq_contrib,
            "model_fit_r2_centered": model_fit_r2_centered,
            "model_fit_dcor": model_fit_dcor,
            "model_corr": model_corr,
            "dcor_a": dc_a,
            "dcor_b": dc_b,
            "max_parent_dcor": max_parent_dcor,
            "min_parent_dcor": min_parent_dcor,
            "parent_dcor_diff": parent_dcor_diff,
            "c1": c1,
            "c2": c2,
            "c1_abs": c1_abs,
            "c2_abs": c2_abs,
            "c1_plus_c2": c1_plus_c2,
            "c1_minus_c2": c1_minus_c2,
            "c1_times_c2": c1_times_c2,
            "same_sign_coeffs": same_sign_coeffs,
            "contrib_a": contrib_a,
            "contrib_b": contrib_b,
            "effective_dominance": effective_dominance,
            "contribution_balance": contribution_balance,
            "norm_a": norm_a,
            "norm_b": norm_b,
            "norm_ab": norm_ab,
            "norm_additive": norm_additive,
            "norm_fit": norm_fit,
            "norm_ab_over_additive": norm_ab_over_additive,
            "norm_fit_over_ab": norm_fit_over_ab,
            "cos_a_b": cos_a_b,
            "cos_ab_a": cos_ab_a,
            "cos_ab_b": cos_ab_b,
            "cos_ab_additive": cos_ab_additive,
            "cos_ab_fit": cos_ab_fit,
            "fit_residual_norm": fit_residual_norm,
            "additive_residual_norm": additive_residual_norm,
            "orthogonal_residual_norm": orthogonal_residual_norm,
            "orthogonal_residual_fraction": orthogonal_residual_fraction,
            "cos_fit_resid_a": cos_fit_resid_a,
            "cos_fit_resid_b": cos_fit_resid_b,
            "cos_fit_resid_ab": cos_fit_resid_ab,
            "cos_add_resid_a": cos_add_resid_a,
            "cos_add_resid_b": cos_add_resid_b,
            "cos_add_resid_ab": cos_add_resid_ab,
            "signed_residual_alignment": signed_residual_alignment,
        })

    df = pd.DataFrame(records)
    if len(df) and {"pert_a", "pert_b"}.issubset(df.columns):
        df.index = df["pert_a"] + sep + df["pert_b"]
    return df


def classify_gi(
    df: "pd.DataFrame",
    *,
    pred_col: str = "predicted_gi",
    fit_mask=None,
    inplace: bool = True,
    synergy_threshold: float = 0.779,
) -> "pd.DataFrame":
    """Classify GI metrics with biologically motivated equal-weight scores.

    The function writes one column, ``pred_col`` (default ``predicted_gi``), and
    returns the dataframe. Rows outside ``fit_mask`` or with missing/non-finite
    required metrics receive ``NA``. Synergy is first scored as one broad class,
    then split into similar or dissimilar phenotype by the median
    ``singles_similarity`` among broad synergy predictions.
    """
    import numpy as np
    import pandas as pd

    score_specs = {
        "approximately additive": [
            (-1, "additive_residual_norm"),
            (-1, "norm_ab_over_additive"),
        ],
        "epistasis": [
            (1, "dominance"),
            (1, "parent_dcor_diff"),
            (-1, "equality_contribution"),
        ],
        "redundant": [
            (1, "singles_similarity"),
            (-1, "norm_ab_over_additive"),
        ],
        "suppression": [
            (-1, "norm_ab_over_additive"),
            (1, "norm_fit_over_ab"),
            (-1, "magnitude"),
        ],
        "neomorphic": [
            (-1, "model_fit_dcor"),
            (-1, "norm_fit"),
        ],
        "potentiation": [
            (1, "norm_ab_over_additive"),
            (1, "signed_residual_alignment"),
            (-1, "singles_similarity"),
        ],
        "synergy": [
            (1, "magnitude"),
            (1, "norm_ab_over_additive"),
            (2, "singles_similarity"),
        ],
    }

    out = df if inplace else df.copy()

    fallbacks = {
        "model_fit_dcor": "model_fit",
        "signed_residual_alignment": "cos_add_resid_ab",
    }
    for new_col, fallback_col in fallbacks.items():
        if new_col not in out.columns and fallback_col in out.columns:
            out[new_col] = out[fallback_col]

    score_metrics = sorted({metric for terms in score_specs.values() for _, metric in terms})
    missing = [metric for metric in score_metrics if metric not in out.columns]
    if missing:
        raise ValueError(f"Missing GI classification metric columns: {missing}")

    def _zscore(x):
        x = pd.Series(x, dtype=float)
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        return (x - x.mean()) / sd

    def _wscore(*terms):
        total = 0.0
        for sign, value in terms:
            total = total + sign * value
        return total / len(terms)

    finite = np.isfinite(out[score_metrics].astype(float)).all(axis=1)
    if fit_mask is None:
        valid = finite
    else:
        fit_mask = pd.Series(fit_mask, index=out.index).astype(bool)
        valid = finite & fit_mask
    out[pred_col] = pd.NA
    if not valid.any():
        return out

    z = {metric: _zscore(out.loc[valid, metric]) for metric in score_metrics}
    class_scores = pd.DataFrame(index=out.index[valid])
    for cls, terms in score_specs.items():
        class_scores[cls] = _wscore(*[(sign, z[metric]) for sign, metric in terms])

    pred = class_scores.idxmax(axis=1).astype(object)
    synergy_mask = pred == "synergy"
    threshold = synergy_threshold

    similar = synergy_mask & (
        out.loc[pred.index, "singles_similarity"].to_numpy() >= threshold
    )
    dissimilar = synergy_mask & (
        out.loc[pred.index, "singles_similarity"].to_numpy() < threshold
    )
    pred.loc[pred.index[similar]] = "synergy (similar phenotype)"
    pred.loc[pred.index[dissimilar]] = "synergy (dissimilar phenotype)"

    out.loc[pred.index, pred_col] = pred.to_numpy()
    return out



class HardConcreteGate(nn.Module):
    def __init__(self, shape, init_p=0.5, temperature=2.0 / 3.0, gamma=-0.1, zeta=1.1):
        super().__init__()
        self.shape = shape
        self.gamma = float(gamma)
        self.zeta = float(zeta)

        init_logit = math.log(init_p) - math.log(1.0 - init_p)
        self.log_alpha = nn.Parameter(torch.full(shape, init_logit))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    @torch.no_grad()
    def set_temperature(self, t: float):
        self.temperature.fill_(float(t))

    def _sample_relaxed(self):
        dev = self.log_alpha.device
        u = torch.rand(self.shape, device=dev)
        temp = self.temperature.to(dev)
        s = torch.sigmoid((self.log_alpha + torch.log(u) - torch.log1p(-u)) / temp)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar

    def forward(self, deterministic: bool = False):
        temp = self.temperature.to(self.log_alpha.device)
        if deterministic:
            s = torch.sigmoid(self.log_alpha / temp)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
        else:
            s_bar = self._sample_relaxed()
        z_hard = torch.clamp(s_bar, 0.0, 1.0)
        return z_hard + (s_bar - z_hard).detach()

    def expected_L0(self):
        c = math.log(-self.gamma / self.zeta)
        temp = self.temperature.to(self.log_alpha.device)
        return torch.sigmoid(self.log_alpha - temp * c)

    def l1_logit(self):
        return torch.norm(torch.sigmoid(self.log_alpha), p=1)

    def expected_L2(self, ungated):
        pi = self.expected_L0()
        return (pi * (ungated ** 2)).sum()


class QRCov(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        # Create on default device; module `.to(...)` will move params as needed
        self.U_unproj = nn.Parameter(0.01 * torch.randn(d, r))
        self.log_s = nn.Parameter(torch.linspace(0.0, -2.0, r))

    def _canonicalize(self):
        Q, R = torch.linalg.qr(self.U_unproj, mode="reduced")
        diag = torch.diag(R)
        signs = torch.where(torch.sign(diag) == 0, torch.ones_like(diag), torch.sign(diag))
        Q = Q * signs
        s = torch.exp(self.log_s)
        perm = torch.argsort(s, descending=True)
        return Q[:, perm], s[perm]

    def forward(self):
        Q, s = self._canonicalize()
        return Q @ torch.diag(s)


def build_treat_effect_map(
    treat_effect_adata,
    perturbation_dict: "dict[str, int]",
    gene_names: "list[str]",
    exclude: "list[str] | set[str]" = (),
    sep: str = "+",
    top_k: "int | None" = None,
) -> "dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]":
    """
    Build the {(p0, p1): (gene_idx, target)} map consumed by
    VAE(use_de_align_loss=True, treat_effect_map=...).

    perturbation_dict : PerturbMatchingDataset.perturbation_dict (single-gene
        name -> embedding index). Keys are (p0, -1) for singles and
        (min(p0,p1), max(p0,p1)) for combos, matching the gene-index pairs
        assigned to each cell's `p` tensor.
    exclude : perturbation labels to drop (e.g. held-out combos used for
        ate_held) — compared after sorting each label's "+"-parts, so caller
        doesn't need to match the raw obs string ordering.
    top_k : per perturbation, keep only the top_k genes by |observed effect|
        (mirrors GEARS' num_de_genes). None (default) keeps *all* genes, so the
        de-align loss supervises the full effect profile — needed for the GI
        residual structure, which lives across all genes, not just the top-DE
        ones. The (pred-target)^4 power still emphasizes the genes that move.
    """
    import pandas as pd

    def canon(label: str) -> str:
        return sep.join(sorted(x.strip() for x in str(label).split(sep) if x.strip()))

    exclude_canon = {canon(e) for e in exclude}

    X = treat_effect_adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    te_var = pd.Index(treat_effect_adata.var_names)
    gene_pos = te_var.get_indexer(list(gene_names))  # -1 where gene absent from treat_effect

    out: "dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]" = {}
    for i, label in enumerate(treat_effect_adata.obs_names):
        if canon(label) in exclude_canon:
            continue
        parts = [x.strip() for x in str(label).split(sep) if x.strip()]
        if not parts or any(part not in perturbation_dict for part in parts):
            continue
        if len(parts) == 1:
            key = (perturbation_dict[parts[0]], -1)
        elif len(parts) == 2:
            a, b = sorted(perturbation_dict[part] for part in parts)
            key = (a, b)
        else:
            continue

        row = np.full(len(gene_names), np.nan, dtype=np.float32)
        valid = gene_pos >= 0
        row[valid] = X[i, gene_pos[valid]]

        finite = np.isfinite(row)
        if not finite.any():
            continue
        if top_k is None:
            top_idx = np.where(finite)[0]                    # all genes
        else:
            k = min(top_k, int(finite.sum()))
            scored = np.where(finite, np.abs(row), -np.inf)
            top_idx = np.argpartition(scored, -k)[-k:]
            top_idx = top_idx[np.argsort(-scored[top_idx])]

        out[key] = (
            torch.from_numpy(top_idx.astype(np.int64)),
            torch.from_numpy(row[top_idx].astype(np.float32)),
        )
    return out


def build_go_term_map(
    gene2go: "dict[str, set[str]]",
    perturbation_dict: "dict[str, int]",
) -> "tuple[torch.Tensor, torch.Tensor, int]":
    """
    Build (term_indices, offsets, n_terms) for nn.EmbeddingBag, ordered to match
    perturbation_dict's gene -> embedding-row index (same order as p_emb / rho).

    Genes absent from gene2go, or with no annotated GO terms, get an empty bag —
    nn.EmbeddingBag returns zeros for empty bags, so they just fall back to the
    old flat N(0,1) prior on that row.

    Parameters
    ----------
    gene2go : gene symbol -> set of GO term IDs (e.g. gears_data/gene2go_all.pkl)
    perturbation_dict : PerturbMatchingDataset.perturbation_dict
    """
    # Restrict the vocabulary to terms actually used by genes in perturbation_dict —
    # gene2go typically covers the whole genome (tens of thousands of genes/terms),
    # but only a handful of genes are ever perturbed here.
    all_terms = sorted({
        t for gene in perturbation_dict if gene in gene2go for t in gene2go[gene]
    })
    term_to_idx = {t: i for i, t in enumerate(all_terms)}

    P = len(perturbation_dict)
    idx_by_gene: list[list[int]] = [[] for _ in range(P)]
    for gene, gi in perturbation_dict.items():
        idx_by_gene[gi] = [term_to_idx[t] for t in gene2go.get(gene, ())]

    flat: list[int] = []
    offsets: list[int] = []
    for terms in idx_by_gene:
        offsets.append(len(flat))
        flat.extend(terms)

    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        len(all_terms),
    )


class VAE(PerturbModelBase):
    def __init__(
        self,
        input_dim,
        latent_dim,
        perturbs,
        conds,
        tau,
        l0_lambda=1.0,
        l1_lambda=1e-3,
        l2_lambda=1e-3,
        H_lambda=1e-1,
        ce_lambda=10,
        ntc_lambda=1.0,
        pert_lambda=1.0,
        cov_lambda=1e-5,
        gate_row_repulsion_lambda=1e-2,
        gate_init_p=0.5,
        rank=6,
        use_conditions=False,
        shift='linear',
        use_de_align_loss=False,
        de_align_lambda=1e-3,
        de_align_direction_lambda=1e-3,
        de_align_direction_tau=0.5,
        de_align_n_bg=64,
        synergy_l2_lambda=1e-3,
        parent_scale_lambda=0,
        treat_effect_map=None,
        combinatorial=False,
        synergy=False,
        synergy_rank=4,
        synergy_hidden=32,
        z_var=1.0,
        use_go_prior=False,
        go_term_indices=None,
        go_term_offsets=None,
        n_go_terms=0,
    ):
        super().__init__()
        assert shift in ['poe', 'linear'], "shift must be 'poe' or 'linear'"

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.perturbs = perturbs
        self.conds = conds
        self.use_conditions = use_conditions
        self.rank = rank
        self.shift = shift
        self.combinatorial = combinatorial
        self.synergy = bool(synergy)
        self.synergy_rank = int(synergy_rank)      # retained for checkpoint compat; unused by MLP synergy
        self.synergy_hidden = int(synergy_hidden)

        # GO-term prior mean for rho (see build_go_term_map). go_term_emb is a real
        # submodule (learned weights, part of state_dict/checkpoints); go_term_indices/
        # go_term_offsets are data (nn.EmbeddingBag "offsets" format, one bag per
        # perturbation, row-ordered to match p_emb) kept as plain attributes — not
        # register_buffer — so they're excluded from state_dict, same as
        # treat_effect_map: a checkpoint reload only needs go_term_emb's *weights* to
        # be usable at eval time; resuming training with the prior active requires
        # re-supplying go_term_indices/go_term_offsets (rebuild via build_go_term_map).
        self.use_go_prior = bool(use_go_prior)
        self.n_go_terms = int(n_go_terms)
        self.go_term_indices = go_term_indices.long() if go_term_indices is not None else None
        self.go_term_offsets = go_term_offsets.long() if go_term_offsets is not None else None
        if self.use_go_prior:
            self.go_term_emb = nn.EmbeddingBag(self.n_go_terms, latent_dim, mode="mean")

        # reg weights
        self.l0_lambda = float(l0_lambda)
        self.l1_lambda = float(l1_lambda)
        self.l2_lambda = float(l2_lambda)
        self.H_lambda = float(H_lambda)
        self.ce_lambda = float(ce_lambda)
        self.ntc_lambda = float(ntc_lambda)
        self.pert_lambda = float(pert_lambda)
        self.cov_lambda = float(cov_lambda)
        self.gate_row_repulsion_lambda = float(gate_row_repulsion_lambda)
        self.use_de_align_loss = bool(use_de_align_loss)
        self.de_align_lambda = float(de_align_lambda)
        self.de_align_direction_lambda = float(de_align_direction_lambda)
        self.de_align_direction_tau = float(de_align_direction_tau)
        self.de_align_n_bg = int(de_align_n_bg)
        # Additivity prior on the combo shift: delta_ab = c_a*delta_a + c_b*delta_b + syn.
        # Nothing else in the objective penalises syn != 0 or c != 1, so the model is
        # free to inject non-additivity into genuinely additive combos -- it barely costs
        # the ELBO, but it destroys the additive signature the GI classifier reads
        # (additive_residual_norm / norm_ab_over_additive / model_fit), which is why
        # true-additive combos get misread as interacting. These shrink the *total*
        # delta_ab toward delta_a + delta_b -- and the total is exactly what the
        # classifier sees, since it only ever gets the gene-space triple.
        self.synergy_l2_lambda = float(synergy_l2_lambda)
        self.parent_scale_lambda = float(parent_scale_lambda)
        # {(p0, p1): (gene_idx LongTensor (K,), target FloatTensor (K,))}, p1=-1 for singles.
        # Build with build_treat_effect_map(); must exclude held-out combo labels.
        self.treat_effect_map = treat_effect_map
        self.z_var = float(z_var)

        # embeddings
        self.p_emb = nn.Embedding(perturbs, latent_dim)

        # encoders / decoders
        def hidden_enc(in_d):
            return nn.Sequential(
                nn.Linear(in_d, 128), nn.LeakyReLU(), nn.LayerNorm(128),
                nn.Linear(128, 128), nn.LeakyReLU(), nn.LayerNorm(128),
            )

        # separate encoders — cell_encoder gets only CE/recon gradients, no z0 KL interference
        self.cell_encoder = hidden_enc(input_dim)   # q(z):  individual cells (gene-space residual)
        self.cond_encoder = hidden_enc(input_dim)   # q(z0): condition mean
        self.z0_head = nn.Linear(128, latent_dim * 2)
        self.z_head  = nn.Linear(128, latent_dim * 2)

        self.z_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, input_dim),
        )

        self.rho_enc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        if self.synergy:
            # Nonlinear synergy: a full MLP over [rho_a, rho_b] -> latent shift.
            # No low-rank / bilinear assumption, so it can express asymmetric
            # (redundant/epistatic) and novel (neomorphic) interaction directions
            # the old rank-r bilinear could not. Symmetrized at call time.
            self.syn_mlp = nn.Sequential(
                nn.Linear(2 * latent_dim, self.synergy_hidden),
                nn.LeakyReLU(),
                nn.Linear(self.synergy_hidden, latent_dim),
            )
            # Parent scales c_a, c_b: scalar (per-pair) reweightings of each
            # parent's shift. Takes the *pair* [rho_self, rho_other] (2*latent_dim)
            # rather than the antisymmetric (rho_a - rho_b) the old Linear(latent_dim, 1)
            # used. That antisymmetry forced c_a = 1 + tanh(s), c_b = 1 - tanh(s), i.e.
            # c_a + c_b == 2 identically -- the model could only shuffle weight between
            # parents (dominance) and could NEVER scale the additive part up or down.
            # Potentiation (both parents amplified) and suppression (both damped) were
            # therefore structurally inexpressible via c and had to be contorted out of
            # the free-form syn vector. Applying this layer to both orderings keeps the
            # swap symmetry c_a(a,b) == c_b(b,a) while leaving the two scales free.
            self.parent_coeff = nn.Linear(2 * latent_dim, 1, bias=False)
            # Start ADDITIVE. Default Linear init gives score std ~1 over a 2*latent_dim
            # input, so tanh(score) would be O(0.7) and c would span ~(0.3, 1.8) at
            # step 0 -- i.e. the model would begin with large *random* non-additivity,
            # which is exactly the prior we're trying to impose against. Shrinking the
            # init puts score ~= 0 => c_a = c_b ~= 1, so delta_ab starts at delta_a +
            # delta_b and the model must be *pushed* off additivity by the data (and pay
            # parent_scale_lambda to stay there). Gradients still flow normally.
            with torch.no_grad():
                self.parent_coeff.weight.mul_(0.01)
        else:
            self.syn_mlp = None
            self.parent_coeff = None
        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, 128), nn.LeakyReLU(),
            nn.Linear(128, perturbs),
        )

        # gating (always on)
        self.tau_init = float(tau)
        self.gate = HardConcreteGate(
            shape=(self.perturbs, self.latent_dim),
            init_p=gate_init_p,
            temperature=self.tau_init if self.tau_init is not None else (2.0 / 3.0),
        )

        # low-rank covariance (params live on module device)
        self.qr = QRCov(self.perturbs, self.rank)

        # KL warmup weight — set externally by VAETrainer each epoch
        self.kl_weight = 1.0
        self.warmup_done = False
        # L0/gate sparsity weight — set externally by VAETrainer via delayed schedule
        # (ramps 0→1 only after KL warmup ends to prevent gate collapse on warmup exit)
        self.l0_weight = 0.0



    # --- helpers ---
    @torch.no_grad()
    def _corr(self, M, eps=1e-8):
        d = torch.sqrt(torch.clamp(torch.diag(M), min=eps))
        denom = torch.outer(d, d)
        return M / torch.clamp(denom, min=eps)

    def _mednorm(self, x: torch.Tensor) -> torch.Tensor:
        lib = x.sum(dim=1, keepdim=True)
        med = torch.median(lib).item()
        return torch.log1p(x / torch.clamp(lib, min=1e-8) * med)

    @staticmethod
    def _dedupe_ntc(x_ntc: torch.Tensor) -> torch.Tensor:
        """
        Collapse the batch-matched, with-replacement NTC sample back down to
        the distinct real cells actually drawn (duplicate rows are bit-identical
        copies from the same source pool, so this is exact) — keeps NTC decode
        cost independent of x_p's batch size. Must be called on the *raw* x_ntc
        (before _mednorm) in both model() and guide() so both dedupe on identical
        values and agree on row order — model()'s replayed "z0"/"z" sites must
        line up position-for-position with guide()'s.
        """
        return torch.unique(x_ntc, dim=0)

    @torch.no_grad()
    def _get_cov(self):
        L = self.qr().detach()
        sigma_P = pyro.param("sigma_fac").detach().to(L.device)
        P_ = L.size(0)
        eyeP = torch.eye(P_, dtype=L.dtype, device=L.device)
        Sigma_P = L @ L.T + (sigma_P**2) * eyeP
        return Sigma_P
    
    def __poe(self, z0_loc, lin_shift, W_var):
        # PoE mean: z0_loc + alpha * lin_shift, alpha = 1/(1+W_var)
        # z_var is controlled separately via z_var_scale in the model
        alpha = 1.0 / (1.0 + W_var)
        z_loc = z0_loc + alpha * lin_shift
        return z_loc

    def __decode(self, x, z, weight, theta, d_key):
        logits_gene = weight(z)
        total_counts = x.sum(-1, keepdim=True)
        mu_prob = torch.softmax(logits_gene, dim=-1)
        mu = total_counts * mu_prob
        pyro.deterministic(d_key, mu)
        logits_nb = (mu + 1e-6).log() - theta.log()
        return logits_nb, mu

    @staticmethod
    def _lognorm_t(x: torch.Tensor) -> torch.Tensor:
        """log2-CPM normalise, torch/differentiable — matches PerturbModelBase._lognorm
        and preprocessing.treat_effect(method='perturbseq')."""
        lib = x.sum(-1, keepdim=True)
        return torch.log2(1e4 * x / torch.clamp(lib, min=1e-8) + 1.0)

    def _de_align_lookup(self, device: torch.device):
        """
        Precompute (once per device, then cached) the dense tables that let
        _de_align_loss group a batch by perturbation without a Python loop:

        gene_idx/target/mask : (M, top_k) — M = number of treat_effect_map
            entries, top_k = widest entry's gene count (others zero-padded,
            masked off).
        single_group_id : (P,)   -> row in the above tables, or -1
        combo_group_id  : (P, P) -> row in the above tables, or -1 (indexed by
            [min(p0,p1), max(p0,p1)], matching treat_effect_map's key convention)
        """
        cache = getattr(self, "_de_align_cache", None)
        if cache is not None and cache[0] == device:
            return cache[1]

        keys = list(self.treat_effect_map.keys())
        M = len(keys)
        top_k = max((idx.numel() for idx, _ in self.treat_effect_map.values()), default=0)

        gene_idx_mat = torch.zeros(M, top_k, dtype=torch.long)
        target_mat = torch.zeros(M, top_k, dtype=torch.float32)
        mask_mat = torch.zeros(M, top_k, dtype=torch.float32)
        single_group_id = torch.full((self.perturbs,), -1, dtype=torch.long)
        combo_group_id = torch.full((self.perturbs, self.perturbs), -1, dtype=torch.long)
        # perturbation gene index (or -1) for each group row, to rebuild its shift
        group_p0 = torch.zeros(M, dtype=torch.long)
        group_p1 = torch.full((M,), -1, dtype=torch.long)

        for gi, (key, (idx, tgt)) in enumerate(zip(keys, self.treat_effect_map.values())):
            k = idx.numel()
            gene_idx_mat[gi, :k] = idx
            target_mat[gi, :k] = tgt
            mask_mat[gi, :k] = 1.0
            a, b = key
            group_p0[gi] = a
            group_p1[gi] = b  # -1 for singles
            if b == -1:
                single_group_id[a] = gi
            else:
                combo_group_id[a, b] = gi

        tables = (
            gene_idx_mat.to(device), target_mat.to(device), mask_mat.to(device),
            single_group_id.to(device), combo_group_id.to(device),
            group_p0.to(device), group_p1.to(device), M,
        )
        self._de_align_cache = (device, tables)
        return tables

    def _de_align_loss(
        self,
        z0_bg: torch.Tensor,
        A: torch.Tensor,
        W: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """
        GEARS-style DE-gene supervision on the actual COUNTERFACTUAL path.

        For each perturbation present in the batch, predict its effect the same
        way eval does — apply the learned shift to real NTC backgrounds,
            z_cf = z0*(1 - gate_p) + shift_p,
        decode, and compare lognorm(decode(z_cf)) - lognorm(decode(z0)) to the
        observed treat_effect on that perturbation's top-DE genes, using GEARS'
        (pred-y)^4 + direction_lambda * sign-mismatch loss (utils.loss_fct).

        This trains the exact quantity eval measures — A*W (+ parent_coeff/synergy
        for combos) applied to backgrounds — unlike the previous version, which
        supervised decode(cell_encoder(x_p)) (the reconstruction path) and reached
        A*W only second-hand through KL(q(z)||p(z|z0)).

        z0_bg : (n_ntc, d) abducted NTC backgrounds (detached — treated as fixed
                controls, GEARS-style; de_align shapes shift+decoder, not encoder).
        A, W  : (P, d) per-perturbation shift factor and gate (training-time,
                stochastic — same tensors model() uses to build z_loc).
        rho   : (P, d) per-perturbation rho sample (training-time, same tensor
                model() uses for parent_coeff/synergy) — needed so combo shifts
                here match model()'s combinatorial branch exactly, including
                syn_mlp; without it de-align silently never trains synergy.
        """
        if not self.treat_effect_map:
            return z0_bg.new_zeros(())

        device = z0_bg.device
        (gene_idx_mat, target_mat, mask_mat, single_group_id, combo_group_id,
         group_p0, group_p1, M) = self._de_align_lookup(device)

        # groups (perturbations) present in this batch
        if p.ndim == 1:
            gid = single_group_id[p.clamp(min=0)]
        else:
            p0 = p[:, 0].clamp(min=0)
            p1 = p[:, 1]
            is_single = p1 == -1
            p1c = p1.clamp(min=0)
            a = torch.minimum(p0, p1c)
            b = torch.maximum(p0, p1c)
            gid = torch.where(is_single, single_group_id[p0], combo_group_id[a, b])
        present = torch.unique(gid[gid >= 0])
        if present.numel() == 0:
            return z0_bg.new_zeros(())

        # fixed-control backgrounds; subsample to bound decode cost
        z0_bg = z0_bg.detach()
        n_bg = z0_bg.shape[0]
        if n_bg > self.de_align_n_bg:
            sel = torch.randperm(n_bg, device=device)[: self.de_align_n_bg]
            z0_bg = z0_bg[sel]
            n_bg = self.de_align_n_bg

        # per-present-group shift from A, W, rho — matches model()'s combinatorial
        # branch exactly: singles get shift_a alone; combos get
        # c_a*shift_a + c_b*shift_b + syn(rho_a, rho_b). _parent_scales_from_rho /
        # _synergy_shift_from_rho degrade to (1, 1, 0) when synergy=False, so this
        # formula is correct (== plain additive) in both cases — no branching needed.
        gp0 = group_p0[present]
        gp1 = group_p1[present]
        combo_m = (gp1 >= 0).unsqueeze(-1).to(z0_bg.dtype)   # (nP, 1)
        gp1c = gp1.clamp(min=0)

        shift_a = A[gp0] * W[gp0]
        shift_b = A[gp1c] * W[gp1c]
        c_a, c_b = self._parent_scales_from_rho(rho[gp0], rho[gp1c])
        syn = self._synergy_shift_from_rho(rho[gp0], rho[gp1c])
        combo_shift = c_a * shift_a + c_b * shift_b + syn
        shift = shift_a * (1.0 - combo_m) + combo_shift * combo_m   # (nP, d)

        nP = present.numel()
        # z_cf[g, b] = z0_bg[b] + shift[g]   (pure-additive, matches model()/_apply_shift)
        z_cf = z0_bg.unsqueeze(0) + shift.unsqueeze(1)  # (nP, n_bg, d)
        mu_cf = torch.softmax(self.z_decoder(z_cf.reshape(nP * n_bg, -1)), dim=-1)
        ln_cf = self._lognorm_t(mu_cf).reshape(nP, n_bg, -1).mean(1)                # (nP, G)

        mu_base = torch.softmax(self.z_decoder(z0_bg), dim=-1)                      # (n_bg, G)
        base_ln = self._lognorm_t(mu_base).mean(0)                                  # (G,)

        delta = ln_cf - base_ln.unsqueeze(0)                                        # (nP, G)

        idx = gene_idx_mat[present]
        tgt = target_mat[present]
        mask = mask_mat[present]
        pred = torch.gather(delta, 1, idx)

        denom = mask.sum().clamp(min=1)
        mse = ((pred - tgt).pow(4) * mask).sum() / denom
        # torch.sign(pred) has zero gradient everywhere (flat step function), so a
        # raw sign-mismatch term is inert regardless of de_align_direction_lambda.
        # tanh(pred/tau) is a differentiable surrogate for sign(pred) (-> exact sign
        # as tau -> 0); target side stays exact sign since it's a constant, no
        # gradient needed there.
        pred_dir = torch.tanh(pred / self.de_align_direction_tau)
        direction = ((torch.sign(tgt) - pred_dir).pow(2) * mask).sum() / denom
        return mse + self.de_align_direction_lambda * direction

    def _synergy_shift_from_rho(
        self,
        rho_a: torch.Tensor,
        rho_b: torch.Tensor,
    ) -> torch.Tensor:
        if not self.synergy:
            return torch.zeros_like(rho_a)
        # Symmetric nonlinear synergy: average the MLP over both orderings so
        # syn(a, b) == syn(b, a) (A+B and B+A are the same combo).
        ab = torch.cat([rho_a, rho_b], dim=-1)
        ba = torch.cat([rho_b, rho_a], dim=-1)
        return 0.5 * (self.syn_mlp(ab) + self.syn_mlp(ba))

    def _parent_scales_from_rho(
        self,
        rho_a: torch.Tensor,
        rho_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.synergy:
            ones = rho_a.new_ones((*rho_a.shape[:-1], 1))
            return ones, ones

        # Free, symmetric parent scales. Scoring [rho_self, rho_other] in both
        # orderings gives c_a(a, b) == c_b(b, a) (swap-equivariant) WITHOUT the
        # old antisymmetry that pinned c_a + c_b to 2. Writing the layer weight as
        # W = [W1; W2], the old form is exactly the special case W2 = -W1; with W1,
        # W2 free, c_a and c_b are independent, so both can rise above 1
        # (potentiation) or both fall below 1 (suppression).
        # tanh keeps each scale in (0, 2): a parent can be damped or up to doubled,
        # but can't flip sign or blow up. Deviation from 1 is penalised by
        # parent_scale_lambda (see model()), so additive (c = 1) is the default and
        # the model pays to move away from it.
        ab = torch.cat([rho_a, rho_b], dim=-1)
        ba = torch.cat([rho_b, rho_a], dim=-1)
        c_a = 1.0 + torch.tanh(self.parent_coeff(ab))
        c_b = 1.0 + torch.tanh(self.parent_coeff(ba))
        return c_a, c_b

    # --- model/guide ---
    def model(self, x_p, x_ntc, p, c, c_ntc):
        pyro.module("VAE", self)
        device = x_p.device

        theta_uncon = pyro.param("theta_uncon", torch.ones(self.input_dim, device=device))
        theta = F.softplus(theta_uncon) + 1e-3

        L = self.qr()
        sigma = pyro.param("sigma_fac", torch.full((), 0.05, device=device), constraint=constraints.positive)
        row_cov = L @ L.T + sigma.pow(2) * torch.eye(self.perturbs, device=device)

        # mild stabilizers
        pyro.factor("half_normal_energy", -0.5 * (sigma / 0.05) ** 2)
        pyro.factor("cov_ridge", - self.cov_lambda * (L ** 2).sum())

        if self.use_go_prior and self.go_term_indices is not None:
            # GO-informed prior mean: KL(q(rho)||p(rho)) pulls the guide's rho
            # posterior toward this instead of toward zero — a real Bayesian
            # shrinkage prior, not a competing loss term (see build_go_term_map).
            mu_rho = self.go_term_emb(
                self.go_term_indices.to(device), self.go_term_offsets.to(device)
            )  # (P, d)
        else:
            mu_rho = torch.zeros_like(self.p_emb.weight)

        with pyro.plate("perturbations", self.perturbs):
            rho_single = pyro.sample(
                "rho",
                dist.Normal(mu_rho, torch.ones_like(mu_rho)).to_event(1),
                infer={"scale": self.kl_weight},
            )
        chol_P = torch.linalg.cholesky(row_cov)
        A = chol_P @ rho_single  # (P, d)

        W = self.gate(deterministic=False)  # (P, d) in [0,1]
        pyro.deterministic("W", W)

        pi = self.gate.expected_L0()
        expected_l0 = pi.sum()
        expected_l2 = self.gate.expected_L2(A)
        pyro.factor("l0_penalty", - self.l0_weight * self.l0_lambda * expected_l0)
        pyro.factor("l1_penalty", - self.l0_weight * self.l1_lambda * self.gate.l1_logit())
        pyro.factor("l2_penalty", - self.l0_weight * self.l2_lambda * expected_l2)

        # column entropy encouragement
        col_usage = pi.sum(dim=0)
        p_col = col_usage / (col_usage.sum() + 1e-8)
        entropy_col = -(p_col * (p_col + 1e-8).log()).sum()
        pyro.factor("col_entropy", + self.H_lambda * entropy_col)
        # encourage perturbations to use different latent gates
        if self.perturbs > 1:
            W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
            sim = W_norm @ W_norm.T
            off_diag = sim[~torch.eye(self.perturbs, device=device, dtype=torch.bool)]
            repulsion = (off_diag ** 2).mean()
            pyro.factor("gate_row_repulsion", - self.gate_row_repulsion_lambda * repulsion)

        # ── per-perturbed-cell shift ─────────────────────────────────────
        x_ntc_unique = self._dedupe_ntc(x_ntc)
        n_ntc = x_ntc_unique.size(0)
        n     = x_p.size(0)
        n_all = n_ntc + n

        if p.ndim > 1:  # combinatorial
            p0      = p[:, 0].clamp(min=0)
            p1_raw  = p[:, 1]
            p1      = p1_raw.clamp(min=0)
            valid_m = (p1_raw != -1)                                   # True for combo cells

            shift_a = A[p0] * W[p0]                                    # (n, d)
            shift_b = A[p1] * W[p1]                                    # (n, d)

            # Pure additive default: δab = δa + δb (c1 = c2 = 1 for combo cells; c2 = 0 for single-pert)
            lin_shift_p = shift_a + valid_m.float().unsqueeze(-1) * shift_b
            if self.synergy:
                c_a, c_b = self._parent_scales_from_rho(rho_single[p0], rho_single[p1])
                syn_shift_p = self._synergy_shift_from_rho(rho_single[p0], rho_single[p1])
                combo_mask = valid_m.float().unsqueeze(-1)
                scaled_shift_p = c_a * shift_a + c_b * shift_b + syn_shift_p
                lin_shift_p = shift_a * (1.0 - combo_mask) + scaled_shift_p * combo_mask
                syn_shift_p = syn_shift_p * combo_mask
                parent_scale_penalty = (((c_a - 1.0) ** 2) + ((c_b - 1.0) ** 2)) * combo_mask
                # Additivity prior. Own lambdas (NOT the gate's l2_lambda, which is
                # already doing duty on the gate and couldn't be tuned independently).
                # Both default to 0.0 => exactly the old behaviour, so this is a clean A/B.
                # Summed over combo cells to match the ELBO's summed likelihood, keeping
                # the prior/likelihood ratio invariant to batch size.
                if self.synergy_l2_lambda > 0.0:
                    pyro.factor("synergy_l2",
                                -self.synergy_l2_lambda * syn_shift_p.pow(2).sum())
                if self.parent_scale_lambda > 0.0:
                    pyro.factor("parent_scale_l2",
                                -self.parent_scale_lambda * parent_scale_penalty.sum())
                pyro.deterministic("syn_shift", syn_shift_p)
                pyro.deterministic("parent_scale", torch.cat([c_a, c_b], dim=-1))
        else:
            lin_shift_p = A[p] * W[p]                                  # (n, d)

        # ── n_all tensors: NTC cells first (no shift), perturbed cells second ─
        zeros_ntc     = torch.zeros(n_ntc, self.latent_dim, device=device)
        lin_shift_all = torch.cat([zeros_ntc, lin_shift_p], dim=0)  # (n_all, d)

        # isotropic z0 prior — condition adjustment done in gene space via the residual guide
        z0_loc_all   = torch.zeros(n_all, self.latent_dim, device=device)
        z0_scale_all = torch.ones( n_all, self.latent_dim, device=device)

        z_var = torch.full((self.latent_dim,), self.z_var, device=device)

        if self.shift == 'poe':
            #Here to not be in cell plate
            W_scale = pyro.param("W_scale", 0.1 * torch.ones(self.latent_dim, device=device), constraint=constraints.positive)
            pyro.factor("W_scale_prior", -0.5 * (W_scale / 0.1).pow(2).sum())

        # ── cells plate: one z per cell (NTC: no shift ⟹ z≈z0; perturbed: z=z0+shift) ─
        # Pure-additive shift: z = z0 + A*W. The background z0 is NOT masked
        # (dropped the old z0*(1-W) term) — with a non-binary gate that masking
        # attenuated z0 (~0.5x) and made the effective shift W*(A - z0)
        # background-dependent; additive matches the latent-arithmetic that works
        # (scGen/sVAE). The gate W still scales/selects shift dims via A*W.
        with pyro.plate("cells", n_all):
            #TODO this is not proper POE for what we need. should be using z0_mu and z0_std
            #but gradient won't flow this way since its not from pyro.sample. Temp fix using
            #sampled z0 but its not ideal since it won't have the same mean/var as the guide's z0 distribution.
            #A better fix would be to implement the POE logic manually in the guide and model instead of relying on pyro.sample for z0
            z0     = pyro.sample("z0", dist.Normal(z0_loc_all, z0_scale_all).to_event(1), infer={"scale": self.kl_weight})

            if self.shift == 'poe':
                z_loc = self.__poe(z0, lin_shift_all, W_scale.pow(2))
            elif self.shift == 'linear':
                z_loc = z0 + lin_shift_all
            else:
                raise ValueError("Invalid shift type")

            z = pyro.sample("z", dist.Normal(z_loc, torch.sqrt(z_var)).to_event(1), infer={"scale": self.kl_weight})

        # ── observations and CE loss (outside plate) ─────────────────────
        z_ntc_z = z[:n_ntc]
        z_p_z   = z[n_ntc:]

        cls_logits = self.cls_head(z_p_z)
        pyro.deterministic("cls_logits", cls_logits)

        if p.ndim == 1:
            CE_loss = F.cross_entropy(cls_logits, p, reduction="sum")
        else:
            B_ce, Pdim = cls_logits.shape
            target = torch.zeros((B_ce, Pdim), device=device, dtype=cls_logits.dtype)
            mask   = (p != -1)
            target.scatter_(1, p.clamp(min=0), mask.to(target.dtype))
            CE_loss = F.binary_cross_entropy_with_logits(cls_logits, target, reduction="sum")
        pyro.factor("CE_loss", -self.ce_lambda * CE_loss)

        logits_ntc, mu_ntc = self.__decode(x_ntc_unique, z_ntc_z, self.z_decoder, theta, "x_ntc")
        ntc_logprob = dist.NegativeBinomial(total_count=theta, logits=logits_ntc).log_prob(x_ntc_unique.float()).sum()
        pyro.factor("X_ntc_logprob", self.ntc_lambda * ntc_logprob)

        logits_p, mu_p = self.__decode(x_p, z_p_z, self.z_decoder, theta, "x_p")
        p_logprob = dist.NegativeBinomial(total_count=theta, logits=logits_p).log_prob(x_p.float()).sum()
        pyro.factor("X_p_logprob", self.pert_lambda * p_logprob)

        if self.use_de_align_loss and self.treat_effect_map:
            # Counterfactual-path supervision: apply the learned shift (A, W,
            # rho_single) to the NTC backgrounds z_ntc_z and match the observed
            # effect — trains the shift (incl. synergy for combos) directly (what
            # eval uses), not decode(cell_encoder(x_p)).
            de_loss = self._de_align_loss(z_ntc_z, A, W, rho_single, p)
            pyro.factor("de_align_loss", -self.de_align_lambda * de_loss)

    def guide(self, x_p, x_ntc, p, c, c_ntc):
        pyro.module("VAE", self)

        # x_ntc arrives batch-matched to x_p (one real NTC cell per perturbed
        # cell, with replacement — see PerturbMatchingDataset.get_collate_fn).
        # x_ntc_paired keeps that full, row-aligned version for the perturbed
        # branch's z0; x_ntc itself is deduped down to the distinct real cells
        # actually drawn, matching model()'s _dedupe_ntc so both agree on the
        # "cells" plate's NTC row count/order.
        x_ntc_paired = self._mednorm(x_ntc)
        x_p          = self._mednorm(x_p)
        x_ntc        = self._mednorm(self._dedupe_ntc(x_ntc))

        n_ntc = x_ntc.size(0)
        n     = x_p.size(0)
        n_all = n_ntc + n

        q_loc = self.rho_enc(self.p_emb.weight)
        with pyro.plate("perturbations", self.perturbs):
            pyro.sample("rho", dist.Normal(q_loc, torch.ones_like(q_loc)).to_event(1), infer={"scale": self.kl_weight})


        if self.use_conditions:
            # NTC: z0 = z — encode each distinct NTC cell through cond_encoder (same
            # path as perturbed z0). No residual: cond_encoder carries per-cell background.
            z0_mu_ntc, z0_logvar_ntc = self.z0_head(self.cond_encoder(x_ntc)).chunk(2, dim=-1)
            z0_std_ntc = (0.5 * z0_logvar_ntc).exp()
            z_mu_ntc   = z0_mu_ntc
            z_std_ntc  = z0_std_ntc

            # Perturbed: z0 from a paired real NTC cell (with replacement, row-aligned
            # to x_p and already drawn from that cell's own condition at the collate
            # level — see PerturbMatchingDataset.get_collate_fn). z is a plain encode
            # of x_p itself, no hand-subtracted residual.
            z0_mu_p, z0_logvar_p = self.z0_head(self.cond_encoder(x_ntc_paired)).chunk(2, dim=-1)
            z0_std_p = (0.5 * z0_logvar_p).exp()
            z_mu_p, z_logvar_p = self.z_head(self.cell_encoder(x_p)).chunk(2, dim=-1)
            z_std_p  = (0.5 * z_logvar_p).exp()
        else:
            # NTC: z0 = z via cond_encoder(x_ntc) — per-cell background, on the
            # distinct real NTC cells actually drawn this batch.
            z0_mu_ntc, z0_logvar_ntc = self.z0_head(self.cond_encoder(x_ntc)).chunk(2, dim=-1)
            z0_std_ntc = (0.5 * z0_logvar_ntc).exp()
            z_mu_ntc   = z0_mu_ntc
            z_std_ntc  = z0_std_ntc

            # Perturbed: z0 from a paired real NTC cell (with replacement, row-aligned
            # to x_p at the collate level) — genuine per-cell background instead of the
            # population mean. z is then a plain encode of x_p itself, no hand-subtracted
            # residual: z0 already carries a real per-row background, so the
            # background/shift decomposition is left to the model's KL rather than
            # being manually centered here.
            z0_mu_p, z0_logvar_p = self.z0_head(self.cond_encoder(x_ntc_paired)).chunk(2, dim=-1)
            z0_std_p = (0.5 * z0_logvar_p).exp()
            z_mu_p, z_logvar_p = self.z_head(self.cell_encoder(x_p)).chunk(2, dim=-1)
            z_std_p  = (0.5 * z_logvar_p).exp()

        z0_mu_all  = torch.cat([z0_mu_ntc, z0_mu_p],  dim=0)
        z0_std_all = torch.cat([z0_std_ntc, z0_std_p], dim=0)

        z_mu_all  = torch.cat([z_mu_ntc, z_mu_p],  dim=0)
        z_std_all = torch.cat([z_std_ntc, z_std_p], dim=0)

        with pyro.plate("cells", n_all):
            pyro.sample("z0", dist.Normal(z0_mu_all, z0_std_all).to_event(1), infer={"scale": self.kl_weight})
            pyro.sample("z",  dist.Normal(z_mu_all,  z_std_all).to_event(1),  infer={"scale": self.kl_weight})

    @torch.no_grad()
    def _abduct_z0(self, x_ntc):
        x_ntc_n = self._mednorm(x_ntc)
        z_mu, _ = self.z0_head(self.cond_encoder(x_ntc_n)).chunk(2, dim=-1)
        return z_mu

    @torch.no_grad()
    def init_p_emb_from_adata(self, adata) -> None:
        """Seed p_emb with PCA of per-perturbation pseudobulk expression differences vs NTC.

        Baseline is NTC cells from the untreated condition only. Perturbed cells are
        also taken from untreated only, falling back to all conditions if a perturbation
        has no untreated cells. In combinatorial mode individual genes are model rows;
        'A+B' labels contribute to both gene A and gene B deltas.
        """
        import scipy.sparse as sp
        from sklearn.decomposition import TruncatedSVD

        ntc_label       = get_config("ntc_label")
        pert_key        = get_config("pert_key")
        treatment_key   = get_config("treatment_key")
        untreated_label = get_config("untreated_label")

        X = adata.X
        if sp.issparse(X):
            X = X.toarray().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        lib    = X.sum(1, keepdims=True)
        X_norm = np.log1p(X / (lib + 1e-8) * float(np.median(lib)))

        obs_pert = adata.obs[pert_key].values
        obs_cond = adata.obs[treatment_key].values
        is_untreated = obs_cond == untreated_label

        # Baseline: NTC cells in untreated condition
        baseline_mask = (obs_pert == ntc_label) & is_untreated
        if not baseline_mask.any():
            baseline_mask = obs_pert == ntc_label
        ntc_mean = X_norm[baseline_mask].mean(0)

        def _pert_mean(pert_mask):
            """Mean expression for perturbed cells, preferring untreated condition."""
            untreated_pert = pert_mask & is_untreated
            return X_norm[untreated_pert if untreated_pert.any() else pert_mask].mean(0)

        if self.combinatorial:
            all_genes: set = set()
            for p in obs_pert[obs_pert != ntc_label]:
                s = str(p)
                for g in (s.split("+") if "+" in s else [s]):
                    all_genes.add(g.strip())
            non_ntc = sorted(all_genes)
            if len(non_ntc) != self.perturbs:
                raise ValueError(
                    f"Perturbation count mismatch: {len(non_ntc)} genes in adata, model expects {self.perturbs}"
                )
            deltas = np.zeros((self.perturbs, X_norm.shape[1]), dtype=np.float32)
            for i, gene in enumerate(non_ntc):
                mask = np.array(
                    [gene in (str(p).split("+") if "+" in str(p) else [str(p)])
                     for p in obs_pert], dtype=bool
                ) & (obs_pert != ntc_label)
                if mask.any():
                    deltas[i] = _pert_mean(mask) - ntc_mean
        else:
            non_ntc = sorted(p for p in np.unique(obs_pert) if p != ntc_label)
            if len(non_ntc) != self.perturbs:
                raise ValueError(
                    f"Perturbation count mismatch: {len(non_ntc)} labels in adata, model expects {self.perturbs}"
                )
            deltas = np.zeros((self.perturbs, X_norm.shape[1]), dtype=np.float32)
            for i, p in enumerate(non_ntc):
                mask = obs_pert == p
                if mask.any():
                    deltas[i] = _pert_mean(mask) - ntc_mean

        r   = min(self.perturbs - 1, self.latent_dim)
        emb = TruncatedSVD(n_components=r, random_state=0).fit_transform(deltas)  # (P, r)
        if r < self.latent_dim:
            pad = 0.01 * np.random.default_rng(0).standard_normal(
                (self.perturbs, self.latent_dim - r)
            ).astype(np.float32)
            emb = np.concatenate([emb, pad], axis=1)

        emb = (emb / (emb.std(0, keepdims=True) + 1e-8)).astype(np.float32)
        self.p_emb.weight.data.copy_(torch.tensor(emb, dtype=self.p_emb.weight.dtype))

    # ------------------------------ PerturbModelBase interface ----------------------------- #

    @torch.no_grad()
    def checkpoint_ctor_args(self) -> dict:
        # NOTE: go_term_emb's trained weights round-trip via state_dict, so a
        # reload is usable at eval time as-is. But go_term_indices/go_term_offsets
        # aren't stored here (data-shaped, like treat_effect_map) — resuming
        # training with the GO prior active needs them re-supplied (rebuild via
        # build_go_term_map); otherwise model() silently falls back to the flat
        # N(0,1) prior on rho.
        return {
            "input_dim":          self.input_dim,
            "latent_dim":         self.latent_dim,
            "perturbs":           self.perturbs,
            "conds":              self.conds,
            "use_conditions":     bool(self.use_conditions),
            "rank":               self.rank,
            "shift":              self.shift,
            "combinatorial":      self.combinatorial,
            "synergy":            self.synergy,
            "synergy_rank":       self.synergy_rank,
            "synergy_hidden":     self.synergy_hidden,
            "z_var":              self.z_var,
            "use_go_prior":       self.use_go_prior,
            "n_go_terms":         self.n_go_terms,
        }

    def get_z(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        x_norm = self._mednorm(x)
        z_mu, _ = self.z_head(self.cell_encoder(x_norm)).chunk(2, dim=-1)
        return z_mu

    @torch.no_grad()
    def get_recon(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Reconstructed expected counts at observed library size."""
        z_mu = self.get_z(x, p)
        logits_gene = self.z_decoder(z_mu)
        mu_prob = F.softmax(logits_gene, dim=-1)
        library = x.sum(dim=-1, keepdim=True)
        return library * mu_prob

    def _get_rho_embed(
        self,
        pert_indices: np.ndarray,
        z: np.ndarray,
        p_indices: np.ndarray,
        all_perts: np.ndarray,
    ) -> np.ndarray:
        """
        A*W action embeddings for the selected perturbations.

        PerturbSimpleDataset sorts non-NTC perts alphabetically, matching
        VAE's internal row order (0..P-1).  We recover VAE rows by ranking
        pert_indices among all non-NTC global indices.
        """
        from .._configs import get_config
        ntc_label    = get_config("ntc_label")
        all_non_ntc  = np.where(all_perts != ntc_label)[0]
        vae_rows     = np.searchsorted(all_non_ntc, pert_indices)

        Sigma_P = self._get_cov().cpu()
        q_rho   = self.rho_enc(self.p_emb.weight).detach().cpu()   # (P_vae, d)
        chol_P  = torch.linalg.cholesky(Sigma_P.to(q_rho.dtype))
        A       = (chol_P @ q_rho).detach().numpy()                 # (P_vae, d)
        W       = self.gate(deterministic=True).detach().cpu().numpy()
        return (A * W)[vae_rows].astype(np.float32)

    # ── counterfactual hooks (PerturbModelBase interface) ─────────────────────

    @torch.no_grad()
    def _abduct_ntc(self, x_ntc: torch.Tensor) -> torch.Tensor:
        return self._abduct_z0(x_ntc)

    @torch.no_grad()
    def _get_all_pert_shifts(self, device: torch.device) -> torch.Tensor:
        """Cholesky + gate computed once; returns packed single-pert shift tensors."""
        L      = self.qr().detach().to(device)
        sigma  = pyro.param("sigma_fac").detach().to(device)
        chol_P = torch.linalg.cholesky(L @ L.T + sigma ** 2 * torch.eye(self.perturbs, device=device))
        q_rho = self.rho_enc(self.p_emb.weight).detach().to(device)
        A = chol_P @ q_rho
        W = self.gate(deterministic=True).detach().to(device)

        syn = torch.zeros_like(A)

        out = torch.stack([A * W, W, q_rho, syn], dim=1)  # (P, 4, d) for unpacking in _apply_shift
        return torch.cat([out.new_zeros((1, 4, self.latent_dim)), out], dim=0) #0 is NTC for this function

    @torch.no_grad()
    def _apply_shift(self, u: torch.Tensor, m_p: torch.Tensor) -> torch.Tensor:
        """Apply perturbation shift via PoE or linear. Pure-additive background
        (no u*(1-W) masking) — matches model()'s z = z0 + A*W."""

        m_p, w_p, a_p, s_p = (
            m_p[:, 0, :],
            m_p[:, 1, :],
            m_p[:, 2, :],
            m_p[:, 3, :],
        )

        if self.shift == "poe":
            W_scale = pyro.param(
                "W_scale",
                0.1 * torch.ones(self.latent_dim),
                constraint=constraints.positive,
            ).to(u.device)
            return self.__poe(u, m_p, W_scale.pow(2)) + s_p

        elif self.shift == "linear":
            return u + m_p + s_p

        raise ValueError("Invalid shift type")

    @torch.no_grad()
    def _combine_pert_shifts(
        self,
        m1: torch.Tensor,
        m2: torch.Tensor,
        pert_idx_0: int | None = None,
        pert_idx_1: int | None = None,
    ) -> torch.Tensor:
        """Combine two single-pert shift tensors, adding optional bilinear synergy."""
        shift_a = m1[:, 0, :]
        shift_b = m2[:, 0, :]
        # The gate slot (index 1) is retained for tuple-shape compatibility but is
        # vestigial: _apply_shift is now pure-additive and no longer masks the
        # background with it. Keep the OR combination for anyone reading it.
        w_a = m1[:, 1, :]
        w_b = m2[:, 1, :]
        gate  = w_a + w_b - w_a * w_b
        rho_a = m1[:, 2, :]
        rho_b = m2[:, 2, :]
        if self.synergy:
            c_a, c_b = self._parent_scales_from_rho(rho_a, rho_b)
            syn = self._synergy_shift_from_rho(rho_a, rho_b)
            shift = c_a * shift_a + c_b * shift_b
        else:
            syn = torch.zeros_like(shift_a)
            shift = shift_a + shift_b

        return torch.stack(
            [shift, gate, torch.zeros_like(shift), syn],
            dim=1,
        )

    @torch.no_grad()
    def _decode_to_expr(self, z: torch.Tensor, lib_size: float) -> torch.Tensor:
        return lib_size * torch.softmax(self.z_decoder(z), dim=-1)

    # ------------------------------ PerturbModelBase overrides ---------------------------- #

    @torch.no_grad()
    def _run_inference(self, dataset, device, batch_size=1024):
        """
        Encode z = z0_head(cond_encoder(x)) for NTC cells (z0 = z, no shift) and
        z = z_head(cell_encoder(x)) for perturbed cells — matching guide(), which
        no longer subtracts any background mean before cell_encoder.
        """
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        x_list, p_list = [], []
        for x, p in loader:
            x_list.append(x)
            p_list.append(p)

        x_all = torch.cat(x_list, dim=0).to(device, dtype=torch.float32)
        p_all = torch.cat(p_list, dim=0)

        lib_all = x_all.sum(dim=1, keepdim=True)
        med = torch.median(lib_all).item()
        x_norm_all = torch.log1p(x_all / lib_all * med)

        N, G = x_all.shape
        if p_all.ndim == 2:
            ntc_mask = (p_all[:, 0] == dataset.ntc_idx) & (p_all[:, 1] == -1)
        else:
            ntc_mask = p_all == dataset.ntc_idx

        z_out     = torch.empty(N, self.latent_dim)
        recon_out = torch.empty(N, G)

        for is_ntc, mask in [(True, ntc_mask), (False, ~ntc_mask)]:
            indices = mask.nonzero(as_tuple=True)[0]
            for i in range(0, len(indices), batch_size):
                idx_b = indices[i : i + batch_size]
                xb    = x_norm_all[idx_b]
                lib_b = lib_all[idx_b]
                if is_ntc:
                    # NTC: z = z0 = cond_encoder(x_ntc), matching guide
                    z_mu, _ = self.z0_head(self.cond_encoder(xb)).chunk(2, dim=-1)
                else:
                    # perturbed: z = z_head(cell_encoder(x_p)), matching guide
                    z_mu, _ = self.z_head(self.cell_encoder(xb)).chunk(2, dim=-1)
                logits = self.z_decoder(z_mu)
                z_out[idx_b]     = z_mu.detach().cpu()
                recon_out[idx_b] = (lib_b * F.softmax(logits, dim=-1)).detach().cpu()

        return (
            z_out.numpy().astype(np.float32),
            recon_out.numpy().astype(np.float32),
            p_all.numpy(),
        )

    # ------------------------------ Sherlock-specific helpers ----------------------------- #
    @torch.no_grad()
    def _counterfactual_effect_size(self, x_ntc_cond, device=None):
        if device is None:
            device = next(self.parameters()).device

        if isinstance(x_ntc_cond, np.ndarray):
            x_ntc = torch.as_tensor(x_ntc_cond, dtype=torch.float32, device=device)
        else:
            x_ntc = x_ntc_cond.to(device=device, dtype=torch.float32)

        n_cells, G = x_ntc.shape
        num_p = self.perturbs

        if n_cells < 2:
            return np.zeros((num_p, G), dtype=np.float32)

        lib = x_ntc.sum(dim=1, keepdim=True) + 1e-8
        var_total = torch.log1p(x_ntc / lib * 1e4).var(dim=0, unbiased=True)  # (G,)

        # abduct backgrounds and shifts once — no repeated Cholesky or re-encoding
        u_ntc  = self._abduct_ntc(x_ntc)            # (n_cells, d)
        shifts = self._get_all_pert_shifts(device)[1:]   # (P, d)

        probs0   = self._decode_to_expr(u_ntc, 1.0)    # (n_cells, G) — proportions
        mu0_norm = torch.log1p(probs0 * 1e4)

        ev_pg = torch.zeros(num_p, G, device=device, dtype=torch.float32)
        for p_idx in range(num_p):
            z_cf     = self._apply_shift(u_ntc, shifts[[p_idx]])    # (n_cells, d)
            probs1   = self._decode_to_expr(z_cf, 1.0)              # (n_cells, G)
            mu1_norm = torch.log1p(probs1 * 1e4)
            mean_shift = (mu1_norm - mu0_norm).mean(dim=0)
            ev_pg[p_idx] = mean_shift.pow(2) / (var_total + 1e-8)

        return ev_pg.cpu().numpy().astype(np.float32)
    
    @staticmethod
    def _silhouette_pair(z_c1: np.ndarray, z_c2: np.ndarray) -> float:
        """
        Mean silhouette score for a 2-condition split in z-space.
        Scale- and rotation-invariant → comparable across model runs.
        Returns float in [-1, 1]: 1=separated, 0=boundary, <0=mixed.
        """
        n1, n2 = len(z_c1), len(z_c2)
        if n1 < 2 or n2 < 2:
            return np.nan
        d11 = np.linalg.norm(z_c1[:, np.newaxis] - z_c1[np.newaxis], axis=-1)
        d22 = np.linalg.norm(z_c2[:, np.newaxis] - z_c2[np.newaxis], axis=-1)
        d12 = np.linalg.norm(z_c1[:, np.newaxis] - z_c2[np.newaxis], axis=-1)
        a1 = d11.sum(axis=1) / (n1 - 1)
        b1 = d12.mean(axis=1)
        s1 = (b1 - a1) / np.maximum(a1, b1)
        a2 = d22.sum(axis=1) / (n2 - 1)
        b2 = d12.mean(axis=0)
        s2 = (b2 - a2) / np.maximum(a2, b2)
        return float(np.concatenate([s1, s2]).mean())

    @torch.no_grad()
    def condition_perturbation_interaction(self, adata, ds, obsm_key):
        """
        Condition-perturbation interaction via silhouette score in z-space.

        score[p, c1, c2] = mean silhouette of cells with perturbation p,
                        labeled by condition (c1 vs c2).

        Range [-1, 1]:
        +1 : conditions perfectly separated for this drug
        0 : conditions on the boundary
        <0 : conditions mixed

        Returns: (P, C, C) float32 array, NaN where a perturbation has <2 cells
                in one of the two conditions.
        """
        p_key = get_config("pert_key")
        treatment_key = get_config("treatment_key")

        z_all = np.array(adata.obsm[obsm_key], dtype=np.float32)
        pert_col = np.array(adata.obs[p_key].values, dtype=str)
        cond_col = np.array(adata.obs[treatment_key].values, dtype=str)

        P, C = self.perturbs, self.conds

        idx_to_pert = {v: k for k, v in ds.perturbation_dict.items()}
        idx_to_cond = {v: k for k, v in ds.condition_dict.items()}
        pert_names = [idx_to_pert[i] for i in range(P)]
        cond_names = [idx_to_cond[i] for i in range(C)]

        valid = ~np.isnan(z_all).any(axis=1)

        score = np.full((P, C, C), np.nan, dtype=np.float32)
        for p, pname in enumerate(pert_names):
            for c1 in range(C):
                for c2 in range(c1 + 1, C):
                    mask_c1 = valid & (pert_col == pname) & (cond_col == cond_names[c1])
                    mask_c2 = valid & (pert_col == pname) & (cond_col == cond_names[c2])
                    s = self._silhouette_pair(z_all[mask_c1], z_all[mask_c2])
                    score[p, c1, c2] = s
                    score[p, c2, c1] = s

        return score.astype(np.float32)

    @torch.no_grad()
    def get_gi(
        self,
        adata,
        min_cells: int = 5,
        mode: str = "hybrid",
        observed_singles: str = "decoded",
        obsm_key: str = "z",
        approximate: bool = False,
        device=None,
    ) -> "pd.DataFrame":
        """Norman-style genetic-interaction metrics; regressed by the shared
        ``regress_gi_params`` (the *same* downstream GEARS uses), on plain
        log2-CPM deltas over all genes (no z-scoring).

        Singles and doubles depend on ``mode``:

        - ``mode='hybrid'`` (default): **observed** singles + generated doubles.
          Regressing the generated double against real singles keeps the GI
          residual discriminative (fully-generated collapses it: the additive
          part is exactly the singles it's regressed against). Best on the
          classification metrics; component singles are always observed even for
          held-out combos, so this still handles OOD doubles.
        - ``mode='generated'``: singles *and* doubles generated (fully
          counterfactual — the strictly-fair-vs-GEARS setting, but limited by how
          non-additive the learned combo shift is).
        - ``mode='observed'``: singles *and* doubles taken from the decoded
          observed q(z) (mean-of-decode over each perturbation's own cells). No
          counterfactual shift is applied anywhere, so this needs the combo cells
          to be present in ``adata`` (not OOD-capable), but it measures GI purely
          from the model's reconstruction of the real data.

        All three effect vectors share one baseline (population-average decoded
        NTC), so the additive residual is clean. Assumes a single condition.

        Parameters
        ----------
        min_cells : minimum observed cells per perturbation to score a pair.
        mode : 'hybrid', 'generated', or 'observed'.
        observed_singles : for hybrid, 'decoded' (decode of mean q(z) — stays in
            decoder space, consistent with the generated double) or 'raw' (mean
            of log-normed raw counts — preserves real single structure, which the
            similarity-based classes like synergy-dissimilar depend on).
        obsm_key : obs latents used by observed_singles='decoded' (needs eval()).
        approximate : True = 25 K-means centroids for the NTC background; False
            (default) = every NTC cell (exact population average).
        """
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.cluster import KMeans
        from .._configs import get_config

        if mode not in ("hybrid", "generated", "observed"):
            raise ValueError("mode must be 'hybrid', 'generated', or 'observed'.")
        if observed_singles not in ("decoded", "raw"):
            raise ValueError("observed_singles must be 'decoded' or 'raw'.")

        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")
        sep = "+"

        def canon(s):
            s = str(s)
            return sep.join(sorted(s.split(sep))) if sep in s else s

        if device is None:
            device = next(self.parameters()).device

        from ._datasets import PerturbSimpleDataset
        ds = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label, combinatorial=True)
        idx2pert = ds.idx_to_pert()
        name2idx = {n: i for i, n in idx2pert.items()}
        all_shifts = self._get_all_pert_shifts(device)   # (P+1, 4, d), row 0 = NTC

        labels = np.asarray(adata.obs[p_key].astype(str))

        # ── NTC background + one shared baseline ─────────────────────────────
        ntc_mask = labels == str(ntc_label)
        x_ntc = adata.X[ntc_mask]
        x_ntc = x_ntc.toarray() if hasattr(x_ntc, "toarray") else np.asarray(x_ntc)
        u_ntc = self._abduct_ntc(torch.tensor(x_ntc, dtype=torch.float32, device=device))
        if approximate:
            n_k = min(25, u_ntc.shape[0])
            km = KMeans(n_clusters=n_k, n_init=10, random_state=0).fit(u_ntc.detach().cpu().numpy())
            centroids = torch.tensor(km.cluster_centers_, dtype=u_ntc.dtype, device=device)
            cnt = np.bincount(km.labels_, minlength=n_k)
            weights = (cnt / cnt.sum())[:, None]
        else:
            centroids = u_ntc
            weights = np.full((u_ntc.shape[0], 1), 1.0 / u_ntc.shape[0])

        def wln(z):  # weighted-mean log2-CPM of decode(z) over backgrounds -> (G,)
            mu = self._decode_to_expr(z, 1.0)                       # proportions
            return (weights * self._lognorm(mu.detach().cpu().numpy())).sum(0)

        def mean_ln_decode(Z):  # mean over rows of log2-CPM(decode(Z)) -> (G,)
            out = None
            for i in range(0, Z.shape[0], 4096):
                ln = self._lognorm(self._decode_to_expr(Z[i:i + 4096], 1.0).detach().cpu().numpy())
                out = ln.sum(0) if out is None else out + ln.sum(0)
            return out / Z.shape[0]

        baseline_ln = wln(centroids)                       # decoded NTC baseline (mean-of-decode)
        raw_baseline_ln = self._lognorm(x_ntc).mean(0)     # raw NTC baseline (mean-of-lognorm)

        def gen_effect(shift_tuple):
            return wln(self._apply_shift(centroids, shift_tuple)) - baseline_ln

        z_obs = np.asarray(adata.obsm[obsm_key], dtype=np.float32) if (
            (mode == "hybrid" and observed_singles == "decoded")
            or mode == "observed") else None

        def obs_single(gene):
            mask = labels == str(gene)
            if mask.sum() == 0:
                return None
            if observed_singles == "decoded":
                # mean-of-decode over the cells (NOT decode-of-mean): matches the
                # mean-of-decode baseline, so a null single -> 0 (decode-of-mean vs
                # mean-of-decode baseline leaves a Jensen offset that biases the fit).
                Z = torch.tensor(z_obs[mask], dtype=torch.float32, device=device)
                return mean_ln_decode(Z) - baseline_ln
            # raw: real log-normed expression minus the *raw* NTC baseline (both in
            # observed space -> null single -> 0), not the decoded baseline.
            xr = adata.X[mask]
            xr = xr.toarray() if hasattr(xr, "toarray") else np.asarray(xr)
            return self._lognorm(xr).mean(0) - raw_baseline_ln

        # ── build effect vectors ─────────────────────────────────────────────
        effects: dict[str, np.ndarray] = {}
        genes = [n for i, n in idx2pert.items() if n != ntc_label]
        for g in genes:
            if mode == "generated":
                effects[canon(g)] = gen_effect(all_shifts[[name2idx[g]]])
            else:
                v = obs_single(g)
                if v is not None:
                    effects[canon(g)] = v

        combo_labels = sorted({canon(l) for l in set(labels) - {str(ntc_label)} if sep in str(l)})
        canon_labels = np.array([canon(l) for l in labels]) if mode == "observed" else None
        for c in combo_labels:
            idxs = [name2idx[p] for p in c.split(sep) if p in name2idx]
            if len(idxs) != 2:
                continue
            if mode == "observed":
                # decoded q(z) of the combo's own observed cells (mean-of-decode),
                # matching the decoded-NTC baseline — no counterfactual shift.
                cmask = canon_labels == c
                if cmask.sum() == 0:
                    continue
                Z = torch.tensor(z_obs[cmask], dtype=torch.float32, device=device)
                effects[c] = mean_ln_decode(Z) - baseline_ln
                continue
            m_p = self._combine_pert_shifts(
                all_shifts[[idxs[0]]], all_shifts[[idxs[1]]],
                pert_idx_0=idxs[0], pert_idx_1=idxs[1],
            )
            effects[c] = gen_effect(m_p)

        combos = [c for c in combo_labels
                  if c in effects and all(canon(p) in effects for p in c.split(sep))]

        raw = adata.obs[p_key].astype(str).value_counts().to_dict()
        n_obs = {canon(k): int(v) for k, v in raw.items() if str(k) != str(ntc_label)}

        return regress_gi_params(
            effects,
            combos=combos,
            sep=sep,
            n_obs=n_obs,
            min_cells=min_cells,
            progress=True,
        )

    @torch.no_grad()
    def _eval(self, adata, obsm_key: str, device: torch.device) -> dict:
        from scipy.stats import spearmanr, pearsonr
        from ._datasets import PerturbMatchingDataset

        p_key         = get_config("pert_key")
        ntc_label     = get_config("ntc_label")
        treatment_key = get_config("treatment_key")

        self.to(device)
        ds = PerturbMatchingDataset(adata, combinatorial=self.combinatorial)
        result = {}

        # ── classifier accuracy ───────────────────────────────────────
        P_sub = adata[adata.obs[p_key].values != ntc_label]
        z_np  = P_sub.obsm[obsm_key]
        z     = torch.as_tensor(z_np, dtype=torch.float32, device=device)
        logits = self.cls_head(z)

        if self.combinatorial:
            p_idx_2d = torch.from_numpy(ds.P_indices).to(device)   # (N, 2)
            target = torch.zeros(z.size(0), self.perturbs, device=device)
            comp_mask = p_idx_2d != -1
            target.scatter_(1, p_idx_2d.clamp(min=0), comp_mask.to(target.dtype))
            pred = (logits.sigmoid() > 0.5).to(target.dtype)
            result["acc"] = (pred == target).all(dim=1).float().mean().item()
        else:
            true_idx = np.array(
                [ds.perturbation_dict[n] for n in P_sub.obs[p_key].values], dtype=np.int64
            )
            y_true = torch.as_tensor(true_idx, dtype=torch.long, device=device)
            result["acc"] = (logits.argmax(dim=-1) == y_true).float().mean().item()

        # ── rho_corr from learned covariance + A*W embed ─────────────
        L        = self.qr().cpu()
        sigma_P  = pyro.param("sigma_fac").cpu().detach()
        Sigma_P  = L @ L.T + sigma_P ** 2 * torch.eye(L.size(0))
        rho_corr = self._corr(Sigma_P).numpy()
        result["rho_corr"] = rho_corr

        q_rho  = self.rho_enc(self.p_emb.weight).detach().cpu()
        chol_P = torch.linalg.cholesky(Sigma_P.to(q_rho.dtype))
        A      = (chol_P @ q_rho).detach().numpy()
        W_np   = self.gate(deterministic=True).detach().cpu().numpy()
        result["rho_embed"] = (A * W_np).astype(np.float32)

        # ── pseudobulk z_corr for non-NTC cells (matches rho_corr dims) ─
        num_p      = self.perturbs
        P_idx      = torch.from_numpy(ds.P_indices)
        z_all_pert = torch.as_tensor(z_np, dtype=torch.float32)
        z_bar      = torch.zeros(num_p, self.latent_dim)

        if self.combinatorial:
            single_mask = P_idx[:, 1] == -1
            P_idx_1d    = P_idx[single_mask, 0]
            z_single    = z_all_pert[single_mask]
            for pidx in range(num_p):
                mask = P_idx_1d == pidx
                if mask.any():
                    z_bar[pidx] = z_single[mask].mean(0)
        else:
            for pidx in range(num_p):
                mask = P_idx == pidx
                if mask.any():
                    z_bar[pidx] = z_all_pert[mask].mean(0)

        nz = z_bar.norm(dim=1) > 1e-12
        z_corr_mat = torch.full((num_p, num_p), float("nan"))
        if nz.sum() >= 2:
            sub_corr = torch.corrcoef(z_bar[nz])
            nz_idx = torch.where(nz)[0]
            z_corr_mat[nz_idx.unsqueeze(1), nz_idx.unsqueeze(0)] = sub_corr
        torch.diagonal(z_corr_mat).fill_(1.0)
        result["z_corr"] = z_corr_mat.numpy()

        # ── z_corr vs rho_corr ────────────────────────────────────────
        def _flat_triu(M):
            M   = torch.tensor(M) if not torch.is_tensor(M) else M
            idx = torch.triu_indices(M.size(0), M.size(1), offset=1)
            return M[idx[0], idx[1]]

        r_s, p_s = spearmanr(_flat_triu(rho_corr), _flat_triu(z_corr_mat))
        r_p, p_p = pearsonr( _flat_triu(rho_corr), _flat_triu(z_corr_mat))
        result["z_rho_corr"] = {
            "spearman_r": r_s, "spearman_p": p_s,
            "pearson_r":  r_p, "pearson_p":  p_p,
        }

        # ── gate weights W ────────────────────────────────────────────
        result["W"] = self.gate(deterministic=True).cpu().detach().numpy()
        result["synergy_enabled"] = bool(self.synergy)
        result["synergy_rank"] = int(self.synergy_rank)

        # ── explained variance (condition × perturbation × gene) ─────
        x_ntc_mat = ds.X_ntc  # list over cond: cells x genes

        cond_unique = adata.obs[treatment_key].unique()
        conds = torch.tensor([ds.condition_dict[c] for c in cond_unique], dtype=torch.int32)

        ev_list = []
        device = next(self.parameters()).device

        for _, cond_idx in zip(cond_unique, conds):
            x_ntc_cond = x_ntc_mat[cond_idx]
            ev_pg = self._counterfactual_effect_size(
                x_ntc_cond=x_ntc_cond,
                device=device,
            )
            ev_list.append(ev_pg)
        result['counterfactual_effect_size'] = np.stack(ev_list, axis=0)  # (C, P, G)

        # Condition-Perturbation Interaction (P, C, C)
        cpi = self.condition_perturbation_interaction(adata, ds, obsm_key)
        result['condition_perturbation_interaction'] = cpi  # (P, C, C)

        result['conditions'] = np.array(cond_unique)
        idx_to_pert = {idx: name for name, idx in ds.perturbation_dict.items()}
        result['rho_perts'] = np.array([idx_to_pert[i] for i in range(len(idx_to_pert))])

        if self.synergy and self.combinatorial:
            combo_rows = ds.P_indices[ds.P_indices[:, 1] != -1]
            unique_pairs = sorted({
                (min(int(r[0]), int(r[1])), max(int(r[0]), int(r[1])))
                for r in combo_rows
            })
            if unique_pairs:
                pair_idx = torch.tensor(unique_pairs, dtype=torch.long, device=device)
                q_rho_d = self.rho_enc(self.p_emb.weight).detach().to(device)
                parent_c_a, parent_c_b = self._parent_scales_from_rho(
                    q_rho_d[pair_idx[:, 0]],
                    q_rho_d[pair_idx[:, 1]],
                )
                syn_shift = self._synergy_shift_from_rho(
                    q_rho_d[pair_idx[:, 0]],
                    q_rho_d[pair_idx[:, 1]],
                )
                result["synergy_pair_names"] = np.array([
                    f"{idx_to_pert[i]}+{idx_to_pert[j]}" for i, j in unique_pairs
                ])
                result["synergy_pair_norm"] = syn_shift.norm(dim=1).detach().cpu().numpy()
                result["parent_scale_a"] = parent_c_a.squeeze(-1).detach().cpu().numpy()
                result["parent_scale_b"] = parent_c_b.squeeze(-1).detach().cpu().numpy()

        # # ── geometric synergy classification ──────────────────────────
        # if self.combinatorial:
        #     result['synergy'] = self.get_synergy_new(adata, obsm_key)

        return result
