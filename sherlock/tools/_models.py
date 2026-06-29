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
            (1, "singles_to_doubles"),
            (-1, "additive_residual_norm"),
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
            (1, "signed_residual_alignment"),
            (1, "singles_similarity"),
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
    if synergy_mask.any():
        threshold = out.loc[pred.index[synergy_mask], "singles_similarity"].median()
    else:
        threshold = out.loc[valid, "singles_similarity"].median()

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
        treat_effect_map=None,
        combinatorial=False,
        synergy=False,
        synergy_rank=4,
        z_var=1.0,
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
        self.synergy_rank = int(synergy_rank)

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
        self.treat_effect_map = treat_effect_map
        self.z_var = float(z_var)

        # embeddings
        self.p_emb = nn.Embedding(perturbs, latent_dim)
        # per-condition NTC mean in mednorm space; populated by init_cond_means_from_adata
        self.register_buffer("cond_x_mean", torch.zeros(conds, input_dim))
        # global NTC mean across all conditions; populated by init_global_mean_from_adata
        self.register_buffer("global_x_mean", torch.zeros(input_dim))

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
            if self.synergy_rank <= 0:
                raise ValueError("synergy_rank must be positive when synergy=True")
            self.syn_factor = nn.Linear(latent_dim, self.synergy_rank, bias=False)
            self.syn_proj = nn.Linear(self.synergy_rank, latent_dim, bias=False)
            self.parent_coeff = nn.Linear(latent_dim, 1, bias=False)
            nn.init.normal_(self.syn_factor.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.syn_proj.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.parent_coeff.weight)
        else:
            self.syn_factor = None
            self.syn_proj = None
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
        return logits_nb

    def _synergy_shift_from_rho(
        self,
        rho_a: torch.Tensor,
        rho_b: torch.Tensor,
    ) -> torch.Tensor:
        if not self.synergy:
            return torch.zeros_like(rho_a)
        h_a = self.syn_factor(rho_a)
        h_b = self.syn_factor(rho_b)
        return self.syn_proj(h_a * h_b)

    def _parent_scales_from_rho(
        self,
        rho_a: torch.Tensor,
        rho_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.synergy:
            ones = rho_a.new_ones((*rho_a.shape[:-1], 1))
            return ones, ones

        score = self.parent_coeff(rho_a - rho_b)
        c_a = 1.0 + torch.tanh(score)
        c_b = 1.0 + torch.tanh(-score)
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

        with pyro.plate("perturbations", self.perturbs):
            rho_single = pyro.sample(
                "rho",
                dist.Normal(torch.zeros_like(self.p_emb.weight), torch.ones_like(self.p_emb.weight)).to_event(1),
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
        n_ntc = x_ntc.size(0)
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
                pyro.factor("synergy_l2", -self.l2_lambda * syn_shift_p.pow(2).sum())
                pyro.factor("parent_scale_l2", -self.l2_lambda * parent_scale_penalty.sum())
                pyro.deterministic("syn_shift", syn_shift_p)
                pyro.deterministic("parent_scale", torch.cat([c_a, c_b], dim=-1))
            W_b_eff     = W[p1] * valid_m.float().unsqueeze(-1)
            W_p_eff     = W[p0] + W_b_eff - W[p0] * W_b_eff
        else:
            lin_shift_p = A[p] * W[p]                                  # (n, d)
            W_p_eff     = W[p]                                         # (n, d)

        # ── n_all tensors: NTC cells first (W=0), perturbed cells second ─
        zeros_ntc     = torch.zeros(n_ntc, self.latent_dim, device=device)
        W_all         = torch.cat([zeros_ntc, W_p_eff],    dim=0)  # (n_all, d)
        lin_shift_all = torch.cat([zeros_ntc, lin_shift_p], dim=0)  # (n_all, d)

        # isotropic z0 prior — condition adjustment done in gene space via the residual guide
        z0_loc_all   = torch.zeros(n_all, self.latent_dim, device=device)
        z0_scale_all = torch.ones( n_all, self.latent_dim, device=device)

        z_var = torch.full((self.latent_dim,), self.z_var, device=device)

        if self.shift == 'poe':
            #Here to not be in cell plate
            W_scale = pyro.param("W_scale", 0.1 * torch.ones(self.latent_dim, device=device), constraint=constraints.positive)
            pyro.factor("W_scale_prior", -0.5 * (W_scale / 0.1).pow(2).sum())

        # ── cells plate: one z per cell (NTC: W=0 ⟹ z≈z0; perturbed: z=shift(z0)) ─
        with pyro.plate("cells", n_all):
            #TODO this is not proper POE for what we need. should be using z0_mu and z0_std
            #but gradient won't flow this way since its not from pyro.sample. Temp fix using
            #sampled z0 but its not ideal since it won't have the same mean/var as the guide's z0 distribution.
            #A better fix would be to implement the POE logic manually in the guide and model instead of relying on pyro.sample for z0
            z0     = pyro.sample("z0", dist.Normal(z0_loc_all, z0_scale_all).to_event(1), infer={"scale": self.kl_weight})
            z0_mod = z0 * (1.0 - W_all)

            if self.shift == 'poe':
                z_loc = self.__poe(z0_mod, lin_shift_all, W_scale.pow(2))
            elif self.shift == 'linear':
                z_loc = z0_mod + lin_shift_all
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

        logits_ntc  = self.__decode(x_ntc, z_ntc_z, self.z_decoder, theta, "x_ntc")
        ntc_logprob = dist.NegativeBinomial(total_count=theta, logits=logits_ntc).log_prob(x_ntc.float()).sum()
        pyro.factor("X_ntc_logprob", self.ntc_lambda * ntc_logprob)

        logits_p  = self.__decode(x_p, z_p_z, self.z_decoder, theta, "x_p")
        p_logprob = dist.NegativeBinomial(total_count=theta, logits=logits_p).log_prob(x_p.float()).sum()
        pyro.factor("X_p_logprob", self.pert_lambda * p_logprob)

    def guide(self, x_p, x_ntc, p, c, c_ntc):
        pyro.module("VAE", self)

        x_p   = self._mednorm(x_p)
        x_ntc = self._mednorm(x_ntc)

        n_ntc = x_ntc.size(0)
        n     = x_p.size(0)
        n_all = n_ntc + n

        q_loc = self.rho_enc(self.p_emb.weight)
        with pyro.plate("perturbations", self.perturbs):
            pyro.sample("rho", dist.Normal(q_loc, torch.ones_like(q_loc)).to_event(1), infer={"scale": self.kl_weight})


        if self.use_conditions:
            # NTC: z0 = z — encode each NTC cell through cond_encoder (same path as
            # perturbed z0). No residual: cond_encoder carries per-cell background.
            z0_mu_ntc, z0_logvar_ntc = self.z0_head(self.cond_encoder(x_ntc)).chunk(2, dim=-1)
            z0_std_ntc = (0.5 * z0_logvar_ntc).exp()
            z_mu_ntc   = z0_mu_ntc
            z_std_ntc  = z0_std_ntc

            # Perturbed: z0 from fixed per-condition mean buffer; z from per-cell residual.
            cond_mean_p = self.cond_x_mean[c]   # (n, G) — fixed buffer, no gradient
            z0_mu_p, z0_logvar_p = self.z0_head(self.cond_encoder(cond_mean_p)).chunk(2, dim=-1)
            z0_std_p = (0.5 * z0_logvar_p).exp()
            x_p_enc  = x_p - cond_mean_p
            z_mu_p, z_logvar_p = self.z_head(self.cell_encoder(x_p_enc)).chunk(2, dim=-1)
            z_std_p  = (0.5 * z_logvar_p).exp()
        else:
            # no conditions: mirrors use_conditions path with global NTC mean as background
            # NTC: z0 = z via cond_encoder(x_ntc) — per-cell background
            z0_mu_ntc, z0_logvar_ntc = self.z0_head(self.cond_encoder(x_ntc)).chunk(2, dim=-1)
            z0_std_ntc = (0.5 * z0_logvar_ntc).exp()
            z_mu_ntc   = z0_mu_ntc
            z_std_ntc  = z0_std_ntc

            # Perturbed: z0 from global NTC mean buffer; z from per-cell residual
            global_mean_p = self.global_x_mean.unsqueeze(0).expand(n, -1)
            z0_mu_p, z0_logvar_p = self.z0_head(self.cond_encoder(global_mean_p)).chunk(2, dim=-1)
            z0_std_p = (0.5 * z0_logvar_p).exp()
            z_mu_p, z_logvar_p = self.z_head(self.cell_encoder(x_p - global_mean_p)).chunk(2, dim=-1)
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

    def init_cond_means_from_adata(self, adata, dataset) -> None:
        """Populate cond_x_mean and global_x_mean buffers with NTC means in mednorm space."""
        import scipy.sparse as sp

        ntc_label     = get_config("ntc_label")
        pert_key      = get_config("pert_key")
        treatment_key = get_config("treatment_key")

        X = adata.X
        if sp.issparse(X):
            X = X.toarray().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        lib    = X.sum(1, keepdims=True)
        med    = float(np.median(lib))
        X_norm = np.log1p(X / (lib + 1e-8) * med)

        obs_pert = adata.obs[pert_key].values
        obs_cond = adata.obs[treatment_key].values

        means = np.zeros((self.conds, self.input_dim), dtype=np.float32)
        for cond_label, cond_idx in dataset.condition_dict.items():
            mask = (obs_pert == ntc_label) & (obs_cond == cond_label)
            if mask.any():
                means[cond_idx] = X_norm[mask].mean(0)

        self.cond_x_mean.copy_(torch.tensor(means))

        ntc_mask = obs_pert == ntc_label
        global_mean = X_norm[ntc_mask].mean(0) if ntc_mask.any() else np.zeros(self.input_dim, dtype=np.float32)
        self.global_x_mean.copy_(torch.tensor(global_mean))

    def init_global_mean_from_adata(self, adata) -> None:
        """Populate global_x_mean buffer with the NTC mean in mednorm space (no-conditions path)."""
        import scipy.sparse as sp

        ntc_label = get_config("ntc_label")
        pert_key  = get_config("pert_key")

        X = adata.X
        if sp.issparse(X):
            X = X.toarray().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        lib    = X.sum(1, keepdims=True)
        med    = float(np.median(lib))
        X_norm = np.log1p(X / (lib + 1e-8) * med)

        ntc_mask = adata.obs[pert_key].values == ntc_label
        global_mean = X_norm[ntc_mask].mean(0) if ntc_mask.any() else np.zeros(self.input_dim, dtype=np.float32)
        self.global_x_mean.copy_(torch.tensor(global_mean))

    # ------------------------------ PerturbModelBase interface ----------------------------- #

    @torch.no_grad()
    def checkpoint_ctor_args(self) -> dict:
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
            "z_var":              self.z_var,
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
        """Apply perturbation shift via PoE or linear."""

        m_p, w_p, a_p, s_p = (
            m_p[:, 0, :],
            m_p[:, 1, :],
            m_p[:, 2, :],
            m_p[:, 3, :],
        )

        u_mod = u * (1.0 - w_p)

        if self.shift == "poe":
            W_scale = pyro.param(
                "W_scale",
                0.1 * torch.ones(self.latent_dim),
                constraint=constraints.positive,
            ).to(u.device)
            return self.__poe(u_mod, m_p, W_scale.pow(2)) + s_p

        elif self.shift == "linear":
            return u_mod + m_p + s_p

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
        gate  = (m1[:, 1, :] + m2[:, 1, :]).clamp(0.0, 1.0)
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
        Encode z as z_head(cell_encoder(x - x_cond_mean)) when use_conditions=True,
        else z_head(cell_encoder(x)). NTC cells have no perturbation shift; perturbed cells do.
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

        # Load condition indices if available for perturbed-cell residual subtraction
        c_all = None
        if self.use_conditions:
            c_indices = getattr(dataset, "C_indices", None)
            if c_indices is not None:
                c_all = np.array(c_indices, dtype=np.int64)

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
                elif self.use_conditions and c_all is not None:
                    # perturbed with conditions: subtract per-condition buffer mean
                    c_batch = torch.from_numpy(c_all[idx_b.cpu().numpy()]).to(xb.device)
                    xb_enc  = xb - self.cond_x_mean[c_batch]
                    z_mu, _ = self.z_head(self.cell_encoder(xb_enc)).chunk(2, dim=-1)
                else:
                    # perturbed without conditions: subtract global NTC mean
                    xb_enc = xb - self.global_x_mean.to(xb.device)
                    z_mu, _ = self.z_head(self.cell_encoder(xb_enc)).chunk(2, dim=-1)
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
        obsm_key: str,
        min_cells: int = 5,
        observed: bool | str = True,
    ) -> "pd.DataFrame":
        """Compute gene-space GI metrics for every combo pair via Norman-style regression.

        Parameters
        ----------
        observed
            If True, use observed encoded z means from ``adata.obsm[obsm_key]``.
            If False, use generated perturbation latents from the model's learned
            perturbation shifts. If ``"hybrid"``, use observed single-perturbation
            z means and generated double-perturbation latents. The default True
            path preserves the historical behavior used by previous runs.
        """
        import numpy as np
        import pandas as pd
        import scipy.sparse as sp
        import torch
        import dcor as _dcor
        from sklearn.linear_model import TheilSenRegressor
        from tqdm.auto import tqdm
        from ._datasets import PerturbMatchingDataset

        p_key     = get_config("pert_key")
        ntc_label = get_config("ntc_label")
        eps_      = 1e-8

        mode = observed
        if isinstance(mode, str):
            mode = mode.lower()
        valid_modes = {True, False, "hybrid"}
        if mode not in valid_modes:
            raise ValueError('observed must be True, False, or "hybrid".')

        def _norm(x):
            return float(np.linalg.norm(x))

        def _cos(x, y):
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            if denom < eps_:
                return np.nan
            return float(np.dot(x, y) / denom)

        _device = next(self.parameters()).device

        labels   = np.array(adata.obs[p_key].values, dtype=str)
        z_all    = np.array(adata.obsm[obsm_key], dtype=np.float32)
        mean_ntc = z_all[labels == ntc_label].mean(axis=0)

        # P_idx from PerturbMatchingDataset is aligned to non-NTC cells.
        z_pert = z_all[labels != ntc_label]

        # Norman-style control normalization from observed NTC raw counts
        X_ntc = adata.X[labels == ntc_label]
        lib_ntc = np.asarray(X_ntc.sum(axis=1)).ravel()
        target_umi = float(np.median(lib_ntc[lib_ntc > eps_]))

        scale = target_umi / np.maximum(lib_ntc, eps_)
        if sp.issparse(X_ntc):
            X_ntc_scaled = X_ntc.multiply(scale[:, None])
            ctrl_mean = np.asarray(X_ntc_scaled.mean(axis=0)).ravel()
            ctrl_sq_mean = np.asarray(X_ntc_scaled.power(2).mean(axis=0)).ravel()
        else:
            X_ntc_scaled = np.asarray(X_ntc, dtype=np.float64) * scale[:, None]
            ctrl_mean = X_ntc_scaled.mean(axis=0)
            ctrl_sq_mean = (X_ntc_scaled ** 2).mean(axis=0)

        ctrl_std = np.sqrt(np.maximum(ctrl_sq_mean - ctrl_mean ** 2, eps_))

        def _norm_decoded_expr(x):
            x = np.asarray(x, dtype=np.float64).ravel()
            x = x * (target_umi / max(float(x.sum()), eps_))
            return (x - ctrl_mean) / (ctrl_std + eps_)

        def _decode_z_to_norm_expr(z):
            z_t = torch.tensor(z, dtype=torch.float32, device=_device)
            expr_raw = self._decode_to_expr(
                z_t.unsqueeze(0), 1.0
            ).squeeze(0).detach()
            return _norm_decoded_expr(expr_raw.cpu().numpy())

        ds          = PerturbMatchingDataset(adata, combinatorial=True)
        idx_to_pert = {v: k for k, v in ds.perturbation_dict.items()}
        P_idx       = ds.P_indices

        if len(P_idx) != len(z_pert):
            raise ValueError(
                f"P_idx length {len(P_idx)} does not match non-NTC z length {len(z_pert)}. "
                "Use z_pert = z_all[labels != ntc_label], or check PerturbMatchingDataset alignment."
            )

        # Exclude perturbation target genes from regression/metrics
        pert_genes = set()
        for p in idx_to_pert.values():
            if str(p) != str(ntc_label):
                for g in str(p).split("+"):
                    g = g.strip()
                    if g:
                        pert_genes.add(g)

        gene_names = np.asarray(adata.var_names.astype(str))
        reg_gene_mask = ~np.isin(gene_names, list(pert_genes))

        expr_ntc = _decode_z_to_norm_expr(mean_ntc)

        generated_z_cache = {}

        if mode is False or mode == "hybrid":
            base_z = torch.tensor(
                mean_ntc,
                dtype=torch.float32,
                device=_device,
            ).unsqueeze(0)
            all_shifts = self._get_all_pert_shifts(_device)

            def _generated_single_z(idx):
                key = ("single", int(idx))
                if key not in generated_z_cache:
                    shift = all_shifts[[int(idx) + 1]]
                    generated_z_cache[key] = (
                        self._apply_shift(base_z, shift)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                return generated_z_cache[key]

            def _generated_combo_z(idx_a, idx_b):
                idx_a = int(idx_a)
                idx_b = int(idx_b)
                key = ("combo", min(idx_a, idx_b), max(idx_a, idx_b))
                if key not in generated_z_cache:
                    shift_a = all_shifts[[idx_a + 1]]
                    shift_b = all_shifts[[idx_b + 1]]
                    combo_shift = self._combine_pert_shifts(
                        shift_a,
                        shift_b,
                        pert_idx_0=idx_a,
                        pert_idx_1=idx_b,
                    )
                    generated_z_cache[key] = (
                        self._apply_shift(base_z, combo_shift)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                return generated_z_cache[key]

        combo_rows  = P_idx[P_idx[:, 1] != -1]
        unique_keys = {
            (min(int(r[0]), int(r[1])), max(int(r[0]), int(r[1])))
            for r in combo_rows
        }

        records = []

        desc = "get_gi" if mode is True else ("get_gi_hybrid" if mode == "hybrid" else "get_gi_generated")
        for (i, j) in tqdm(sorted(unique_keys), desc=desc, unit="pair"):
            name_a = idx_to_pert[i]
            name_b = idx_to_pert[j]

            mask_a  = (P_idx[:, 0] == i) & (P_idx[:, 1] == -1)
            mask_b  = (P_idx[:, 0] == j) & (P_idx[:, 1] == -1)
            mask_ab = ((P_idx[:, 0] == i) & (P_idx[:, 1] == j)) | \
                    ((P_idx[:, 0] == j) & (P_idx[:, 1] == i))

            n_a, n_b, n_ab = int(mask_a.sum()), int(mask_b.sum()), int(mask_ab.sum())

            if n_a < min_cells or n_b < min_cells or n_ab < min_cells:
                records.append({
                    "pert_a": name_a,
                    "pert_b": name_b,
                    "n_a": n_a,
                    "n_b": n_b,
                    "n_ab": n_ab,
                })
                continue

            if mode is True:
                # Observed encoded-z means. This is the historical path.
                mean_z_a  = z_pert[mask_a].mean(axis=0)
                mean_z_b  = z_pert[mask_b].mean(axis=0)
                mean_z_ab = z_pert[mask_ab].mean(axis=0)
            elif mode == "hybrid":
                # Observed single perturbations, generated double perturbation.
                mean_z_a  = z_pert[mask_a].mean(axis=0)
                mean_z_b  = z_pert[mask_b].mean(axis=0)
                mean_z_ab = _generated_combo_z(i, j)
            else:
                # Generated counterfactual latents using the learned perturbation shifts.
                mean_z_a  = _generated_single_z(i)
                mean_z_b  = _generated_single_z(j)
                mean_z_ab = _generated_combo_z(i, j)

            expr_a  = _decode_z_to_norm_expr(mean_z_a)
            expr_b  = _decode_z_to_norm_expr(mean_z_b)
            expr_ab = _decode_z_to_norm_expr(mean_z_ab)

            da_dec = (expr_a  - expr_ntc).astype(np.float64)
            db_dec = (expr_b  - expr_ntc).astype(np.float64)
            dab    = (expr_ab - expr_ntc).astype(np.float64)

            # Remove perturbation target genes from expression-space metrics.
            da_dec = da_dec[reg_gene_mask]
            db_dec = db_dec[reg_gene_mask]
            dab    = dab[reg_gene_mask]

            additive = da_dec + db_dec

            # Norman model: dab ≈ c1*da_dec + c2*db_dec.
            # Norman's paper used a robust Theil-Sen fit for the model coefficients.
            X = np.column_stack([da_dec, db_dec])
            ts = TheilSenRegressor(
                fit_intercept=False,
                max_subpopulation=5000, #100000
                max_iter=300, #1000
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

            # Parent/double dcor details
            max_parent_dcor = max(dc_a, dc_b)
            min_parent_dcor = min(dc_a, dc_b)
            parent_dcor_diff = abs(dc_a - dc_b)

            # Coefficient structure
            c1_abs = abs(c1)
            c2_abs = abs(c2)
            c1_plus_c2 = c1 + c2
            c1_minus_c2 = c1 - c2
            c1_times_c2 = c1 * c2
            same_sign_coeffs = float(np.sign(c1) == np.sign(c2))

            # Norms
            norm_a = _norm(da_dec)
            norm_b = _norm(db_dec)
            norm_ab = _norm(dab)
            norm_additive = _norm(additive)
            norm_fit = _norm(y_pred)

            norm_ab_over_additive = norm_ab / (norm_additive + eps_)
            norm_fit_over_ab = norm_fit / (norm_ab + eps_)

            # Effective fitted contributions
            contrib_a = c1_abs * norm_a
            contrib_b = c2_abs * norm_b

            effective_dominance = float(
                abs(np.log10((contrib_a + eps_) / (contrib_b + eps_)))
            )
            contribution_balance = float(
                min(contrib_a, contrib_b) / (max(contrib_a, contrib_b) + eps_)
            )

            # Cosine geometry
            cos_a_b = _cos(da_dec, db_dec)
            cos_ab_a = _cos(dab, da_dec)
            cos_ab_b = _cos(dab, db_dec)
            cos_ab_additive = _cos(dab, additive)
            cos_ab_fit = _cos(dab, y_pred)

            # Residual geometry
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

                # Norman six
                "magnitude": magnitude,
                "dominance": dominance,
                "model_fit": model_fit,
                "singles_similarity": singles_sim,
                "singles_to_doubles": singles_to_dbl,
                "equality_contribution": eq_contrib,
                "model_fit_r2_centered": model_fit_r2_centered,
                "model_fit_dcor": model_fit_dcor,
                "model_corr": model_corr,

                # Parent-to-double details
                "dcor_a": dc_a,
                "dcor_b": dc_b,
                "max_parent_dcor": max_parent_dcor,
                "min_parent_dcor": min_parent_dcor,
                "parent_dcor_diff": parent_dcor_diff,

                # Coefficients
                "c1": c1,
                "c2": c2,
                "c1_abs": c1_abs,
                "c2_abs": c2_abs,
                "c1_plus_c2": c1_plus_c2,
                "c1_minus_c2": c1_minus_c2,
                "c1_times_c2": c1_times_c2,
                "same_sign_coeffs": same_sign_coeffs,

                # Effective contribution
                "contrib_a": contrib_a,
                "contrib_b": contrib_b,
                "effective_dominance": effective_dominance,
                "contribution_balance": contribution_balance,

                # Norms
                "norm_a": norm_a,
                "norm_b": norm_b,
                "norm_ab": norm_ab,
                "norm_additive": norm_additive,
                "norm_fit": norm_fit,
                "norm_ab_over_additive": norm_ab_over_additive,
                "norm_fit_over_ab": norm_fit_over_ab,

                # Cosines
                "cos_a_b": cos_a_b,
                "cos_ab_a": cos_ab_a,
                "cos_ab_b": cos_ab_b,
                "cos_ab_additive": cos_ab_additive,
                "cos_ab_fit": cos_ab_fit,

                # Residuals
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
        df.index = df["pert_a"] + "+" + df["pert_b"]
        return df

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
