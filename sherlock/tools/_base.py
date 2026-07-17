"""
Abstract base class for perturbation models.

Subclasses must implement:
    get_z(x, p=None, **kwargs)     → (B, d)
    get_recon(x, p=None, **kwargs) → (B, G)

The concrete eval() handles storing z in adata.obsm, x_pred in
adata.layers, and z_corr / perts in adata.uns via get_corr().
Model-specific extras are added via the overridable _eval() hook.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    import anndata


class PerturbModelBase(ABC, nn.Module):
    """Abstract base for all perturbation models in the sherlock package."""

    def checkpoint_ctor_args(self) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} must implement checkpoint_ctor_args() to support save/load"
        )

    # ── abstract interface ────────────────────────────────────────────

    @abstractmethod
    @torch.no_grad()
    def get_z(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Mean posterior latent z.

        Parameters
        ----------
        x  : (B, G) raw counts
        p  : (B,)   integer perturbation indices (optional; model-dependent)
        **kwargs : any additional model-specific inputs

        Returns
        -------
        z_mean : (B, d)
        """
        ...

    @abstractmethod
    @torch.no_grad()
    def get_recon(self, x: torch.Tensor, p: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Reconstructed expected counts.

        Parameters
        ----------
        x  : (B, G) raw counts (used for library-size scaling)
        p  : (B,)   integer perturbation indices (optional; model-dependent)
        **kwargs : any additional model-specific inputs

        Returns
        -------
        mu : (B, G) expected counts at observed library size
        """
        ...

    # ── shared utility ────────────────────────────────────────────────

    @torch.no_grad()
    def get_corr(
        self,
        z: np.ndarray,
        p_indices: np.ndarray,
        all_perts: np.ndarray,
        perts: np.ndarray | None = None,
        remove_ntc: bool = True,
        use_rho: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perturbation-level Pearson correlation matrix.

        Parameters
        ----------
        z          : (N, d) encoder embeddings for every cell
        p_indices  : (N,)   integer perturbation index per cell
        all_perts  : (n_perturbs,) perturbation name for each integer index
        perts      : optional subset of pert names to include; None = all
        remove_ntc : exclude NTC perturbation (default True)
        use_rho    : use model-specific embedding via _get_rho_embed (default
                     True); False falls back to pseudobulk mean encoder z

        Returns
        -------
        corr  : (P, P) float32 Pearson correlation matrix
        perts : (P,)   string array of included perturbation names
        embed : (P, d) float32 embedding used to compute corr
        """
        from .._configs import get_config
        ntc_label = get_config("ntc_label")

        selected = np.isin(all_perts, perts) if perts is not None else np.ones(len(all_perts), bool)
        if remove_ntc:
            selected &= (all_perts != ntc_label)

        sel_idx   = np.where(selected)[0]
        sel_perts = all_perts[selected]
        P         = len(sel_idx)

        if use_rho:
            embed = self._get_rho_embed(sel_idx, z, p_indices, all_perts)
        else:
            d     = z.shape[1]
            embed = np.zeros((P, d), dtype=np.float32)
            for j, pidx in enumerate(sel_idx):
                mask = p_indices == pidx
                if mask.any():
                    embed[j] = z[mask].mean(0)

        if P < 2:
            corr = np.ones((P, P), dtype=np.float32) if P == 1 else np.empty((0, 0), dtype=np.float32)
            return corr, sel_perts, embed

        X        = embed - embed.mean(axis=1, keepdims=True)
        row_norm = np.linalg.norm(X, axis=1)
        nz       = row_norm > 1e-12
        corr     = np.zeros((P, P), dtype=np.float32)
        if nz.sum() >= 2:
            Xn     = X[nz] / row_norm[nz, None]
            Cnz    = np.clip(Xn @ Xn.T, -1.0, 1.0).astype(np.float32)
            nz_idx = np.where(nz)[0]
            corr[np.ix_(nz_idx, nz_idx)] = Cnz
        np.fill_diagonal(corr, 1.0)
        return corr, sel_perts, embed

    def _get_rho_embed(
        self,
        pert_indices: np.ndarray,
        z: np.ndarray,
        p_indices: np.ndarray,
        all_perts: np.ndarray,
    ) -> np.ndarray:
        """
        Model-specific embedding for correlation, shape (P, d).

        Called by get_corr(use_rho=True).  Subclasses must implement this
        to use the model's learned representation (e.g. action_prior_mean,
        A*W) instead of pseudobulk encoder z.

        Parameters
        ----------
        pert_indices : selected global perturbation integer indices (non-NTC)
        z            : (N, d) encoder embeddings for every cell
        p_indices    : (N,)   integer perturbation index per cell
        all_perts    : (n_perturbs,) perturbation name per integer index
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_rho_embed to use get_corr(use_rho=True)"
        )

    @staticmethod
    def _lognorm(x: np.ndarray) -> np.ndarray:
        """log2-CPM normalise a (B, G) count matrix."""
        lib = x.sum(axis=1, keepdims=True)
        return np.log2(1e4 * x / np.clip(lib, 1e-8, None) + 1.0)

    # ── counterfactual hooks (override in subclasses) ─────────────────

    @torch.no_grad()
    def _abduct_ntc(self, x_ntc: torch.Tensor) -> torch.Tensor | None:
        """
        Encode NTC cells → background latent vectors u_i of shape (N_ntc, d).
        Return None to opt out of counterfactual prediction.
        """
        return None

    @torch.no_grad()
    def _apply_shift(self, u: torch.Tensor, m_p: torch.Tensor) -> torch.Tensor:
        """
        Apply perturbation shift m_p to background u.
        Default is pure addition (sVAE+ faithful).
        Override in models that use a different fusion (e.g. PoE).
        u : (N, d), m_p : (1, d) or (N, d) → returns (N, d)
        """
        return u + m_p

    @torch.no_grad()
    def _combine_pert_shifts(
        self,
        m1: torch.Tensor,
        m2: torch.Tensor,
        pert_idx_0: int | None = None,
        pert_idx_1: int | None = None,
    ) -> torch.Tensor:
        """
        Combine two shift tensors for a combinatorial perturbation.
        Default (cVAE/sVAE): additive sum — both are plain (1, d) shift vectors.
        VAE overrides to apply learned c1/c2 scaling using the pert indices.
        """
        return m1 + m2

    @torch.no_grad()
    def _get_all_pert_shifts(self, device: torch.device) -> torch.Tensor | None:
        """
        Return all perturbation shift vectors of shape (P, d).
        Computed once per predict_counterfactual_effects call.
        Return None to opt out of counterfactual prediction.
        """
        return None

    @torch.no_grad()
    def _decode_to_expr(self, z: torch.Tensor, lib_size: float) -> torch.Tensor | None:
        """
        Decode latent z (K, d) → expected counts (K, G) at scalar *lib_size*.
        Return None to opt out of counterfactual prediction.
        """
        return None

    @torch.no_grad()
    def predict_counterfactual_effects(
        self,
        adata,
        n_centroids: int = 25,
        observed_effect=None,
        device: torch.device | None = None,
        approximate: bool = True,
    ) -> dict:
        """
        NTC-population counterfactual effects over abducted control backgrounds.

        Computes the perturbation-by-gene counterfactual effect matrix

            Delta_cf[p, g] = lognorm_g(E_k[f(c_k + m_p)])
                            - lognorm_g(E_k[f(c_k)])

        where c_k are the control backgrounds. With approximate=True (default),
        c_k are `n_centroids` K-means centroids of the abducted NTC backgrounds
        (weighted by cluster size) — a cheap quadrature of the population
        expectation. With approximate=False, c_k are *all* abducted NTC cells
        (uniform weights) — the exact population average, slower but no K-means
        approximation. Empirically the two agree closely (~0.01 ATE); the mean
        alone (n_centroids=1) is notably worse due to the nonlinear decoder
        (Jensen's inequality).

        If `observed_effect` is provided, also computes Pearson correlation between
        the predicted counterfactual effect matrix and the observed/reference effect
        matrix over aligned perturbations and genes.

        Parameters
        ----------
        adata : AnnData
            Full dataset; NTC cells are extracted automatically using the
            configured pert_key / ntc_label.
        n_centroids : int
            Number of K-means centroids (only used when approximate=True).
        observed_effect : AnnData-like or pd.DataFrame, optional
            Observed treatment-effect matrix with perturbations as rows and genes
            as columns. If AnnData-like, uses `.X`, `.obs_names`, `.var_names`.
        device : torch.device, optional
            Device to run on; defaults to the device of the model's parameters.
        approximate : bool
            If True (default), summarize control backgrounds with K-means
            centroids. If False, use every abducted NTC cell as a background
            (exact population average, no clustering — slower).

        Returns
        -------
        If observed_effect is None and return_effect_df=False:
            dict {pert_name: (G,) counterfactual effect vector}

        Otherwise:
            dict with keys:
                "effects"       : {pert_name: (G,) effect vector}
                "effect_df"     : pd.DataFrame, predicted effects
                "corr"          : Pearson correlation with observed_effect, or nan
                "observed_df"   : aligned observed effects, if observed_effect given
                "predicted_df"  : aligned predicted effects, if observed_effect given
        """
        from sklearn.cluster import KMeans
        import pandas as pd
        from scipy.stats import pearsonr
        from ._datasets import PerturbSimpleDataset
        from .._configs import get_config

        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        self.to(device)

        ntc_label = get_config("ntc_label")
        p_key = get_config("pert_key")
        var_names = adata.var_names

        # Use individual-gene indexing for combinatorial models (cVAE/sVAE) so
        # non_ntc_idx aligns with their pert_emb / action_prior_mean rows.
        combinatorial = getattr(self, "combinatorial", False)
        ds = PerturbSimpleDataset(adata, pert_key=p_key, ntc_label=ntc_label,
                                  combinatorial=combinatorial)
        idx2pert = ds.idx_to_pert()

        ntc_mask = adata.obs[p_key] == ntc_label
        ntc_adata = adata[ntc_mask]

        # Stratify NTC cells by condition so that per-condition baselines are
        # computed separately.  In the single-condition case this is a no-op.
        try:
            treatment_key = get_config("treatment_key")
            cond_labels = ntc_adata.obs[treatment_key].values
            unique_conds = np.unique(cond_labels)
        except Exception:
            cond_labels = np.zeros(ntc_mask.sum(), dtype=int)
            unique_conds = np.array([0])

        non_ntc_items = [(i, n) for i, n in idx2pert.items() if n != ntc_label]
        non_ntc_idx   = [i for i, _ in non_ntc_items]
        non_ntc_names = [n for _, n in non_ntc_items]

        # For combinatorial datasets, map individual gene names → dataset indices and
        # collect combined labels (e.g. "GENE1+GENE2") present in adata.obs.
        name_to_ds_idx = {n: i for i, n in non_ntc_items}
        combo_labels = (
            [lbl for lbl in sorted(set(adata.obs[p_key].values) - {ntc_label}) if "+" in lbl]
            if combinatorial else []
        )

        # Compute all perturbation shifts once (shared across conditions).
        all_shifts = self._get_all_pert_shifts(device)  # (P+1, ...) with index 0 = NTC
        if all_shifts is None:
            return {}

        # Per-condition: abduct NTC backgrounds, K-means centroids, effects.
        # Effects are then averaged across conditions weighted by NTC cell count.
        cond_effects_all: dict[str, dict[str, np.ndarray]] = {}
        cond_sizes: dict[str, int] = {}

        for cond in unique_conds:
            cond_key = str(cond)
            cond_ntc_mask = cond_labels == cond
            x_ntc_cond_raw = ntc_adata[cond_ntc_mask].X
            if hasattr(x_ntc_cond_raw, "toarray"):
                x_ntc_cond_raw = x_ntc_cond_raw.toarray()
            x_ntc_cond_t = torch.tensor(
                np.asarray(x_ntc_cond_raw), dtype=torch.float32, device=device
            )

            u_ntc = self._abduct_ntc(x_ntc_cond_t)
            if u_ntc is None:
                return {}

            if approximate:
                # K-means quadrature of the control-background distribution
                n_k = min(n_centroids, u_ntc.shape[0])
                km = KMeans(n_clusters=n_k, n_init=10, random_state=0).fit(
                    u_ntc.detach().cpu().numpy()
                )
                centroids = torch.tensor(km.cluster_centers_, dtype=u_ntc.dtype, device=device)
                counts = np.bincount(km.labels_, minlength=n_k)
                weights = torch.tensor(counts / counts.sum(), dtype=u_ntc.dtype, device=device)
            else:
                # exact population average: every abducted NTC cell, uniform weight
                centroids = u_ntc
                n_k = u_ntc.shape[0]
                weights = torch.full(
                    (n_k,), 1.0 / n_k, dtype=u_ntc.dtype, device=device
                )

            lib_med = float(x_ntc_cond_t.sum(-1).median().item())

            # NTC baseline: decode centroids with no shift, assuming m_NTC = 0
            mu_ntc_k = self._decode_to_expr(centroids, lib_med)
            if mu_ntc_k is None:
                return {}

            # Normalize each centroid before averaging to match treat_effect's
            # E[log2-CPM(x)] operation order (Jensen's: f(E[x]) != E[f(x)])
            ntc_ln_k = self._lognorm(mu_ntc_k.detach().cpu().numpy())   # (K, G)
            ntc_ln   = (weights.cpu().numpy()[:, None] * ntc_ln_k).sum(0)  # (G,)

            # per-perturbation counterfactual effect for this condition
            cond_effects: dict[str, np.ndarray] = {}
            for idx, name in zip(non_ntc_idx, non_ntc_names):
                if idx >= all_shifts.shape[0]:
                    continue
                m_p    = all_shifts[[idx]]                          # (1, ...)
                z_cf_k = self._apply_shift(centroids, m_p)          # (K, d)
                mu_cf_k = self._decode_to_expr(z_cf_k, lib_med)     # (K, G)
                if mu_cf_k is None:
                    continue
                cf_ln_k = self._lognorm(mu_cf_k.detach().cpu().numpy())   # (K, G)
                cf_ln   = (weights.cpu().numpy()[:, None] * cf_ln_k).sum(0)  # (G,)
                cond_effects[name] = cf_ln - ntc_ln                 # (G,)

            # Combined-label effects: sum component shifts (e.g. GENE1+GENE2)
            for label in combo_labels:
                parts = [p.strip() for p in label.split("+")]
                idxs  = [name_to_ds_idx[p] for p in parts if p in name_to_ds_idx]
                if not idxs:
                    continue
                m_p = all_shifts[[idxs[0]]]
                for i in idxs[1:]:
                    m_p = self._combine_pert_shifts(
                        m_p, all_shifts[[i]],
                        pert_idx_0=idxs[0], pert_idx_1=i,
                    )
                z_cf_k  = self._apply_shift(centroids, m_p)
                mu_cf_k = self._decode_to_expr(z_cf_k, lib_med)
                if mu_cf_k is None:
                    continue
                cf_ln_k = self._lognorm(mu_cf_k.detach().cpu().numpy())
                cf_ln   = (weights.cpu().numpy()[:, None] * cf_ln_k).sum(0)
                cond_effects[label] = cf_ln - ntc_ln

            cond_effects_all[cond_key] = cond_effects
            cond_sizes[cond_key] = int(cond_ntc_mask.sum())

        # Weighted average of per-condition effects (weight = NTC cell count).
        all_pert_names = {nm for ce in cond_effects_all.values() for nm in ce}
        pred_effects: dict[str, np.ndarray] = {}
        for name in all_pert_names:
            weighted: np.ndarray | None = None
            weight_sum = 0
            for cond_key, cond_effects in cond_effects_all.items():
                if name not in cond_effects:
                    continue
                w = cond_sizes[cond_key]
                weighted = cond_effects[name] * w if weighted is None else weighted + cond_effects[name] * w
                weight_sum += w
            if weighted is not None and weight_sum > 0:
                pred_effects[name] = weighted / weight_sum

        # Build predicted effect DataFrame.
        if len(pred_effects) == 0:
            effect_df = pd.DataFrame()
        else:
            n_genes = next(iter(pred_effects.values())).shape[0]
            if var_names is None:
                var_names = [f"gene_{j}" for j in range(n_genes)]

            effect_df = pd.DataFrame.from_dict(
                pred_effects,
                orient="index",
                columns=list(var_names),
            )

        out = {
            "effects": pred_effects,
            "effect_df": effect_df,
            "corr": float("nan"),
        }

        if observed_effect is None:
            return out

        # Convert observed/reference effects to DataFrame.
        if isinstance(observed_effect, pd.DataFrame):
            obs_df = observed_effect.copy()
        else:
            obs_X = observed_effect.X
            if hasattr(obs_X, "toarray"):
                obs_X = obs_X.toarray()
            obs_df = pd.DataFrame(
                np.asarray(obs_X),
                index=observed_effect.obs_names,
                columns=observed_effect.var_names,
            )

        common_perts = effect_df.index.intersection(obs_df.index)
        common_genes = effect_df.columns.intersection(obs_df.columns)

        pred_aligned = effect_df.loc[common_perts, common_genes]
        obs_aligned = obs_df.loc[common_perts, common_genes]

        x = pred_aligned.values.ravel()
        y = obs_aligned.values.ravel()
        valid = np.isfinite(x) & np.isfinite(y)

        corr = pearsonr(x[valid], y[valid])[0] if valid.sum() > 2 else float("nan")

        out.update(
            {
                "corr": float(corr),
                "predicted_df": pred_aligned,
                "observed_df": obs_aligned,
            }
        )

        return out

    # ── inference loop (overridable) ──────────────────────────────────

    def _run_inference(
        self,
        dataset: Dataset,
        device: torch.device,
        batch_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the get_z / get_recon forward passes over *dataset*.

        Override in subclasses if the model needs a different batch format
        (e.g. models that require additional inputs per batch).

        Returns
        -------
        z_all    : (N, d)
        recon_all: (N, G)
        p_all    : (N,)  integer perturbation indices
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        z_parts, recon_parts, p_parts = [], [], []

        with torch.no_grad():
            for x, p in loader:
                x = x.to(device, dtype=torch.float32)
                p = p.to(device)
                z_parts.append(self.get_z(x, p).cpu().numpy())
                recon_parts.append(self.get_recon(x, p).cpu().numpy())
                p_parts.append(p.cpu().numpy())

        return (
            np.concatenate(z_parts,     axis=0).astype(np.float32),
            np.concatenate(recon_parts, axis=0).astype(np.float32),
            np.concatenate(p_parts,     axis=0),
        )

    # ── universal evaluation ──────────────────────────────────────────

    def eval(
        self,
        adata: anndata.AnnData | None = None,
        obsm_key: str = "z",
        uns_key: str = "results",
        device: torch.device | None = None,
        batch_size: int = 1024,
        dataset: Dataset | None = None,
    ):
        """
        Evaluate on *adata*, writing results in-place.

        Called with *no arguments* it falls back to nn.Module.eval()
        (toggles training mode).  Called with *adata* it runs:

          1. _run_inference() → z, x_pred via get_z() / get_recon()
             stored in adata.obsm[obsm_key] and adata.layers["x_pred"]
          2. get_corr()        → adata.uns[uns_key]["z_corr"] / "rho_corr"
          3. _eval() hook      → model-specific extras merged into uns

        Parameters
        ----------
        adata      : AnnData to evaluate on.
        obsm_key   : key in adata.obsm to store z.
        uns_key    : key in adata.uns to store the results dict.
        device     : torch device; defaults to cpu.
        batch_size : inference batch size.
        dataset    : dataset to use; if None, PerturbSimpleDataset(adata)
                     is constructed automatically.
        """
        if adata is None:
            return super().eval()

        from ._datasets import PerturbSimpleDataset
        from .._configs import get_config

        if device is None:
            device = torch.device("cpu")

        super().eval()
        self.to(device)

        ntc_label = get_config("ntc_label")
        if dataset is None:
            p_key = get_config("pert_key")
            combinatorial = getattr(self, "combinatorial", False)
            dataset = PerturbSimpleDataset(
                adata, pert_key=p_key, ntc_label=ntc_label, combinatorial=combinatorial
            )

        z_all, recon_all, p_all = self._run_inference(dataset, device, batch_size)

        adata.obsm[obsm_key] = z_all
        adata.layers["x_pred"] = recon_all

        idx2pert = dataset.idx_to_pert()
        perts = np.array([idx2pert[i] for i in range(dataset.n_perturbs)])

        # For combinatorial datasets p_all is (N, 2): [gene_idx, -1] for single-pert,
        # [idx1, idx2] for combo, [0, -1] for NTC.  Restrict z_corr to single-pert
        # cells (p[:,1]==-1) so individual-gene embeddings are clean.
        if p_all.ndim == 2:
            single_mask = p_all[:, 1] == -1
            p_for_corr = p_all[single_mask, 0]
            z_for_corr = z_all[single_mask]
        else:
            p_for_corr = p_all
            z_for_corr = z_all

        z_corr, z_perts, z_embed = self.get_corr(z_for_corr, p_for_corr, perts, use_rho=False)
        try:
            rho_corr, rho_perts, rho_embed = self.get_corr(z_for_corr, p_for_corr, perts, use_rho=True)
        except NotImplementedError:
            rho_corr = rho_embed = None
            rho_perts = z_perts

        uns_data: dict = {
            "z_corr":    z_corr,
            "z_perts":   z_perts,
            "z_embed":   z_embed,
            "rho_corr":  rho_corr,
            "rho_perts": rho_perts,
            "rho_embed": rho_embed,
        }

        uns_data.update(self._eval(adata, obsm_key=obsm_key, device=device))
        adata.uns[uns_key] = uns_data

    def _eval(
        self,
        adata,
        obsm_key: str,
        device: torch.device,
    ) -> dict:
        """
        Hook for model-specific evaluation metrics.

        Called at the end of eval() after z, x_pred and z_corr are stored.
        Override in subclasses to add model-specific results (accuracy, gate
        weights, explained variance, counterfactuals, …).  The returned dict
        is merged into adata.uns[uns_key].  Default returns {}.
        """
        return {}
