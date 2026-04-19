import copy
import math
import numpy as np
import pandas as pd
import pyro
import pyro.poutine as poutine
import torch
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class VAETrainer:
    def __init__(
        self,
        vae,
        dataloader: DataLoader,
        treat_effect,
        lr: float = 5e-4,
        num_epochs: int = 10,
        validate_every: int = 10,
        verbose: bool = True,
        device=torch.device('cpu'),
        seed: int = 0,
        non_blocking_copy: bool = True,
        tau_init: float | None = None,
        tau_end: float = 0.10,
        tau_anneal_steps: int | None = None,
        patience: int = 20,
    ):
        self.vae = vae
        self.dataloader = dataloader
        self.treat_effect = treat_effect
        self.num_epochs = int(num_epochs)
        self.validate_every = max(1, int(validate_every))
        self.verbose = verbose
        self.non_blocking_copy = non_blocking_copy
        self.patience = patience

        self.device = device
        self.vae.to(self.device)

        pyro.clear_param_store()
        self.svi = SVI(self.vae.model, self.vae.guide, Adam({"lr": lr}), loss=Trace_ELBO())

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # gate temperature schedule
        self.tau_init = float(tau_init) if tau_init is not None else float(self.vae.tau_init)
        self.tau_end = float(tau_end)
        total_steps = len(dataloader) * self.num_epochs
        self.tau_anneal_steps = int(tau_anneal_steps or total_steps)
        self.global_step = 0
        self.vae.gate.set_temperature(self.tau_init)

        # last validated metrics (persist between evals)
        self._last_valid = dict(
            CE=np.nan, acc=np.nan, R2_p=np.nan, R2_ntc=np.nan, ATE=np.nan, pi25=np.nan, pi99=np.nan
        )
        self._last_tau = self.tau_init

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _current_tau(self, step) -> float:
        if self.tau_anneal_steps <= 0 or self.tau_init <= self.tau_end:
            return self.tau_end
        k = math.log(self.tau_init / self.tau_end) / self.tau_anneal_steps
        return max(self.tau_end, self.tau_init * math.exp(-k * step))

    def _move_to_device(self, X_p, X_ntc, P, C):
        nb = self.non_blocking_copy
        return (
            X_p.to(self.device, dtype=torch.float32, non_blocking=nb),
            X_ntc.to(self.device, dtype=torch.float32, non_blocking=nb),
            P.to(self.device, dtype=torch.long, non_blocking=nb),
            C.to(self.device, dtype=torch.long, non_blocking=nb),
        )

    def _validate(self, val_loader):
        self.vae.eval()
        preds_p, preds_ntc, actuals_p, actuals_ntc = [], [], [], []
        val_ce_sum, val_correct, val_n = 0.0, 0, 0
        all_P = []
        pi25 = pi99 = np.nan  # percentiles of gate probability π

        with torch.no_grad():
            for X_p, X_ntc, P, C in val_loader:
                X_p, X_ntc, P, C = self._move_to_device(X_p, X_ntc, P, C)
                all_P.append(P.detach().cpu().numpy())

                guide_tr = poutine.trace(self.vae.guide).get_trace(X_p, X_ntc, P, C)
                z = guide_tr.nodes["z"]["value"]
                z0 = guide_tr.nodes["z0"]["value"]

                # CE / acc
                cls_logits = self.vae.cls_head(z)
                if P.ndim == 1:
                    val_ce_sum += F.cross_entropy(cls_logits, P.long(), reduction="sum").item()
                    val_correct += (cls_logits.argmax(dim=-1) == P).sum().item()
                    val_n += P.size(0)

                else:
                    B, Pn = cls_logits.shape
                    mask = (P != -1)
                    k = mask.sum(dim=1).max().item()  # typically 2

                    # multi-hot targets for BCE
                    tgt = torch.zeros((B, Pn), device=cls_logits.device, dtype=cls_logits.dtype)
                    tgt.scatter_(1, P.clamp(min=0).long(), mask.float())

                    val_ce_sum += F.binary_cross_entropy_with_logits(cls_logits, tgt, reduction="sum").item()

                    # "set" accuracy: both true labels must be in top-k predictions
                    topk = cls_logits.topk(k=k, dim=1).indices  # (B,k)
                    hit = (topk.unsqueeze(2) == P.clamp(min=0).long().unsqueeze(1)) & mask.unsqueeze(1)
                    val_correct += (hit.any(dim=1).sum(dim=1) == mask.sum(dim=1)).sum().item()
                    val_n += B

                # decode means for R²/ATE
                total_ntc = X_ntc.sum(-1, keepdim=True)
                mu_ntc = total_ntc * torch.softmax(self.vae.z_decoder(z0), dim=-1)
                total_p = X_p.sum(-1, keepdim=True)
                mu_p = total_p * torch.softmax(self.vae.z_decoder(z), dim=-1)

                preds_p.append(mu_p.detach().cpu())
                actuals_p.append(X_p.detach().cpu())
                preds_ntc.append(mu_ntc.detach().cpu())
                actuals_ntc.append(X_ntc.detach().cpu())

            # gate π percentiles (over perturbation×latent dims)
            pi = self.vae.gate.expected_L0().detach().flatten().cpu()
            if pi.numel() > 0:
                pi25 = torch.quantile(pi, 0.25).item()
                pi99 = torch.quantile(pi, 0.99).item()

        actuals_p = torch.cat(actuals_p, 0).numpy()
        preds_p = torch.cat(preds_p, 0).numpy()
        actuals_ntc = torch.cat(actuals_ntc, 0).numpy()
        preds_ntc = torch.cat(preds_ntc, 0).numpy()

        r2_p = r2_score(actuals_p.ravel(), preds_p.ravel())
        r2_ntc = r2_score(actuals_ntc.ravel(), preds_ntc.ravel())

        def lognorm(x):
            lib = x.sum(axis=1, keepdims=True)
            return np.log2(1e4 * x / np.clip(lib, 1e-8, None) + 1.0)

        preds_p_n = lognorm(preds_p)
        preds_ntc_n = lognorm(preds_ntc)

        ce_loss = val_ce_sum / max(1, val_n)
        acc = val_correct / max(1, val_n)

        return {
            "preds_p_n": preds_p_n,
            "preds_ntc_n": preds_ntc_n,
            "ce_loss": ce_loss,
            "acc": acc,
            "r2_p": r2_p,
            "r2_ntc": r2_ntc,
            "all_P": np.concatenate(all_P) if len(all_P) else np.array([]),
            "pi25": pi25,
            "pi99": pi99,
        }

    def fit(self):
        print(f"Training VAE on {self.device} …")
        dataset_size = len(self.dataloader.dataset)

        best_score = float("inf")
        best_state = None
        best_param_store = None
        patience = self.patience
        patience_counter = 0

        epoch_bar = tqdm(range(1, self.num_epochs + 1), desc="Epochs", dynamic_ncols=True)

        for epoch in epoch_bar:
            self.vae.train()
            epoch_loss = 0.0

            for X_p, X_ntc, P, C in self.dataloader:  # no inner tqdm
                X_p, X_ntc, P, C = self._move_to_device(X_p, X_ntc, P, C)

                # anneal gate temperature (gating is always on)
                tau_curr = self._current_tau(self.global_step)
                self.vae.gate.set_temperature(tau_curr)
                self._last_tau = tau_curr

                epoch_loss += self.svi.step(X_p, X_ntc, P, C)
                self.global_step += 1

            avg_elbo = epoch_loss / dataset_size

            # validate on schedule; remember metrics for persistence
            if epoch % self.validate_every == 0:
                stats = self._validate(getattr(self, "val_dataloader", self.dataloader))

                # ATE
                ate = float("nan")
                preds_p_n, preds_ntc_n, all_P = stats["preds_p_n"], stats["preds_ntc_n"], stats["all_P"]
                ds = self.dataloader.dataset
                n_genes = preds_p_n.shape[1] if preds_p_n.size else 0

                # index -> name (so effect_df uses string labels)
                idx2pert = {i: name for name, i in getattr(ds, "perturbation_dict", {}).items()}

                if all_P.ndim == 1:
                    combos = all_P.astype(np.int64).reshape(-1, 1)  # (N,1)
                else:
                    combos = all_P.astype(np.int64).copy()          # (N,2)
                    # canonicalize order so (a,b) == (b,a); keep (-1) as second slot
                    a = combos[:, 0]
                    b = combos[:, 1]
                    swap = (b != -1) & (a > b)
                    combos[swap, 0], combos[swap, 1] = combos[swap, 1], combos[swap, 0]

                uniq_combos = np.unique(combos, axis=0)
                n_perts = uniq_combos.shape[0]

                effect = np.zeros((n_perts, n_genes))
                for j, key in enumerate(uniq_combos):
                    if key.shape[0] == 1:
                        mask = (combos[:, 0] == key[0])
                    else:
                        mask = (combos[:, 0] == key[0]) & (combos[:, 1] == key[1])

                    if not mask.any():
                        continue
                    effect[j] = preds_p_n[mask].mean(0) - preds_ntc_n[mask].mean(0)

                # string names for combo index
                def combo_name(k):
                    p1 = idx2pert.get(int(k[0]), str(int(k[0])))
                    if k.shape[0] == 1 or int(k[1]) == -1:
                        return p1
                    p2 = idx2pert.get(int(k[1]), str(int(k[1])))
                    return f"{p1}+{p2}"

                effect_df = pd.DataFrame(
                    effect,
                    index=[combo_name(k) for k in uniq_combos],
                    columns=self.treat_effect.var_names,
                )
                effect_aligned = effect_df.loc[self.treat_effect.obs_names, self.treat_effect.var_names]
                x = effect_aligned.values.ravel()
                y = self.treat_effect.X.ravel()
                ate = pearsonr(x, y)[0]

                self._last_valid = dict(
                    CE=stats["ce_loss"],
                    acc=stats["acc"],
                    R2_p=stats["r2_p"],
                    R2_ntc=stats["r2_ntc"],
                    ATE=ate,
                    pi25=stats["pi25"],
                    pi99=stats["pi99"],
                )


            # persistent tqdm display (no NA flicker)
            epoch_bar.set_postfix(
                ELBO=f"{avg_elbo:.4f}",
                CE=f"{self._last_valid['CE']:.4f}" if not math.isnan(self._last_valid["CE"]) else "nan",
                acc=f"{self._last_valid['acc']:.3f}" if not math.isnan(self._last_valid["acc"]) else "nan",
                R2_p=f"{self._last_valid['R2_p']:.4f}" if not math.isnan(self._last_valid['R2_p']) else "nan",
                R2_ntc=f"{self._last_valid['R2_ntc']:.4f}" if not math.isnan(self._last_valid['R2_ntc']) else "nan",
                ATE=f"{self._last_valid['ATE']:.4f}" if not math.isnan(self._last_valid['ATE']) else "nan",
                pi25=f"{self._last_valid['pi25']:.2f}" if not math.isnan(self._last_valid['pi25']) else "nan",
                pi99=f"{self._last_valid['pi99']:.2f}" if not math.isnan(self._last_valid['pi99']) else "nan",
                tau=f"{self._last_tau:.3f}",
            )

            # early stopping by ELBO
            score = avg_elbo
            if score < best_score or epoch < self.num_epochs // 2:
                best_score = score
                best_state = copy.deepcopy(self.vae)
                best_param_store = copy.deepcopy(pyro.get_param_store().get_state())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹ Early stopping at epoch {epoch} (best ELBO={best_score:.4f})")
                    pyro.clear_param_store()
                    if best_param_store is not None:
                        pyro.get_param_store().set_state(best_param_store)
                    return best_state, best_param_store

        pyro.clear_param_store()
        if best_param_store is not None:
            pyro.get_param_store().set_state(best_param_store)
        return best_state, best_param_store


class cVAETrainer:
    """
    Trains a cVAE model using Pyro SVI (Trace_ELBO).

    Parameters
    ----------
    model        : cVAE instance.
    dataloader   : DataLoader yielding (x, p) batches.
    treat_effect : AnnData with ground-truth ATEs used for validation.
    lr           : Adam learning rate.
    num_epochs   : maximum training epochs.
    validate_every : epoch interval between validation passes.
    patience     : early-stopping patience (epochs without ELBO improvement).
    device       : torch device.
    seed         : RNG seed.
    """

    def __init__(
        self,
        model,
        dataloader: DataLoader,
        treat_effect,
        lr: float = 3e-4,
        num_epochs: int = 200,
        validate_every: int = 10,
        patience: int = 30,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        non_blocking_copy: bool = True,
    ):
        self.model = model
        self.dataloader = dataloader
        self.treat_effect = treat_effect
        self.num_epochs = int(num_epochs)
        self.validate_every = max(1, int(validate_every))
        self.patience = patience
        self.non_blocking_copy = non_blocking_copy

        self.device = device
        self.model.to(self.device)

        pyro.clear_param_store()
        self.svi = SVI(self.model.model, self.model.guide, Adam({"lr": lr}), loss=Trace_ELBO())

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self._last_valid = dict(R2=float("nan"), ATE=float("nan"))
        self.history: list[dict] = []

    def _to_device(self, x, p):
        nb = self.non_blocking_copy
        return (
            x.to(self.device, dtype=torch.float32, non_blocking=nb),
            p.to(self.device, dtype=torch.long, non_blocking=nb),
        )

    def _validate(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        preds_all, actuals_all, p_all = [], [], []

        with torch.no_grad():
            for x, p in val_loader:
                x, p = self._to_device(x, p)
                recon = self.model.get_recon(x, p)
                preds_all.append(recon.cpu().numpy())
                actuals_all.append(x.cpu().numpy())
                p_all.append(p.cpu().numpy())

        preds   = np.concatenate(preds_all,   axis=0)
        actuals = np.concatenate(actuals_all, axis=0)
        p_idx   = np.concatenate(p_all,       axis=0)

        r2 = r2_score(actuals.ravel(), preds.ravel())

        # ── ATE correlation ───────────────────────────────────────────
        ate = float("nan")
        ds = val_loader.dataset
        idx2pert = ds.idx_to_pert()
        ntc_idx  = ds.ntc_idx

        def _lognorm(x):
            lib = x.sum(axis=1, keepdims=True)
            return np.log2(1e4 * x / np.clip(lib, 1e-8, None) + 1.0)

        preds_ln = _lognorm(preds)
        ntc_mask = p_idx == ntc_idx

        if ntc_mask.sum() > 0:
            ntc_mean = preds_ln[ntc_mask].mean(axis=0)
            pred_effects: dict[str, np.ndarray] = {}
            for pidx in np.unique(p_idx[~ntc_mask]):
                name = idx2pert.get(int(pidx))
                if name is None:
                    continue
                pred_effects[name] = preds_ln[p_idx == pidx].mean(axis=0) - ntc_mean

            common_perts = [n for n in pred_effects if n in self.treat_effect.obs_names]
            common_genes = [g for g in ds.var_names if g in self.treat_effect.var_names]

            if len(common_perts) >= 2 and len(common_genes) >= 1:
                te_df   = pd.DataFrame(
                    self.treat_effect[common_perts, common_genes].X,
                    index=common_perts, columns=common_genes,
                )
                pred_mat = np.stack(
                    [pred_effects[n][[ds.var_names.index(g) for g in common_genes]]
                     for n in common_perts],
                    axis=0,
                )
                ate = pearsonr(pred_mat.ravel(), te_df.values.ravel())[0]

        return {"r2": r2, "ate": ate}

    def fit(self, val_loader: DataLoader | None = None) -> tuple:
        """
        Train the cVAE.

        Parameters
        ----------
        val_loader : optional validation DataLoader; falls back to train loader.

        Returns
        -------
        (best_model, best_param_store)
        """
        _val_loader = val_loader if val_loader is not None else self.dataloader
        dataset_size = len(self.dataloader.dataset)

        best_elbo = float("inf")
        best_state = None
        best_param_store = None
        patience_counter = 0

        epoch_bar = tqdm(range(1, self.num_epochs + 1), desc="cVAE", dynamic_ncols=True)

        for epoch in epoch_bar:
            self.model.train()
            epoch_loss = 0.0

            for x, p in self.dataloader:
                x, p = self._to_device(x, p)
                epoch_loss += self.svi.step(x, p)

            avg_elbo = epoch_loss / dataset_size

            if epoch % self.validate_every == 0:
                stats = self._validate(_val_loader)
                self._last_valid["R2"]  = stats["r2"]
                self._last_valid["ATE"] = stats["ate"]

            self.history.append(
                {"epoch": epoch, "elbo": avg_elbo, **self._last_valid}
            )

            epoch_bar.set_postfix(
                ELBO=f"{avg_elbo:.4f}",
                R2=f"{self._last_valid['R2']:.4f}"
                if not math.isnan(self._last_valid["R2"]) else "nan",
                ATE=f"{self._last_valid['ATE']:.4f}"
                if not math.isnan(self._last_valid["ATE"]) else "nan",
            )

            if avg_elbo < best_elbo or epoch < self.num_epochs // 4:
                best_elbo = avg_elbo
                best_state = copy.deepcopy(self.model)
                best_param_store = copy.deepcopy(pyro.get_param_store().get_state())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch} (best ELBO={best_elbo:.4f})")
                    break

        pyro.clear_param_store()
        if best_param_store is not None:
            pyro.get_param_store().set_state(best_param_store)
        return best_state, best_param_store


# sVAETrainer shares the same training loop as cVAETrainer:
# both use PerturbSimpleDataset yielding (x, p) batches with SVI.
sVAETrainer = cVAETrainer
