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
                val_ce_sum += F.cross_entropy(cls_logits, P, reduction="sum").item()
                val_correct += (cls_logits.argmax(dim=-1) == P).sum().item()
                val_n += P.size(0)

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
                n_perts = len(getattr(ds, "perturbation_dict", {}))
                n_genes = preds_p_n.shape[1] if preds_p_n.size else 0
                if n_perts > 0 and n_genes > 0 and all_P.size > 0:
                    effect = np.zeros((n_perts, n_genes))
                    for pert_name, j in ds.perturbation_dict.items():
                        mask = (all_P == j)
                        if not mask.any():
                            continue
                        effect[j] = preds_p_n[mask].mean(0) - preds_ntc_n[mask].mean(0)
                    effect_df = pd.DataFrame(
                        effect,
                        index=list(ds.perturbation_dict.keys()),
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
