import math, os, time
from typing import Optional, Sequence

import numpy as np
import copy
import torch
from torch.utils.data import DataLoader
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro, pyro.poutine as poutine
from sklearn.metrics import r2_score
from tqdm.auto import tqdm
import torch.nn.functional as F
import anndata
import pandas as pd
from scipy.stats import pearsonr

# tiny util to fetch a param safely (no device thrash)
def _get_param(name, device):
    ps = pyro.get_param_store()
    t = ps[name]
    return t if t.device == device else t.to(device)

class VAETrainer:

    def __init__(
        self,
        vae,
        dataloader: DataLoader,
        treat_effect: anndata.AnnData,
        lr: float = 5e-4,
        num_epochs: int = 10,
        init_coeff: float = 1e-2,
        final_coeff: float = 0.0,
        tau_init: float = 0.67,   # good default for Hard-Concrete
        tau_end: float = 0.10,
        gate_start: int = 0,
        tau_anneal_steps: Optional[int] = None,
        ramp_steps: Optional[int] = None,
        verbose: bool = True,
        device: Optional[torch.device] = None,
        seed: int = 0,
        validate_every: int = 10,        
        pin_memory: bool = True,        
        non_blocking_copy: bool = True,
    ):
        self.vae        = vae
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.init_coeff = init_coeff
        self.non_blocking_copy = non_blocking_copy
        self.final_coeff = final_coeff
        self.tau_init  = float(tau_init)
        self.tau_end   = float(tau_end)
        self.tau_anneal_steps = tau_anneal_steps
        self.verbose = verbose
        self.treat_effect = treat_effect

        self.gate_start = gate_start
        self.validate_every = max(1, int(validate_every))
        self.pin_memory = pin_memory
        self.non_blocking_copy = non_blocking_copy

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(self.device)

        # total #steps used for linear/exp schedules
        total_steps = len(dataloader) * num_epochs
        self.ramp_steps = ramp_steps if ramp_steps is not None else total_steps
        self.tau_anneal_steps = self.tau_anneal_steps or total_steps

        # Pyro bits ----------------------------------------------------------
        pyro.clear_param_store()
        self.svi = SVI(
            self.vae.model,
            self.vae.guide,
            Adam({"lr": lr}),
            loss=Trace_ELBO(),   # keep simple, single particle for speed
        )

        # reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # kick-off centre coeff at max
        self.vae.center_coeff.fill_(self.init_coeff)

        # if Hard-Concrete gate exists, initialize temperature
        if getattr(self.vae, "gate_on", False) and hasattr(self.vae, "gate"):
            self.vae.gate.set_temperature(self.tau_init)

        # CUDA perf niceties
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.global_step = 0

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _update_center_coeff_decay(self):
        """Linear decay: init_coeff → final_coeff over ramp_steps steps."""
        prog = min(self.global_step / self.ramp_steps, 1.0)  # 0 → 1
        coeff = self.init_coeff - (self.init_coeff - self.final_coeff) * prog
        self.vae.center_coeff.fill_(coeff)

    def _current_tau(self, step) -> float:
        """Exponential decay: τ_t = max(τ_end, τ0 * exp(-k*t))."""
        if self.tau_anneal_steps <= 0 or self.tau_init <= self.tau_end:
            return self.tau_end
        k = math.log(self.tau_init / self.tau_end) / self.tau_anneal_steps
        return max(self.tau_end, self.tau_init * math.exp(-k * step))

    def _move_to_device(self, X_p, X_ntc, P, C):
        nb = self.non_blocking_copy
        X_p  = X_p.to(self.device, dtype=torch.float32, non_blocking=nb)
        X_ntc = X_ntc.to(self.device, dtype=torch.float32, non_blocking=nb)
        P    = P.to(self.device, dtype=torch.long, non_blocking=nb)
        C    = C.to(self.device, dtype=torch.long, non_blocking=nb)
        return X_p, X_ntc, P, C

    def _validate_fast(self, val_loader):
        """
        Fast validation without poutine.replay:
        - Run guide once to get z0, z.
        - Compute logits/means directly via decoder & cls_head.
        - Use deterministic gates (mean) to reduce noise.
        """
        self.vae.eval()
        preds_p, preds_ntc = [], []
        actuals_p, actuals_ntc = [], []
        val_ce_sum, val_correct, val_n = 0.0, 0, 0
        all_P = []

        # use deterministic gates for stability during eval
        use_det_gates = getattr(self.vae, "gate_on", False) and hasattr(self.vae, "gate")

        with torch.no_grad():
            # reconstruct theta from param store
            theta_uncon = _get_param("theta_uncon", self.device)
            theta = F.softplus(theta_uncon) + 1e-3

            for X_p, X_ntc, P, C in val_loader:
                X_p, X_ntc, P, C = self._move_to_device(X_p, X_ntc, P, C)
                all_P.append(P.detach().cpu().numpy())

                # trace the guide once
                guide_tr = poutine.trace(self.vae.guide).get_trace(X_p, X_ntc, P, C)

                # fetch z and z0 from guide trace
                z  = guide_tr.nodes["z"]["value"]
                z0 = guide_tr.nodes["z0"]["value"]

                # classification logits
                cls_logits = self.vae.cls_head(z)
                ce_batch   = F.cross_entropy(cls_logits, P, reduction="sum")
                val_ce_sum += ce_batch.item()
                val_correct += (cls_logits.argmax(dim=-1) == P).sum().item()
                val_n += P.size(0)

                # decode means (match __decode math)
                logits_ntc = self.vae._VAE__decode(X_ntc, z0, self.vae.z_decoder, theta, 'x_ntc')
                logits_p   = self.vae._VAE__decode(X_p,  z,  self.vae.z_decoder, theta, 'x_p')

                # convert logits->mu deterministically (same as in __decode)
                # __decode already pushed mu via pyro.deterministic; we recompute here:
                with torch.no_grad():
                    # negative binomial parameterization used only for logits; we want mu
                    total_ntc = X_ntc.sum(-1, keepdim=True)
                    mu_ntc = total_ntc * torch.softmax(self.vae.z_decoder(z0), dim=-1)
                    total_p = X_p.sum(-1, keepdim=True)
                    mu_p = total_p * torch.softmax(self.vae.z_decoder(z), dim=-1)

                preds_p.append(mu_p.detach().cpu())
                actuals_p.append(X_p.detach().cpu())
                preds_ntc.append(mu_ntc.detach().cpu())
                actuals_ntc.append(X_ntc.detach().cpu())

        # aggregate
        actuals_p  = torch.cat(actuals_p,  dim=0).numpy()
        preds_p    = torch.cat(preds_p,    dim=0).numpy()
        actuals_ntc = torch.cat(actuals_ntc, dim=0).numpy()
        preds_ntc   = torch.cat(preds_ntc,   dim=0).numpy()

        r2_p    = r2_score(actuals_p.flatten(),   preds_p.flatten())
        r2_ntc  = r2_score(actuals_ntc.flatten(), preds_ntc.flatten())

        # normalize to Perturb-seq space for ATE metric
        lib_p = preds_p.sum(axis=1, keepdims=True)
        preds_p_n = 1e4 * preds_p / lib_p
        preds_p_n = np.log2(preds_p_n + 1.0)

        lib_ntc = preds_ntc.sum(axis=1, keepdims=True)
        preds_ntc_n = 1e4 * preds_ntc / lib_ntc
        preds_ntc_n = np.log2(preds_ntc_n + 1.0)

        ce_loss  = val_ce_sum / max(1, val_n)
        acc      = val_correct / max(1, val_n)

        return {
            "actuals_p": actuals_p,
            "preds_p": preds_p,
            "preds_p_n": preds_p_n,
            "preds_ntc_n": preds_ntc_n,
            "ce_loss": ce_loss,
            "acc": acc,
            "r2_p": r2_p,
            "r2_ntc": r2_ntc,
            "all_P": np.concatenate(all_P) if len(all_P) else np.array([]),
        }

    def fit(self):
        print(f"Training VAE on {self.device} …")
        dataset_size = len(self.dataloader.dataset)

        best_score = float("inf")
        best_state = None
        best_param_store = None
        patience = self.num_epochs                # keep same semantics
        patience_counter = 0

        # ───────── outer progress bar over epochs ─────────
        epoch_bar = tqdm(range(1, self.num_epochs + 1), desc="Epochs", dynamic_ncols=True)

        for epoch in epoch_bar:
            self.vae.train()
            epoch_loss = 0.0

            # ----- switch gate ON exactly at gate_start ------------
            if (not getattr(self.vae, "gate_on", False)) and epoch >= self.gate_start:
                self.vae.gate_on = True
                if self.verbose:
                    print(f"▶  Hard-Concrete gate ENABLED at epoch {epoch}")

            # ───── transient batch bar (cleared each epoch) ─────
            batch_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=False, position=1, dynamic_ncols=True)

            for X_p, X_ntc, P, C in batch_bar:
                # fast host→GPU copies
                X_p, X_ntc, P, C = self._move_to_device(X_p, X_ntc, P, C)

                # anneal center coeff & temperature
                self._update_center_coeff_decay()
                if getattr(self.vae, "gate_on", False) and hasattr(self.vae, "gate"):
                    tau_curr = self._current_tau(self.global_step)
                    self.vae.gate.set_temperature(tau_curr)
                else:
                    tau_curr = self.tau_init

                # optimisation step
                batch_loss = self.svi.step(X_p, X_ntc, P, C)
                epoch_loss += batch_loss
                self.global_step += 1
                batch_bar.set_postfix(elbo_per_cell=batch_loss / X_p.size(0), tau=f"{tau_curr:.3f}")

            avg_elbo = epoch_loss / dataset_size

            # ──────────── validation (every N epochs) ────────────
            do_validate = (epoch % self.validate_every == 0)
            ce_loss = acc = r2_p = r2_ntc = ate = float("nan")
            pi25 = pi99 = float("nan")

            if do_validate:
                val_stats = self._validate_fast(getattr(self, "val_dataloader", self.dataloader))
                ce_loss = val_stats["ce_loss"]
                acc     = val_stats["acc"]
                r2_p    = val_stats["r2_p"]
                r2_ntc  = val_stats["r2_ntc"]

                # Treatment effect correlation (ATE)
                preds_p_n   = val_stats["preds_p_n"]
                preds_ntc_n = val_stats["preds_ntc_n"]
                all_P       = val_stats["all_P"]

                dataset = self.dataloader.dataset
                n_perts   = len(dataset.perturbation_dict)
                n_genes   = preds_p_n.shape[1] if preds_p_n.size else 0

                if n_perts > 0 and n_genes > 0 and all_P.size > 0:
                    effect = np.zeros((n_perts, n_genes))
                    for pert_name, j in dataset.perturbation_dict.items():
                        mask = (all_P == j)
                        if not mask.any(): continue
                        mean_p   = preds_p_n[mask].mean(axis=0)
                        mean_ntc = preds_ntc_n[mask].mean(axis=0)
                        effect[j] = mean_p - mean_ntc

                    effect_df = pd.DataFrame(
                        effect,
                        index=list(dataset.perturbation_dict.keys()),
                        columns=self.treat_effect.var_names
                    )
                    effect_aligned = effect_df.loc[self.treat_effect.obs_names, self.treat_effect.var_names]
                    x = effect_aligned.values.flatten()
                    y = self.treat_effect.X.flatten()
                    ate, _ = pearsonr(x, y)

                # Inspect gate probabilities if available
                if hasattr(self.vae, "gate"):
                    # expected_L0 returns probabilities (P,d); summarize for logs
                    q = self.vae.gate.expected_L0().detach().cpu().flatten()
                    if q.numel() > 0:
                        pi25 = torch.quantile(q, 0.25).item()
                        pi99 = torch.quantile(q, 0.99).item()

            coeff_now = float(self.vae.center_coeff.item())
            curr_tau = float(self.vae.gate.temperature.item()) if getattr(self.vae, "gate_on", False) and hasattr(self.vae, "gate") else float("nan")

            epoch_bar.set_postfix(
                ELBO=f"{avg_elbo:.4f}",
                CE=f"{ce_loss:.4f}" if not math.isnan(ce_loss) else "nan",
                acc=f"{acc:.3f}" if not math.isnan(acc) else "nan",
                R2_p=f"{r2_p:.4f}" if not math.isnan(r2_p) else "nan",
                R2_ntc=f"{r2_ntc:.4f}" if not math.isnan(r2_ntc) else "nan",
                ATE=f"{ate:.4f}" if not math.isnan(ate) else "nan",
                pi25=f"{pi25:.2f}" if not math.isnan(pi25) else "nan",
                pi99=f"{pi99:.2f}" if not math.isnan(pi99) else "nan",
                lambda_center=f"{coeff_now:.6f}",
                tau=f"{curr_tau:.3f}" if not math.isnan(curr_tau) else "nan",
            )

            # use ELBO as score (smaller is better)
            score = avg_elbo
            if score < best_score or epoch < self.num_epochs // 2:
                best_score = score
                best_state = copy.deepcopy(self.vae)
                best_param_store = copy.deepcopy(pyro.get_param_store().get_state())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹ Early stopping at epoch {epoch} (best score={best_score:.4f})")
                    pyro.clear_param_store()
                    if best_param_store is not None:
                        pyro.get_param_store().set_state(best_param_store)
                    return best_state

        pyro.clear_param_store()
        if best_param_store is not None:
            pyro.get_param_store().set_state(best_param_store)
        return best_state, best_param_store