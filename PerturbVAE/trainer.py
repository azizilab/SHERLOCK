import math, os, time
from typing import Optional, Sequence

import numpy as np
import copy
import torch
from torch.utils.data import DataLoader
from pyro.infer import SVI, Trace_ELBO, RenyiELBO
from pyro.optim import Adam, ClippedAdam
import pyro, pyro.poutine as poutine
from sklearn.metrics import r2_score
from tqdm import tqdm
from PerturbVAE.model import VAE
from tqdm.auto import tqdm
import torch.nn.functional as F
import anndata
import pandas as pd
from scipy.stats import pearsonr




class VAETrainer:

    def __init__(
        self,
        vae: VAE,
        dataloader: DataLoader,
        treat_effect: anndata.AnnData,
        lr: float = 5e-4,
        num_epochs: int = 10,
        init_coeff: float = 1e-2,
        final_coeff: float = 0.0,
        tau_init: float = 1.0,
        tau_end: float = 0.1,
        gate_start: int = 0,
        tau_anneal_steps: Optional[int] = None,
        ramp_steps: Optional[int] = None,
        verbose: bool = True,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ):
        self.vae        = vae
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.init_coeff = init_coeff
        self.final_coeff = final_coeff
        self.tau_init  = tau_init
        self.tau_end   = tau_end
        self.tau_anneal_steps = tau_anneal_steps
        self.verbose = verbose
        self.treat_effect = treat_effect

        self.gate_start = gate_start         # # warm-up epochs
        self.tau_hi = tau_init
        self.tau_lo = tau_end

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.vae.to(self.device)

        # total #steps used for linear schedule
        self.ramp_steps = (
            ramp_steps
            if ramp_steps is not None
            else len(dataloader) * num_epochs
        )

        # Pyro bits ----------------------------------------------------------
        pyro.clear_param_store()
        self.svi = SVI(
            self.vae.model,
            self.vae.guide,
            # ClippedAdam({"lr": lr, 'clip_norm': 5.0}),
            Adam({"lr": lr}),
            loss=Trace_ELBO(),
            # loss=RenyiELBO(alpha=0, num_particles=10)
        )

        # reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # kick-off centre coeff at max
        self.vae.center_coeff.fill_(self.init_coeff)

        self.global_step = 0

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _update_center_coeff_decay(self):
        """
        Linear decay: init_coeff  →  final_coeff over `self.ramp_steps` steps.
        """
        prog = min(self.global_step / self.ramp_steps, 1.0)  # 0 → 1
        coeff = self.init_coeff - (self.init_coeff - self.final_coeff) * prog
        self.vae.center_coeff.fill_(coeff)
    
    def _update_center_coeff_ramp(self):
        """
        Linearly ramps the center-loss coefficient *up* from `init_coeff`
        to `final_coeff` over `self.ramp_steps` training steps.
        """
        prog  = min(self.global_step / self.ramp_steps, 1.0)          # 0 → 1
        coeff = self.init_coeff + (self.final_coeff - self.init_coeff) * prog
        self.vae.center_coeff.fill_(coeff)
        
    def _current_tau(self, step) -> float:
        """Exponential decay: τ_t = max(τ_min, τ0 * exp(-k*t))."""
        k = math.log(self.tau_init / self.tau_end) / self.tau_anneal_steps
        return max(self.tau_end, self.tau_init * math.exp(-k * step))


    def fit(self):
        print(f"Training VAE on {self.device} …")
        dataset_size = len(self.dataloader.dataset)
        step = 0

        best_score = float("inf")       # best harmonic mean so far
        best_state = None                # best model state_dict
        param_store = None
        patience = self.num_epochs                 
        patience_counter = 0


        # ───────── outer progress bar over epochs ─────────
        epoch_bar = tqdm(
            range(1, self.num_epochs + 1),
            desc="Epochs",
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            epoch_loss = 0.0
            preds_p, actuals_p = [], []
            preds_ntc, actuals_ntc = [], []
            pis = torch.zeros([self.vae.perturbs, self.vae.latent_dim])

            # ----- switch gate ON exactly at gate_start ------------
            if (not self.vae.gate_on) and epoch >= self.gate_start:
                self.vae.gate_on = True
                if self.verbose:
                    print(f"▶  Gate training ENABLED at epoch {epoch}")

            # ----- anneal tau when gate is active ------------------
            if self.vae.gate_on:
                # linear tau: τ_hi → τ_lo over remaining epochs
                t = (epoch - self.gate_start) / max(1, self.num_epochs - self.gate_start)
                tau_now = self.tau_hi * (self.tau_lo / self.tau_hi) ** t
                pyro.get_param_store()["tau_temp"] = torch.tensor(tau_now,
                                                                  device=self.device)
            else:
                tau_now = self.tau_hi


            # ───── transient batch bar (cleared each epoch) ─────
            batch_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}",
                leave=False,
                position=1,
                dynamic_ncols=True,
            )

            for X_p, X_ntc, P, C in batch_bar:
                X_p = X_p.to(self.device, dtype=torch.float32)
                X_ntc = X_ntc.to(self.device, dtype=torch.float32)
                P = P.to(self.device, dtype=torch.long)
                C = C.to(self.device, dtype=torch.long)

                # anneal coeffs & temp
                # self._update_center_coeff_ramp()
                tau_curr = self._current_tau(step) if self.tau_anneal_steps else self.tau_init
                pyro.get_param_store()["tau_temp"] = torch.tensor(tau_curr, device=self.device)
                step += 1

                # optimisation step
                batch_loss = self.svi.step(X_p, X_ntc, P, C)
                epoch_loss += batch_loss
                self.global_step += 1
                batch_bar.set_postfix(elbo_per_cell=batch_loss / X_p.size(0))

            # ──────────── validation ────────────
            val_loader = getattr(self, "val_dataloader", self.dataloader)
            val_ce_sum, val_correct, val_n = 0.0, 0, 0

            all_P = []

            with torch.no_grad():
                all_P = []
                val_ce_sum, val_correct, val_n = 0.0, 0, 0

                for batch_idx, (X_p, X_ntc, P, C) in enumerate(val_loader):
                    # --- move to device ---
                    X_p  = X_p.to(self.device, dtype=torch.float32)
                    X_ntc = X_ntc.to(self.device, dtype=torch.float32)
                    P    = P.to(self.device, dtype=torch.long)
                    C    = C.to(self.device, dtype=torch.long)

                    all_P.append(P.cpu().numpy())

                    # --- guide → model replay ---
                    guide_tr = poutine.trace(self.vae.guide).get_trace(X_p, X_ntc, P, C)
                    model_tr = poutine.trace(
                        poutine.replay(self.vae.model, guide_tr)
                    ).get_trace(X_p, X_ntc, P, C)

                    # --- decoded counts directly from model trace ---
                    x_p   = model_tr.nodes["x_p"]["value"]
                    x_ntc = model_tr.nodes["x_ntc"]["value"]

                    actuals_p.append(X_p.cpu())
                    preds_p.append(x_p.detach().cpu())
                    actuals_ntc.append(X_ntc.cpu())
                    preds_ntc.append(x_ntc.detach().cpu())

                    # --- classification metric ---
                    cls_logits = model_tr.nodes["cls_logits"]["value"]
                    ce_batch   = F.cross_entropy(cls_logits, P, reduction="sum")
                    val_ce_sum += ce_batch.item()
                    val_correct += (cls_logits.argmax(dim=-1) == P).sum().item()
                    val_n += P.size(0)





            # ──────────── epoch-level metrics ────────────
            actuals_p = np.concatenate([a.numpy() for a in actuals_p], axis=0)
            preds_p   = np.concatenate([p.numpy() for p in preds_p],   axis=0)
            r2_p      = r2_score(actuals_p.flatten(), preds_p.flatten())
            actuals_ntc = np.concatenate([a.numpy() for a in actuals_ntc], axis=0)
            preds_ntc   = np.concatenate([p.numpy() for p in preds_ntc],   axis=0)
            r2_ntc      = r2_score(actuals_ntc.flatten(), preds_ntc.flatten())

            # convert to perturb-seq space for ATE metric
            # ---- normalize preds into Perturb-seq space ----
            lib_p = preds_p.sum(axis=1, keepdims=True)
            preds_p = 1e4 * preds_p / lib_p
            preds_p = np.log2(preds_p + 1.0)

            lib_ntc = preds_ntc.sum(axis=1, keepdims=True)
            preds_ntc = 1e4 * preds_ntc / lib_ntc
            preds_ntc = np.log2(preds_ntc + 1.0)
            ###############


            avg_elbo = epoch_loss / dataset_size
            ce_loss  = val_ce_sum / val_n
            acc      = val_correct / val_n
            coeff_now = self.vae.center_coeff.item()


            #Treatment effect
            dataset = self.dataloader.dataset
            n_perts   = len(dataset.perturbation_dict)
            n_genes   = preds_p.shape[1]

            effect = np.zeros((n_perts, n_genes))

            # Loop over perturbations by index
            P = np.concatenate(all_P)
            for pert_name, j in dataset.perturbation_dict.items():
                mask = (P == j)                         # cells for this perturbation
                if not mask.any():
                    continue                            # skip if no cells for this perturbation
                mean_p   = preds_p[mask].mean(axis=0)   # mean of perturbed cells
                mean_ntc = preds_ntc[mask].mean(axis=0) # mean of matched controls
                effect[j] = mean_p - mean_ntc

            effect_df = pd.DataFrame(
                effect, 
                index=list(dataset.perturbation_dict.keys()), 
                columns=self.treat_effect.var_names
            )

            effect_aligned = effect_df.loc[self.treat_effect.obs_names, self.treat_effect.var_names]

            x = effect_aligned.values.flatten()
            y = self.treat_effect.X.flatten()

            # Pearson correlation
            ate, pval = pearsonr(x, y)

            # ───── Pi percentiles (print NaN during warm-up) ─────
            # pis = pis.flatten() / (batch_idx+1)
            # pi25 = torch.quantile(pis, 0.25).item()
            # pi99 = torch.quantile(pis, 0.99).item()
            if "q_pi_logits" in pyro.get_param_store():
                q_pi = F.sigmoid(pyro.param("q_pi_logits").detach())
                pi25 = torch.quantile(q_pi, 0.25).item()
                pi99 = torch.quantile(q_pi, 0.99).item()
            else:                     # gate not yet enabled
                pi25 = float("nan")
                pi99 = float("nan")


            epoch_bar.set_postfix(
                ELBO=f"{avg_elbo:.4f}",
                CE=f"{ce_loss:.4f}",
                acc=f"{acc:.3f}",
                R2_p=f"{r2_p:.4f}",
                R2_ntc=f"{r2_ntc:.4f}",
                ATE=f"{ate:.4f}",
                pi25=f"{pi25:.2f}",
                pi99=f"{pi99:.2f}",
                lambda_center=f"{coeff_now:.6f}",
                tau=f"{tau_now:.4f}",
            )

            # ───── harmonic mean of accuracy and R² ─────
            # if (acc + r2) > 0:
            #     score = 2 * acc * r2 / (acc + r2)
            # else:
            #     score = 0.0

            score = avg_elbo

            if score < best_score or epoch < self.num_epochs // 2:
                best_score = score
                best_state = copy.deepcopy(self.vae)
                param_store = copy.deepcopy(pyro.get_param_store().get_state())
                patience_counter = 0

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹ Early stopping at epoch {epoch} (best score={best_score:.4f})")
                    pyro.clear_param_store()                                     
                    pyro.get_param_store().set_state(param_store)
                    return best_state
        pyro.clear_param_store()                                     
        pyro.get_param_store().set_state(param_store)
        return best_state, param_store