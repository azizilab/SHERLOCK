import math, os, time
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam
import pyro, pyro.poutine as poutine
from sklearn.metrics import r2_score
from tqdm import tqdm
from PerturbVAE.model import VAE
from tqdm.auto import tqdm
import torch.nn.functional as F



class VAETrainer:

    def __init__(
        self,
        vae: VAE,
        dataloader: DataLoader,
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

        self.gate_start = gate_start         # # warm-up epochs
        self.tau_hi = vae.tau_hi
        self.tau_lo = vae.tau_lo

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

        # ───────── outer progress bar over epochs ─────────
        epoch_bar = tqdm(
            range(1, self.num_epochs + 1),
            desc="Epochs",
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            epoch_loss = 0.0
            preds, actuals = [], []

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

            with torch.no_grad():
                for X_p, X_ntc, P, C in val_loader:
                    X_p = X_p.to(self.device, dtype=torch.float32)
                    X_ntc = X_ntc.to(self.device, dtype=torch.float32)
                    P = P.to(self.device, dtype=torch.long)
                    C = C.to(self.device, dtype=torch.long)

                    # guide → model replay
                    guide_tr = poutine.trace(self.vae.guide).get_trace(X_p, X_ntc, P, C)
                    model_tr = poutine.trace(
                        poutine.replay(self.vae.model, guide_tr)
                    ).get_trace(X_p, X_ntc, P, C)

                    # regression predictions
                    x_mu = model_tr.nodes["x_mu"]["value"]
                    actuals.append(X_p.cpu())
                    preds.append(x_mu.cpu())

                    # ───── classification metrics
                    cls_logits = model_tr.nodes["cls_logits"]["value"]
                    ce_batch   = F.cross_entropy(cls_logits, P, reduction="sum")
                    val_ce_sum += ce_batch.item()
                    val_correct += (cls_logits.argmax(dim=-1) == P).sum().item()
                    val_n += P.size(0)

            # ──────────── epoch-level metrics ────────────
            actuals = np.concatenate([a.numpy() for a in actuals], axis=0)
            preds   = np.concatenate([p.numpy() for p in preds],   axis=0)
            r2      = r2_score(actuals.flatten(), preds.flatten())
            avg_elbo = epoch_loss / dataset_size
            ce_loss  = val_ce_sum / val_n
            acc      = val_correct / val_n
            coeff_now = self.vae.center_coeff.item()


            # ───── Pi percentiles (print NaN during warm-up) ─────
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
                R2=f"{r2:.4f}",
                pi25=f"{pi25:.2f}",
                pi99=f"{pi99:.2f}",
                lambda_center=f"{coeff_now:.6f}",
                tau=f"{tau_now:.4f}",
            )