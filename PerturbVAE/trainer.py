import math, os, time
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro, pyro.poutine as poutine
from sklearn.metrics import r2_score
from tqdm import tqdm
from PerturbVAE.model import VAE


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
        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            actuals, preds = [], []

            # ---------------- mini-batches --------------------------------
            for X, P, C in tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                leave=False,
            ):
                # send data to the right device
                X = X.to(self.device, dtype=torch.float32)
                P = P.to(self.device, dtype=torch.long)
                C = C.to(self.device, dtype=torch.long)

                # ramp centre-loss coefficient BEFORE forward pass
                self._update_center_coeff_ramp()
                tau_curr = self._current_tau(step) if self.tau_anneal_steps else self.tau_init
                pyro.get_param_store()['tau_temp'] = torch.tensor(tau_curr, device=self.device)
                step += 1
                # one SVI step (forward+backward+optim)
                batch_loss = self.svi.step(X, P, C)
                epoch_loss += batch_loss
                self.global_step += 1

                # quick reconstruction for R²
                with torch.no_grad():
                    z_loc, _ = self.vae.z_encoder(
                        torch.cat([X, self.vae.p_emb(P)], dim=-1)
                    )
                    logits = self.vae.z_decoder(z_loc)
                    mu = torch.softmax(logits, dim=-1)
                    tot = X.sum(-1, keepdim=True)
                    x_mu = tot * mu

                actuals.append(self._to_numpy(X))
                preds.append(self._to_numpy(x_mu))

            self.vae.canonicalise_columns()
            actuals = np.concatenate(actuals, axis=0)
            preds   = np.concatenate(preds,   axis=0)
            r2      = r2_score(actuals.flatten(), preds.flatten())
            avg_loss = epoch_loss / dataset_size

            coeff_now = self.vae.center_coeff.item()
            if self.verbose:
                print(
                    f"Epoch {epoch:02d} | "
                    f"ELBO per cell: {avg_loss:.4f} | "
                    f"R²: {r2:.4f} | λ_center: {coeff_now:.6f}"
                    f" | τ: {tau_curr:.4f}"
                )
        