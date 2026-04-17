from __future__ import annotations
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from sherlock.benchmarks._metrics import lognorm_ln, compute_ate_from_preds_ln

class PyroTrainer:
    def __init__(
        self,
        model,
        dataloader,
        val_dataloader=None,
        treat_effect=None,
        lr=5e-4,
        num_epochs=50,
        validate_every=5,
        patience=20,
        device="cpu",
        early_stop_metric="ate",
    ):
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.treat_effect = treat_effect
        self.device = torch.device(device) if isinstance(device, str) else device

        self.num_epochs = int(num_epochs)
        self.validate_every = max(1, int(validate_every))
        self.patience = int(patience)
        self.early_stop_metric = early_stop_metric

        self.model.to(self.device)

        pyro.clear_param_store()
        self.svi = SVI(
            self.model.model,
            self.model.guide,
            Adam({"lr": float(lr)}),
            loss=Trace_ELBO(),
        )

        self._last_valid = {"ATE": np.nan, "val_loss": np.nan, "R2": np.nan}

    def _unwrap_dataset(self, ds):
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        return ds

    def _score(self, *, ate, val_loss, r2):
        if self.early_stop_metric == "ate":
            if not np.isnan(ate): return float(ate), True
            if not np.isnan(val_loss): return -float(val_loss), True
            return float("-inf"), True
        if self.early_stop_metric == "r2":
            if not np.isnan(r2): return float(r2), True
            if not np.isnan(val_loss): return -float(val_loss), True
            return float("-inf"), True
        if self.early_stop_metric == "val_loss":
            if not np.isnan(val_loss): return -float(val_loss), True
            return float("-inf"), True
        raise ValueError(self.early_stop_metric)

    def _train_epoch(self):
        self.model.train()
        losses = []
        for batch in self.dataloader:
            # batch can be (x, p) or more later; your dataset defines this
            batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
            loss = self.svi.step(*batch)
            losses.append(float(loss))
        return float(np.mean(losses)) if losses else np.nan

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        preds_ln, trues_ln, p_all = [], [], []
        losses = []

        for batch in self.val_dataloader:
            batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
            # loss without stepping:
            loss = self.svi.evaluate_loss(*batch)
            losses.append(float(loss))

            # assumes batch = (x, p)
            x, p = batch
            x_pred = self.model.predict_mu(x, p)

            preds_ln.append(lognorm_ln(x_pred).cpu())
            trues_ln.append(lognorm_ln(x).cpu())
            p_all.append(p.cpu())

        val_loss = float(np.mean(losses)) if losses else np.nan
        preds_ln = torch.cat(preds_ln).numpy() if preds_ln else np.array([])
        trues_ln = torch.cat(trues_ln).numpy() if trues_ln else np.array([])
        p_all = torch.cat(p_all).numpy() if p_all else np.array([])

        r2 = r2_score(trues_ln.ravel(), preds_ln.ravel()) if preds_ln.size else np.nan
        ds = self._unwrap_dataset(self.val_dataloader.dataset)
        ate = compute_ate_from_preds_ln(preds_ln, p_all, ds, self.treat_effect)
        return val_loss, r2, ate

    def fit(self):
        best_score = -float("inf")
        best_state = None
        pat = 0

        bar = tqdm(range(1, self.num_epochs + 1), desc="Epochs", dynamic_ncols=True)
        for epoch in bar:
            train_loss = self._train_epoch()

            ate = val_loss = r2 = np.nan
            ran_val = self.val_dataloader is not None and (epoch % self.validate_every == 0)
            if ran_val:
                val_loss, r2, ate = self._validate()
                score, _ = self._score(ate=ate, val_loss=val_loss, r2=r2)

                if score > best_score:
                    best_score = score
                    best_state = pyro.get_param_store().get_state()
                    pat = 0
                else:
                    pat += 1

                self._last_valid = {"ATE": float(ate), "val_loss": float(val_loss), "R2": float(r2)}

            bar.set_postfix(
                train=f"{train_loss:.4f}" if not np.isnan(train_loss) else "-",
                val=f"{val_loss:.4f}" if not np.isnan(val_loss) else "-",
                r2=f"{r2:.4f}" if not np.isnan(r2) else "-",
                ate=f"{ate:.4f}" if not np.isnan(ate) else "-",
                pat=f"{pat}/{self.patience}" if ran_val else "-",
            )

            if ran_val and pat >= self.patience:
                break

        if best_state is not None:
            pyro.get_param_store().set_state(best_state)

        return self.model