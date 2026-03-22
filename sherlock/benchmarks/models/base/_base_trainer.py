# sherlock/benchmarks/trainers/_base_trainer.py
from __future__ import annotations
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score
from sherlock.benchmarks._metrics import lognorm_ln, compute_ate_from_preds_ln

class BaseTrainer:
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
        device: str | torch.device="cpu",
        non_blocking_copy: bool=True,
        early_stop_metric: str = "ate",
    ):
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.treat_effect = treat_effect
        self.early_stop_metric = early_stop_metric

        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.validate_every = max(1, int(validate_every))
        self.patience = int(patience)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.nb = non_blocking_copy

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self._last_valid = {"ATE": np.nan, "val_loss": np.nan, "R2": np.nan}

    def _move(self, x, p):
        return (
            x.to(self.device, dtype=torch.float32, non_blocking=self.nb),
            p.to(self.device, dtype=torch.long, non_blocking=self.nb),
        )
    
    def _unwrap_dataset(self, ds):
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        return ds

    def _train_epoch(self, epoch: int):
        self.model.train()
        losses = []
        for x, p in self.dataloader:
            x, p = self._move(x, p)
            self.optimizer.zero_grad()
            loss, _logs = self.model.loss(x, p, epoch=epoch)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else np.nan

    def _validate(self, epoch: int):
        self.model.eval()
        losses = []
        preds_ln = []
        trues_ln = []
        p_all = []

        with torch.no_grad():
            for x, p in self.val_dataloader:
                x, p = self._move(x, p)

                loss, _ = self.model.loss(x, p, epoch=epoch)
                losses.append(float(loss.item()))

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
    
    def _score(self, *, ate, val_loss, r2):
        """
        Returns (score, higher_is_better)
        """
        if self.early_stop_metric == "ate":
            # If ATE unavailable, fallback to minimizing val_loss
            if not np.isnan(ate):
                return float(ate), True
            if not np.isnan(val_loss):
                return -float(val_loss), True
            return float("-inf"), True

        if self.early_stop_metric == "r2":
            if not np.isnan(r2):
                return float(r2), True
            if not np.isnan(val_loss):
                return -float(val_loss), True
            return float("-inf"), True

        if self.early_stop_metric == "val_loss":
            if not np.isnan(val_loss):
                return -float(val_loss), True
            return float("-inf"), True

        raise ValueError(f"Unknown early_stop_metric={self.early_stop_metric}")
    
    def fit(self):
        best_score = -float("inf")
        best_state = None
        pat = 0

        bar = tqdm(range(1, self.num_epochs + 1), desc="Epochs", dynamic_ncols=True)

        for epoch in bar:
            train_loss = self._train_epoch(epoch)

            ate = val_loss = r2 = np.nan
            score = np.nan

            ran_val = self.val_dataloader is not None and (epoch % self.validate_every == 0)
            if ran_val:
                val_loss, r2, ate = self._validate(epoch)
                score, _ = self._score(ate=ate, val_loss=val_loss, r2=r2)

                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
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
            self.model.load_state_dict(best_state)

        return self.model