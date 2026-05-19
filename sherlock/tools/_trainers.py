import copy
import math
import numpy as np
import pyro
import pyro.poutine as poutine
import torch
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class VAETrainer:
    def __init__(
        self,
        vae,
        dataloader: DataLoader,
        treat_effect,
        adata=None,
        lr: float = 5e-4,
        num_epochs: int = 10,
        validate_every: int = 10,
        verbose: bool = True,
        device=torch.device('cpu'),
        seed: int | None = None,
        non_blocking_copy: bool = True,
        tau_init: float | None = None,
        tau_end: float = 0.3,
        tau_anneal_steps: int | None = None,
        patience: int = 20,
        n_epochs_kl_warmup: int = 50,
        n_epochs_l0_warmup: int = 50,
    ):
        self.vae = vae
        self.dataloader = dataloader
        self.treat_effect = treat_effect
        self.adata = adata
        self.num_epochs = int(num_epochs)
        self.validate_every = max(1, int(validate_every))
        self.verbose = verbose
        self.non_blocking_copy = non_blocking_copy
        self.patience = patience

        self.device = device
        self.vae.to(self.device)

        pyro.clear_param_store()
        self.svi = SVI(self.vae.model, self.vae.guide, Adam({"lr": lr}), loss=Trace_ELBO())

        if seed is not None:
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

        # KL / L0 warmup schedules
        self.n_epochs_kl_warmup = max(1, int(n_epochs_kl_warmup))
        self.n_epochs_l0_warmup = max(1, int(n_epochs_l0_warmup))

        # last validated metrics (persist between evals)
        self._last_valid = dict(
            CE=np.nan, acc=np.nan, R2_p=np.nan, R2_ntc=np.nan, ATE=np.nan, pi25=np.nan, pi99=np.nan
        )
        self._last_tau = self.tau_init

        # per-validation history for ATE trajectory plotting
        self.history: list[dict] = []

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _kl_weight(self, epoch: int) -> float:
        return min(1.0, epoch / self.n_epochs_kl_warmup)

    def _l0_weight(self, epoch: int) -> float:
        """Gate sparsity weight: waits until KL warmup ends, then ramps 0→1 over n_epochs_l0_warmup epochs."""
        delay = self.n_epochs_kl_warmup
        if epoch <= delay:
            return 0.0
        return min(1.0, (epoch - delay) / self.n_epochs_l0_warmup)

    def _current_tau(self, step) -> float:
        if self.tau_anneal_steps <= 0 or self.tau_init <= self.tau_end:
            return self.tau_end
        k = math.log(self.tau_init / self.tau_end) / self.tau_anneal_steps
        return max(self.tau_end, self.tau_init * math.exp(-k * step))

    def _move_to_device(self, X_p, X_ntc, P, C, C_ntc):
        nb = self.non_blocking_copy
        return (
            X_p.to(self.device, dtype=torch.float32, non_blocking=nb),
            X_ntc.to(self.device, dtype=torch.float32, non_blocking=nb),
            P.to(self.device, dtype=torch.long, non_blocking=nb),
            C.to(self.device, dtype=torch.long, non_blocking=nb),
            C_ntc.to(self.device, dtype=torch.long, non_blocking=nb),
        )

    def _validate(self, val_loader):
        self.vae.eval()
        preds_p, preds_ntc, actuals_p, actuals_ntc = [], [], [], []
        val_ce_sum, val_correct, val_n = 0.0, 0, 0
        all_P = []
        pi25 = pi99 = np.nan  # percentiles of gate probability π

        with torch.no_grad():
            for X_p, X_ntc, P, C, C_ntc in val_loader:
                X_p, X_ntc, P, C, C_ntc = self._move_to_device(X_p, X_ntc, P, C, C_ntc)
                all_P.append(P.detach().cpu().numpy())
                n_ntc = X_ntc.size(0)

                guide_tr = poutine.trace(self.vae.guide).get_trace(X_p, X_ntc, P, C, C_ntc)
                z     = guide_tr.nodes["z"]["value"]   # (n_ntc + n, d)
                z_ntc = z[:n_ntc]
                z_p   = z[n_ntc:]

                # CE / acc (perturbed cells only)
                cls_logits = self.vae.cls_head(z_p)
                if P.ndim == 1:
                    val_ce_sum += F.cross_entropy(cls_logits, P.long(), reduction="sum").item()
                    val_correct += (cls_logits.argmax(dim=-1) == P).sum().item()
                    val_n += P.size(0)

                else:
                    B, Pn = cls_logits.shape
                    mask = (P != -1)
                    k = mask.sum(dim=1).max().item()

                    tgt = torch.zeros((B, Pn), device=cls_logits.device, dtype=cls_logits.dtype)
                    tgt.scatter_(1, P.clamp(min=0).long(), mask.float())

                    val_ce_sum += F.binary_cross_entropy_with_logits(cls_logits, tgt, reduction="sum").item()

                    topk = cls_logits.topk(k=k, dim=1).indices
                    hit = (topk.unsqueeze(2) == P.clamp(min=0).long().unsqueeze(1)) & mask.unsqueeze(1)
                    val_correct += (hit.any(dim=1).sum(dim=1) == mask.sum(dim=1)).sum().item()
                    val_n += B

                # decode means for R²
                total_ntc = X_ntc.sum(-1, keepdim=True)
                mu_ntc = total_ntc * torch.softmax(self.vae.z_decoder(z_ntc), dim=-1)
                total_p = X_p.sum(-1, keepdim=True)
                mu_p = total_p * torch.softmax(self.vae.z_decoder(z_p), dim=-1)

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

            kl_weight = self._kl_weight(epoch)
            l0_weight = self._l0_weight(epoch)
            self.vae.kl_weight = kl_weight
            self.vae.l0_weight = l0_weight

            for X_p, X_ntc, P, C, C_ntc in self.dataloader:  # no inner tqdm
                X_p, X_ntc, P, C, C_ntc = self._move_to_device(X_p, X_ntc, P, C, C_ntc)

                # anneal gate temperature (gating is always on)
                tau_curr = self._current_tau(self.global_step)
                self.vae.gate.set_temperature(tau_curr)
                self._last_tau = tau_curr

                epoch_loss += self.svi.step(X_p, X_ntc, P, C, C_ntc)
                self.global_step += 1

            avg_elbo = epoch_loss / dataset_size

            # validate on schedule; remember metrics for persistence
            if epoch % self.validate_every == 0:
                stats = self._validate(getattr(self, "val_dataloader", self.dataloader))

                # ATE via counterfactual prediction
                ate = float("nan")
                if self.adata is not None:
                    cf = self.vae.predict_counterfactual_effects(
                        self.adata,
                        observed_effect=self.treat_effect,
                        device=self.device,
                    )
                    ate = float(cf.get("corr", float("nan")))

                self._last_valid = dict(
                    CE=stats["ce_loss"],
                    acc=stats["acc"],
                    R2_p=stats["r2_p"],
                    R2_ntc=stats["r2_ntc"],
                    ATE=ate,
                    pi25=stats["pi25"],
                    pi99=stats["pi99"],
                )

                self.history.append({"epoch": epoch, "ATE": ate, "pi25": stats["pi25"], "pi99": stats["pi99"]})

            # persistent tqdm display (no NA flicker)
            epoch_bar.set_postfix(
                ELBO=f"{avg_elbo:.4f}",
                KLw=f"{kl_weight:.2f}",
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
    Trains a cVAE model using plain PyTorch Adam with KL warmup.

    Mirrors sVAETrainer: calls model.loss(x, p, kl_weight, n_obs) directly.

    Parameters
    ----------
    model              : cVAE instance.
    dataloader         : DataLoader yielding (x, p) batches.
    treat_effect       : AnnData with ground-truth ATEs used for validation.
    lr                 : Adam learning rate.
    num_epochs         : maximum training epochs.
    validate_every     : epoch interval between validation passes.
    patience           : early-stopping patience (epochs without loss improvement).
    n_epochs_kl_warmup : epochs over which KL weight ramps 0 → 1.
    device             : torch device.
    seed               : RNG seed.
    """

    def __init__(
        self,
        model,
        dataloader: DataLoader,
        treat_effect,
        adata=None,
        lr: float = 3e-4,
        num_epochs: int = 200,
        validate_every: int = 10,
        patience: int = 30,
        n_epochs_kl_warmup: int = 50,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
        non_blocking_copy: bool = True,
        desc_name: str = "cVAE",
    ):
        self.model = model
        self.dataloader = dataloader
        self.treat_effect = treat_effect
        self.adata = adata
        self.num_epochs = int(num_epochs)
        self.validate_every = max(1, int(validate_every))
        self.patience = patience
        self.n_epochs_kl_warmup = max(1, int(n_epochs_kl_warmup))
        self.n_obs = len(dataloader.dataset)
        self.non_blocking_copy = non_blocking_copy
        self.desc_name = desc_name

        self.device = device
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self._last_valid = dict(R2=float("nan"), ATE=float("nan"))
        self.history: list[dict] = []

    def _kl_weight(self, epoch: int) -> float:
        return min(1.0, epoch / self.n_epochs_kl_warmup)

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

        ate = float("nan")
        if self.adata is not None:
            cf = self.model.predict_counterfactual_effects(
                self.adata,
                observed_effect=self.treat_effect,
                device=self.device,
            )
            ate = float(cf.get("corr", float("nan")))

        return {"r2": r2, "ate": ate}

    def fit(self, val_loader: DataLoader | None = None) -> tuple:
        """
        Train the cVAE.

        Returns
        -------
        (best_model, None)  — None in place of Pyro param store.
        """
        _val_loader = val_loader if val_loader is not None else self.dataloader

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        epoch_bar = tqdm(range(1, self.num_epochs + 1), desc=self.desc_name, dynamic_ncols=True)

        for epoch in epoch_bar:
            self.model.train()
            kl_weight = self._kl_weight(epoch)
            epoch_loss = 0.0

            for x, p in self.dataloader:
                x, p = self._to_device(x, p)
                self.optimizer.zero_grad()
                loss = self.model.loss(x, p, kl_weight=kl_weight, n_obs=self.n_obs)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)

            if epoch % self.validate_every == 0:
                stats = self._validate(_val_loader)
                self._last_valid["R2"]  = stats["r2"]
                self._last_valid["ATE"] = stats["ate"]

            self.history.append(
                {"epoch": epoch, "loss": avg_loss, **self._last_valid}
            )

            epoch_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                KLw=f"{kl_weight:.2f}",
                R2=f"{self._last_valid['R2']:.4f}"
                if not math.isnan(self._last_valid["R2"]) else "nan",
                ATE=f"{self._last_valid['ATE']:.4f}"
                if not math.isnan(self._last_valid["ATE"]) else "nan",
            )

            if avg_loss < best_loss or epoch < self.num_epochs // 4:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch} (best loss={best_loss:.4f})")
                    break

        return best_state, None


class sVAETrainer:
    """
    Trains an sVAE model using plain PyTorch Adam with KL warmup.

    Matches the training protocol of Genentech/sVAE demo.py:
    linear KL annealing from 0 → 1 over the first n_epochs_kl_warmup epochs,
    then full KL weight. No Pyro — calls model.loss(x, p, kl_weight, n_obs)
    directly.

    Parameters
    ----------
    model              : sVAE instance.
    dataloader         : DataLoader yielding (x, p) batches.
    treat_effect       : AnnData with ground-truth ATEs for validation.
    lr                 : Adam learning rate.
    num_epochs         : maximum training epochs.
    validate_every     : epoch interval between validation passes.
    patience           : early-stopping patience (epochs without loss improvement).
    n_epochs_kl_warmup : epochs over which KL weight ramps 0 → 1 (default 50).
    device             : torch device.
    seed               : RNG seed.
    """

    def __init__(
        self,
        model,
        dataloader: DataLoader,
        treat_effect,
        adata=None,
        lr: float = 3e-4,
        num_epochs: int = 200,
        validate_every: int = 10,
        patience: int = 30,
        n_epochs_kl_warmup: int = 50,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
        non_blocking_copy: bool = True,
        desc_name: str = "sVAE",
    ):
        self.model = model
        self.dataloader = dataloader
        self.treat_effect = treat_effect
        self.adata = adata
        self.num_epochs = int(num_epochs)
        self.validate_every = max(1, int(validate_every))
        self.patience = patience
        self.n_epochs_kl_warmup = max(1, int(n_epochs_kl_warmup))
        self.n_obs = len(dataloader.dataset)
        self.non_blocking_copy = non_blocking_copy
        self.desc_name = desc_name

        self.device = device
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self._last_valid = dict(R2=float("nan"), ATE=float("nan"))
        self.history: list[dict] = []

    def _kl_weight(self, epoch: int) -> float:
        return min(1.0, epoch / self.n_epochs_kl_warmup)

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

        ate = float("nan")
        if self.adata is not None:
            cf = self.model.predict_counterfactual_effects(
                self.adata,
                observed_effect=self.treat_effect,
                device=self.device,
            )
            ate = float(cf.get("corr", float("nan")))

        return {"r2": r2, "ate": ate}

    def fit(self, val_loader: DataLoader | None = None) -> tuple:
        """
        Train the sVAE.

        Returns
        -------
        (best_model, None)  — None in place of Pyro param store.
        """
        _val_loader = val_loader if val_loader is not None else self.dataloader

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        epoch_bar = tqdm(range(1, self.num_epochs + 1), desc=self.desc_name, dynamic_ncols=True)

        for epoch in epoch_bar:
            self.model.train()
            kl_weight = self._kl_weight(epoch)
            epoch_loss = 0.0

            for x, p in self.dataloader:
                x, p = self._to_device(x, p)
                self.optimizer.zero_grad()
                loss = self.model.loss(x, p, kl_weight=kl_weight, n_obs=self.n_obs)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)

            if epoch % self.validate_every == 0:
                stats = self._validate(_val_loader)
                self._last_valid["R2"]  = stats["r2"]
                self._last_valid["ATE"] = stats["ate"]

            self.history.append(
                {"epoch": epoch, "loss": avg_loss, **self._last_valid}
            )

            epoch_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                KLw=f"{kl_weight:.2f}",
                R2=f"{self._last_valid['R2']:.4f}"
                if not math.isnan(self._last_valid["R2"]) else "nan",
                ATE=f"{self._last_valid['ATE']:.4f}"
                if not math.isnan(self._last_valid["ATE"]) else "nan",
            )

            if avg_loss < best_loss or epoch < self.num_epochs // 4:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch} (best loss={best_loss:.4f})")
                    break

        return best_state, None


class SCGENTrainer:
    """
    Trains SCGENModel: unconditional VAE + post-training delta fitting.

    The VAE is trained without perturbation labels (p is ignored each batch).
    After finding the best checkpoint, fit_deltas(adata) is called once to
    compute and store the per-perturbation delta vectors.

    Parameters
    ----------
    model          : SCGENModel instance.
    dataloader     : DataLoader yielding (x, p) batches (p ignored during training).
    treat_effect   : AnnData with ground-truth ATEs used for ATE validation.
    adata          : full AnnData; used for fit_deltas and ATE computation.
    lr             : Adam learning rate.
    num_epochs     : maximum training epochs.
    validate_every : epoch interval between validation passes.
    patience       : early-stopping patience.
    device         : torch device.
    seed           : RNG seed.
    """

    def __init__(
        self,
        model,
        dataloader,
        treat_effect,
        adata=None,
        lr: float = 1e-4,
        num_epochs: int = 200,
        validate_every: int = 10,
        patience: int = 30,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
        non_blocking_copy: bool = True,
    ):
        self.model             = model
        self.dataloader        = dataloader
        self.treat_effect      = treat_effect
        self.adata             = adata
        self.num_epochs        = int(num_epochs)
        self.validate_every    = max(1, int(validate_every))
        self.patience          = patience
        self.non_blocking_copy = non_blocking_copy

        self.device = device
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if seed is not None:
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
            p.to(self.device, dtype=torch.long,    non_blocking=nb),
        )

    def _validate(self, val_loader) -> dict:
        from sklearn.metrics import r2_score

        self.model.eval()
        preds_all, actuals_all = [], []
        with torch.no_grad():
            for x, p in val_loader:
                x, _ = self._to_device(x, p)
                preds_all.append(self.model.get_recon(x).cpu().numpy())
                actuals_all.append(x.cpu().numpy())

        r2 = r2_score(
            np.concatenate(actuals_all).ravel(),
            np.concatenate(preds_all).ravel(),
        )

        ate = float("nan")
        if self.adata is not None and self.treat_effect is not None:
            self.model.fit_deltas(self.adata, self.device)
            cf  = self.model.predict_counterfactual_effects(
                self.adata,
                observed_effect=self.treat_effect,
                device=self.device,
            )
            ate = float(cf.get("corr", float("nan")))

        return {"r2": r2, "ate": ate}

    def fit(self, val_loader=None) -> tuple:
        _val_loader = val_loader if val_loader is not None else self.dataloader

        best_loss        = float("inf")
        best_state       = None
        patience_counter = 0

        epoch_bar = tqdm(range(1, self.num_epochs + 1), desc="scGEN", dynamic_ncols=True)

        for epoch in epoch_bar:
            self.model.train()
            epoch_loss = 0.0
            for x, p in self.dataloader:
                x, _ = self._to_device(x, p)   # p unused during training
                self.optimizer.zero_grad()
                loss = self.model.loss(x)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)

            if epoch % self.validate_every == 0:
                stats = self._validate(_val_loader)
                self._last_valid["R2"]  = stats["r2"]
                self._last_valid["ATE"] = stats["ate"]

            self.history.append({"epoch": epoch, "loss": avg_loss, **self._last_valid})
            epoch_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                R2 =f"{self._last_valid['R2']:.4f}"  if not math.isnan(self._last_valid["R2"])  else "nan",
                ATE=f"{self._last_valid['ATE']:.4f}" if not math.isnan(self._last_valid["ATE"]) else "nan",
            )

            if avg_loss < best_loss or epoch < self.num_epochs // 4:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch} (best loss={best_loss:.4f})")
                    break

        # Store deltas on best checkpoint before returning
        if best_state is not None and self.adata is not None:
            best_state.fit_deltas(self.adata, self.device)

        return best_state, None
