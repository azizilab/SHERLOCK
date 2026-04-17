import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import numpy as np
from torch.utils.data import Subset
from sklearn.metrics import r2_score

class CVAETrainer:
    def __init__(
            self,
            cVAE,
            dataloader: DataLoader, 
            treat_effect,
            val_dataloader: DataLoader = None,
            lr: float = 5e-4,
            num_epochs: int = 10,
            validate_every: int = 10,
            device=torch.device("cpu"),
            patience: int = 20,
            non_blocking_copy=True,
        ):

        self.cVAE = cVAE
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.treat_effect = treat_effect

        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.validate_every = max(1, int(validate_every))
        self.device = device
        self.patience = int(patience)
        self.non_blocking_copy = non_blocking_copy

        self.cVAE.to(self.device)
        self.optimizer = torch.optim.Adam(self.cVAE.parameters(), lr=self.lr)

        self._last_valid = dict(
            ATE=np.nan, 
            val_loss=np.nan,
            R2=np.nan
        )

    def _move_to_device(self, x, p):
        nb = self.non_blocking_copy
        return ( 
            x.to(self.device, dtype=torch.float32, non_blocking=nb), 
            p.to(self.device, dtype=torch.long, non_blocking=nb)
        )
    
    def _unwrap_dataset(self, ds):
        while isinstance(ds, Subset):
            ds = ds.dataset
        return ds

    def _kl_divergence(self, mu_q, logvar_q):
        kl = 0.5 * torch.sum(torch.exp(logvar_q) + mu_q**2 - 1.0 - logvar_q, dim=1).mean()
        return kl
    
    def _nb_nll(self,x, mu, theta, eps=1e-8):
        x = x.clamp_min(0.0)
        mu = mu.clamp_min(eps)
        theta = theta.clamp_min(eps)

        t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        t2 = theta * (torch.log(theta) - torch.log(mu + theta))
        t3 = x * (torch.log(mu) - torch.log(mu + theta))
        log_probs = t1 + t2 + t3
        return -log_probs.mean()
    
    def _lognorm_ln(self, x, eps=1e-8):
        lib = x.sum(dim=1, keepdim=True).clamp_min(eps)      # library size per cell
        x = 1e4 * x / lib
        return torch.log(x + 1.0)
    
    def _compute_ate_from_preds(self, preds_ln, p_all, ds):
        
        pert_dict = getattr(ds, "perturbation_dict", {})
        ntc_idx = getattr(ds, "ntc_idx", None)

        if ntc_idx is None or len(pert_dict) == 0:
            return np.nan
        
        n_perts = len(pert_dict)
        n_genes = preds_ln.shape[1] if preds_ln.size else 0

        if n_perts == 0 or n_genes == 0 or p_all.size == 0:
            return np.nan
        
        # NTC mean
        ntc_mask = (p_all == ntc_idx)
        if not ntc_mask.any():
            return np.nan
        ntc_mean = preds_ln[ntc_mask].mean(axis=0)

        # effects per perturbation
        effect = np.zeros((n_perts, n_genes), dtype=np.float32)
        for pert_name, j in pert_dict.items():
            if j == ntc_idx:
                continue
            mask = (p_all == j)

            if not mask.any():
                continue

            effect[j] = preds_ln[mask].mean(0) - ntc_mean
        
        idx_to_pert = {j: name for name, j in pert_dict.items()}
        row_names = [idx_to_pert[j] for j in range(len(pert_dict))]
        # DataFrame for easy alignment
        effect_df = pd.DataFrame(
            effect,
            index=row_names,
            columns=self.treat_effect.var_names,
        )

        # align to ground truth treat_effect 
        effect_aligned = effect_df.loc[self.treat_effect.obs_names, self.treat_effect.var_names]
        x = effect_aligned.values.ravel()
        y = np.array(self.treat_effect.X).ravel()


        # Pearson correlation
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        
        return np.corrcoef(x, y)[0, 1]


    def _train(self, epoch):

        self.cVAE.train()
        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_kl_div = 0.0


        for x, p in self.dataloader:
            x, p = self._move_to_device(x, p)

            self.optimizer.zero_grad()

            mu_nb, theta_nb, mu_q, logvar_q, _, _ = self.cVAE(x, p)
            reconstruction_loss = self._nb_nll(x, mu_nb, theta_nb)

            beta = min(1.0, epoch / 50.0)  # 50-epoch warmup
            kl_div = self._kl_divergence(mu_q, logvar_q)

            loss = reconstruction_loss + beta * kl_div
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_rec_loss += reconstruction_loss.item()
            epoch_kl_div += kl_div.item()

        n_batches = len(self.dataloader)
        print(
            f"Epoch {epoch} | "
            f"Average loss: {epoch_loss / n_batches:.4f}, "
            f"Reconstruction loss: {epoch_rec_loss / n_batches:.4f} |"
            f"KL Divergence: {epoch_kl_div / n_batches:.4f} |"
        )

        return epoch_loss / n_batches

    def _validate(self, validation_loader):
        self.cVAE.eval()
        preds_ln, losses, trues_ln, p_all = [], [], [], []
    
        with torch.no_grad():
            for x, p in validation_loader:
                x, p = self._move_to_device(x, p)
            
                # Forward pass 
                mu_nb, theta_nb, mu_q, logvar_q, _, _ = self.cVAE(x, p)

                # Calculating loss
                reconstruction_loss = self._nb_nll(x, mu_nb, theta_nb)
                kl_div = self._kl_divergence(mu_q, logvar_q)
                loss = reconstruction_loss + kl_div
                losses.append(loss.item())

                # Prior path for ATE (generative)
                pred_ln = self._lognorm_ln(mu_nb)
                true_ln = self._lognorm_ln(x)
                
                preds_ln.append(pred_ln.cpu())
                trues_ln.append(true_ln.cpu())
                p_all.append(p.cpu())
            
            val_loss = float(np.mean(losses)) if len(losses) else np.nan

            preds_ln = torch.cat(preds_ln, dim=0).numpy() if len(preds_ln) else np.array([])
            trues_ln = torch.cat(trues_ln, dim=0).numpy() if len(trues_ln) else np.array([])
            all_P = torch.cat(p_all, dim=0).numpy() if len(p_all) else np.array([])

            r2 = r2_score(trues_ln.ravel(), preds_ln.ravel()) if preds_ln.size else np.nan

        return {
            "preds_ln": preds_ln,
            "trues_ln": trues_ln,
            "all_p": all_P,
            "val_loss": val_loss,
            "r2": r2
        }
    

    def fit(self):
        print(f"Training cVAE on {self.device} …")

        best_score = -float("inf")
        best_state = None
        patience_counter = 0

        epoch_bar = tqdm(range(1, self.num_epochs + 1), desc="Epochs", dynamic_ncols=True)

        for epoch in epoch_bar:
            training_loss = self._train(epoch)

            ate = np.nan
            val_loss = np.nan
            r2 = np.nan
            score = np.nan

            ran_validation = (self.val_dataloader is not None) and (epoch % self.validate_every == 0)
            
            if ran_validation:
                stats = self._validate(self.val_dataloader)

                val_loss = stats["val_loss"]
                r2 = stats["r2"]

                ds = self._unwrap_dataset(self.val_dataloader.dataset)
                ate = self._compute_ate_from_preds(stats["preds_ln"], stats["all_p"], ds)

                score = ate

                if(not np.isnan(score)) and (score > best_score):
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in self.cVAE.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

            self._last_valid = dict(
                ATE=ate ,
                val_loss=val_loss,
                R2=r2
            )

            epoch_bar.set_postfix(
                ate=("-" if np.isnan(ate) else f"{ate:.4f}"),
                val_loss=("-" if np.isnan(val_loss) else f"{val_loss:.4f}"),
                train=f"{training_loss:.4f}", 
                score=("—" if np.isnan(score) else f"{score:.4f}"),
                r2=("-" if np.isnan(r2) else f"{r2:.4f}"),
                pat=f"{patience_counter}/{self.patience}"
            )
            if ran_validation and patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch} (best ATE={best_score:.4f})")
                break
        if best_state is not None:
            self.cVAE.load_state_dict(best_state)

        return self.cVAE