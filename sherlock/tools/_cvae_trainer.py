import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm 
import numpy as np
from torch.utils.data import Subset

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
        self.lr = lr
        self.num_epochs = int(num_epochs)
        self.validate_every = max(1, int(validate_every))
        self.device = device
        self.patience = int(patience)
        self.non_blocking_copy = non_blocking_copy

        self.cVAE.to(self.device)
        self.optimizer = torch.optim.Adam(self.cVAE.parameters(), lr=self.lr)

        self._last_valid = dict(
            ATE=np.nan, 
            val_loss=np.nan
        )

    def _move_to_device(self, X_p, X_ntc, p, c):
        nb = self.non_blocking_copy
        return ( 
            X_p.to(self.device, dtype=torch.float32, non_blocking=nb), 
            X_ntc.to(self.device, dtype=torch.float32, non_blocking=nb), 
            p.to(self.device, dtype=torch.long, non_blocking=nb), 
            c.to(self.device, dtype=torch.long, non_blocking=nb)
        )
    
    def _unwrap_dataset(self, ds):
        while isinstance(ds, Subset):
            ds = ds.dataset
        return ds

    def _kl_divergence(self, mu_q, logvar_q, mu_p, logvar_p):
        log_terms = logvar_p - logvar_q
        var_ratio = torch.exp(logvar_q - logvar_p)
        mu_diff_sq = (mu_p - mu_q).pow(2) * torch.exp(-logvar_p)
        kl_div = 0.5 * torch.sum(log_terms + var_ratio + mu_diff_sq - 1, dim=1)
        return kl_div.mean()
    
    def _nb_nll(self,x, mu, theta, eps=1e-8):
        x = x.clamp_min(0.0)
        mu = mu.clamp_min(eps)
        theta = theta.clamp_min(eps)

        t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
        t2 = theta * (torch.log(theta) - torch.log(mu + theta))
        t3 = x * (torch.log(mu) - torch.log(mu + theta))
        log_probs = t1 + t2 + t3
        return -log_probs.mean()
    
    def _lognorm(self, x, eps=1e-8):
        lib = x.sum(dim=1, keepdim=True).clamp_min(eps)      # library size per cell
        x = 1e4 * x / lib
        return torch.log2(x + 1.0)

    def _train(self, epoch):
        self.cVAE.train()

        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_kl_div = 0.0


        for batch_idx, (X_p, X_ntc, p, c) in enumerate(self.dataloader):
            X_p, X_ntc, p, c = self._move_to_device(X_p, X_ntc, p, c)

            X_p_n = self._lognorm(X_p)
            X_ntc_n = self._lognorm(X_ntc)

            self.optimizer.zero_grad()
            mu_nb, theta_nb, mu_q, logvar_q, mu_p, logvar_p = self.cVAE(X_ntc_n, X_p_n, p, c)
            reconstruction_loss = self._nb_nll(X_p, mu_nb, theta_nb)

            beta = min(1.0, epoch / 10.0)  # 10-epoch warmup
            kl_div = self._kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
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
        val_loss = 0
        preds_p, preds_ntc, all_P = [], [], []
        first_batch = True

        with torch.no_grad():
            for X_p, X_ntc, p, c in validation_loader:
                X_p, X_ntc, p, c = self._move_to_device(X_p, X_ntc, p, c)
                X_p_n = self._lognorm(X_p)
                X_ntc_n = self._lognorm(X_ntc)

                # Forward pass 
                mu_nb, theta_nb, mu_q, logvar_q, mu_p, logvar_p = self.cVAE(X_ntc_n, X_p_n, p, c)

                # Calculating loss
                reconstruction_loss = self._nb_nll(X_p, mu_nb, theta_nb)
                kl_div = self._kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
                beta = 1.0
                loss = reconstruction_loss + kl_div * beta
                val_loss += loss.item()

                # ATE computed from predictged perturbed expression minus observed NTC
                pred_p = self._lognorm(mu_nb)
                pred_ntc = self._lognorm(X_ntc)

                if first_batch:
                    p_alt = (p + 1) % self.cVAE.perturbs
                    mu_alt, theta_alt = self.cVAE.decoder(mu_q, p_alt, c)
                    pred_alt = self._lognorm(mu_alt)
                    diff = (pred_alt - pred_p).abs().mean().item()
                    print("decoder sensitivity |pred(p+1)-pred(p)| mean abs:", diff)
                    first_batch = False

                preds_p.append(pred_p.cpu())
                preds_ntc.append(pred_ntc.cpu())
                all_P.append(p.cpu())


            pred_p = torch.cat(preds_p).numpy()
            pred_ntc = torch.cat(preds_ntc).numpy()
            all_P = torch.cat(all_P).numpy()

            val_loss = val_loss / max(1, len(validation_loader))
            print(f'Validation loss: {val_loss:.4f}')

        return {
            "pred_p": pred_p,
            "pred_ntc": pred_ntc,
            "all_p": all_P,
            "val_loss": val_loss
        }
    

    def fit(self):
        print(f"Training cVAE on {self.device} …")

        best_score = float("inf")
        best_state = None
        patience_counter = 0

        epoch_bar = tqdm(range(1, self.num_epochs + 1), desc="Epochs", dynamic_ncols=True)

        for epoch in epoch_bar:
            training_loss = self._train(epoch)
            ate = np.nan
            val_loss = np.nan


            if self.val_dataloader is not None and epoch % self.validate_every == 0:
                stats = self._validate(self.val_dataloader)


                preds_p_n, preds_ntc_n, all_P, val_loss = stats["pred_p"], stats["pred_ntc"], stats["all_p"], stats["val_loss"]
                score = val_loss

                print("VALIDATE ran. pred_p shape:", preds_p_n.shape, "pred_ntc shape:", preds_ntc_n.shape, "all_P shape:", all_P.shape)

                ds = self._unwrap_dataset(self.val_dataloader.dataset)
                pert_dict = getattr(ds, "perturbation_dict", {})
                n_perts = len(pert_dict)
                n_genes = preds_p_n.shape[1] if preds_p_n.size else 0


                if n_perts > 0 and n_genes > 0 and all_P.size > 0:
                    effect = np.zeros((n_perts, n_genes))
                    for pert_name, j in pert_dict.items():
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
                    y = np.array(self.treat_effect.X).ravel()
                    ate = np.corrcoef(x.flatten(), y.flatten())[0, 1]

                    G_pred = preds_p_n.shape[1]
                    G_te = len(self.treat_effect.var_names)
                    print("G_pred:", G_pred, "G_treat_effect:", G_te)

                
            else: 
                score = training_loss
            
            self._last_valid = dict(
                ATE=ate ,
                val_loss=val_loss
            )
            epoch_bar.set_postfix(
                ate=f"{self._last_valid['ATE']:.4f}",
                val_loss=f"{self._last_valid['val_loss']:.4f}",
                train=f"{training_loss:.4f}", 
                score=f"{score:.4f}")
            

            if score < best_score or epoch < self.num_epochs // 2:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in self.cVAE.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch} (best={best_score:.4f})")
                    break
        # after filling effect
        print(
            "effect stats:",
            "mean", effect.mean(),
            "std",  effect.std(),
            "min",  effect.min(),
            "max",  effect.max(),
        )
        print("unique P in val:", len(np.unique(all_P)))



        if best_state is not None:
            self.cVAE.load_state_dict(best_state)
        return self.cVAE
