## Sherlock

# VAE Training – Parameter Reference

Reference for **all arguments** used by `run_single(...)`:

---

## Parameters

| Name | Default(s) | Used by | What it controls / When to change |
|---|---|---|---|
| **adata** | *(required)* | `run_single` | Input `AnnData` (cells × features). Must contain `.X` (and any fields your dataset expects). Pre-filter/normalize per your pipeline. |
| **batch_size** | `4096` | `run_single` | Minibatch size. Increase for faster epochs if memory allows; decrease if you hit OOM or unstable gradients. |
| **device** | `cpu` (`run_single`), `cpu` (`Trainer`) | both | Compute device (`torch.device("cpu")` / `"cuda:0"`). Use CUDA for GPU training. |
| **gate_init_p** | `0.5` | `VAE` | Initial open probability for the Hard-Concrete gate. Lower to start sparser; higher to start more open. |
| **H_lambda** | `1e-3` | `VAE` | Column-usage entropy encouragement (balances latent dimension usage). Raise if a few columns dominate; lower if over-spreading hurts. |
| **latent_dim** | `16` (run/model) | `run_single`, `VAE` | Size of latent bottleneck. Larger → richer representation; smaller → stronger regularization. |
| **l0_lambda** | `10.0` | `VAE` | Expected-L0 sparsity penalty on the gate (perturbation × latent). Prioritize this over **l1/l2**. Increase for sparser gating. |
| **l1_lambda** | `1e-3` | `VAE` | L1 penalty on gate logits. Increase to further encourage sparsity. |
| **l2_lambda** | `1e-3` | `VAE` | L2 penalty on expected squared gate usage. Increase for smoother/smaller gates. |
| **lr** | `1e-3` (`run_single`), `5e-4` (`Trainer`) | both | Optimizer learning rate. Lower if training diverges/oscillates; raise slightly if learning is slow. |
| **num_epochs** | `200` (`run_single`), `10` (`Trainer`) | both | Maximum epochs. Use with `patience` for early stopping. Increase for harder datasets. |
| **num_workers** | `0` | `run_single` | DataLoader workers. Increase (e.g., 4–8) for faster CPU data loading; Will be slow to initialize for non-zero values |
| **patience** | `20` | both | Early-stopping patience (epochs without improvement). Increase to allow longer training; decrease to stop earlier. |
| **seed** | `0` | `Trainer` | RNG seed for reproducibility (init, shuffles). Change to explore different random starts. |
| **shuffle** | `True` | `run_single` | Shuffle samples each epoch. Keep `True` for SGD; set `False` for deterministic passes/eval. |
| **use_conditions** | `False` | `run_single`, `VAE` | Enable conditional priors/embeddings over `conds`. Set `True` if you provide condition indices (e.g., batch/cell line). |
| **validate_every** | `10` | both | Run validation every N epochs. Increase to validate less often (faster), decrease for closer monitoring. |
| **verbose** | `True` | `Trainer` | Print training/validation logs. Turn off for silent runs. |

---
