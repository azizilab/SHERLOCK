import torch
import pyro
import pyro.poutine as poutine
from pyro.infer.elbo import ELBO

def _normalize_log1p(x, eps=1e-8):
    lib = x.sum(-1, keepdim=True).clamp_min(1.0)
    return torch.log1p(1e6 * x / lib + eps)


def _mmd1d(A, B, sigmas=(0.5, 1.0, 2.0)):
    # Deterministic 1D MMD across selected axes; vectorized over genes.
    diffs_AA = (A.unsqueeze(1) - A.unsqueeze(0))**2     # (B_p,B_p,K)
    diffs_BB = (B.unsqueeze(1) - B.unsqueeze(0))**2
    diffs_AB = (A.unsqueeze(1) - B.unsqueeze(0))**2
    mmd = 0.0
    for s in sigmas:
        inv2 = 1.0 / (2.0 * s * s)
        K_AA = torch.exp(-inv2 * diffs_AA).mean((0,1))   # (K,)
        K_BB = torch.exp(-inv2 * diffs_BB).mean((0,1))
        K_AB = torch.exp(-inv2 * diffs_AB).mean((0,1))
        mmd = mmd + (K_AA + K_BB - 2.0 * K_AB)
    return mmd / len(sigmas)


def _w1_1d(A, B):
    A_s, _ = torch.sort(A, dim=0)
    B_s, _ = torch.sort(B, dim=0)
    return (A_s - B_s).abs().mean(dim=0) 


def sparsemax(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    k = torch.arange(1, z.size(dim)+1, device=logits.device, dtype=z.dtype)
    k = k.view([1 if i!=dim else -1 for i in range(z.dim())])
    cumsum = z_sorted.cumsum(dim)
    cond = 1 + k * z_sorted > cumsum
    k_z = cond.sum(dim=dim, keepdim=True)
    tau = (cumsum.gather(dim, k_z-1) - 1) / k_z.clamp_min(1)
    return (z - tau).clamp_min(0.0)

def modules_only_weights_from_pi(q_pi_logits, M, mode="sparsemax", tau=1.0, eps=1e-8):
    if mode == "softmax":
        pi = torch.softmax(q_pi_logits / tau, dim=1)     
    elif mode == "sparsemax":
        pi = sparsemax(q_pi_logits)                      
    else:
        raise ValueError("mode must be 'softmax' or 'sparsemax'")

    S = pi @ M.T                                        
    Wg = S / (S.sum(dim=1, keepdim=True) + eps)         
    return Wg, pi

def gene_distribution_loss(
    mu, x_p, x_ntc, p, q_pi_logits, dec,               
    lam_match=1e-3, lam_margin=1e-3, margin=0.05,
    mode="sparsemax", tau_pi=1.0, eps=1e-8
):
    mu_n, xp_n, x0_n = _normalize_log1p(mu), _normalize_log1p(x_p.float()), _normalize_log1p(x_ntc.float())
    uniq, inv = torch.unique(p, return_inverse=True)     # uniq: (U,)
    if uniq.numel() == 0: return mu.new_tensor(0.0)

    M = dec[2].M.to(mu)                                  # (G,d) constant mask
    Wg, pi_used = modules_only_weights_from_pi(q_pi_logits[uniq], M, mode=mode, tau=tau_pi)   # (U,G)

    loss = mu.new_tensor(0.0)
    U = uniq.numel()
    for ui in range(U):
        cell_mask = (inv == ui)
        w = Wg[ui]                                      
        sel = w > 0 # only in-module genes
        if sel.sum() == 0: continue
        w = w[sel] / (w[sel].sum() + eps)        
        x_pert_pred = mu_n[cell_mask][:, sel]                  
        x_pert_obs =  xp_n[cell_mask][:, sel]
        x_ctrl =  x0_n[cell_mask][:, sel]
        
        d_match    = (w * _w1_1d(x_pert_pred, x_pert_obs)).sum()
        d_separate = (w * _w1_1d(x_pert_pred, x_ctrl)).sum()
        loss = loss + lam_match*d_match + lam_margin*torch.relu(margin - d_separate)

    return loss
