import torch
import pyro
import pyro.poutine as poutine
from pyro.infer.elbo import ELBO

def mmd_rbf_linear(x, y, sigma=None):
    """
    Un-biased linear-time MMD²(x ‖ y) with an RBF kernel.
    x, y : (B, D)
    """
    B = x.size(0)
    x = x.reshape(B, -1)
    y = y.reshape(B, -1)

    if sigma is None:                         # median heuristic
        with torch.no_grad():
            d2 = torch.cdist(x, y).pow(2)
            sigma = torch.sqrt(0.5 * torch.median(d2[d2 > 0])) + 1e-6
    gamma = 1.0 / (2 * sigma ** 2)

    idx = torch.randperm(B, device=x.device)  # random pairing
    x2, y2 = x[idx], y[idx]

    k_xx = torch.exp(-gamma * (x - x2).pow(2).sum(-1))
    k_yy = torch.exp(-gamma * (y - y2).pow(2).sum(-1))
    k_xy = torch.exp(-gamma * (x - y ).pow(2).sum(-1))

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


# Regularized ELBO with MMD to avoid posterior collapse
class ELBO_MMD_reg(ELBO):
    """
        L = recon- (1-alpha) * sum_i KL[q(z_i|x) || p(z_i)] - (alpha + lam - 1) * MMD[q(z) || p(z)]
    """
    def __init__(self,
                 alpha: float = 0.5,
                 lambda_: float = 1.0,
                 recon_weight: float = 1.0,
                 num_particles: int = 1,
                 kernel=mmd_rbf_linear):
        super().__init__(num_particles=num_particles,
                         vectorize_particles=False,
                         keep_graph=False)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.recon_w = recon_weight
        self.kernel = kernel

    @staticmethod
    def _iter_latent_sites(trace):
        for name, site in trace.nodes.items():
            if site["type"] == "sample" and not site["is_observed"]:
                yield name, site

    # flatten-and-concat helper
    @staticmethod
    def _concat_latents(latent_dict):
        flat = [v.reshape(v.size(0), -1) for v in latent_dict.values()]
        return torch.cat(flat, dim=-1) if flat else None  # (B, D_total)

    def loss_and_grads(self, model, guide, *args, **kwargs):
        # forward passes
        guide_tr = poutine.trace(guide).get_trace(*args, **kwargs)
        model_tr = poutine.trace(poutine.replay(model, guide_tr)).get_trace(*args, **kwargs)

        # NLL of observed
        recon_loss = -sum( 
            site["log_prob"].sum()
            for site in model_tr.nodes.values()
            if site["type"] == "sample" and site["is_observed"]
        )

        # KL
        kl = 0.
        for name, q_site in self._iter_latent_sites(guide_tr):
            p_site = model_tr.nodes[name]
            kl = kl + (q_site["log_prob"] - p_site["log_prob"]).sum()

        # Posterior MMD
        
        # posterior samples
        q_latents = {name: site["value"]
                     for name, site in self._iter_latent_sites(guide_tr)}
        z_q = self._concat_latents(q_latents)         # (B, D_q)

        # prior samples
        with torch.no_grad():                         
            z_p_parts = {}
            for name, p_site in self._iter_latent_sites(model_tr):
                fn = p_site["fn"]
                sample_shape = q_latents[name].shape[:-len(fn.event_shape)]
                z_p_parts[name] = fn.rsample(sample_shape)
        z_p = self._concat_latents(z_p_parts)

        mmd = self.kernel(z_q.detach(), z_p.detach()) if z_q is not None else 0.
        
        loss = self.recon_w * recon_loss + (1. - self.alpha) * kl + (self.alpha + self.lambda_ - 1.) * mmd

        loss.backward()
        return loss.detach()