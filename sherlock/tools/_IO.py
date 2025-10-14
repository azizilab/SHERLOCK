import torch
import pyro

def save_results(results, path: str) -> None:
    """
    Save everything needed to reconstruct and load a VAE from just `path`.
    Stores:
      - constructor args (input_dim, latent_dim, perturbs, conds, tau, reg weights, etc.)
      - model state_dict
    """
    
    model = results['model']

    ctor = {
        "input_dim":  model.input_dim,
        "latent_dim": model.latent_dim,
        "perturbs":   model.perturbs,
        "conds":      model.conds,
        "use_conditions": bool(model.use_conditions),
    }

    payload = {
        "class": "VAE",
        "ctor": ctor,
        "state_dict": model.state_dict(),
        "meta": {
            "torch": torch.__version__,
        },
    }

    payload["pyro_param_store"] = results['param_store']

    torch.save(payload, path)


def load_results(path: str):
    """
    Load a VAE from a checkpoint saved with `save_results`, returning the constructed model.
    """
    ckpt = torch.load(path, weights_only=False)

    if "ctor" not in ckpt or "state_dict" not in ckpt:
        raise ValueError("Checkpoint is missing required fields ('ctor', 'state_dict').")

    try:
        from ._models import VAE  
    except Exception:
        assert False, "Could not import VAE model class from ._models."

    model = VAE(**ckpt["ctor"], tau=0.5)  # tau is dummy if not used
    model.load_state_dict(ckpt["state_dict"])

    pyro.get_param_store().set_state(ckpt["pyro_param_store"])

    return model
