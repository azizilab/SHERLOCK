import torch
import pyro
from pathlib import Path
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
        "rank":       model.rank,
        "shift":     model.shift,
    }

    payload = {
        "class": "VAE",
        "ctor": ctor,
        "state_dict": model.state_dict(),
        "meta": {"torch": torch.__version__},
        "pyro_param_store": results['param_store'],
        "training_config": results['training_config'],
    }

    payload["pyro_param_store"] = results['param_store']
    
    

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

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


def load_training_config(path: str) -> dict:
    """
    Load only the training configuration dictionary from a saved model checkpoint.
    
    Parameters
    ----------
    path : str
        Path to the saved model checkpoint (.pth file)
    
    Returns
    -------
    dict
        Training configuration dictionary containing all training parameters,
        model hyperparameters, and training history
        
    Examples
    --------
    >>> config = slk.tl.load_training_config('../data/model_20241107_103826.pth')
    >>> print(f"Learning rate: {config['lr']}")
    >>> print(f"L0 lambda: {config['l0_lambda']}")
    >>> print(f"Final loss: {config['train_history']['total_loss'][-1]}")
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    if "training_config" not in ckpt:
        # Return empty dict if no training config was saved (backwards compatibility)
        print(f"Warning: No training_config found in checkpoint {path}")
        return {}

    return ckpt["training_config"]

