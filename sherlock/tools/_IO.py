import torch
from pathlib import Path

_CLASS_REGISTRY = {}


def _get_class(name: str):
    if name not in _CLASS_REGISTRY:
        if name == "VAE":
            from ._models import VAE
            _CLASS_REGISTRY["VAE"] = VAE
        elif name == "sVAE":
            from ._svae import sVAE
            _CLASS_REGISTRY["sVAE"] = sVAE
        elif name == "cVAE":
            from ._cvae import cVAE
            _CLASS_REGISTRY["cVAE"] = cVAE
        else:
            raise ValueError(f"Unknown model class: {name!r}")
    return _CLASS_REGISTRY[name]


def save_results(results, path: str, model=None, param_store=None) -> None:
    """
    Save a trained model checkpoint to *path*.

    Works for any PerturbModelBase subclass that implements checkpoint_ctor_args().
    For Pyro-based models (VAE), pass param_store in the results dict or as a kwarg.
    For plain-PyTorch models (sVAE), param_store is not required.
    """
    if results is None:
        if model is None:
            raise ValueError("If 'results' is not provided, 'model' must be given.")
        results = {"model": model}
        if param_store is not None:
            results["param_store"] = param_store

    model = results["model"]

    payload = {
        "class":      type(model).__name__,
        "ctor":       model.checkpoint_ctor_args(),
        "state_dict": model.state_dict(),
        "meta":       {"torch": torch.__version__},
    }

    if results.get("param_store") is not None:
        payload["pyro_param_store"] = results["param_store"]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_results(path: str):
    """
    Load a model checkpoint saved with save_results, returning the constructed model.
    """
    ckpt = torch.load(path, weights_only=False)

    if "ctor" not in ckpt or "state_dict" not in ckpt:
        raise ValueError("Checkpoint is missing required fields ('ctor', 'state_dict').")

    class_name = ckpt.get("class", "VAE")

    # ── ContrastiveVI (current) and ContrastiveVIWrapper (legacy) ─────────────
    if class_name in ("ContrastiveVI", "ContrastiveVIWrapper"):
        from ._contrastivevi import ContrastiveVI

        ctor = dict(ckpt["ctor"])
        if "n_perturbs" not in ctor:
            import warnings
            warnings.warn(
                "Loading a legacy ContrastiveVIWrapper checkpoint; "
                "pert_mu / pert_log_var will not be available.",
                UserWarning,
                stacklevel=2,
            )
            sd = ckpt["state_dict"]
            ctor["n_perturbs"] = sd["pert_mu"].shape[0] if "pert_mu" in sd else 1

        model = ContrastiveVI(**ctor)
        model.load_state_dict(ckpt["state_dict"])

        if "pyro_param_store" in ckpt:
            import pyro
            pyro.get_param_store().set_state(ckpt["pyro_param_store"])

        return model

    # ── VAE / cVAE / sVAE ─────────────────────────────────────────────────────
    cls = _get_class(class_name)
    extra_kwargs = {}
    if class_name == "VAE":
        extra_kwargs["tau"] = 0.5  # dummy; not stored but required by ctor

    model = cls(**ckpt["ctor"], **extra_kwargs)
    model.load_state_dict(ckpt["state_dict"])

    if "pyro_param_store" in ckpt:
        import pyro
        pyro.get_param_store().set_state(ckpt["pyro_param_store"])

    return model
