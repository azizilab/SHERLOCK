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
        elif name == "SCGENModel":
            from ._scgen import SCGENModel
            _CLASS_REGISTRY["SCGENModel"] = SCGENModel
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
        param_store = results["param_store"]
        # pyro.param() access after get_state() re-adds weakrefs to unconstrained
        # params; strip them before pickling (mirrors what get_state() does).
        for p in param_store.get("params", {}).values():
            p.__dict__.pop("unconstrained", None)
        payload["pyro_param_store"] = param_store

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

        # A freshly constructed nn.Module defaults to train() mode; models
        # with BatchNorm (see load below) give wrong inference results in
        # that mode, so force eval() before handing the model back.
        model.eval()
        return model

    # ── VAE / cVAE / sVAE ─────────────────────────────────────────────────────
    cls = _get_class(class_name)
    ctor = dict(ckpt["ctor"])
    extra_kwargs = {}
    if class_name == "VAE":
        if "use_synergy" in ctor and "synergy" not in ctor:
            ctor["synergy"] = ctor.pop("use_synergy")
        else:
            ctor.pop("use_synergy", None)
        if "tau" not in ctor:
            extra_kwargs["tau"] = 0.5  # dummy; not stored but required by ctor

    model = cls(**ctor, **extra_kwargs)
    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError as err:
        if class_name != "VAE":
            raise
        load_result = model.load_state_dict(ckpt["state_dict"], strict=False)
        missing = set(load_result.missing_keys)
        unexpected = set(load_result.unexpected_keys)
        allowed_missing = {"parent_coeff.weight", "parent_coeff.bias"}
        if not missing.issubset(allowed_missing) or unexpected:
            raise err

    if "pyro_param_store" in ckpt:
        import pyro
        pyro.get_param_store().set_state(ckpt["pyro_param_store"])

    # A freshly constructed nn.Module defaults to train() mode. Models using
    # BatchNorm (cVAE/sVAE/scgen) give badly wrong inference results in that
    # mode — BatchNorm normalizes using the current batch's statistics
    # instead of the learned running statistics, which is especially bad for
    # small batches like the K-means centroids used in
    # predict_counterfactual_effects. Force eval() so a loaded checkpoint is
    # always ready for immediate evaluation/inference.
    model.eval()
    return model
