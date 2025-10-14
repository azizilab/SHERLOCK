from dataclasses import dataclass, asdict
import threading
from types import MappingProxyType
from typing import Mapping, Optional, Union, Any

@dataclass(frozen=True)
class _Defaults:
    pert_key: str = "pert"
    treatment_key: str = "treatment"
    ntc_label: str = "NTC"

_DEFAULTS = asdict(_Defaults())          # single place where values live
_CONFIG   = dict(_DEFAULTS)              # mutable runtime copy
_LOCK     = threading.RLock()

# sentinel so users can pass default=None intentionally
class _Missing:
    pass
_MISSING = _Missing()

def get_config(key: str, default: Any = _MISSING) -> str:
    """
    Return the config value for `key`.
    - If `key` is missing and `default` is provided, return `default`.
    - If `key` is missing and no default is provided, raise KeyError.
    """
    with _LOCK:
        if key in _CONFIG:
            return _CONFIG[key]
    if default is _MISSING:
        raise KeyError(f"Unknown config key: {key!r}. Allowed: {sorted(_CONFIG.keys())}")
    return default


from typing import Mapping, Optional, Union

def set_config(
    key_or_updates: Union[str, Mapping[str, str]],
    value: Optional[str] = None,
    /,
    **kwargs: str,
) -> None:
    """
    Set configuration values.

    Usage:
      set_config("pert_key", "perturbation")
      set_config({"pert_key": "perturbation", "ntc_label": "CONTROL"})
      set_config(pert_key="perturbation", ntc_label="CONTROL")
      set_config({"pert_key": "p"}, ntc_label="CONTROL")  # mix mapping + kwargs
    """
    # Normalize inputs to a single payload dict
    if isinstance(key_or_updates, str):
        if value is None:
            raise TypeError("set_config(key, value) requires both key and value.")
        payload = {key_or_updates: value, **kwargs}
    else:
        # key_or_updates is a Mapping
        payload = dict(key_or_updates)
        payload.update(kwargs)

    # Validate keys against known schema
    unknown = payload.keys() - _DEFAULTS.keys()
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")

    # Apply updates atomically
    with _LOCK:
        _CONFIG.update(payload)


def reset_config() -> None:
    with _LOCK:
        _CONFIG.clear()
        _CONFIG.update(_DEFAULTS)

def show() -> None:
    """Print all config key/value pairs."""
    with _LOCK:
        items = sorted(_CONFIG.items(), key=lambda kv: kv[0])
        if not items:
            print("<empty config>")
            return
        width = max(len(k) for k, _ in items)
    for k, v in items:
        print(f"{k.ljust(width)} = {v}")

