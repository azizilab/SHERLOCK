from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Optional

try:
    __version__ = version("sherlock-perturb")
except PackageNotFoundError:  # running from a source checkout without `pip install`
    __version__ = "0.1.0"

from . import datasets
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from . import _configs as configs

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pp", "pl", "configs"]})
