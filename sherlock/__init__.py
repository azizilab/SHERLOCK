from __future__ import annotations

import sys
from importlib.metadata import version
from typing import TYPE_CHECKING, Optional

from . import datasets
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from . import _configs as configs

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pp", "pl", "configs"]})
