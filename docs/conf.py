"""Sphinx configuration for the SHERLOCK documentation site."""

import os
import re
import sys

# -- Path setup --------------------------------------------------------------
# conf.py lives in docs/, and the repository root one level up *is* the
# `sherlock` package. The docs build does not install the package, so expose the
# root under the name `sherlock` through a symlink in the build directory.
import tempfile

_root = os.path.abspath("..")
_shim = os.path.join(tempfile.gettempdir(), "sherlock-docs-path")
os.makedirs(_shim, exist_ok=True)
_link = os.path.join(_shim, "sherlock")
if os.path.realpath(_link) != _root:
    if os.path.islink(_link):
        os.unlink(_link)
    os.symlink(_root, _link)
if _shim not in sys.path:
    sys.path.insert(0, _shim)

# -- Project information -----------------------------------------------------
project = "SHERLOCK"
copyright = "2026, Azizi Lab"
author = "Azizi Lab"
# Single source of truth for the version is pyproject.toml. The docs build does
# not install the package, so read the file directly.
_m = re.search(
    r'(?m)^version = "([^"]+)"',
    open(os.path.join(_root, "pyproject.toml"), encoding="utf-8").read(),
)
release = _m.group(1) if _m else "0.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Mock the deep-learning and single-cell stack so ReadTheDocs can build the API
# reference without installing torch or CUDA. Lightweight dependencies
# (numpy / pandas / scipy / matplotlib / networkx) stay real and are installed
# from requirements-docs.txt.
autodoc_mock_imports = [
    "torch",
    "pyro",
    "scvi",
    "gears",
    "torch_geometric",
    "scanpy",
    "anndata",
    "sklearn",
    "statsmodels",
    "seaborn",
    "dcor",
    "umap",
    "tqdm",
]

# -- Notebook handling -------------------------------------------------------
# The tutorials need large datasets and a GPU, so they are never executed at
# build time; nbsphinx renders the outputs saved in the committed notebooks.
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output (furo theme) ------------------------------------------------
html_theme = "furo"
html_title = "SHERLOCK"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": True,
    "light_logo": "sherlock_logo.png",
    "dark_logo": "sherlock_logo.png",
}
