from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import scanpy as sc

from ..preprocessing import slk_prepare_data

HERE = Path(__file__).parent

# Replogle et al. (2022) working subset (118,641 cells x 1,185 genes, 682 perturbation
# labels) with the pathway annotations used for evaluation in data.uns['pathways'].
REPLOGLE_URL = (
    "https://media.githubusercontent.com/media/azizilab/SHERLOCK/main/"
    "sherlock/datasets/replogle.h5ad"
)


def _cache_dir() -> Path:
    """Directory where downloaded datasets are stored (``$SHERLOCK_DATA_DIR`` or ``~/.cache/sherlock``)."""
    return Path(os.environ.get("SHERLOCK_DATA_DIR", Path.home() / ".cache" / "sherlock"))


def _is_h5ad(path: Path) -> bool:
    # A Git LFS pointer (left behind when a clone is made without git-lfs) is a small
    # text file, not an HDF5 file.
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


def _download(url: str, dest: Path) -> Path:
    from tqdm.auto import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}") as bar:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)
    return dest


def replogle(path: str | os.PathLike | None = None, download: bool = True):
    """Load the Replogle et al. (2022) genome-scale CRISPRi Perturb-seq subset.

    Parameters
    ----------
    path
        Path to ``replogle.h5ad``. If ``None``, the copy shipped with a source checkout
        is used when present; otherwise the file is read from (or downloaded to) the
        cache directory, ``$SHERLOCK_DATA_DIR`` or ``~/.cache/sherlock``.
    download
        Download the file from ``REPLOGLE_URL`` if it is not found locally.

    Returns
    -------
    AnnData with raw counts in ``.X``, perturbation labels mapped to the configured
    ``pert_key``/``ntc_label``, and the eight annotated pathways in ``.uns['pathways']``.
    """
    if path is not None:
        file = Path(path)
    elif _is_h5ad(HERE / "replogle.h5ad"):
        file = HERE / "replogle.h5ad"
    else:
        file = _cache_dir() / "replogle.h5ad"
        if not _is_h5ad(file):
            if not download:
                raise FileNotFoundError(
                    f"{file} not found. Pass `path=` or set `download=True`."
                )
            _download(REPLOGLE_URL, file)

    data = sc.read_h5ad(file)
    slk_prepare_data(data, pert_key="gene", ntc_label="non-targeting", treatment_key=None, inplace=True)
    return data
