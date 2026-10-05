from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import scanpy as sc

from ..preprocessing import slk_prepare_data

HERE = Path(__file__).parent

_DRIVE_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"

# Processed copies of the two public datasets used in the tutorials, hosted on Google Drive.
# Each entry: (Drive file id, path relative to the data directory, size in bytes).
_FILES = {
    # Replogle et al. (2022) working subset: 118,641 cells x 1,185 genes, 682 perturbation
    # labels, and the pathway annotations used for evaluation in .uns['pathways'].
    "replogle": ("1E2bqqkzS2GaocHHYg3KImgEDdu7TbvyM", "replogle.h5ad", 573_798_189),
    # Norman et al. (2019) combinatorial CRISPRa Perturb-seq: raw counts in
    # .layers['counts'], perturbation labels in .obs['perturbation_name'].
    "norman": ("16A6GSoFtAt8LHbq0OnWGAcryjS7qS9WA", "norman/Norman_2019.h5ad", 1_703_064_678),
}


def data_dir() -> Path:
    """Directory where downloaded datasets are stored.

    ``$SHERLOCK_DATA_DIR`` if set, otherwise ``~/.cache/sherlock``.
    """
    return Path(os.environ.get("SHERLOCK_DATA_DIR", Path.home() / ".cache" / "sherlock"))


def _is_h5ad(path: Path) -> bool:
    # Guards against a Git LFS pointer or an HTML error page saved in place of the data.
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


def _download(url: str, dest: Path, size: int | None = None) -> Path:
    from tqdm.auto import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0)) or size
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}") as bar:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
                bar.update(len(chunk))
    if not _is_h5ad(tmp) or (size is not None and tmp.stat().st_size != size):
        tmp.unlink()
        raise OSError(
            f"The download of {dest.name} did not return the expected file. "
            f"Download it manually from {url} and pass its location with `path=`."
        )
    tmp.replace(dest)
    return dest


def _fetch(name: str, path: str | os.PathLike | None, download: bool) -> Path:
    """Locate dataset ``name``: an explicit path, a copy in a source checkout, or the cache."""
    file_id, rel, size = _FILES[name]
    if path is not None:
        return Path(path)
    local = HERE / rel                      # copy kept next to the code in a source checkout
    if _is_h5ad(local):
        return local
    cached = data_dir() / rel
    if _is_h5ad(cached):
        return cached
    if not download:
        raise FileNotFoundError(f"{cached} not found. Pass `path=` or set `download=True`.")
    return _download(_DRIVE_URL.format(file_id), cached, size)


def replogle(path: str | os.PathLike | None = None, download: bool = True):
    """Load the Replogle et al. (2022) genome-scale CRISPRi Perturb-seq subset.

    Parameters
    ----------
    path
        Path to ``replogle.h5ad``. If ``None``, the file is read from the data directory
        (:func:`data_dir`), and downloaded there on first use (about 570 MB).
    download
        Download the file if it is not found locally.

    Returns
    -------
    AnnData with raw counts in ``.X``, perturbation labels mapped to the configured
    ``pert_key``/``ntc_label``, and the eight annotated pathways in ``.uns['pathways']``.
    """
    data = sc.read_h5ad(_fetch("replogle", path, download))
    slk_prepare_data(data, pert_key="gene", ntc_label="non-targeting", treatment_key=None, inplace=True)
    return data


def norman(path: str | os.PathLike | None = None, download: bool = True):
    """Load the Norman et al. (2019) combinatorial CRISPRa Perturb-seq dataset.

    Parameters
    ----------
    path
        Path to ``Norman_2019.h5ad``. If ``None``, the file is read from the data directory
        (:func:`data_dir`, subfolder ``norman/``), and downloaded there on first use (about 1.7 GB).
    download
        Download the file if it is not found locally.

    Returns
    -------
    AnnData with raw counts in ``.X`` (copied from ``.layers['counts']``) and perturbation
    labels mapped to the configured ``pert_key``/``ntc_label``. Pairs are labelled ``A+B``.
    """
    data = sc.read_h5ad(_fetch("norman", path, download))
    if "counts" in data.layers:
        data.X = data.layers["counts"].copy()
    slk_prepare_data(data, pert_key="perturbation_name", ntc_label="control", treatment_key=None, inplace=True)
    return data
