# SHERLOCK: structured representation learning and causal inference of downstream perturbation effects

[![Documentation Status](https://readthedocs.org/projects/sherlock-perturb/badge/?version=latest)](https://sherlock-perturb.readthedocs.io/en/latest/)

SHERLOCK (Single-cell Hierarchical Embedding of peRturbations and Learning Of Causal Knowledge) is
an interpretable deep generative framework for single-cell perturbation data. It models each
genetic or chemical perturbation as a structured intervention on a latent baseline cell state,
with perturbation effects that are correlated across perturbations and sparse across latent
factors. From the fitted model you can

- **group perturbations** by the transcriptional responses they share,
- **estimate downstream effects counterfactually**, by applying a perturbation to control cells'
  inferred baseline states and decoding the result,
- **measure how responses depend on a condition** such as drug exposure or T-cell co-culture, and
- **compose single perturbations** to predict combinations, including unseen ones, and classify
  genetic interactions.

📖 **Documentation:** https://sherlock-perturb.readthedocs.io

## Installation

SHERLOCK requires Python ≥ 3.10 and is published on PyPI as `sherlock-perturb`; the package is
imported as `sherlock`.

```bash
pip install sherlock-perturb
```

Optional extras:

| Extra | Adds |
|---|---|
| `tutorials` | Leiden clustering, UMAP, gene-set enrichment (`gseapy`) and Jupyter, used by the tutorials |
| `benchmarks` | the baseline models: cVAE, sVAE+, contrastiveVI and scGen (`scvi-tools`), and GEARS (`cell-gears`) |
| `all` | `tutorials`, `benchmarks` and `test` |

```bash
pip install "sherlock-perturb[tutorials]"
```

To install from source, or with conda:

```bash
git clone https://github.com/azizilab/SHERLOCK.git
cd SHERLOCK
pip install -e ".[tutorials]"          # or: conda env create -f environment.yml
```

Install PyTorch first if you need a specific CUDA build ([pytorch.org](https://pytorch.org/get-started/locally/)).
See the [installation guide](https://sherlock-perturb.readthedocs.io/en/latest/installation.html)
for details.

## Quickstart

```python
import scanpy as sc
import torch
import sherlock as slk

# Raw counts in .X; one perturbation label per cell, controls labelled e.g. "non-targeting"
adata = sc.read_h5ad("my_screen.h5ad")
slk.pp.slk_prepare_data(adata, pert_key="guide_target", ntc_label="non-targeting", inplace=True)

# Observed average treatment effects, used for validation during training
slk.pp.treat_effect(adata, label_col="pert", control_label="NTC", method="perturbseq", inplace=True)

# Train and evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = slk.tl.run(adata, model="vae", latent_dim=16, device=device)
model = out["model"]
slk.tl.eval(model, adata, device=device)       # writes adata.obsm["z"] and adata.uns["results"]

# Perturbation groups and counterfactual downstream effects
slk.pl.plot_corr(adata)
cf = model.predict_counterfactual_effects(adata, observed_effect=adata.uns["treat_effect"])
slk.tl.ev_sig(adata)
slk.pl.plot_ev_sig(adata)

slk.tl.save_results(out, "sherlock_model.pth")
```

For screens with conditions, also pass `treatment_key` and `untreated_label` to
`slk_prepare_data` and train with `use_conditions=True`. For combinations (`A+B` labels), train
with `combinatorial=True, synergy=True`. See the
[API reference](https://sherlock-perturb.readthedocs.io/en/latest/api/index.html).

## Tutorials

| Tutorial | Data | Analyses |
|---|---|---|
| [Replogle](https://sherlock-perturb.readthedocs.io/en/latest/tutorials/replogle.html) | genome-scale CRISPRi Perturb-seq, K562 (Replogle et al. 2022) | perturbation groups vs annotated pathways, counterfactual effects, downstream targets, enrichment |
| [Glioblastoma drugs ± T cells](https://sherlock-perturb.readthedocs.io/en/latest/tutorials/gbm_drug_conditions.html) | 11 kinase inhibitors in BT333 cells with and without cytotoxic T cells, sci-Plex (Shi et al. 2026) | condition-dependent drug responses, shared downstream genes per condition |
| [Norman](https://sherlock-perturb.readthedocs.io/en/latest/tutorials/norman_combinatorial.html) | combinatorial CRISPRa Perturb-seq, K562 (Norman et al. 2019) | held-out combinations, genetic-interaction classification |

The tutorial notebooks are in [`docs/tutorials/`](docs/tutorials).

## Repository structure

```
├── sherlock/            # the package
│   ├── tools/           #   models, training, evaluation, clustering, interaction scoring (slk.tl)
│   ├── preprocessing/   #   label preparation and observed treatment effects (slk.pp)
│   ├── plotting/        #   plots (slk.pl)
│   ├── datasets/        #   example data loaders (slk.datasets)
│   └── notebooks/       #   analysis notebooks for every dataset in the manuscript
│       ├── replogle_analysis.ipynb, benchmarking.ipynb
│       ├── raefish/     #   spatial CRISPR screen
│       ├── gbm/         #   glioblastoma: EGFR inhibitors, kinome CRISPRi/a, 11 drugs ± T cells
│       └── norman/      #   combinatorial CRISPRa screen
├── docs/                # documentation source: guides, API reference, tutorials
├── tests/               # unit tests
├── pyproject.toml       # packaging (pip install)
└── environment.yml      # conda environment
```

The analysis notebooks in `sherlock/notebooks/` run every analysis in the manuscript; they read
inputs from `sherlock/datasets/` (not tracked by git). A version organized by figure, with
instructions for obtaining the data and saved models, is in
[azizilab/SHERLOCK_Reproducibility](https://github.com/azizilab/SHERLOCK_Reproducibility).

## Citation

If you use SHERLOCK, please cite our preprint:
https://www.biorxiv.org/content/10.64898/2026.09.25.754573v1

_SHERLOCK: Structured representation learning and causal inference of downstream perturbation effects_

```bibtex
@article{zhang2026sherlock,
  title   = {SHERLOCK: Structured representation learning and causal inference of downstream perturbation effects},
  author  = {Zhang, Mingxuan and Myers, Joshua D. and Shi, Lingting and Giglio, Ross M. and
             Chatterjee, Sharanya and McFaline-Figueroa, Jos{\'e} L. and Azizi, Elham},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.25.754573}
}
```

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International License](LICENSE).
