SHERLOCK
========

**SHERLOCK** (Single-cell Hierarchical Embedding of peRturbations and Learning Of
Causal Knowledge) is a deep generative framework for single-cell perturbation
data. It represents each perturbation as a structured intervention on a latent
baseline cell state, which lets it

- group genetic and chemical perturbations by the transcriptional responses they share,
- estimate the downstream effects of each perturbation counterfactually,
- measure how a perturbation's effect changes across conditions, and
- compose single perturbations to predict combinations and classify genetic interactions.

Get started
-----------

- :doc:`overview`: the model and what each analysis reads from it.
- :doc:`installation`: install SHERLOCK with pip or conda.
- :doc:`tutorials/index`: worked examples on three published screens.
- :doc:`api/index`: the ``sherlock`` public API.

Tutorials at a glance
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Tutorial
     - Data
     - Analyses
   * - :doc:`tutorials/replogle`
     - Genome-scale CRISPRi Perturb-seq (K562)
     - perturbation groups, counterfactual effects, downstream targets
   * - :doc:`tutorials/gbm_drug_conditions`
     - 11 kinase inhibitors in glioblastoma cells, with and without T cells (sci-Plex)
     - condition-dependent drug responses
   * - :doc:`tutorials/norman_combinatorial`
     - Combinatorial CRISPRa Perturb-seq (K562)
     - held-out combinations, genetic-interaction classes

Reproducing the manuscript
--------------------------

Code that regenerates every figure of the SHERLOCK manuscript is in
`azizilab/SHERLOCK_Reproducibility <https://github.com/azizilab/SHERLOCK_Reproducibility>`_.

Citation
--------

If you use SHERLOCK, please cite our preprint
(`bioRxiv <https://www.biorxiv.org/content/10.64898/2026.09.25.754573v1>`_):

   Zhang M, Myers JD, Shi L, Giglio RM, Chatterjee S, McFaline-Figueroa JL, Azizi E.
   *SHERLOCK: Structured representation learning and causal inference of downstream
   perturbation effects.* bioRxiv (2026). doi:10.64898/2026.09.25.754573

.. code-block:: bibtex

   @article{zhang2026sherlock,
     title   = {SHERLOCK: Structured representation learning and causal inference of downstream perturbation effects},
     author  = {Zhang, Mingxuan and Myers, Joshua D. and Shi, Lingting and Giglio, Ross M. and
                Chatterjee, Sharanya and McFaline-Figueroa, Jos{\'e} L. and Azizi, Elham},
     journal = {bioRxiv},
     year    = {2026},
     doi     = {10.64898/2026.09.25.754573}
   }

.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   installation
   tutorials/index
   api/index
