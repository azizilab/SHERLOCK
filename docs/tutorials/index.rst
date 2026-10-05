Tutorials
=========

The three tutorials cover the experimental designs SHERLOCK supports. Each one
runs end to end on a published screen and loads a pretrained model when one is
available, so the analysis can be followed without retraining.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Design
     - Tutorial
     - Data
   * - One perturbation per cell
     - :doc:`single_perturbations`
     - Genome-scale CRISPRi Perturb-seq in K562 cells (Replogle et al.)
   * - One perturbation per cell, under several conditions
     - :doc:`conditional_perturbations`
     - 11 kinase inhibitors in glioblastoma cells, with and without cytotoxic
       T cells (Shi et al.)
   * - One or two perturbations per cell
     - :doc:`combinatorial_perturbations`
     - Combinatorial CRISPRa Perturb-seq in K562 cells (Norman et al.)

The Replogle and Norman datasets are downloaded by ``slk.datasets.replogle()``
and ``slk.datasets.norman()``; the glioblastoma dataset is available from the
first authors on request.

.. toctree::
   :maxdepth: 1

   single_perturbations
   conditional_perturbations
   combinatorial_perturbations
