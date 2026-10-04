Tutorials
=========

Each tutorial runs SHERLOCK end to end on a published perturbation screen and
loads a pretrained checkpoint when one is available, so the analysis can be
followed without retraining.

- :doc:`replogle`: genome-scale CRISPRi Perturb-seq. Perturbation groups, their
  agreement with annotated pathways, counterfactual effects and downstream
  targets.
- :doc:`gbm_drug_conditions`: 11 kinase inhibitors in glioblastoma cells with
  and without cytotoxic T cells. How each drug's effect depends on the T-cell
  condition, and which downstream genes two drugs share in each condition.
- :doc:`norman_combinatorial`: combinatorial CRISPRa Perturb-seq. Prediction of
  combinations held out from training and classification of genetic
  interactions.

.. toctree::
   :maxdepth: 1

   replogle
   gbm_drug_conditions
   norman_combinatorial
