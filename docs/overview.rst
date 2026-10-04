Overview
========

SHERLOCK takes single-cell expression counts together with a perturbation label
for every cell and, optionally, a condition label (for example drug exposure,
co-culture or dose). It is a variational autoencoder whose latent space separates
the cell's baseline state from the effect of the perturbation it received.

Model
-----

1. **Baseline state.** Control cells (non-targeting guides, vehicle, untreated)
   define a baseline latent state :math:`z^0` that captures intrinsic
   transcriptional variation. When conditions are given, the baseline is learned
   from the control cells of the matching condition.

2. **Perturbation shift.** Each perturbation :math:`p` shifts the baseline by a
   sparse, gated vector :math:`\Delta_p = W_p \odot A_p`. The embedding
   :math:`A = C_P\,\rho` is correlated across perturbations through a low-rank
   covariance :math:`\Sigma_P`, so perturbations with similar effects are
   similar in the model; the gate :math:`W` restricts each perturbation to a few
   latent factors.

3. **Decoder.** A negative-binomial decoder maps the perturbed latent state
   :math:`z = z^0 + \Delta_p` back to counts.

4. **Combinations.** A combination A+B reuses its two parents: their shifts are
   rescaled by learned coefficients and summed, plus a learned interaction term.
   Training starts from the additive combination and penalizes departures from it.

What you can read from a fitted model
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Analysis
     - Functions
   * - Perturbation groups
     - ``adata.uns['results']['rho_corr']`` from :func:`sherlock.tl.eval`;
       :func:`sherlock.tl.cluster_rho`, :func:`sherlock.pl.plot_corr`
   * - Counterfactual effects
     - ``model.predict_counterfactual_effects`` (abduct a control background,
       apply the perturbation, decode); compared with observed average treatment
       effects from :func:`sherlock.pp.treat_effect`
   * - Downstream targets
     - ``counterfactual_effect_size`` from :func:`sherlock.tl.eval`;
       :func:`sherlock.tl.ev_sig`, :func:`sherlock.pl.plot_ev_sig`,
       :func:`sherlock.pl.plot_cf_bipartite`
   * - Condition dependence
     - ``condition_perturbation_interaction`` from :func:`sherlock.tl.eval`
   * - Genetic interactions
     - ``model.get_gi``, :func:`sherlock.tl.regress_gi_params`,
       :func:`sherlock.tl.classify_gi`

Import namespace
----------------

.. code-block:: python

   import sherlock as slk

   slk.pp   # preprocessing: label preparation, observed treatment effects
   slk.tl   # training, evaluation, clustering, interaction scoring
   slk.pl   # plotting
   slk.configs   # names of the perturbation / condition columns
   slk.datasets  # example data

SHERLOCK reads labels from standardized ``.obs`` columns (``pert``,
``treatment``) and control labels (``NTC``, ``untreated``).
:func:`sherlock.pp.slk_prepare_data` maps your own columns onto them.
