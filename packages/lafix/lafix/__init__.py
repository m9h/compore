"""PRISM-JAX: multi-fixel refinement of CSD peaks with a Laplace fixel posterior.

* :mod:`~lafix.prism_jax` -- the differentiable K-fixel model, Rician NLL,
  priors (including ``lam_iso``) and the Gauss-Newton/L-BFGS refine (``fit_prism``).
* :mod:`~lafix.prism_uncertainty` -- Laplace approximation per fixel and
  posterior sampling of orientations for probabilistic tracking.
* :mod:`~lafix.dipy_refine` -- the DIPY plug-in (``refine_peaks``,
  ``default_config``, ``fi_config``) and the ``dipy_refine_peaks`` CLI backend.
* :mod:`~lafix.tracking` -- peaks -> PAM and EuDX connectivity helpers.

Validated on DiSCo and the 42-subject HCP retest cohort; see
``docs/decisions/008-positioning-vs-force-prism-sbi.md``.
"""
