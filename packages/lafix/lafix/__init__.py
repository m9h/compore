"""lafix — Laplace fixel posteriors for diffusion MRI.

Refine any DIPY ``PeaksAndMetrics`` with a differentiable K-fixel
stick + zeppelin (+ CSF/GM) fit under a Rician likelihood and return the
refined peaks together with a per-fixel Laplace posterior (orientation σ,
intra-axonal-fraction σ) that can be sampled for probabilistic connectomes.

* :mod:`~lafix.model` -- the differentiable K-fixel model, Rician NLL, priors
  and the Gauss-Newton / L-BFGS refine (``FixelConfig``, ``fit_fixels``).
* :mod:`~lafix.uncertainty` -- Laplace approximation per fixel
  (``laplace_fixel_posterior`` → ``FixelPosterior``), posterior direction
  sampling, cone coverage, Laplace evidence.
* :mod:`~lafix.dipy_refine` -- the DIPY plug-in (``refine_peaks``,
  ``default_config``, ``fi_config``) behind the ``dipy_refine_peaks`` CLI.
* :mod:`~lafix.tracking` / :mod:`~lafix.fixel` -- peaks → PAM / EuDX helpers
  and the MRtrix fixel-directory writer.

The forward model follows PRISM (Abouagour, Shah, Garyfallidis, arXiv
2604.00250) and extends it with learned diffusivities, tortuosity and the
Laplace posterior; the per-fixel uncertainty was validated on DiSCo and the
44-subject HCP-YA retest cohort (compore repository, docs/decisions/008).
Submodules are imported lazily so ``import lafix`` stays cheap.
"""
__version__ = "0.1.0"
