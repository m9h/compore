# Changelog

## 0.1.0 — 2026-11-07

First release.

- `dipy_refine_peaks` / `lafix` CLI (a DIPY `Workflow`): refine any
  `PeaksAndMetrics` with the differentiable K-fixel fit and write refined
  peaks, f_i, f_wm, σ_θ, an MRtrix fixel directory and a model JSON.
- `lafix.dipy_refine.refine_peaks`, `default_config`, `fi_config`.
- `lafix.model` (`FixelConfig`, `fit_fixels`, `fit_to_pam`) and
  `lafix.uncertainty` (`laplace_fixel_posterior`, `FixelPosterior`,
  `posterior_connectivity`, `cone_coverage`, Laplace evidence). The former
  module names `lafix.prism_jax` / `lafix.prism_uncertainty` and the names
  `PrismConfig`, `PrismFit`, `fit_prism`, `prism_fit_to_pam` remain as
  deprecated aliases.
- No dependency on `compore` any more: the sphere kernel is vendored in
  `lafix.kernels`.
- The σ calibration factor is opt-in (`--sigma_inflation`, default 1.0 = raw
  Laplace σ); 1.81 is the HCP-YA scan–rescan constant.
- Examples (`python -m lafix.examples.<name>`): `stanford_hardi_quickstart`
  (brain mask ∧ DTI FA > 0.2 by default, `--fa-thr 0` for the whole brain),
  `viz_skyline_sigma` (Skyline, dipy ≥ 1.13).
