# DIPY issue draft: calibrated per-fixel uncertainty for refined peaks (lafix)

**Status:** drafted 2026-11-06 for Morgan to post at https://github.com/dipy/dipy/issues. Not sent. Follow-up PR: `lafix.dipy_refine.refine_peaks` as a `dipy.reconst` or `dipy.workflows` contribution; code at https://github.com/m9h/compore/tree/master/packages/lafix (BSD). Numbers: doc 008 §6.4, §10.6, doc 012 §7.6.

---

**Title:** Proposal: a peaks-refinement step with a calibrated per-fixel Laplace posterior (`refine_peaks`), downstream of any `PeaksAndMetrics`

### Summary

I have a small, DIPY-only-dependency package (`lafix`, "Laplace fixel") that takes any DIPY peaks (CSD, MSMT-CSD, FORCE, or a peaks array) and refines them with a differentiable multi-fixel stick-and-zeppelin fit under a Rician likelihood, returning the refined directions, per-fixel volume fractions, and a per-fixel Laplace posterior (angular σ and fraction σ) with an in-vivo calibration factor measured on the 44 HCP-YA test–retest subjects. I would like to contribute it as a DIPY workflow / reconst step and am asking here first about the preferred shape.

### Why it may be useful to DIPY users

- **No DIPY method reports per-fixel orientation uncertainty**. The Laplace σ here ranks scan–rescan angular error at Spearman ρ = 0.457 ± 0.023 over 44 subjects, and after a constant inflation factor of 1.81 ± 0.15 (measured on the same cohort) its 90 % cones cover the retest direction at the nominal rate. The posterior-mean connectome built from it is more reproducible than the point-estimate connectome (scan–rescan r(log) 0.965 vs 0.923; Dice 0.936 vs 0.855, HCP-YA).
- **Refinement improves whatever it is given.** On the DiSCo protocol, FORCE peaks in → 0.79 → 0.91 connectome correlation after refinement; on Monte Carlo substrate crossings (30–90°) the refined fit is 2.3–8.0° vs 9.7–16.6° for FORCE and 5–23° for MSMT-CSD on identical inputs (doc 008 §6.4 table).
- **Cheap.** Whole Stanford HARDI (154 625 voxels) on CPU: refine 87.8 s, Laplace 31.2 s (doc 012 §7.6); GPU is optional.
- **In-vivo f_i reproducibility is at NODDI level**: HCP-YA 44-subject scan–rescan mean |Δf_i| 0.035 ± 0.008 (ours) vs 0.039 ± 0.007 (NODDI NDI·(1−FWF)), within-subject CoV 6.4 % vs 6.5 %. I would state this as "as reproducible as NODDI", not better.

### What exists

```python
from lafix.dipy_refine import refine_peaks, default_config
pam_refined, post = refine_peaks(data, gtab, pam, mask=mask, config=default_config(gtab), laplace=True)
post.sigma_deg_calibrated          # per fixel, (X, Y, Z, K)
post.sample_dirs(key, n, calibrated=True)   # posterior direction samples for probabilistic connectomes
```
A CLI (`dipy_refine_peaks`, a `dipy.workflows.workflow.Workflow`) writes refined peaks, `*_sigmatheta_calibrated.nii.gz`, and an MRtrix fixel directory. The dependency chain is dipy + optax + a JAX library of ours (compore) for the differentiable forward model; I can reduce it to dipy + jax if that is preferred for a contribution.

### Known limits (so nobody is surprised)

- The calibration constant was measured on HCP-YA (b = 1000/2000/3000, SNR 12–22); on other protocols the raw σ should be re-calibrated or reported uncalibrated.
- With isotropic (CSF/GM) compartments on, f_i scan–rescan CoV is 8.9 %; with them off 6.4 % but the main fixel loses 3.6° — two configurations, both exposed (`default_config`, `fi_config`).
- It refines; it does not detect fixels. Fixel count comes from the input peaks.

### Questions for the maintainers

1. `dipy.reconst` (a `refine_peaks(pam, ...)` function returning a `PeaksAndMetrics` plus a posterior object) or `dipy.workflows` only?
2. Is a JAX optional dependency acceptable for a reconst module (as cvxpy is for MSMT), or should the first PR be numpy/scipy with a slower L-BFGS?
3. Where should calibration constants live and how should their provenance be recorded?

I can open the PR within two weeks of guidance and will include the HCP-YA and DiSCo validation scripts.
