# 015 — lafix 0.1.0: a standalone release of the fixel-posterior tool

## Date
2026-11-07

## Status
Executed (build + clean-venv verification below). Not yet published to PyPI
or tagged: both are the user's call (§5).

## 1. Decision

After doc 014 (DIPY's roadmap is pipelines, viewer and bundle statistics; no
session touches microstructure or per-fixel uncertainty) the user chose to
**release lafix as a tool the community can install now**, instead of first
negotiating a DIPY PR (`docs/outreach/dipy-lafix-issue.md` remains the draft
for that conversation; the PR, if it happens, follows the release). compore
stays the platform; nothing in it is released.

## 2. What changed for the release

| item | before | 0.1.0 |
|---|---|---|
| module names | `lafix.prism_jax`, `lafix.prism_uncertainty`; `PrismConfig`, `PrismFit`, `fit_prism`, `prism_fit_to_pam` | `lafix.model`, `lafix.uncertainty`; `FixelConfig`, `FixelFit`, `fit_fixels`, `fit_to_pam`. Old names stay as deprecated shims (`DeprecationWarning`), so every `validation/*.py` script and the 008/012/013 docs keep working and reading. Reason: the old names borrowed the name of the PRISM preprint whose authors are the DIPY maintainers; the model is a re-implementation plus extensions, and the package should not look like their release. |
| dependency on compore | `g3_sphere` (GPD sphere kernel for `restricted_kernel="sphere"`) | vendored verbatim into `lafix/kernels.py` (100 roots copied from `compore/constants.py` as source text, not through a float32 array; the first attempt lost digits and failed `test_sphere_kernel_limits`). lafix depends on dipy, jax, optax, numpy, scipy, nibabel, h5py only. |
| σ calibration | CLI default `sigma_inflation=1.81`, `*_sigmatheta_calibrated` always written | default 1.0 (raw Laplace σ); the calibrated map is written only when a factor is given; `*_model.json` records the factor and its provenance. Reason: 1.81 is one cohort (HCP-YA), the draft issue already promised this. |
| examples | `packages/lafix/examples/` (not shipped) | `lafix.examples` sub-package: `python -m lafix.examples.stanford_hardi_quickstart out/` and `viz_skyline_sigma` |
| metadata | MIT in pyproject, no LICENSE file, no README beyond 12 lines | `LICENSE` (MIT, 2026 Morgan Hough), full README (outputs, Python API, validation numbers from docs 008/012, calibration, limits, relation to PRISM), `CHANGELOG.md`, classifiers, URLs, extras `cuda` / `viz` / `test`, Python ≥ 3.11 |
| tests | `tests/validation/test_lafix_prism.py` | renamed `test_lafix_model.py`; + shim aliasing test, + `import lafix` lazy/version test; `test_doc014_tools.py` covers the examples |

## 3. Verification log

- `JAX_PLATFORMS=cpu uv run pytest tests/validation/test_lafix_model.py
  tests/validation/test_lafix_uncertainty.py tests/validation/test_lafix_exchange.py
  tests/validation/test_doc014_tools.py`: 62 passed, 5 skipped (GPU / Skyline
  gated), 93 s, before the two release tests were added; 2 more pass after.
- `uv build --package lafix`: `dist/lafix-0.1.0-py3-none-any.whl` (35 kB) and
  `lafix-0.1.0.tar.gz`; wheel contains the 10 modules + `examples/`, the
  LICENSE and the two entry points; no compore files.
- Clean venv (`uv venv --python 3.12` in the scratchpad, `uv pip install
  dist/lafix-0.1.0-py3-none-any.whl`, 604 MB with CPU JAX 0.12.0.1 and dipy
  1.12.1): `import compore` fails as intended; `dipy_refine_peaks --help`
  lists the Workflow arguments; `python -m lafix.examples.stanford_hardi_quickstart`
  runs end to end on 8 CPU threads. Numbers from `quickstart.json`:

  | step | value |
  |---|---|
  | mask (median-Otsu, brain, not WM) | 185 968 voxels, 48 s |
  | CSD peaks (sh 8, 5 peaks, parallel) | 208 s |
  | refine + Laplace (n_fibres 3, 300 iterations) | 291 s |
  | learned D∥ / extra-cellular D∥ | 1.32 / 1.11 µm²/ms |
  | σ_θ main fixel, median / p90 | 5.8° / 57.3° (p90 sits on the 1 rad prior bound: non-WM voxels of a brain mask) |
  | f_i median | 0.44 |
  | files written | peaks.pam5, fintra, fwm, sigmatheta, fixels/ (index, directions, fraction, sigma_theta_deg, f_intra), model.json; **no** `_calibrated` map at the default factor |

  The 291 s is slower than doc 012 §7.6 (88 + 31 s) because the quick start
  masks the whole brain, not white matter (185 968 vs 154 625 voxels, and
  the non-WM voxels converge slowly); the README now quotes both.

## 6. Quick start on a white-matter mask (2026-11-07, after the release commit)

The user chose option 4 of §5. `stanford_hardi_quickstart` now computes a
DTI FA map inside the median-Otsu brain mask and fits brain ∧ FA > 0.2
(`--fa-thr`, 0 restores the brain mask); `brain_mask.nii.gz` and `fa.nii.gz`
are written next to `mask.nii.gz`. Dev venv, `JAX_PLATFORMS=cpu`, 8 threads,
numbers from `quickstart.json`:

| step | brain mask (§3) | FA > 0.2 mask |
|---|---|---|
| mask voxels | 185 968 | 77 664 (brain 185 968) |
| mask step | 48 s | 133 s (median-Otsu + whole-brain DTI) |
| CSD peaks | 208 s | 17 s (data already cached; §3 included the download / first-run cost) |
| refine + Laplace (n_fibres 3, 300 it.) | 291 s | 128 s |
| learned D∥ / extra D∥ | 1.32 / 1.11 µm²/ms | 1.46 / 0.99 µm²/ms |
| σ_θ main fixel median / p90 | 5.8° / 57.3° | 3.7° / 7.9° |
| f_i median | 0.44 | 0.47 |

The 57° tail was the non-WM voxels, as §3 guessed. The refine time did not
halve with the voxel count (128 vs 291 s for 0.42× the voxels) because the
fixed costs (JIT compile, Laplace Hessians per fixel) stay; it is still the
runtime the README quotes for the default example. The DTI fit is the new
dominant cost of the example; it is plain DIPY and not lafix's.

## 4. Not done in this release

- Skyline example untested (dipy 1.13 not installable from PyPI; source
  install blocked in this session). It is shipped as an example, documented as
  such, and covered by the stand-in test.
- No single-shell test: the README's statement that one shell needs
  `--d_par_init` is from the model structure (diffusivities are global and
  weakly identified from one shell), not from a run.
- The step-5 debts of doc 013 §6.4 and the validation scripts' imports
  (`lafix.prism_jax`) are untouched; the shims warn but work.
- No `conda-forge` / `dipy` extension listing; no docs site. The README is
  the documentation.

## 5. Left to the user

1. **Tag and publish**: `git tag lafix-v0.1.0`, `uv publish dist/*` (PyPI
   credentials needed). The name `lafix` was free on PyPI on 2026-11-07
   (`https://pypi.org/pypi/lafix/json` → 404). Rebuild the wheel first if
   §6 is to be included (it only changes the example). The sdist and wheel
   in `dist/` are the verified artefacts.
2. ~~Push the commits~~ — pushed 2026-11-07 (a12add0..370bd11).
3. Send `docs/outreach/dipy-lafix-issue.md` (updated to the 0.1.0 API and
   `pip install lafix`) and `docs/outreach/cuslines-issues.md`.
4. ~~Decide whether the quick start should mask white matter~~ — done, §6.
