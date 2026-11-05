# Changelog

## Unreleased (2026-11-04) — product/platform split, step 1–2 (doc 012 §7)

### Changed
- Core dependencies 29 → 11 (jax, jaxlib, numpy, scipy, equinox, jaxtyping, optimistix,
  lineax, optax, nibabel, dipy). New extras: `cuda` (jax[cuda13]), `sbi`, `sim`, `algebra`,
  `inverse`, `nn`, `viz`, `all`. CPU users install no GPU wheels.
- `dmipy_jax/__init__.py` imports submodules lazily; `import dmipy_jax.prism` no longer
  loads diffrax / scico / flowjax.
- `jaxopt` is optional: `VoxelFitter` (bounded L-BFGS), `ConstrainedOptimizer` and
  `models/fw_dti` import it on use. `annotated-types` removed (no importer).

### Added
- Shipped σ calibration: `prism_uncertainty.SIGMA_INFLATION_HCP_YA = 1.81` (± 0.15,
  44-subject HCP-YA scan–rescan, doc 008 §10.6). `FixelPosterior.sigma_inflation`,
  `.sigma_deg_calibrated`, `.fintra_sd_calibrated`, `sample_dirs(calibrated=True)`;
  `refine_peaks(sigma_inflation=…)` (default: the HCP-YA constant); `dipy_refine_peaks`
  writes `*_sigmatheta_calibrated.nii.gz`, the fixel directory gains
  `sigma_theta_deg_calibrated`, `*_model.json` records the factor. Raw σ unchanged.
- CPU whole-brain runtime (Stanford HARDI, 154,625 voxels): refine 88 s, Laplace 31 s
  (`validation/runtime_hardi_cpu.json`; GPU 31 s / 23 s).
- doc 012 §7: the program to completion (two distributions, upstream, oracle protocol,
  the SBI model-selection experiment, January debts).
- Oracle protocol (step 3): `simulation/oracle.py` with `SubstrateOracle`, `PGSEScheme`,
  `JaxWalkerOracle` / `MCMROracle` / `MCDSOracle`, `get_oracle`, `generate_library` → HDF5
  `SimulationLibrary`; `simulation/emulator.py::LibraryEmulator` (ensemble MLP, `as_model_simulator`,
  `invert`, `save/load`). `validation/lib/substrate_benchmark.py` moved into the wheel as
  `simulation/substrates/spheres.py` (shim kept). Tests in `dmipy_jax/tests/test_oracle.py`.

## Unreleased (2026-10-31) — retrospective and bibliography

### Added
- doc 012: project retrospective (dmipy fork origins, Kidger-stack audit, technique tour)
  and the October 2026 landscape by group; doc 004 now points to it.
- `docs/references.bib`: 6 → 51 entries (Crossref/arXiv records for every group, method and
  JAX package the decisions cite); the Sphinx bibliography lists all entries.
- `signal_models/matrix_formalism.py`: exact finite-pulse ring / disk / sphere kernels
  (Canales-Rodríguez et al. 2026), doc 011 §6.

## Unreleased (2026-10-09) — package review follow-up (doc 010)

### Fixed
- Callaghan cylinder (`signal_models/cylinder_models.py`) returned ~1e-51 instead of the
  restricted signal: orders were never accumulated, coefficients were doubled and the
  zero root was missing. Now a faithful port of dmipy 7b254f4 (rtol 1e-9 in float64).
- Callaghan sphere rederived with the roots of the spherical Bessel derivative; pinned by
  its Stejskal–Tanner and free-diffusion limits (dmipy's version was unrunnable).
- `constants.SPHERE_ROOTS` is the single 100-root table (six mistranscribed roots gone).
- `uv run pytest` works from a clean checkout; six stale test modules fixed or removed;
  two tests that poisoned the session (MagicMock `numpy`/`jax`, global x64) scoped.
- Four files with committed merge-conflict markers resolved.

### Changed
- Cylinder kernels are SI-only; the `max(b) > 5e4` unit heuristic is gone.
- `dmipy_jax.validation` split: PRISM-JAX → `dmipy_jax.prism`, CATERPillar →
  `dmipy_jax.simulation.substrates`, paper code → `validation/lib` (not shipped).
- Dependencies: core list 48 → 29; new extras `data` (boto3, datalad) and
  `baselines` (dmri-amico, torch); torch/timm/opencv/gpjax/uguide/vbjax/pypulseq removed.
- Checkpoints, logs and sample volumes untracked (mirrored in
  `/data/datasets/sbi4dwi-artifacts`, see `docs/artifacts.md`); vendored AMICO deleted.
- Transcranial-ultrasound, EIT and pulseq code moved to `archive/`.
- README rewritten around what the package is now; Sphinx project renamed SBI4DWI.

### Removed
- `dmipy==1.0.4` dependency (reference is `git show 7b254f4:dmipy/...`).
- 17 dead modules (no importer, test or script).

## 0.1.0 — earlier history
See `git log` and `docs/decisions/`.
