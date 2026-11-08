# lafix

**Laplace fixel posteriors for diffusion MRI.** `lafix` takes any DIPY
`PeaksAndMetrics` (CSD, MSMT-CSD, FORCE, or your own peaks), refines it with
a differentiable multi-fixel stick + zeppelin (+ CSF/GM) fit under a Rician
likelihood, and returns the refined peaks together with a **per-fixel
posterior**: an angular standard deviation for every fibre direction and an
uncertainty on the intra-axonal fraction. The posterior can be sampled, so a
tractography pipeline can propagate orientation uncertainty into
probabilistic connectomes instead of tracking a single point estimate.

Whole brain on a CPU takes a few minutes: Stanford HARDI (150 directions)
refines in 88 s + 31 s Laplace on a 154 625-voxel white-matter mask, and in
291 s on the 185 968-voxel brain mask the quick start below uses. A GPU is
optional.

```bash
pip install lafix            # or: uv add lafix
dipy_refine_peaks dwi.nii.gz dwi.bval dwi.bvec mask.nii.gz peaks.pam5 --out_dir out/
```

## What you get

| file | content |
|---|---|
| `<prefix>_desc-refined_peaks.pam5` | refined `PeaksAndMetrics`, drop-in for any DIPY tracker |
| `<prefix>_desc-refined_fintra.nii.gz` | intra-axonal fraction of the white-matter pool, f_i |
| `<prefix>_desc-refined_fwm.nii.gz` | white-matter fraction Σ f_wm (so f_i · f_wm is the voxel intra fraction) |
| `<prefix>_desc-refined_sigmatheta.nii.gz` | Laplace σ_θ of the main fixel, degrees |
| `<prefix>_desc-refined_fixels/` | MRtrix fixel directory: `index.mif`, `directions.mif`, `fraction.mif`, `sigma_theta_deg.mif`, `f_intra.mif` |
| `<prefix>_desc-refined_model.json` | configuration, learned global diffusivities, noise σ |

With `--sigma_inflation 1.81` a `*_sigmatheta_calibrated.nii.gz` map is also
written (see *Calibration*).

## Python

```python
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti
from dipy.io.peaks import load_pam
from lafix import dipy_refine as dr

data, affine = load_nifti("dwi.nii.gz")
mask, _ = load_nifti("mask.nii.gz")
gtab = gradient_table("dwi.bval", bvecs="dwi.bvec")
pam = load_pam("peaks.pam5")

pam_refined, post, fit = dr.refine_peaks(data, gtab, mask.astype(bool), pam, n_fibres=3, affine=affine)

post.sigma_deg                      # (N_voxels, K) angular sd per fixel, degrees
post.fintra_sd                      # (N_voxels,) sd of f_i
dirs = post.sample_dirs(jax.random.PRNGKey(0), n_samples=50)   # (50, N, K, 3) direction samples
```

`post.sample_dirs` feeds `lafix.tracking.peaks_to_pam` so you can run DIPY's
EuDX / probabilistic trackers once per posterior sample and average the
connectomes; `lafix.uncertainty.posterior_connectivity` does the loop.

Two shipped configurations:

- `dr.default_config(n_fibres)`: learned D∥ and extra-cellular D∥, tortuosity,
  CSF and GM balls. Best orientations; use it for tractography.
- `dr.fi_config(n_fibres)`: adds a one-sided prior on the isotropic
  fractions. Best intra-axonal-fraction maps (scan–rescan on par with NODDI
  NDI·(1−FWF)); main fixel about 1.3° worse.

## Quick start on public data

```bash
python -m lafix.examples.stanford_hardi_quickstart out/    # downloads Stanford HARDI, CSD peaks, refine, Laplace
```

It writes the files above plus `out/quickstart.json` with timings.
`python -m lafix.examples.viz_skyline_sigma out/stanford_hardi_desc-refined`
shows the result in DIPY's Skyline viewer (dipy ≥ 1.13, FURY ≥ 2.1).

## How well does it work

Numbers from the validation logs in the compore repository
(`docs/decisions/008-*.md`, `012-*.md`); the inputs to each comparison are
identical.

- **Refinement improves whatever peaks it is given.** DiSCo 1 connectome
  correlation with ground truth: FORCE peaks 0.79 → 0.91 after refinement.
  Monte Carlo substrate crossings at 30–90°: refined 2.3–8.0° angular error
  vs 9.7–16.6° for FORCE and 5–23° for MSMT-CSD.
- **The σ is informative.** Over 44 HCP-YA test–retest subjects the Laplace
  σ_θ ranks the scan–rescan angular error at Spearman ρ = 0.457 ± 0.023.
- **The posterior-mean connectome is more reproducible than the point
  estimate.** HCP-YA scan–rescan r(log counts) 0.965 vs 0.923, Dice 0.936 vs
  0.855.
- **f_i reproducibility is at NODDI level.** Mean |Δf_i| 0.035 ± 0.008 vs
  0.039 ± 0.007 for NODDI, within-subject CoV 6.4 % vs 6.5 %.

## Calibration

The Laplace σ is a local Gaussian approximation and under-states the real
scan–rescan spread. On the 44 HCP-YA subjects (3 T, 1.25 mm,
b = 1000/2000/3000, SNR 12–22) the ratio was **1.81 ± 0.15**; after that
inflation the 90 % cones cover the retest direction at the nominal rate.
`lafix` ships the raw σ by default. Pass `--sigma_inflation 1.81` (or
`lafix.uncertainty.SIGMA_INFLATION_HCP_YA`) on comparable protocols, or
measure your own constant on one retest pair with
`lafix.uncertainty.cone_coverage`. The constant used is recorded in
`*_model.json`.

## Limits

- It refines, it does not detect: the fixel count comes from the input peaks
  (`--peak_frac_min` drops fixels whose refined fraction falls below it).
- The model is single-shell-friendly but needs at least two shells to learn
  diffusivities; with one shell pass `--d_par_init` and the learned values
  stay near it.
- The isotropic compartments trade f_i precision for orientation precision;
  pick the configuration for your question (see above).
- Rician likelihood with one global noise σ, learned.
- σ_θ is bounded at about 57° (1 rad) by a weak prior on the direction
  parameters; a fixel at the bound has no resolvable orientation (grey
  matter, CSF, or a fraction near `--peak_frac_min`). Mask such voxels with
  `f_wm` or the σ itself before averaging.

## How it relates to PRISM

The forward model follows PRISM (Abouagour, Shah & Garyfallidis, arXiv
2604.00250): K sticks with a shared zeppelin, CSF and GM balls, spatial and
sparsity priors, fitted jointly over the mask. `lafix` adds learned
diffusivities, Szafer–Stanisz tortuosity, a Rician likelihood and the
per-fixel Laplace posterior, and it starts from peaks instead of random
directions. PRISM itself had no public code at the time of this release.

## Install notes

Dependencies: `dipy ≥ 1.12.1`, `jax`, `optax`, `numpy`, `scipy`, `nibabel`,
`h5py`. `pip install "lafix[cuda]"` adds the CUDA 13 JAX build. The
companion platform `compore` (differentiable microstructure models,
simulators, simulation-based inference) is **not** required.

Licence: MIT. Issues and code: https://github.com/m9h/compore (directory
`packages/lafix`).
