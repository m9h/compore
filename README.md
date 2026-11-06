# compore + lafix

[![CI](https://github.com/m9h/sbi4dwi/actions/workflows/ci.yml/badge.svg)](https://github.com/m9h/sbi4dwi/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-9cf)](https://github.com/google/jax)

Two packages from one repository (doc 012 §7):

- **compore** ("compartments and pores"): differentiable diffusion-MRI
  microstructure physics in JAX — compartment and pore kernels, Monte Carlo
  substrate oracles behind one interface, Fisher-information acquisition
  design, and simulation-based inference (SBI) with uncertainty tooling.
  The compartment models began as a JAX port of
  [dmipy](https://github.com/AthenaEPI/dmipy) and are tested against its
  closed forms.
- **lafix** ("Laplace fixel", `packages/lafix`): refine any DIPY peaks with a
  differentiable multi-fixel fit and get a calibrated per-fixel Laplace
  posterior. Depends on compore. Whole brain in about two minutes on CPU.

Until 2026-11-05 this repository was named dmipy-jax, then SBI4DWI; the
GitHub URL below still says so until it is renamed.

## What it does

| | module | what you get |
|---|---|---|
| **lafix refine** | `lafix` | Takes any DIPY peaks (CSD, MSMT-CSD, FORCE), fits a K-fixel stick + zeppelin + CSF/GM model per voxel with a Rician likelihood, learned axial diffusivity and spatial priors, and returns refined peaks plus a **Laplace posterior per fixel** (orientation σ, intra-axonal fraction σ). CLI: `dipy_refine_peaks` or `lafix`. |
| **Compartment models** | `compore.signal_models` | Ball, Stick, Zeppelin, Tensor, Soderman and Callaghan cylinders, Stejskal–Tanner / GPD / Callaghan spheres, planes, Watson/Bingham dispersion, NODDI, SANDI. `eqx.Module` pytrees, SI units, `jit`/`vmap`/`grad`-ready. |
| **SBI pipeline** | `compore.pipeline` | `ModelSimulator` (prior → forward → Rician noise → b0-normalised signal) → `train_sbi` (MDN or FlowJAX flow) → checkpoint → `SBIPredictor` on NIfTI volumes. SBC, PPC, conformal, OOD and ensemble checks. |
| **Simulation** | `compore.simulation` | FEM matrix-formalism on meshes, Monte Carlo, differentiable SDE walkers, Karger exchange; CATERPillar and MCMRSimulator.jl substrates behind an oracle boundary. |
| **Acquisition design** | `validation/design_*.py` | Fisher / Bayesian-CRLB protocol search (orbital RESOLVE, exchange protocols). |

## Results that are actually in the repository

Everything below is reproducible from `validation/` and written up with the
raw numbers in `docs/decisions/008-positioning-vs-force-prism-sbi.md` (PRISM,
HCP) and `009-graves-orbit-resolve-design.md` (design).

**Fixel precision at crossings, identical inputs** (doc 008 §6.4; Monte Carlo
substrates with 30–90° crossings, mean angular error):

| method | crossings | single bundle |
|---|---|---|
| lafix plus-x (ours) | **2.3–8.0°** | 6.7–6.9° |
| FORCE (dipy master, seeded) | 9.7–16.6° | 6.7–8.7° |
| SBI_dMRI amortised NPE | 4.6–13.1° | 1.2–2.8° |
| MSMT-CSD | 5–23° | 4.5–5.5° |

**DiSCo connectome** (FORCE authors' protocol, SNR 10): posterior-averaged
tractography r = 0.887 / Dice 0.73 vs FORCE 0.80–0.81 / 0.35 and MSMT-CSD
0.75 / 0.37.

**HCP-YA scan–rescan, 42 clean subjects** (doc 008 §10.6, §13; intra-axonal
fraction f_i vs AMICO-NODDI on the same data):

| estimator | CCC | mean \|Δ\| | within-subject CoV |
|---|---|---|---|
| lafix with isotropic prior (`fi_config`) | **0.867** | **0.036** | 6.8 % |
| NODDI NDI·(1−FWF) | 0.858 | 0.039 | 6.5 % |
| DTI FA (reference) | 0.894 | 0.055 | 11.5 % |

Edge-level connectome reliability (ICC) across the cohort: posterior-mean
tractography 0.64 vs MSMT-CSD 0.47; the fixel σ ranks edge reliability
(Spearman −0.67). The honest decomposition of that gain is in doc 008 §13.2.

## Install

Python 3.12+, [uv](https://docs.astral.sh/uv/). The core install is CPU-only
and is all that compartment models, fitting and `dipy_refine_peaks` need
(whole-brain refine + posterior in about two minutes on 20 cores).

```bash
git clone https://github.com/m9h/sbi4dwi.git && cd sbi4dwi
uv sync                       # core: jax, equinox, optimistix, optax, dipy, nibabel
uv sync --extra cuda          # + CUDA-13 JAX wheels
uv sync --extra sbi           # + flowjax, blackjax, numpyro, h5py (SBI pipeline, UQ)
uv sync --extra sim           # + diffrax, jax-md, pandas (walkers, FEM, substrates)
uv sync --extra data          # + HCP/OpenNeuro loaders (boto3, datalad)
uv sync --all-extras          # everything, incl. baselines (dmri-amico, torch)
```

## Quickstart

### 1. Forward model, vectorised over voxels

```python
import jax, jax.numpy as jnp, numpy as np
from compore.acquisition import JaxAcquisition
from compore.cylinder import C1Stick

rng = np.random.default_rng(0)
g = rng.normal(size=(60, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
acq = JaxAcquisition(bvalues=jnp.full(60, 1000e6), gradient_directions=jnp.asarray(g))  # SI: s/m^2

stick = C1Stick()
predict = jax.jit(jax.vmap(lambda mu, lpar: stick(acq.bvalues, acq.gradient_directions, mu=mu, lambda_par=lpar)))
mu = jnp.stack([jnp.full(1000, jnp.pi / 2), jnp.linspace(0, jnp.pi, 1000)], axis=1)  # [theta, phi]
signal = predict(mu, jnp.full(1000, 1.7e-9))          # (1000, 60)
```

Units are SI throughout: b in s/m², diffusivity in m²/s, lengths in m.

### 2. Refine DIPY peaks and get a fixel posterior

```bash
dipy_refine_peaks dwi.nii.gz dwi.bval dwi.bvec mask.nii.gz msmt_peaks.pam5 \
    --n_fibres 3 --out_dir refined/
```

or in Python, after any DIPY reconstruction that gives a `PeaksAndMetrics`:

```python
from lafix import dipy_refine as dr

cfg = dr.fi_config(n_fibres=3)       # recommended for f_i maps (isotropic-fraction prior)
pam_refined, posterior, fit = dr.refine_peaks(data, gtab, mask, pam, n_fibres=3, cfg=cfg, affine=affine)
# posterior.sigma_deg (N,K) orientation sd, posterior.wm_fracs, posterior.sample_dirs(key, n)
# pam_refined drops into dipy tracking unchanged
```

`default_config` reproduces the orientation-optimal setting (about 1.3° better
than `fi_config`, which trades that for an f_i map on par with NODDI).

### 3. Train an amortised posterior and deploy it

```python
import jax.numpy as jnp
from compore.acquisition import JaxAcquisition
from compore.pipeline.simulator import ModelSimulator
from compore.pipeline.config import SBIPipelineConfig
from compore.pipeline.train import train_sbi
from compore.pipeline.checkpoint import save_checkpoint, load_checkpoint
from compore.pipeline.deploy import SBIPredictor

def forward_fn(params, acq):                     # mono-exponential: params = [D in um^2/ms]
    return jnp.exp(-acq.bvalues * params[0] * 1e-9)

acq = JaxAcquisition(bvalues=jnp.r_[jnp.zeros(4), jnp.full(28, 1000e6)], gradient_directions=jnp.zeros((32, 3)))
names, ranges = ["D"], {"D": (0.1, 3.0)}
sim = ModelSimulator(forward_fn, names, ranges, acq, noise_type="rician", snr=30.0)
cfg = SBIPipelineConfig(model_name="ADC", parameter_names=names, parameter_ranges=ranges,
                        acquisition={"bvalues": acq.bvalues.tolist()},
                        inference_mode="mdn", n_components=4, hidden_dim=64, depth=2,
                        batch_size=256, n_steps=200)
model, losses = train_sbi(cfg, sim, print_every=100)
save_checkpoint(model, cfg, "adc_mdn")
model, cfg = load_checkpoint("adc_mdn")
# SBIPredictor(model, cfg).predict_volume("dwi.nii.gz", "dwi.bval", "dwi.bvec", mask_path="mask.nii.gz", output_dir="out/")
```

Training and deployment b0-normalise identically inside `ModelSimulator` and
`SBIPredictor`; do not normalise by hand in between.

## Testing

```bash
uv run pytest                      # compore/tests + tests (about 6 min on CPU)
JAX_PLATFORMS=cpu uv run pytest    # what CI runs
uv run pytest docs/tutorials       # markdown tutorials as doctests (sybil)
```

Compartment models are checked against closed forms transcribed from the
original dmipy code (`compore/tests/test_callaghan_reference.py`), not
against a dmipy install.

## Layout

```
compore/        the package: signal_models, prism, pipeline, simulation, inference, fitting, io
validation/       paper scripts (validate_*.py, hcp_*.py, design_*.py) and validation/lib (not shipped)
tests/            top-level tests, incl. tests/validation for the PRISM stack
docs/decisions/   ADRs and result logs; 008 = PRISM/HCP results, 009 = orbit design, 010 = package review
archive/          transcranial-ultrasound / EIT / pulseq code kept for reference, not imported
```

`CLAUDE.md` has the conventions (uv only, SI units, Equinox pytrees, b0
normalisation) and the full directory map; `docs/artifacts.md` says where the
untracked checkpoints live.

## Relationship to dmipy and to SBI_dMRI

The compartment models follow dmipy (Fick, Wassermann & Deriche, 2019); the
Callaghan cylinder is verified to 1e-9 against the dmipy code at commit
`7b254f4`. The SBI pipeline follows the design of Manzano-Patrón et al. (2025),
whose SBI_dMRI networks we also run as a baseline in `validation/`. If you use
the models, cite dmipy:

> Fick, Wassermann & Deriche, "The Dmipy Toolbox: Diffusion MRI
> Multi-Compartment Modeling and Microstructure Recovery Made Easy",
> *Frontiers in Neuroinformatics* 13 (2019): 64.

> Manzano-Patrón et al., "Uncertainty mapping and probabilistic tractography
> using Simulation-Based Inference in diffusion MRI", *Medical Image Analysis*
> 103 (2025): 103580. [10.1016/j.media.2025.103580](https://doi.org/10.1016/j.media.2025.103580)

## License

MIT. Copyright (c) 2017 Rutger Fick & Demian Wassermann (dmipy signal models);
Copyright (c) 2024–2026 Morgan Hough (compore / lafix).
