# CLAUDE.md — Agent Guide for compore + lafix

This file orients new Claude Code agents working on this repository (named
dmipy-jax, then SBI4DWI, until 2026-11-05; doc 012 §7.1). Read this first
before touching any code.

## What is this project?

One repository, two distributions (a uv workspace):

- **compore** (`compore/`, "compartments and pores") is the platform: a
  JAX library of differentiable diffusion-MRI microstructure physics,
  Monte Carlo substrate oracles and simulation-based inference (SBI) tooling.
- **lafix** (`packages/lafix/`, "Laplace fixel") is the product: refine any
  DIPY peaks with a differentiable multi-fixel fit and return a calibrated
  per-fixel Laplace posterior. It depends on compore and ships the
  `dipy_refine_peaks` / `lafix` CLI. Its results are doc 008.

The platform combines differentiable biophysical signal models, multi-fidelity
physics simulation and neural posterior estimation to recover tissue
properties (axon diameter, cell density, fibre orientation) from
diffusion-weighted MRI. The codebase spans:

- **Analytical forward models** (Ball, Stick, Zeppelin, Sphere, NODDI, SANDI)
- **Differentiable physics simulations** (FEM, Monte Carlo, SDE walkers)
- **Simulation-Based Inference (SBI)** pipeline (MDN + Normalizing Flows)
- **External oracle integration** (ReMiDi, MCMRSimulator.jl, DIPY)
- **Uncertainty quantification** (SBC, PPC, conformal prediction, OOD, ensembles)
- **Clinical deployment** (NIfTI volume inference, comparison framework)

The principle: **build where differentiable, wrap where not**. JAX models
handle gradient-based optimization. External simulators produce training
data at the HDF5 boundary.

## Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Arrays | JAX | GPU acceleration, autodiff |
| Modules | Equinox (`eqx.Module`) | Pytree-compatible neural nets & models |
| Optimisation | Optimistix / Optax | Deterministic fitting / stochastic training |
| ODEs/SDEs | Diffrax | Differentiable solvers for Bloch/diffusion |
| MCMC | BlackJAX | Bayesian inference (NUTS) |
| Flows | FlowJAX | Normalizing flow posteriors |
| IO | nibabel, h5py, DIPY | NIfTI, HDF5, gradient tables |
| Package management | **uv only** | No pip/conda/poetry — see below |

## Critical conventions

### uv only
All Python operations go through `uv`. Never use `pip`, `conda`, or bare
`python`. Use:
```bash
uv sync              # install deps
uv run python x.py   # run scripts
uv run pytest        # run tests
```

### Testing
```bash
uv run pytest                              # full suite (compore/tests + tests; tests/validation = lafix)
uv run pytest compore/tests/test_oracle.py -v   # single file
uv run pytest docs/tutorials               # markdown tutorials as doctests (sybil)
JAX_PLATFORMS=cpu uv run pytest            # CPU; use this for compile-heavy or agent-run suites
uv run pytest --cov=compore              # coverage is opt-in
```
The root `conftest.py` loads sybil only if it is installed, so `--noconftest`
is no longer needed. Equivalence with the original dmipy code is tested
against vendored closed forms (`compore/tests/test_callaghan_reference.py`);
the reference source is `git show 7b254f4:dmipy/...`, not a dmipy install.

### Equinox patterns
All differentiable objects (models, simulators) are `eqx.Module` subclasses.
They are immutable pytrees — use `eqx.tree_at` for updates, not `__setattr__`.
Plain Python classes (not eqx.Module) are used for pipeline orchestration
(`ModelSimulator`, `ComparisonRunner`, `LibraryGenerator`).

### Units
- b-values: **SI (s/m²)** internally. DIPY and FSL use s/mm² — convert with `/ 1e6`.
- Diffusivity: m²/s (e.g. 2.0e-9 for free water).
- Lengths: metres (e.g. 5e-6 for a 5 μm radius).
- The `JaxAcquisition` class stores b-values in SI.

### b0 normalisation
Training data is b0-normalised in `ModelSimulator.sample_and_simulate()`.
The same normalisation happens at inference time in `SBIPredictor.predict_volume()`.
These **must match** or the network sees different distributions at train vs deploy.

## Directory map

```
packages/lafix/lafix/        # the product distribution `lafix` (depends on compore)
├── model.py                 # the differentiable K-fixel model, Rician NLL, priors, fit_fixels
│                            #   (was prism_jax.py until the 0.1.0 release; shim kept, doc 015)
├── uncertainty.py           # Laplace fixel posterior, SIGMA_INFLATION_HCP_YA, cone coverage,
│                            #   posterior-sampled connectomes, Laplace evidence (was prism_uncertainty.py)
├── kernels.py               # vendored GPD sphere kernel so lafix does not depend on compore
├── dipy_refine.py           # DIPY plug-in: refine_peaks, default_config, fi_config
├── tracking.py fixel.py     # peaks → PAM / EuDX helpers; MRtrix fixel-directory writer
└── workflow.py              # `dipy_refine_peaks` / `lafix` CLI (DIPY Workflow)

compore/                     # the platform distribution `compore`
├── acquisition.py           # JaxAcquisition (b-values SI, gradient dirs, delta/Delta)
├── constants.py             # SPHERE_ROOTS (100-root dmipy table), Bessel-root finders, gamma
├── signal_models/           # Analytical forward models (Ball, Stick, Zeppelin, cylinders,
│                            #   spheres incl. Callaghan, NODDI, SANDI); closed-form tested;
│                            #   matrix_formalism.py = exact finite-pulse ring (myelin water,
│                            #   Canales-Rodríguez 2026) / disk / sphere spectral kernels
├── core/                    # modeling_framework (compose), solvers, surrogate
├── simulation/              # mesh_sim (FEM), monte_carlo, differentiable_walker, simulator,
│   │                        #   oracle.py (SubstrateOracle ABC, jax/mcmr/mcds engines, registry,
│   │                        #   library generation), emulator.py (LibraryEmulator → ModelSimulator)
│   └── substrates/          #   spheres.py (Substrate, SDFs, PGSE walker; was validation/lib/
│                            #   substrate_benchmark.py), caterpillar.py (CATERPillar generators)
├── signal_models/gm_powder.py  # powder-averaged NEXI / SANDI / NEXI+soma on multi-(b,Δ) schemes
├── inference/model_selection.py  # joint p(M, θ | S): Laplace evidence vs amortised classifier + flows
├── io/nexi_c2.py            # ds008771 (reduced NEXI, Connectome 2.0) preproc loader; io/wand.py t_bdel parsing
├── pipeline/                # ModelSimulator → train_sbi → checkpoint → deploy; UQ (sbc, ppc,
│                            #   conformal, ood, ensemble), comparison runner, metrics
├── library/                 # SimulationLibrary (HDF5), generator, matcher
├── inference/               # mdn, flows, mcmc (MCMCInference), score_posterior, trainer
├── fitting/                 # VoxelFitter, GlobalBruteInitializer, algebraic initialisers
├── biophysics/              # neural_exchange, velocity, conductivity (dMRI-side only)
├── io/ inverse/ viz/ ...    # loaders (BIDS, HCP, mesh, multi-TE), AMICO-style inversions, viz
└── tests/                   # package tests (qmri/, scanner/ hold the moved model tests)

validation/                  # paper scripts (validate_*.py, hcp_*.py, design_*.py) …
└── lib/                     # … and their shared code (FORCE/DiSCo loaders, baselines,
                             #   connectivity metrics, synthetic benchmarks). Importable in a
                             #   dev checkout (`from validation.lib import force_disco`), NOT in the wheel
tests/                       # top-level tests, incl. tests/validation for lafix + validation/lib
archive/                     # transcranial ultrasound / EIT / pulseq code, kept but not imported
docs/decisions/              # ADRs and result logs (008 = PRISM/HCP results, 009 = orbit design,
                             #   012 = retrospective + landscape by group, 013 = progress report + roadmap, 014 = DIPY workshop 2027 landscape,
                             #   010 = package review + clean-up plan and its execution log)
docs/artifacts.md            # where the untracked checkpoints/binaries live
```

Dependencies (doc 012 §7, 2026-11-04): compore's core list is only what
`compore.signal_models` / `fitting` and lafix need on CPU; lafix depends on
dipy, jax, optax only (no compore since the 0.1.0 release, doc 015). Extras: `cuda` (jax[cuda13]), `sbi` (flowjax, blackjax, numpyro, h5py),
`sim` (diffrax, jax-md, pandas), `algebra` (sympy), `inverse` (scico, cvxpy),
`nn` (e3nn, healpy, ttax), `viz` (matplotlib, tqdm), `data`, `baselines`,
`test`, `doc`, `all`. The dev environment is `uv sync --all-extras`. Keep the
core import path free of extras: `import lafix` must not pull in
diffrax/scico/flowjax (the package `__init__` is lazy for this reason).

## Key data flow

### SBI training pipeline
```
ModelSimulator                         train_sbi()
  ├── forward_fn(params, acq) → signal    │
  ├── prior_sampler(key, n) → params      │
  └── add_noise(key, signal) → noisy      │
        │                                  │
        ▼                                  ▼
  sample_and_simulate(key, n)         MDN or Flow
        │  (b0-normalise)                  │
        ▼                                  ▼
  (theta, noisy_signals)          _NormalisedMDN / _NormalisedFlow
                                           │
                                           ▼
                                    save_checkpoint()
                                           │
                                           ▼
                                    SBIPredictor.predict_volume()
```

### Oracle / multi-fidelity path
```
OracleSimulator.generate_library()
        │
        ▼
  SimulationLibrary (HDF5)
        │
        ▼
  OracleModelSimulator (adapter)
        │  (k-NN interpolation or neural emulator)
        │
        ▼
  train_sbi()   ←── or HybridLibraryGenerator mixes
                     analytical (70%) + oracle (30%)
```

### SimulationLibrary HDF5 schema
```
/params              (N, P) float32     — tissue parameters
/signals             (N, M) float32     — primary (diffusion) signal
/contrast_signals/   (group, optional)  — multi-contrast data
    /t1              (N, M) float32
    /t2              (N, M) float32
/source              (N,) bytes         — provenance labels
attrs:
    parameter_names  list of str
    contrast_channels list of str
    (arbitrary metadata)
```

## FEM simulator (mesh_sim.py)

`MatrixFormalismSimulator` uses spectral ROM on triangular surface meshes:
1. Constructs FEM stiffness (S) and mass (M) matrices from mesh geometry
2. Generalised eigendecomposition: S v = λ M v (via Cholesky transform)
3. Projects position matrices into eigenspace: V^T M_x V
4. Simulates PGSE via 3 matrix exponentials: pulse1 → gap → pulse2

Key features:
- `__call__(G_amp, delta, Delta, gradient_direction=...)` — arbitrary gradient direction
- `simulate_acquisition(acq)` — vmaps over all measurements in a JaxAcquisition
- `construct_fem_matrices_sparse()` — BCOO sparse for meshes >2K vertices

## Oracle system

Status (2026-11-04, doc 012 §7.3): the protocol exists in
`compore/simulation/oracle.py` and wraps three Monte Carlo engines behind one
interface; the sphere-substrate geometry, SDFs and PGSE walker moved from
`validation/lib/substrate_benchmark.py` into
`compore/simulation/substrates/spheres.py` (the old path is a re-export shim).

- `PGSEScheme(bvals_si, bvecs, delta, Delta)` — SI; rejects s/mm².
- `SubstrateOracle` (ABC): `check_available()`, `simulate(substrate, scheme, D_intra=, D_extra=, n_particles=, seed=) -> OracleResult(S, S_intra, S_extra, S_glia, engine, seconds, meta)`,
  `generate_library(sampler, n, scheme, parameter_names=, descriptor_fn=, out_path=) -> SimulationLibrary`
  (HDF5; metadata carries scheme, engine, measured descriptors, per-compartment signals, skip count).
- Engines: `JaxWalkerOracle` ("jax", always available, differentiable walker, glia supported),
  `MCMROracle` ("mcmr", MCMRSimulator.jl v1.1.0 via `julia +1.13 --project=julia/mcmr julia/mcmr/substrate_pgse.jl`, staged under a workdir),
  `MCDSOracle` ("mcds", Permeable_MCDS CLI, `MCDS_BIN` env or the `_external` build; periodic images of boundary spheres).
  None is a Python dependency; unavailable engines raise on use.
- Registry: `get_oracle("jax" | "mcmr" | "mcds", **kw)`, `available_oracles()`, `register_oracle(name, cls)`.
- Adapter: `compore/simulation/emulator.py::LibraryEmulator.fit(library)` trains an Equinox MLP
  ensemble f(θ, b, ĝ) → log S; `.as_model_simulator(acq)` gives a `ModelSimulator` for `train_sbi`;
  `.invert(y, bvals, bvecs)` is the emulator-in-the-loop LM inversion of doc 008 §11.2;
  `.save/.load` (JSON header + Equinox leaves).
- Parity of the three engines: doc 008 §11.1 (extra-cellular RMS 0.01–0.015 JAX↔MCMR; MCDS freer;
  intra MCDS < JAX < MCMR). Tests: `compore/tests/test_oracle.py`.
- Not built: the k-NN `OracleModelSimulator` and `DIPYMultiTensorOracle` / `ReMiDiOracle` named in
  earlier versions of this file (ReMiDi has no public code; DIPY multi-tensor is a closed form, not an oracle).

## Comparison framework

Two runners:
1. `ComparisonRunner` — method-level comparison (DIPY DTI vs SBI vs dictionary matching)
2. `SimulationComparisonRunner` — simulator-level comparison (analytical vs FEM vs MC vs oracle)

Both produce structured result objects with RMSE, correlation, SSIM, and
per-b-value error analysis.

## Recent development history

The project has been built incrementally over ~2 months:

1. **Core JAX port** — Equinox models, JaxAcquisition, Optimistix fitting
2. **SBI pipeline** — ModelSimulator → train_sbi → checkpoint → deploy
3. **Differentiable simulation** — FEM mesh_sim, Monte Carlo, SDE walkers
4. **UQ suite** — SBC, PPC, conformal prediction, OOD detection, ensembles
5. **Comparison runner** — Multi-method benchmarking framework
6. **Multi-fidelity oracles** — Oracle protocol, DIPY/ReMiDi/MCMR integration,
   hybrid library generation, multi-contrast storage
7. **Score-based posterior** — Score matching for diffusion posterior estimation

## What NOT to do

- Don't use `pip install` — use `uv add`
- Don't mutate `eqx.Module` fields — use `eqx.tree_at`
- Don't hardcode b-value units — always check SI vs FSL
- Don't skip b0 normalisation — train and deploy must match
- Don't make external simulators JAX dependencies — use the oracle boundary
- Don't pass b-values in s/mm² to the signal kernels — they are SI-only (no unit heuristics)
