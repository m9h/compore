# 012. Retrospective: from a dmipy port to a tour of frontier microstructure methods, and the landscape as of October 2026

Date: 2026-10-31. Type: review + survey. Status: live (append dated sections).

This document does three things that no other doc does.

1. Records where the project came from (a JAX rewrite of one specific
   package, dmipy) and what the rewrite set out to do.
2. Audits the second goal that was declared in January 2026: build the
   project on the Kidger scientific-JAX stack and make that stack usable for
   simulation-based inference (SBI) of tissue microstructure.
3. Refreshes the research landscape that doc 004 froze in May 2026, group by
   group, with the citations now in `docs/references.bib`.

Doc 004 remains the historical survey that set the first roadmap. Doc 008 is
still the primary results log. This document is the one to read to understand
why the repository looks the way it does.

## 1. Origins: a fork of dmipy

The git history begins on 2017-03-22 with Rutger Fick's "Code extraction from
Athena Tools" into `AthenaEPI/microstruktur`, the repository that became dmipy
{cite}`fick2019dmipy`. Everything up to commit `fe4e35e` (2024-12-04, a Python
3.8 install fix) is upstream dmipy history. The counts below are from
`git log --format=%ad | uniq -c` on 2026-10-31.

| period | commits | what |
|---|---|---|
| 2017-03 to 2018-03 | 845 | dmipy's own development (Fick, Wassermann, Deriche) |
| 2018-04 to 2024-12 | 74 | dmipy maintenance |
| 2026-01 | 34 | the JAX port begins (`1d55647`, 2026-01-14) |
| 2026-03 | 34 | SBI pipeline, score posteriors, qMRI models, ultrasound detour begins |
| 2026-04 | 43 | transcranial ultrasound / full-waveform inversion detour (now `archive/`) |
| 2026-05 | 40 | FORCE replication and DiSCo benchmarks (doc 004 §9–§24) |
| 2026-09 | 103 | PRISM-JAX, Laplace fixel posterior, substrate Monte Carlo, HCP retest (docs 006–008) |
| 2026-10 | 21 | acquisition design, package review and clean-up, dmipy 2.x comparison, realism ladder, matrix-formalism pores |

So the repository carries dmipy's full history and about 275 commits of our
own. The import name `dmipy_jax` and the `git show 7b254f4:dmipy/...` test
oracle are the visible remnants; the project was renamed SBI4DWI on
2026-03-25 (`f29facf`) when the vendored dmipy was removed.

### 1.1 What the port set out to do (January 2026)

The first JAX README (`git show 1d55647:README.md`) was a feature-parity table
against dmipy, with one addition dmipy never had: NUTS sampling through
BlackJAX. The roadmap of 2026-01-20 (`git show 4dfe499:ROADMAP.md`) had four
phases:

1. **Foundation.** `JaxAcquisition` as a pytree, Stick/Ball/Sphere, model
   composition, Levenberg–Marquardt through Optimistix (ADR 001).
2. **Optimisation and compatibility.** A `fit()` API compatible with dmipy,
   a vmapped brute-force initialiser, the missing models (Tensor, restricted
   cylinder, planes), a polynomial-chaos surrogate, CRLB uncertainty, the goal
   of "scaling to 1M+ voxels".
3. **Inverse and global fitting.** scico operators, an ADMM port of AMICO,
   TV-regularised global reconstruction.
4. **Beyond parity.** Permeable spheres and exchange (Kärger, NEXI),
   concentric spheres, ellipsoids.

Of these, phases 1 and 2 were completed and are what `dmipy_jax.signal_models`,
`core` and `fitting` are today. Phase 3 was built (scico is still imported in
four files) but never used in a result; doc 010 pruned most of it. Phase 4
arrived by a different route: the Kärger kernel (`a78dedb`, 2026-09-15) and
the glial sphere pool (`4fb36c5`, 2026-10-22) were built because the CHUV
realism ladder and the exchange-protocol design needed them, not because the
roadmap said so.

The thing the January plan did not foresee was that the fitting engine would
become a minor part of the project. The parity goal was reached in a week.
What took the following nine months was deciding what a differentiable
microstructure library is *for*.

## 2. The Kidger-stack goal, audited

ADR 002 (2026-01-16) declared that every physical model is an `eqx.Module`,
every array is `jaxtyping`-annotated, fitting goes through Optimistix and
Lineax, and differential equations through Diffrax. ADR 003 added Diffrax
SDE walkers and Bloch simulation. The audit below counts files under
`dmipy_jax/`, `validation/` and `tests/` that import each package
(`grep -rl` on 2026-10-31).

| package | files importing it | role today | verdict |
|---|---|---|---|
| equinox {cite}`kidger2021equinox` | 119 | every model, simulator, flow wrapper and MDN | the decision that stuck |
| optax | 41 | MDN, flow, score-network and emulator training | as planned (ADR 001 split) |
| jaxtyping {cite}`kidger2024jaxtyping` | 36 | shape annotations | partial; not runtime-checked |
| diffrax {cite}`kidger2022thesis` | 15 | SDE walkers, Bloch, Kärger waveform ODE, neural CDE | as planned |
| optimistix {cite}`rader2024optimistix` | 7 | `VoxelFitter` Levenberg–Marquardt, PRISM-JAX refine | as planned, but PRISM-JAX uses its own Gauss–Newton |
| numpyro | 3 | NUTS alternative | marginal |
| lineax {cite}`rader2023lineax` | 2 | linear solves inside Laplace | marginal; most solves are `jnp.linalg` |
| blackjax {cite}`cabezas2024blackjax` | 2 | `MCMCInference` NUTS | built, rarely run |
| flowjax {cite}`ward2024flowjax` | 2 | spline-coupling NPE flows | the SBI backbone; two files is deceptive, it is the whole flow path |
| scico | 4 | AMICO/TV operators | legacy of phase 3 |
| jaxopt | 0 | — | still a dependency, no importer; remove |
| unxt (ADR 002c) | 0 | — | never adopted; convention is SI everywhere |

Two conclusions. First, the stack was adopted for real: Equinox pytrees and
`eqx.tree_at` are the project's native idiom, and the GPU compile pathologies
we hit (unrolled Legendre sums, exact Hessians, see the memory note on GB10)
were solved inside it with `lax.scan`, custom VJPs and Gauss–Newton rather
than by leaving it. Second, the "make the Kidger stack usable for SBI"
ambition was met only in the narrow sense: FlowJAX flows and Equinox MDNs
train on Equinox simulators, and the SBC/PPC/conformal/OOD tools in
`dmipy_jax/pipeline` work on them. We did not produce a reusable SBI library
for the JAX ecosystem; we produced an SBI pipeline inside a dMRI package. The
general-purpose role is held by `sbi` {cite}`tejerocantero2020sbi` (PyTorch),
and the one JAX-native entrant with momentum, Gloeckler et al.'s
simformer-derived work {cite}`gloeckler2026sbiprism`, is in the Macke lab,
not here.

## 3. The tour

Read chronologically, the commit log is a tour of frontier techniques, each
tried against a concrete question. The table records what each stop left
behind.

| when | technique | question it answered | what remains |
|---|---|---|---|
| 2026-01 | Equinox port, Optimistix LM, vmapped brute init, gPC surrogate | can dmipy be made differentiable and fast | `signal_models`, `fitting`, `core`; 8M forward voxels/s claimed in `d380833`, never re-measured |
| 2026-01 | scico / ADMM AMICO, TV global fit | can linearised inversion share the operators | `inverse/`, unused in results |
| 2026-03 | MDN and FlowJAX NPE, b0 normalisation, SBC/PPC/conformal/OOD/ensembles | can we match Nottingham's amortised posterior {cite}`manzanopatron2025sbi` | `pipeline/`, `inference/`; 2.8° orientation error reached (`ffe0f09`) |
| 2026-03 | score-based diffusion posterior, DDPM, v-prediction, equivariant heads, "autoresearch" loop | can score matching beat flows for orientation | `inference/score_posterior.py`; 60° to 15.5° (`b39fc9e`), never beat the spline flow; the autoresearch loop was removed |
| 2026-03 | qMRI models (DESPOT1/2, MP2RAGE, mcDESPOT, QMT) + Laplace UQ | does the pattern transfer to relaxometry | `dmipy_jax/tests/qmri`; no result since |
| 2026-03/04 | transcranial ultrasound, MR-ARFI, j-Wave, 3D full-waveform inversion on SimNIBS heads, CANN brain mechanics, cortical buckling | can the same differentiable-physics stack serve an adjacent modality | `archive/tus`, `biophysics/buckling_layer`; results exist (skull velocity 4312 m/s, `ca6bc41`) but outside the project's claim |
| 2026-05 | FORCE replication {cite}`shah2025force`, DiSCo phantom, MRtrix reference pipelines, dictionary matching | is dictionary matching really the frontier | doc 004 §9–§24; found the library min-angle limit and the upstream ODF bug; r = 0.851 with our dictionary |
| 2026-09 | PRISM-JAX re-implementation {cite}`abouagour2026prism`, Rician NLL, tortuosity coupling, Watson dispersion by Funk–Hecke, decoupled extra-cellular D | can a differentiable analysis-by-synthesis fit beat FORCE and PRISM on identical inputs | `dmipy_jax/prism`; doc 008 §6: 2.3–8.0° at crossings vs FORCE 9.7–16.6° |
| 2026-09 | Laplace fixel posterior (Gauss–Newton curvature), cone-coverage calibration, posterior-sampled tractography, CV-pruned connectomes, Laplace-evidence model selection | can a cheap local posterior do what SBI promised | doc 008 §9, §13; Dice 0.39 to 0.86; evidence selection a negative result |
| 2026-09 | CATERPillar {cite}`nguyenduc2025caterpillar` substrates, periodic SDF Monte Carlo walker, MCMRSimulator.jl {cite}`cottaar2026mcmrsimulator` and Permeable_MCDS parity | is the model misspecified on realistic tissue | `simulation/substrates`, `julia/mcmr`; three-engine table doc 008 §11.1; f_i bias is compartment attribution (§7.2) |
| 2026-09 | emulator-in-the-loop inversion over MC libraries | can a neural emulator replace the analytic kernel | doc 008 §11.2, r = 0.98 on f_intra; not productised |
| 2026-09 | Kärger exchange kernel (matrix exponential and Diffrax waveform ODE), Fisher / Bayesian-CRLB protocol design | can we design the protocol, not just fit it | `signal_models` exchange, `validation/design_*`; doc 008 §9.3, doc 009 |
| 2026-09/10 | HCP-YA 44-subject scan–rescan, AMICO-NODDI baseline, HCP-A protocol transfer, edge ICC | does any of it hold on real data | doc 008 §10.6, §12, §13; f_i on par with NODDI, FORCE behind |
| 2026-10 | package review, SI-only kernels, Callaghan fix, sybil tutorials, dmipy 2.x comparison {cite}`dmrailab2026dmipy` | what is the package, honestly | doc 010, doc 011, CHANGELOG |
| 2026-10 | CHUV realism ladder {cite}`nguyenduc2026ladder`: glia, myelin, beading, permeability; glial sphere pool; Watson–Kärger | does the posterior know when the model is wrong | doc 008 §14 |
| 2026-10 | spectral matrix-formalism pores {cite}`canales2026ring` | exact finite-pulse kernels for myelin water, disks, spheres | `signal_models/matrix_formalism.py`; doc 011 §6 |

Three patterns in that table.

- **Every detour that stayed was one that produced a measured comparison
  against someone else's method on shared data.** FORCE, PRISM, SBI_dMRI,
  MSMT-CSD, NODDI, MCMRSimulator, MCDS. The detours that did not (score
  posteriors, qMRI, ultrasound, scico) are archived or dormant.
- **The amortised-SBI route, which gives the project its name, is not the
  route that produced the headline results.** The flow is "calibrated but
  unresolved" (doc 008 §6.3, §7.1) and the dyadic parameterisation fixed the
  seam artefact without closing the gap; the hybrid flow-proposal driver
  showed the MAP does not need it (doc 008 §9.2). The results that beat the
  field came from a differentiable per-voxel fit with a Laplace posterior,
  which is closer to Microstructure.jl's and PRISM's stance than to
  Nottingham's or Alicante's. The name SBI4DWI therefore over-promises the
  amortised part and under-sells the differentiable-physics part.
- **The simulator side became an oracle federation rather than a single
  differentiable simulator.** The FEM matrix formalism of ADR 003 is still
  there, but the realism results run on our SDF walker checked against two
  external engines, with CATERPillar geometry. This is the "build where
  differentiable, wrap where not" principle of CLAUDE.md working as intended,
  and it is also why the project has no single simulator to publish.

## 4. The landscape, refreshed (2026-10-31)

Doc 004 compared three projects. The field in late 2026 has roughly fifteen
groups whose output changes what we should build. "Measured" means a number
in a decisions doc produced on shared inputs; "cited" means we know of it;
blank means neither.

| group | people | packages / datasets | key 2025–26 reference | what they have that we do not | our contact with it |
|---|---|---|---|---|---|
| Indiana / DIPY | Garyfallidis, Shah, Abouagour, Koudoro | FORCE in DIPY, PRISM | {cite}`shah2025force`, {cite}`abouagour2026prism` | DIPY distribution, the FORCE experiment suite datasets | measured: docs 004, 007, 008; `dipy_refine_peaks` plugs into their peaks |
| Nottingham CoNI + Tübingen Macke | Sotiropoulos, Manzano-Patrón, Gloeckler, Macke | SBI_dMRI, cuDIMOT, `sbi` | {cite}`manzanopatron2025sbi`, {cite}`gloeckler2026sbiprism`, {cite}`tejerocantero2020sbi` | joint model-structure + parameter posterior (their "PRISM"), the `sbi` library | measured: doc 008 §6.2–6.4 (SBI_dMRI); Gloeckler 2026 not yet run |
| Alicante CSIC-UMH | De Santis, Eggl | SBIDTI | {cite}`eggl2026sbi` | SBI for DTI/DKI/AxCaliber at minimal data | cited only (doc 004 §4); no code found since the preprint |
| MGH Yendiki | Gong, Yendiki | Microstructure.jl | {cite}`gong2026microstructurejl` | Julia MCMC + neural estimators, multi-TE and non-exchanging models | audited: doc 004 §23 |
| Oxford FMRIB | Cottaar | MCMRSimulator.jl, MRIBuilder | {cite}`cottaar2026mcmrsimulator` | multi-modal MC (diffusion, relaxation, off-resonance), Pulseq CLI | measured: doc 007 §8.5–8.6, doc 008 §11.1 (parity) |
| Lausanne CHUV / EPFL Jelescu + Thiran | Nguyen-Duc, Uhl, Rafael-Patiño, Brammerloh, Le Boeuf Fló | CATERPillar, OCTOPUS, MCDS, NEXI, graymatter_swissknife, reduced-NEXI data on Connectome 2.0 | {cite}`nguyenduc2025caterpillar`, {cite}`nguyenduc2026ladder`, {cite}`brammerloh2026octopus`, {cite}`uhl2026reducednexi`, {cite}`jelescu2022nexi` | realistic substrates, grey-matter exchange models, XAI protocol reduction | measured: doc 008 §11, §14; OCTOPUS unreleased (doc 006 §1.4) |
| Granada / Cardiff / EPFL | Canales-Rodríguez, Tax, Jones, Górriz | ring / spectral Laplacian kernels, Robust-Monte-Carlo-Simulations | {cite}`canales2026ring` | exact finite-pulse pore signals | ported: `matrix_formalism.py`, doc 011 §6 |
| Cardiff CUBRIC | Palombo, Jallais, Jones | μGUIDE, SANDI, WAND dataset | {cite}`jallais2024muguide`, {cite}`palombo2020sandi` | SBI with degeneracy detection, 300 mT/m human data | cited; WAND is on disk (`/data/raw/wand`), unused |
| NYU | Novikov, Fieremans, Coelho | Standard Model, SMI, RotInv | {cite}`novikov2019review`, {cite}`coelho2022smi` | the SM parameterisation the ladder paper validates | cited; SMI is the baseline in {cite}`nguyenduc2026ladder`, not run here |
| MGH Huang + Lee | Huang, Lee | Connectome 2.0 (ds006181, ds007739), GPA myelin-water kernel | {cite}`huang2021connectome2`, {cite}`lee2026gpa` | 500 mT/m human data, realistic-axon MC | cited; ds006181 identified for the Cima.X plan; GPA superseded by the exact ring kernel (Table II of {cite}`canales2026ring`) |
| UCSC Marinescu | Khole, Marinescu, with Ianus and Li | ReMiDi, Spinverse | {cite}`khole2025remidi`, {cite}`khole2026spinverse` | differentiable MC with permeability, mesh gradients | cited: doc 006 §5; no code released |
| Paris: Polytechnique / Inria / NeuroSpin | Li, Grebenkov; Poupon, Ginsburger | SpinDoctor, MEDUSA | {cite}`li2019spindoctor`, {cite}`ginsburger2019medusa` | FEM matrix formalism reference, sphere-packed phantoms | SpinDoctor is the ancestor of our FEM simulator; neither run |
| UCL | Alexander, Zhang, Drobnjak, Kerkelä | NODDI, MISST, Disimpy, AMICO-NODDI (with Verona) | {cite}`zhang2012noddi`, {cite}`kerkela2020disimpy`, {cite}`daducci2015amico` | the clinical baseline | measured: AMICO-NODDI on 42 HCP subjects, doc 008 §10.6 |
| dmrai-lab | Fick | dmipy 2.x (fit/sim/design), surface-relaxivity paper | {cite}`dmrailab2026dmipy`, {cite}`fick2026twoobservables` | matrix-method pores, MC replay compartments, sequence design; AGPL | compared: doc 011 |
| Munich TUM | Nurdinova, Karampinos | GPU JEMRIS | {cite}`nurdinova2025jemris` | Bloch-level sequence simulation | cited: doc 006 §5 |
| new in 2026, untriaged | Sen (microTorch), Li (cellular-level review), Wan (global inversion), Kitware (kwneuro) | {cite}`sen2026microtorch`, {cite}`li2026cellular`, {cite}`wan2026globalinversion`, {cite}`kitware2026kwneuro` | self-supervised PyTorch fitting; a whole-brain global inversion; a Python framework | not read |

Corrections to the list the user gave on 2026-10-30: the Nottingham group is
Nottingham, not Birmingham; dmipy belongs to Fick's dmrai-lab, not to
Lausanne; the Lausanne group's packages are CATERPillar, OCTOPUS, MCDS and
the NEXI family; the Oxford package is MCMRSimulator.jl (FDT is FSL's
tensor-fitting tool, unrelated).

Groups we must keep up with, in order of how often their output has changed
our plan: Lausanne (every substrate result), Indiana (every fixel result),
Nottingham/Tübingen (the SBI baseline and now model-structure inference),
Oxford (the oracle), MGH Huang and Cardiff (the only sources of >80 mT/m human
data), Fick (the closest package to ours). NYU matters as the model
definition, not as software.

## 5. What this implies

The project started as a port, became a benchmark lab, and now has one
result-bearing product (PRISM-JAX refine with a Laplace fixel posterior, as a
DIPY plug-in) and one platform (differentiable kernels + oracle substrates +
SBI/UQ tooling) that supports the lab but has not been packaged as a
contribution of its own.

Three honest options, which are not exclusive:

1. **Ship the product.** PRISM-JAX refine + `dipy_refine_peaks` + the HCP
   retest numbers are a paper and a DIPY contribution today. Nothing in §4
   does the same thing on the same inputs. This needs the deferred doc 010
   items and a DIPY-facing API freeze, not more research.
2. **Make the platform claim honest.** Either rename again (the SBI label
   describes the smaller half) or make SBI earn it: the Gloeckler 2026 model-
   structure posterior is the natural next baseline, and the exchange and
   glial compartments plus the realism ladder give us a model-selection
   problem where amortisation should pay. The ds008771 reduced-NEXI
   scan–rescan is the dataset to do it on.
3. **Keep touring, but write the tour up.** The techniques table in §3 and the
   negative results in doc 008 (§6.3, §7.1, §6.19) are of value to the groups
   in §4 and to anyone choosing a JAX SciML stack for inverse problems. That
   is a methods/software paper distinct from option 1.

Recommendation: 1 first, since it is nearly free; then 2 on ds008771 and
WAND, which also serves the Cima.X demonstration discussed on 2026-10-30.
Option 3 falls out of maintaining this document.

## 6. Housekeeping done with this document

- `docs/references.bib` grew from 6 to 51 entries (fetched from Crossref and
  arXiv on 2026-10-31; keys are `firstauthorYEARshort`). Not in it: MicroKAN
  (NeuroImage 2026, PII S1053811926003472, no DOI resolved) and the
  Nottingham SBI tractography preprint beyond its journal version.
- `docs/bibliography.md` now lists every entry, not only the cited ones.
- `docs/decisions/README.md` gained the 012 row; doc 004 points here.
- `jaxopt` is flagged above as a dependency with no importer; removal is left
  for the next clean-up pass rather than done in a docs commit.

## 7. The program to completion (2026-11-04)

Accepted on 2026-11-04: the repository is a product and a platform, and
"complete the program" means finishing both as separately installable things
and settling the SBI question with one decisive experiment. Steps 1 and 2 are
engineering with no research risk; step 3 is refactoring of code that runs;
step 4 is the one open research question and the one that decides the name.

### 7.1 Step 1: cut the wheel in two (begun 2026-11-04)

One repository, two distributions. A slim core that imports with jax,
equinox, optimistix, lineax, optax, numpy, scipy, nibabel and dipy, holding
the kernels, acquisition, fitting, the Laplace machinery and PRISM refine.
Everything else behind extras, later a second distribution.

Done today, name-independent:

- `pyproject.toml`: core dependency list 29 → 11; extras `cuda`, `sbi`,
  `sim`, `algebra`, `inverse`, `nn`, `viz`, `all` (plus the existing `data`,
  `baselines`, `test`, `doc`). `jax[cuda13]` moved to the `cuda` extra so CPU
  users install nothing GPU-specific. `jaxopt` and `annotated-types` removed
  (no importer for the latter; the former is now optional, imported lazily by
  the two bounded L-BFGS fitters and `models/fw_dti`).
- `dmipy_jax/__init__.py` loads submodules lazily, so `import dmipy_jax.prism`
  no longer imports diffrax, scico or flowjax.
- Verified in a fresh venv with the core dependencies only: see §7.5.

File-level map of what the second distribution will contain when the split
is made physical (not done; needs the name):

| stays in the core | moves to the platform distribution |
|---|---|
| `acquisition`, `constants`, `signal_models/`, `composer`, `cylinder`, `gaussian`, `sphere` | `simulation/` (walkers, FEM, substrates, scanner), `library/`, `biophysics/` |
| `fitting/` (VoxelFitter, Optimistix, brute init), `core/modeling_framework` | `pipeline/` (SBI train/deploy, SBC, PPC, conformal, OOD, ensembles), `inference/` |
| `prism/`, `workflows/refine_peaks_flow`, `io/fixel`, `io/bids` | `inverse/`, `algebra/`, `nn/`, `pinns/`, `reconstruction/`, `tractography/`, `bayesian/` |
| `tests/validation/test_prism_*` | `validation/` (paper scripts) and the oracle protocol of §7.3 |

Candidate names, for the user to decide; none contain "SBI", "dmipy" or "JAX":

- for the product: a word that says fixel + posterior (the refine stage is
  the only tool in §4 that reports a per-fixel σ);
- for the platform: a word that says compartments and pores, or analysis by
  synthesis;
- the import name `dmipy_jax` changes with the platform name, in the same
  commit as `scratchpad/rename_to_sbi4dwi.sh` is retired.

### 7.2 Step 2: finish the product and send it upstream

| item | state 2026-11-04 |
|---|---|
| ship the σ calibration | **done**: `prism_uncertainty.SIGMA_INFLATION_HCP_YA = 1.81` (± 0.15, 44 subjects, doc 008 §10.6) with provenance; `FixelPosterior.sigma_inflation`, `.sigma_deg_calibrated`, `.fintra_sd_calibrated`, `sample_dirs(calibrated=True)`; `refine_peaks(sigma_inflation=…)` defaults to the HCP-YA constant; the CLI writes `*_sigmatheta_calibrated.nii.gz`, the fixel directory gains `sigma_theta_deg_calibrated`, `*_model.json` records the factor. Raw σ is unchanged everywhere. Test added. |
| CPU path and runtime | `validation/benchmark_runtime_hardi.py --skip-force` on CPU, whole Stanford HARDI; result in §7.5 |
| API freeze | `refine_peaks`, `default_config`, `fi_config`, `FixelPosterior`, `write_fixel_directory`, `dipy_refine_peaks` are the public surface; no further signature changes without a CHANGELOG entry |
| DIPY contribution | open an issue on DIPY with the doc 008 §6.4 and §10.6 tables, then the PR; raw material in `docs/outreach/force-dipy-notes.md` |
| paper | doc 008 is the draft's results section |

### 7.3 Step 3: build the platform's missing skeleton

The oracle protocol in CLAUDE.md (`OracleSimulator`, registry,
`OracleModelSimulator`) does not exist. What exists: `SimulationLibrary`
(HDF5), `CATERPillarOracle`, and three working runners for CATERPillar, MCMRSimulator.jl
and Permeable_MCDS as scripts in `validation/`, plus the MC-library emulator
of doc 008 §11.2. Moving the three runners behind the ABC with the emulator
as the adapter turns scripts into the platform's defining feature: three
Monte Carlo engines behind one differentiable interface.

**Done 2026-11-04.** `dmipy_jax/simulation/oracle.py`: `PGSEScheme`,
`OracleResult`, `SubstrateOracle` (ABC with `simulate`, `generate_library`
→ `SimulationLibrary` HDF5 with measured descriptors and per-compartment
signals), `JaxWalkerOracle`, `MCMROracle` (stages the CSV/JSON interface of
`substrate_pgse.jl` under a workdir), `MCDSOracle` (geometry with periodic
images, scheme and config writers from `run_mcds_parity.py`), registry.
`dmipy_jax/simulation/emulator.py`: `LibraryEmulator` (ensemble MLP from
`emulator_inversion.py`) with `as_model_simulator` for `train_sbi`, `invert`,
`save/load`. The substrate geometry and walker moved into the wheel as
`simulation/substrates/spheres.py`; `validation/lib/substrate_benchmark.py`
re-exports it so the ladder, permeability and library scripts are unchanged.
Tests: `dmipy_jax/tests/test_oracle.py` (registry, physics sanity of the JAX
oracle on a grid of straight axons, library round-trip, emulator fit +
`ModelSimulator` adapter + persistence) and the existing substrate suites,
20 passed. Engine smoke run on the same tiny substrate recorded in §7.6.

### 7.4 Step 4: one result where amortisation wins, or the SBI label goes

Per-voxel MAP + Laplace beat the flow on every orientation task (doc 008
§6.3, §7.1, §9.2). The place the flow should win is model selection, where
Laplace evidence failed (§6.19): a joint posterior over model structure and
parameters across the realism-ladder compartments (stick, zeppelin, glial
sphere, exchange, myelin ring), benchmarked against
{cite}`gloeckler2026sbiprism`, on ds008771 scan–rescan and WAND. Needs the
WAND loader to read `t_bdel` and a multi-TE forward model with the ring
kernel. If the flow loses there too, the SBI tooling becomes a validation
utility and the name goes.

### 7.5 Step 5: close the January debts

Remove `scico` (done as an extra; the modules remain), decide ADR 002c
(close it: SI everywhere works), runtime-check the jaxtyping annotations,
retire or archive the dormant qMRI and score-posterior modules.

### 7.6 Numbers from the step-1/2 runs (2026-11-04)

Whole-brain Stanford HARDI runtime (`validation/benchmark_runtime_hardi.py`,
K = 3, 300 iterations, 154,625 mask voxels, 160 measurements; values copied
from `validation/runtime_hardi.json` (GB10, Slurm 1782) and
`validation/runtime_hardi_cpu.json` (20-core Grace CPU, `--skip-force`)):

| stage | GB10 GPU | CPU (20 cores) |
|---|---|---|
| CSD peaks (dipy, 6 processes) | 16.0 s | 6.6 s |
| PRISM-JAX refine | 30.5 s | 87.8 s (0.57 ms/voxel) |
| Laplace fixel posterior | 23.2 s | 31.2 s |
| FORCE library + fit (reference) | 241 + 172 s | not run |
| median main-peak CSD↔refined agreement | 6.2° | 6.2° |
| learned D∥ (µm²/ms), mean f_i | 1.33, 0.420 | 1.32, 0.420 |

A whole brain in two minutes on CPU means the product can ship as a plain
DIPY-compatible package with no GPU requirement; the GPU is a 3× convenience.

Core-only install check (`uv venv` + `uv pip install -e .` with no extras):
the environment holds 34 distributions; `import dmipy_jax.prism.dipy_refine`,
`dmipy_jax.workflows.refine_peaks_flow` and `dmipy_jax.signal_models` load
none of diffrax, scico, flowjax, blackjax, sympy, pandas, matplotlib, jaxopt,
healpy or numpyro (h5py arrives through dipy). `tests/validation/test_prism_*`
pass in that environment except the MSMT-CSD oracle test, which needs cvxpy
and now skips without it.

Oracle engines end to end (2026-11-04, `oracle_smoke.py` in the session
scratchpad; four straight sphere-chain axons r = 0.8 µm in a 6 µm box, 13
measurements b 0/1000/2500, δ 6 / Δ 12 ms, D 2 µm²/ms, 2000 walkers per
compartment; values copied from `oracle_smoke.json`). This is a wiring check,
not a parity study: the parity numbers on CATERPillar substrates are doc 008
§11.1.

| engine | wall time | S_intra, b 2500, ⊥ (|g_z| < 0.3) | S_extra, b 2500 | RMS vs JAX (intra / extra) |
|---|---|---|---|---|
| jax (CPU, dt 10 µs) | 9.8 s | 0.881 | 0.036 | — |
| mcds (8 threads, 5000 steps) | 0.9 s | 0.895 | 0.031 | 0.038 / 0.033 |
| mcmr (Julia 1.13, 8 threads, incl. compile) | 82 s | 0.994 | 0.023 | 0.396 / 0.020 |

MCMR's intra signal is the known over-restriction of overlapping-sphere
chains (doc 007 §8.6: the chain is treated as separate spheres, so axial
diffusion is blocked and the perpendicular signal is nearly 1); the
extra-cellular agreement to 0.02 is the same as on the real substrates. The
MCDS wrapper initially failed because MCDS selects its geometry parser by
file extension (`.swc`); fixed in the oracle.
