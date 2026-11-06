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

**Decided 2026-11-05: the platform is `compore` (compartments and pores), the
product is `lafix` (Laplace fixel).** Both were free on PyPI on 2026-11-04.
`scripts/rename_to_compore_lafix.sh` performs the move: `dmipy_jax` →
`compore`; `prism/`, `io/fixel.py` and the refine workflow → `packages/lafix/lafix/`
as a second distribution in a uv workspace, with `dipy_refine_peaks` and
`lafix` as its entry points. Historical logs (docs 001–012, CHANGELOG) keep
the old names. The GitHub repository and the working directory are renamed
by the user. The candidate list that led there:

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

#### 7.4.1 Groundwork (2026-11-05)

**Data.** ds008771 (CC0) is on disk at `/data/raw/openneuro/ds008771`:
the preprocessed derivatives of all 15 sessions (8 subjects, 7 with scan–
rescan; 17 GB) synced from the OpenNeuro S3 export, the small annexed
sidecars fetched from the OpenNeuro API (snapshot 1.0.5) because they are
absent from the S3 export. Gaps in the public export, verified by direct
object requests (AccessDenied): the preprocessed `.bvec` for sub-01 and
sub-02 (both sessions), the dwi-space DKT parcellation for sub-01 (both) and
sub-02 ses-01, and the transforms for those sessions. Powder-averaged models
do not need directions, so sub-01/02 remain usable for NEXI; the cortical
mask is missing for three sessions. Loader: `compore/io/nexi_c2.py`
(`load_scheme`, `load_dwi`, `load_dseg`, `cortex_mask`, `REDUCED_FEATURES`,
`fetch_small_files`, `available`).

**Scheme** (from the `.bval`/`.delta` files, δ = 5 ms per the paper, TE 66 ms,
TR 3.3 s, 2 mm): 513 volumes, 13 b0, 15 (b, Δ) features; the 8 XAI-selected
features plus b0 are 263 volumes. Counts × (b s/mm², Δ ms):
3×(0,12) 20×(1000,12) 30×(2500,12) | 2×(0,15) 32×(5000,15) | 2×(0,20)
34×(7350,20) | 3×(0,27) 20×(1000,27) 30×(2500,27) 32×(5000,27) 34×(7500,27)
44×(10000,27) | 3×(0,45) 20×(1000,45) 30×(2500,45) 32×(5000,45) 34×(7500,45)
44×(10000,45) 64×(12500,45).

**Forward models** (`compore/signal_models/gm_powder.py`): powder-averaged
NEXI (Kärger stick ⇄ ball, closed-form 2×2 exponential per orientation,
64-node Gauss–Legendre over cos θ), SANDI (stick + GPD sphere + ball) and
NEXI + impermeable soma, plus a Rician-mean helper and the prior boxes of
Uhl 2026. Tests (7): stick powder vs quadrature 1e-8; NEXI slow-exchange
limit = stick + ball (1e-4, quadrature), fast-exchange limit = single
Gaussian (1e-4); per-orientation agreement with the WM Kärger kernel
(`validation/lib/prism_exchange.karger_pgse`, 1e-6); **agreement with the
authors' reference integrand** (graymatter_swissknife `M`, Apache-2.0,
re-expressed in SI with `scipy.integrate.quad_vec`) to rtol 2e-5; SANDI and
NEXI+soma reductions; gradients finite under jit. Installing the reference
package itself was not permitted in this session; its formula was read from
the public source.

**WAND**: the sidecars carry Δ only as the rounded string `t_bdel` and no δ;
`compore/io/wand.wand_timings` now parses the exact Δ from
`SeriesDescription` (17.3 / 30 / 42 / 55 ms) and leaves δ as a constant to be
set from the WAND documentation (`WAND_AXCALIBER_SMALL_DELTA_S`, None → NaN).

**Experiment design.** Three candidate GM models M ∈ {NEXI, SANDI,
NEXI+soma} with the prior boxes above. Per cortical voxel, powder averages
over the 15 (b, Δ) shells (`validation/nexi_c2_powder.py` →
`/data/datasets/compore/ds008771/powder/<sub>_<ses>.npz`, with the b0 noise
sd per voxel). Three estimators of the model posterior and parameters:

1. NLLS per model + Laplace evidence (the lafix machinery on powder data);
2. amortised joint posterior p(M, θ | S) from one flow trained on simulations
   of all three models (the Gloeckler 2026 stance, our implementation);
3. the authors' point estimate (NEXI NLLS) as the reference parameter map.

Scored on: (a) scan–rescan reproducibility of p(M | S) and of θ under M
(Bland–Altman as in the paper, plus ICC across the 7 rescanned subjects),
(b) posterior predictive checks on held-out shells (fit the 8 reduced
features, predict the 7 others), (c) simulation-based calibration of both
posteriors, (d) agreement of estimator 2 with estimator 1 where the evidence
is decisive. Amortisation "wins" if (b) and (c) hold where the Laplace
evidence was indecisive on the realism ladder (doc 008 §6.19), and the model
probabilities are at least as reproducible as the parameters.

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

#### 7.4.2 Results (2026-11-06)

Numbers copied from `validation/nexi_c2_model_selection.json` and
`validation/wand_model_selection.json` (driver
`validation/nexi_c2_model_selection.py`, batched estimators of doc 013 §6.1;
whole study 17 min on the GB10: ds008771 9 min, WAND 8 min). Training:
300 000 joint simulations, classifier 4000 steps, flows 6000 steps, SNR drawn
from the data's own range (ds008771 9.5–82.5, WAND 15.1–75.6). Test: 400
simulated voxels, Laplace on the first 120 (0.15 s per voxel batched, was
3.5 s serial).

**Phase A, simulated truth**

| | ds008771 (15 features, δ 5 ms, Δ 12–45 ms) | WAND (8 features, δ 7 ms, Δ 17.3–55 ms) |
|---|---|---|
| amortised 3-way accuracy | 0.780 (0.767 on the Laplace subset) | 0.745 (0.750 on the subset) |
| Laplace-evidence accuracy (n = 120) | 0.608 | 0.450 |
| confusion, rows truth NEXI / SANDI / NEXI+soma | 121 2 3 / 1 115 22 / 12 48 76 | 120 0 6 / 1 115 22 / 26 47 63 |
| reliability, top bin p ≥ 0.87 | acc 0.937 (n 159) | acc 0.910 (n 111) |
| reliability, bin 0.46–0.60 | acc 0.453 (n 75) | acc 0.551 (n 78) |
| SBC KS p, 15 parameters | all ≥ 0.10 | all ≥ 0.30 except NEXI t_ex 0.02 |

Parameter recovery (RMSE; 90 % coverage), amortised / Laplace:

| model, param | ds008771 RMSE | ds008771 cov90 | WAND RMSE | WAND cov90 |
|---|---|---|---|---|
| NEXI f | 0.061 / 0.062 | 0.754 / 0.917 | 0.072 / 0.086 | 0.810 / 1.000 |
| NEXI t_ex (ms) | 12.0 / 11.7 | 0.857 / 0.917 | 13.2 / 14.6 | 0.849 / 0.972 |
| NEXI D_i (µm²/ms) | 0.44 / 0.45 | 0.857 / 0.944 | 0.53 / 0.66 | 0.817 / 1.000 |
| SANDI f_n | 0.048 / 0.038 | 0.884 / 0.933 | 0.088 / 0.098 | 0.884 / 1.000 |
| SANDI f_s | 0.022 / 0.018 | 0.870 / 0.978 | 0.061 / 0.074 | 0.862 / 0.933 |
| SANDI R_s (µm) | 0.91 / 1.12 | 0.906 / 0.911 | 1.29 / 2.51 | 0.913 / 0.956 |
| NEXI+soma f_n | 0.064 / 0.084 | 0.904 / 0.949 | 0.086 / 0.077 | 0.882 / 0.974 |
| NEXI+soma f_s | 0.029 / 0.028 | 0.853 / 0.923 | 0.058 / 0.100 | 0.897 / 0.949 |
| NEXI+soma t_ex (ms) | 17.7 / 22.4 | 0.904 / 0.846 | 18.2 / 22.4 | 0.897 / 0.897 |
| NEXI+soma R_s (µm) | 1.34 / 1.52 | 0.860 / 0.897 | 1.63 / 2.57 | 0.890 / 0.897 |

Reading: on both schemes the classifier beats the Laplace evidence on model
identity by 17 and 30 points; its flows recover parameters as well as or
better than the Laplace fit and are SBC-calibrated (one marginal KS
rejection in 30 tests). The Laplace intervals are too wide (coverage
0.92–1.00 at nominal 0.90), the flow's slightly too narrow for NEXI f
(0.75–0.81). The errors are the physically expected ones: NEXI+soma ↔
SANDI when t_ex is long, NEXI+soma → NEXI when f_s is small.

**Phase B, data: voxel-wise amortised posterior**

ds008771, 11 sessions (sub-08 ses-01 skipped: reduced protocol only),
54 866–62 672 cortical voxels each; arg-max share NEXI / SANDI / NEXI+soma
and mean p:

| session | arg-max share | mean p | session | arg-max share | mean p |
|---|---|---|---|---|---|
| sub-02 ses-02 | 0.659 / 0.009 / 0.332 | 0.599 / 0.013 / 0.388 | sub-05 ses-02 | 0.625 / 0.049 / 0.326 | 0.580 / 0.053 / 0.366 |
| sub-03 ses-01 | 0.675 / 0.038 / 0.287 | 0.621 / 0.041 / 0.338 | sub-06 ses-01 | 0.694 / 0.020 / 0.287 | 0.637 / 0.023 / 0.340 |
| sub-03 ses-02 | 0.679 / 0.019 / 0.302 | 0.629 / 0.023 / 0.348 | sub-06 ses-02 | 0.695 / 0.019 / 0.287 | 0.632 / 0.023 / 0.344 |
| sub-04 ses-01 | 0.630 / 0.030 / 0.340 | 0.580 / 0.033 / 0.387 | sub-07 ses-01 | 0.606 / 0.024 / 0.370 | 0.568 / 0.031 / 0.401 |
| sub-04 ses-02 | 0.706 / 0.028 / 0.266 | 0.643 / 0.030 / 0.326 | sub-07 ses-02 | 0.603 / 0.021 / 0.375 | 0.562 / 0.028 / 0.410 |
| sub-05 ses-01 | 0.681 / 0.033 / 0.286 | 0.629 / 0.038 / 0.333 | WAND sub-08033 | 0.808 / 0.086 / 0.106 | 0.668 / 0.088 / 0.244 |

**Phase B, data: 62 DKT ROIs per session, both estimators on the ROI median
signal**

ds008771, ranges over the 11 sessions: ROI-mean p(NEXI / SANDI / NEXI+soma)
amortised 0.54–0.73 / 0.08–0.23 / 0.15–0.23; Laplace on the full 15
features 0.00–0.02 / 0.00–0.08 / 0.92–1.00; Laplace on the 8 reduced
features 0.00–0.03 / 0.00–0.04 / 0.95–1.00. ROIs where the Laplace evidence
picks NEXI+soma: 57–62 of 62 in every session. Held-out PPC (fit the 8
reduced features, predict the other 7): median RMSE over ROIs NEXI
0.0046–0.0060, SANDI 0.0076–0.0096, NEXI+soma 0.0033–0.0045; NEXI+soma is
the best predictor in 37–57 of 62 ROIs, NEXI in 5–23, SANDI in 0–3. The
reduced scheme gives the same ROI decision as the full one.

WAND (one session, no held-out split): ROI-mean p amortised
0.213 / 0.235 / 0.552, Laplace 0.047 / 0.361 / 0.593; ROI arg-max counts
amortised 13 / 7 / 42, Laplace 2 / 21 / 39. On the second scheme the two
estimators agree at the ROI level on NEXI+soma.

NEXI parameters at the ROI level (medians over ROIs, Laplace / amortised):

| | ds008771 (range over sessions) | WAND |
|---|---|---|
| f | 0.350–0.375 / 0.277–0.305 | 0.320 / 0.317 |
| t_ex (ms) | 34.2–48.9 / 52.8–67.7 | 70.0 / 68.2 |
| D_i (µm²/ms) | 3.50 / 3.25–3.34 | 2.80 / 2.86 |
| D_e (µm²/ms) | 0.85–0.93 / 0.85–0.94 | 0.88 / 0.86 |

Two things in this table are prior-box effects, not biology: the Laplace
D_i sits at the box top (3.5) in every ds008771 ROI, so its Laplace
approximation (and evidence) is invalid there, and the WAND t_ex sits at the
box top (70 ms) for both estimators. The boxes (`gm_powder.PRIOR_BOXES`)
need widening before any parameter from this run is quoted.

**Scan–rescan, ds008771, 5 subjects × 62 ROIs (n = 310)**

| quantity | mean | bias | LoA (1.96 sd) | ICC |
|---|---|---|---|---|
| p(NEXI) amortised | 0.662 | −0.005 | 0.436 | **0.801** |
| p(NEXI) Laplace | 0.003 | 0.006 | 0.114 | 0.006 (no variance: p ≈ 0 everywhere) |
| t_ex Laplace (ms) | 40.4 | 2.20 | 16.4 | 0.935 |
| t_ex amortised (ms) | 52.7 | 1.34 | 11.3 | **0.937** |
| f Laplace | 0.382 | −0.007 | 0.110 | 0.783 |
| f amortised | 0.297 | 0.002 | 0.031 | **0.881** |
| D_i Laplace (µm²/ms) | 3.43 | −0.012 | 0.232 | 0.882 |
| D_i amortised | 3.22 | −0.027 | 0.158 | **0.927** |
| D_e Laplace | 0.882 | −0.001 | 0.100 | 0.893 |
| D_e amortised | 0.865 | 0.003 | 0.049 | **0.940** |

**Against the §7.4.1 criterion and the doc 013 §6.3 decision rule**

- (a) holds on both schemes: classifier accuracy above the Laplace evidence
  (0.78 vs 0.61; 0.75 vs 0.45), SBC uniform (one p = 0.02 in 30 tests).
- (b) half holds. The amortised p(NEXI) is far more reproducible than the
  Laplace p(NEXI) (ICC 0.80 vs 0.01), and every amortised parameter is more
  reproducible than its Laplace counterpart (ICC +0.00 to +0.10, LoA 1.5–3.5×
  narrower). But the ICC of p(NEXI) (0.80) is 0.14 below the ICC of t_ex
  (0.94), outside the pre-registered 0.1 margin, and its LoA of ±0.44 on a
  0–1 scale is wide.
- The two estimators disagree on the model at the ROI level on ds008771
  (amortised NEXI, Laplace and PPC NEXI+soma) and agree on WAND. Part of the
  ds008771 disagreement is an evaluation artefact: the classifier is given
  the ROI noise level clipped to the training range (SNR 82.5) while the
  true ROI-median SNR is in the hundreds, and an ROI median over
  heterogeneous voxels is not a single-voxel signal. The Laplace side has its
  own defect (D_i at the bound). Neither estimator's ROI verdict is clean.

What this settles: amortised joint model + parameter inference is the better
estimator of p(M) and of θ in vivo on this problem, by every measure that
compares the two estimators on the same data. What it does not settle is
whether a per-voxel model probability is reproducible enough to report as a
map; ICC 0.80 is "good" by the usual scale and below the parameters. The
name decision (doc 013 §4.1 item 4) is Morgan's; the recommendation recorded
here is to keep `compore.inference.model_selection` as the platform's
flagship with the model-probability map reported at ROI resolution, and to
re-run after widening the D_i and t_ex boxes and training the classifier
over the ROI SNR range.

Artefacts: `validation/{nexi_c2,wand}_model_selection.json`; per-session maps
(`p_model`, `theta_med`, `label`, `ijk`) in
`/data/datasets/compore/{ds008771,wand}/model_selection/`; WAND powder data
and dwi-space DKT labels in `/data/datasets/compore/wand/powder/`.
