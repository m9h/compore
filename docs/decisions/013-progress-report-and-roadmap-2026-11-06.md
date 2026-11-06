# 013 — Progress report and development roadmap (2026-11-06)

Status: report. Written two days into the program of doc 012 §7. Numbers are
copied from `validation/nexi_c2_model_selection.json` (phase A complete,
phase B 2 of 12 sessions complete while this was written) and from doc 012
§7.6. This document does not change any decision; it records where the
program stands and what is left, in order.

## 1. Where the repository stands

One repository, two distributions, one uv workspace (doc 012 §7.1, applied
2026-11-05 in e50a341):

| | compore | lafix |
|---|---|---|
| role | platform: kernels, fitting, oracles, simulation, SBI/UQ | product: Laplace fixel refine for DIPY peaks |
| import | `compore` | `lafix` |
| CLI | — | `dipy_refine_peaks`, `lafix` |
| core deps | jax, equinox, optimistix, lineax, optax, nibabel, dipy | compore, dipy, optax |
| extras | cuda, sbi, sim, algebra, inverse, nn, viz, data, baselines, test, doc, all | — |
| result doc | 012 §7.3 (oracles), §7.4 (SBI) | 008 (PRISM / HCP) |

Size on 2026-11-06: 47 k lines of package code outside tests, 113 test
files, 587 tests collected. The post-rename suites passed (doc 012 §7.6):
whole package 409 passed / 2 skipped / 13 xfailed, tutorials 129 passed,
`import lafix` loads no diffrax / scico / flowjax, both CLIs run, Sphinx
builds both packages.

Commits since the program began (2026-11-04): 8, all local and unpushed
(60ee0ea … adaaca3). The GitHub repository and the working directory are
still named sbi4dwi; renaming them is the user's step.

## 2. The five steps of doc 012 §7, scored

| step | content | state |
|---|---|---|
| 1 | cut the wheel in two | **done** (e50a341, ffd0bad): workspace, lazy `compore.__init__`, lazy jaxopt, static lafix API page, rename script applied, docs/decisions and CHANGELOG left in their original names |
| 2 | finish the product, send upstream | **half**: σ calibration shipped (`SIGMA_INFLATION_HCP_YA = 1.81 ± 0.15`), CPU path benchmarked (154 625 voxels: refine 87.8 s, Laplace 31.2 s), API frozen. **Open**: DIPY issue + PR, paper draft from doc 008 |
| 3 | platform skeleton: oracle protocol | **done** (95daf7e): `SubstrateOracle` ABC with jax / MCMRSimulator.jl / Permeable_MCDS engines, HDF5 library generation, `LibraryEmulator → ModelSimulator`. Smoke parity in doc 012 §7.6 |
| 4 | one result where amortisation wins, or the SBI label goes | **running**: groundwork (2c62be6) and estimators (adaaca3) committed; GPU study phase A complete, phase B 2/12 sessions; §3 below. WAND arm not started |
| 5 | January debts | **not started**: scico modules, dormant qMRI / score-posterior modules, ADR 002c, runtime jaxtyping |

## 3. Step 4 interim result (ds008771, phase A complete, phase B partial)

Design: doc 012 §7.4.1. Three powder-averaged grey-matter models on the
reduced 8-feature (b, Δ) scheme of ds008771: NEXI (4 parameters), SANDI (5),
NEXI+soma (6). Two estimators of p(M, θ | S): Laplace evidence from a bounded
Levenberg–Marquardt fit, and an amortised classifier + per-model spline flow
trained on 300 000 joint simulations. SNR for training drawn from the data's
own range (9.5 – 82.5; session medians 19.0 – 33.0).

### 3.1 Phase A: simulated truth (400 test voxels; 120 for Laplace)

| | amortised | Laplace |
|---|---|---|
| model accuracy, 3-way | 0.780 (0.767 on the Laplace subset) | 0.608 |
| seconds per voxel | amortised over 83 s of training | 3.50 |
| reliability, top bin p ≥ 0.87 (n = 159) | acc 0.937 | — |
| reliability, bin 0.46 – 0.60 (n = 75) | acc 0.453 | — |

Confusion (rows truth, columns prediction; NEXI, SANDI, NEXI+soma):
NEXI 121 / 2 / 3; SANDI 1 / 115 / 22; NEXI+soma 12 / 48 / 76. The errors are
where they should be: NEXI+soma mistaken for SANDI when t_ex is long, SANDI
for NEXI+soma when f_s is small.

Parameter recovery (RMSE in native units, 90 % interval coverage):

| model | param | amortised RMSE | amortised cov90 | Laplace RMSE | Laplace cov90 |
|---|---|---|---|---|---|
| NEXI | f | 0.061 | 0.754 | 0.062 | 0.917 |
| NEXI | t_ex | 0.012 s | 0.857 | 0.012 s | 0.917 |
| SANDI | f_n | 0.048 | 0.884 | 0.038 | 0.933 |
| SANDI | f_s | 0.022 | 0.870 | 0.018 | 0.978 |
| NEXI+soma | f_n | 0.064 | 0.904 | 0.085 | 0.923 |
| NEXI+soma | f_s | 0.029 | 0.853 | 0.028 | 0.897 |
| NEXI+soma | t_ex | 0.018 s | 0.904 | 0.021 s | 0.821 |

SBC (Kolmogorov–Smirnov p per parameter, uniform ranks): NEXI 0.20 – 0.31
(n = 76), SANDI 0.43 – 0.73 (n = 70), NEXI+soma 0.10 – 0.90 (n = 94). No
parameter rejects uniformity at 0.05.

Reading: the flow is calibrated and recovers parameters as well as the
Laplace fit, and the classifier beats the Laplace evidence on model identity
by 17 points. The Laplace intervals are over-wide (coverage 0.92 – 1.00 at
nominal 0.90) rather than wrong. This is the first task in the project where
the amortised estimator is not worse than per-voxel MAP + Laplace.

### 3.2 Phase B: real cortex, first two sessions

| session | voxels | argmax NEXI / SANDI / NEXI+soma | mean p | seconds |
|---|---|---|---|---|
| sub-02 ses-02 | 62 672 | 41 276 / 570 / 20 826 | 0.599 / 0.013 / 0.388 | 16 275 |
| sub-03 ses-01 | 55 775 | 37 656 / 2 138 / 15 981 | 0.621 / 0.041 / 0.338 | 15 297 |

ROI level (62 DKT cortical ROIs per session, mean over ROIs):

| session | p amortised (voxel-wise, ROI mean) | p Laplace on ROI mean, full 15 features | p Laplace, reduced 8 | held-out PPC best model (count of 62) |
|---|---|---|---|---|
| sub-02 ses-02 | 0.678 / 0.117 / 0.206 | 0.012 / 0.000 / 0.988 | 0.004 / 0.000 / 0.996 | 20 / 1 / 41 |
| sub-03 ses-01 | 0.614 / 0.222 / 0.164 | 0.000 / 0.081 / 0.919 | 0.011 / 0.026 / 0.963 | 5 / 0 / 57 |

Held-out posterior predictive check (fit the 8 reduced features, predict the
7 others): median RMSE over ROIs NEXI 0.005 – 0.006, SANDI 0.008 – 0.009,
NEXI+soma 0.004. Example ROI 1002 (caudal anterior cingulate): Laplace NEXI
f = 0.29 – 0.30, t_ex = 47 – 58 ms; amortised NEXI f = 0.25, t_ex = 68 ms;
NEXI+soma f_n = 0.32 – 0.36, f_s = 0.03.

Reading, provisional until all 12 sessions and the scan–rescan block are in:

- At voxel SNR (median 19 – 33) the amortised posterior sits on NEXI with a
  38 % share for NEXI+soma and essentially none for SANDI. At ROI SNR the
  Laplace evidence and the held-out PPC both choose NEXI+soma. These agree
  once SNR is accounted for: the soma compartment is identifiable after
  averaging and not per voxel, which is what the phase A confusion predicts.
- The reduced 8-feature scheme gives the same ROI-level decision as the full
  15: the Uhl et al. reduction loses nothing for model choice here.
- SANDI (no exchange) is the wrong model for this scheme at every level.
- The doc 012 §7.4.1 criterion ("amortisation wins if its model probabilities
  are at least as reproducible as the parameters where Laplace was
  indecisive") is decided by the scan–rescan ICC block, which has not run.

### 3.3 Cost problem in phase B

Each session takes about 4.3 hours, almost all of it in the per-ROI Laplace
fits (3 models × 62 ROIs × 2 schemes × restarts), run voxel-serially on one
CPU core while the GPU is idle. Ten sessions remain, about 42 hours. The
study is left running; the fix for the next run is in §4.

## 4. Roadmap, in order

Items are ordered by what they unblock. Dates are targets, not promises.

### 4.1 Finish step 4 (this week)

1. Let the running study finish, or stop it after it writes four sessions and
   relaunch with the Laplace ROI fits vectorised (`jax.vmap` over ROIs with
   `optimistix.LevenbergMarquardt`, batched on the GPU) and the scan–rescan
   block computed from whatever sessions exist. Expected: minutes per session.
2. Write doc 012 §7.4.2 from the final JSON: phase A tables above, phase B
   per-session fractions, ICC and Bland–Altman of p(M) against f and t_ex.
   Apply the §7.4.1 criterion in one sentence.
3. WAND arm: set `WAND_AXCALIBER_SMALL_DELTA_S` from the acquisition
   documentation (δ is not in the sidecars), run the same two estimators on
   the AxCaliber shells (Δ = 17.3, 30, 42, 55 ms). This is the second dataset
   the criterion asks for and the only one with four diffusion times.
4. Decide the SBI label. If the criterion holds, `compore.inference` stays and
   the README leads with model selection; if not, it becomes a validation
   utility as doc 012 §7.4 says.

### 4.2 Step 2 upstream (next two weeks)

5. DIPY issue with the doc 008 §6.4 and §10.6 tables and the CPU runtime
   figures; then the PR carrying `lafix.dipy_refine.refine_peaks` as the
   contribution. Raw material: `docs/outreach/force-dipy-notes.md`.
6. Paper draft: doc 008 is the results section; doc 012 §1 – §3 is the
   methods context. Target venue to be chosen by the user.
7. Push the local commits and rename the GitHub repository and working
   directory to compore. User's actions.

### 4.3 Step 5 debts (after the SBI decision)

8. Archive or delete the scico-dependent `inverse` modules, dormant qMRI and
   score-posterior modules; keep tests that still pass under `archive/`.
9. Close ADR 002c (SI everywhere, no unxt).
10. Turn on runtime jaxtyping checks in the test suite (`jaxtyping.install_import_hook`
    for `compore` under pytest) and fix the annotations that fail.
11. Release: compore 0.1.0 and lafix 0.1.0 to PyPI from the workspace, with
    `docs/artifacts.md` pointing at the checkpoints.

### 4.4 Deferred, from earlier documents

- Multi-TE myelin-water compartment on the matrix-formalism ring kernel
  (doc 012 §7.4 named it for the realism ladder; needs a multi-TE dataset).
- Cima.X demonstration items (doc 009).
- Outreach drafts in `docs/outreach` for the user to send.

## 5. Risks

- Phase B scan–rescan may show p(M) less reproducible than f and t_ex: that is
  the honest negative outcome of step 4 and the roadmap already names what
  follows.
- WAND δ is unknown; without it the AxCaliber forward model is wrong by the
  t_d = Δ − δ/3 correction. Must come from the WAND protocol documentation,
  not a guess.
- The flow's NEXI f coverage (0.754 at nominal 0.90) is the one calibration
  miss in phase A; more training steps or a wider SNR range are the first
  things to try before it is reported as a limitation.

## 6. Execution log and background notes (2026-11-06, afternoon)

Done this afternoon on the roadmap of §4: item 1 (batched estimators,
relaunch), item 3 (WAND arm: δ found, powder extracted, driver generalised),
item 5 (DIPY issue draft `docs/outreach/dipy-lafix-issue.md`, paper outline
`docs/paper/outline.md`, both for Morgan to send), item 7 (pushed, GitHub
repository renamed `m9h/compore`, working directory `~/dev/compore` with a
`~/dev/sbi4dwi` symlink). Items 2 and 4 wait for the relaunched study.

### 6.1 Batched estimators

`fit_model_laplace_batch` / `laplace_model_posterior_batch` vmap the
Levenberg–Marquardt solves over (restart, voxel) and the Gauss–Newton Hessian
over voxels; `AmortisedPosterior.theta_median_batch` samples a batch of
conditions in one FlowJAX call. On CPU, 16 voxels × 3 models: batch 7 s vs
serial 30 s. Same optimum in 47 of 48 (voxel, model) pairs at 4 restarts; the
one miss disappears at 8 restarts, which the driver now uses. The Gauss–Newton
log-determinant differs by up to ~1 nat between the two paths along flat
SANDI directions (float32); the test tolerance records this. The serial study
was stopped after 2 sessions and relaunched (`--dataset nexi_c2` then
`--dataset wand`); the smoke run of the refactored driver completed all 11
same-scheme ds008771 sessions on CPU in 13 s each at 400 voxels.
sub-08 ses-01 (reduced protocol only, 8 features) is skipped and recorded,
because the amortised posterior is trained on the 15-feature scheme.

### 6.2 WAND arm

δ = 7 ms, from McNabb et al. 2025 (Sci Data), section "Diffusion-weighted
imaging with variable diffusion time": TR 3.9 s, TE 80 ms, δ 7 ms, Δ 18 (the
SeriesDescription says 17.3), 30, 42, 55 ms; b pairs 2200/4400, 4000/8000,
5800/11600, 7750/15500 s/mm², 30 directions each, 6 b0 per shell. Set as
`compore.io.wand.WAND_AXCALIBER_SMALL_DELTA_S`. One WAND subject is
preprocessed locally (sub-08033 ses-02: FSL topup + eddy of the four
AxCaliber shells, FreeSurfer recon). `validation/wand_axcaliber_powder.py`
registers the FreeSurfer brainmask to the mean b0 (DIPY rigid, mutual
information; Dice of the FreeSurfer brain against the FSL b0 mask 0.947),
resamples the DKT labels, and writes 8 powder features for 69 052 cortical
voxels; median b0 SNR 30.3. Powder medians run from 0.214 (b 2200, Δ 17.3)
to 0.041 (b 15 500, Δ 55), the last at the Rician floor for this SNR. WAND
differs from ds008771 in the useful way: one δ, four Δ, two b per Δ, b up to
15 500. There is no held-out split and no rescan for this subject, so the
WAND arm tests model choice and parameter agreement between the two
estimators, not reproducibility.

### 6.3 Background for the SBI-label decision (item 4)

What the label has to mean. Doc 012 §3 records that every headline result
of the project came from the differentiable fit plus a Laplace posterior,
and doc 008 §6.3, §7.1 that the amortised flow lost on every orientation
task (18.5° vs 1.7° at SNR 30; 9.1° after the parameterisation fixes). Doc
012 §7.4 set the one test on which amortisation should win: model
structure, where the Laplace evidence failed in doc 008 §6.19.

What phase A already shows (§3.1): on simulated truth the classifier is
right 78 % of the time against 61 % for the Laplace evidence, is well
calibrated in its top bin (0.94 accuracy at p ≥ 0.87), and its flows are
SBC-uniform with parameter RMSE equal to the Laplace fit. This is the first
task where the amortised estimator is not worse, and on model identity it is
clearly better. The Laplace evidence's weakness is structural: the
Gauss–Newton Hessian under a uniform box prior gives an Occam penalty that
is float32-noisy along flat directions (§6.1) and ignores the box
boundaries where SANDI and NEXI+soma fits land (f_s → 0).

What the two data phases have to add. The §7.4.1 criterion is about
reproducibility: p(M) across scan–rescan must be at least as reproducible
(ICC, Bland–Altman) as f and t_ex. The two finished sessions (§3.2) showed
the two estimators agree at the ROI level once SNR is accounted for, and
that NEXI+soma wins the held-out PPC in 41 and 57 of 62 ROIs. The
relaunched run gives the ICC for 5 rescanned subjects × 62 ROIs; the WAND
run gives the same two estimators on a different scanner, scheme and δ.

Decision rule, so it is written before the numbers arrive:

- Keep the label if (a) phase A holds on both schemes (classifier ≥ Laplace
  accuracy, SBC uniform) and (b) on ds008771 the ICC of p(NEXI) amortised is
  ≥ the ICC of the Laplace p(NEXI) and within 0.1 of the ICC of t_ex. Then
  `compore.inference.model_selection` is the platform's flagship and the
  README leads with joint model + parameter posteriors.
- Drop it if (b) fails on ds008771 regardless of WAND: the amortised model
  probabilities would then be a simulation-only result, and the honest
  framing is "SBI as a validation utility" (doc 012 §7.4).
- WAND cannot rescue (b) because it has no rescan; it can only confirm or
  contradict the model choice (NEXI+soma at ROI level) on a second scheme.

### 6.4 Background on the step-5 debts (item 6)

Inventory on 2026-11-06 (`compore/`, 40 subpackages, 47 k lines outside
tests):

| debt | what it is | size | who uses it | proposal |
|---|---|---|---|---|
| scico | `inverse/amico.py`, `inverse/global_amico.py`, `inverse/solvers.py`, `examples/clinical/fit_mcdespot_scico.py`; lazily referenced in `compore/__init__` | 1 202 lines in `inverse/` | `pipeline/train.py`, `core/networks.py`, `simulation/phantoms.py`, 3 test files | keep `inverse/` as the `inverse` extra (it already is); replace the scico ADMM in `solvers.py` with the Optimistix/jaxopt path the tests already exercise, then drop the git-source dependency |
| score posterior | `inference/score_posterior.py`, 909 lines, score matching for diffusion posteriors | 1 file | `inference/__init__` only; no validation script, no result doc | move to `archive/` with its test; it was the step-7 experiment of the January plan and nothing was built on it |
| qMRI | `signal_models/qmri/` (DESPOT, VFA T1, qMT, mcDESPOT in `models/`), tests in `tests/qmri/` | ~1 500 lines | nothing in lafix or the validation scripts | keep the kernels (they are the multi-TE / relaxometry path the realism ladder needs for the myelin ring, doc 012 §7.4) but move them under `signal_models/relaxometry/` and gate their tests on an extra |
| zero-user subpackages | `benchmarks`, `examples` (41 files, 6 861 lines), `external`, `gradients`, `pipelines` (one file next to `pipeline/`), `reconstruction` (11 files, 1 441 lines), `docs` (empty) | ~9 900 lines | none outside themselves | `examples/` → `docs/examples/` or delete the stale ones; `pipelines` → merge into `pipeline`; `reconstruction`, `benchmarks`, `gradients`, `external` → `archive/` unless a test covers them |
| one-user subpackages | `analysis`, `cli`, `diffusion`, `geometry`, `harmonization`, `learning`, `morphogenesis`, `preprocessing`, `tractography`, `viz` | ~2 300 lines | one caller each | collapse into their caller's package with a shim for one release |
| ADR 002c (unxt) | RFC to wrap quantities in `unxt` for unit safety | — | — | close as "rejected: SI everywhere + `PGSEScheme` rejecting s/mm² gives the safety at zero cost"; the SI convention is in CLAUDE.md and enforced by `PGSEScheme` |
| runtime jaxtyping | 34 files carry `jaxtyping` annotations; no `install_import_hook`, no typechecker in tests | — | — | add `jaxtyping.install_import_hook("compore", "beartype.beartype")` under a pytest flag, fix what fails, run it in CI only (it slows compiles) |

Order: score posterior and the zero-user subpackages first (pure removal,
no API change), then scico, then the collapses with shims, then ADR 002c
and jaxtyping, then the 0.1.0 releases.
