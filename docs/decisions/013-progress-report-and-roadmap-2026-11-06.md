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
