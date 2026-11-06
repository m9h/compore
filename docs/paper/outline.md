# Paper outline: lafix — calibrated per-fixel posteriors for diffusion MRI tractography

Drafted 2026-11-06 (doc 013 §4.2 item 6). Results section = doc 008; methods context = doc 012 §1–3. Target venue: Morgan's choice (Imaging Neuroscience, MRM, or NeuroImage). Numbers below are copied from the decision docs; do not re-derive.

## Title candidates
- Calibrated per-fixel uncertainty for any peaks: a differentiable refinement step with a Laplace posterior
- lafix: Laplace fixel posteriors improve connectome reproducibility

## Abstract (150 words, to write last)
Claim 1: refining any DIPY peaks with a differentiable multi-fixel fit improves orientation precision (crossings 2.3–8.0° vs 9.7–16.6° FORCE, 5–23° MSMT-CSD on identical Monte Carlo inputs). Claim 2: the Laplace posterior is calibrated after a single in-vivo constant (1.81 ± 0.15, 44 HCP-YA subjects) and predicts scan–rescan error (ρ = 0.457). Claim 3: posterior-mean connectomes are more reproducible than point estimates (r(log) 0.965 vs 0.923, Dice 0.936 vs 0.855). Claim 4: in-vivo f_i is as reproducible as NODDI (|Δ| 0.035 vs 0.039).

## 1 Introduction
- Fixel orientation is the input to tractography; no widely used package reports per-fixel uncertainty (MSMT-CSD, FORCE, NODDI deliver points).
- Amortised SBI promised posteriors; on orientation tasks the per-voxel MAP + Laplace beats the flows (doc 008 §6.3, §7.1) — the paper's methodological point is that a differentiable fit with a Gauss–Newton Laplace step is sufficient and cheaper.
- Contribution: method, calibration constant, reproducibility evidence, open code (BSD, DIPY plug-in).

## 2 Methods
- 2.1 Forward model: K-fixel stick + zeppelin (+ optional CSF/GM balls), Rician NLL, sigmoid-bounded parameters, spatial regularisation; Optimistix LM (doc 008 §4; lafix/prism_jax.py).
- 2.2 Initialisation from any peaks (CSD/MSMT/FORCE); fixel count from the input.
- 2.3 Laplace posterior: Gauss–Newton Hessian in unit-box coordinates, per-fixel angular σ, fraction σ (lafix/prism_uncertainty.py); evidence for K.
- 2.4 Calibration: scan–rescan angular error vs σ on HCP-YA; inflation factor; cone coverage.
- 2.5 Posterior-sampled connectomes: direction samples → tractography → edge mean/CV.
- 2.6 Data: Monte Carlo substrates (MCMRSimulator.jl / Permeable_MCDS / JAX walker, doc 008 §11.1), DiSCo, Stanford HARDI, HCP-YA 44 test–retest.
- 2.7 Baselines: MSMT-CSD (DIPY), FORCE (dipy master, seeded), NODDI (AMICO/dmipy), SBI_dMRI flow.

## 3 Results (tables exist)
- 3.1 Precision on substrates and DiSCo — doc 008 §6.4 table.
- 3.2 Calibration: ρ, inflation 1.81 ± 0.15, coverage — doc 008 §10.6.
- 3.3 Scan–rescan HCP-YA 44: fixel Δθ, count agreement, connectome r/Dice, posterior-mean connectome — doc 008 §10.6 second table.
- 3.4 f_i reproducibility vs NODDI, paired Wilcoxon (41/42 subjects lower |Δ|) — doc 008 §10.6.
- 3.5 Runtime: Stanford HARDI CPU 87.8 s + 31.2 s — doc 012 §7.6.
- 3.6 Where amortised flows stand (brief; doc 008 §7.1 addendum 2).

## 4 Discussion
- What the σ means (local curvature, under-dispersed by 1.8× in vivo: motion, physiology, model error).
- Isotropic compartments trade-off (CoV 8.9 % vs 6.4 %; main fixel 9.0° vs 12.6°).
- Limits: fixel detection not addressed; calibration protocol-specific; Gaussian posterior.
- Relation to FORCE (matching-limited, doc 005), MSMT-CSD, NODDI; to SBI (doc 012 §3).

## 5 Software
- lafix on PyPI, DIPY plug-in and CLI; compore for the forward models; validation scripts.

## Figures
1. Pipeline schematic (peaks in → refined fixels + σ cones out).
2. Substrate crossings: angular error vs crossing angle, four methods.
3. Calibration: σ vs scan–rescan Δθ, before/after inflation; coverage curve.
4. HCP-YA: posterior-mean vs point connectome scan–rescan scatter.
5. f_i Bland–Altman vs NODDI.
6. Example subject: σ map over the centrum semiovale.

## To do before submission
- Decide venue and length; write abstract last.
- Re-run the DiSCo table with lafix 0.1.0 (post-rename) to confirm the numbers reproduce.
- Data/code availability statement: HCP-YA (BALSA terms), DiSCo (public), substrates (HDF5 libraries to Zenodo).
