# 14. DIPY Workshop 2027 programme: what the Indiana group is shipping, read against compore/lafix

## Date

2026-11-06

## Status

Landscape note (extends doc 012 §4 "Indiana / DIPY" row). Sources: the
workshop pages https://workshop.dipy.org/ (2027, 15–19 March 2027) and
https://workshop.dipy.org/2026.html (16–20 March 2026), the `dipy/dipy`
master tree and PR list on 2026-11-05/06, arXiv, PMC. Everything below is
what the public record shows today; sessions marked "no public artefact"
have only a schedule line.

## 1. Programme (2027) and what each name is

| day | session | speaker | what it is | public artefact | relevance to us |
|---|---|---|---|---|---|
| 1 | DIPY SEIRA | Koudoro | unknown; new name, not in the 2026 programme, no code or paper found | none | watch |
| 1 | SUCOR | Chigurupati | susceptibility distortion correction by 1-D optimal transport along PE from reversed-PE b0 pairs; bending-energy regulariser, strength picked by Morozov discrepancy; HCP MI with T1 0.341 vs TOPUP 0.317, ~12 s on one CPU core | arXiv 2603.16758 (Mar 2026); DIPY PR #4258 is the sibling fieldmap-less SyN SDC (`dipy.align.sdc`); SuCor itself not yet in master | preprocessing only; nothing to do, but the ds008771/WAND pipelines use topup and could offer a timing comparison |
| 1 | Skyline (part of "DIPY Auto") | DIPY team | `dipy.viz.skyline` + `dipy_skyline` CLI, FURY 2.1 / imgui viewer that replaces Horizon (deprecated 1.13); shows slices, tractograms with QuickBundlesX, peaks, FORCE ODF glyphs | in `dipy/dipy` master (1.13.0.dev0), PRs #4291, #4294 | `lafix` writes peaks + MRtrix fixel dirs; a Skyline layer for per-fixel σ cones would be the natural demo once the PR lands |
| 1 | DIPY Auto | Koudoro (2026) | "graphical user interface and automated pipelines" | no repo found; `skoudoro/dmri-pipeline-test` (Jul 2026) may be a precursor | a `dipy_refine_peaks` step is the obvious plug-in point if Auto chains DIPY workflows |
| 2 | FORCE | Shah | dictionary matching, `dipy.reconst.force` + `dipy.sims.force`, `dipy_fit_force` | dipy ≥ 1.12 | measured against in docs 004, 005, 007, 008 |
| 2 | Patch2Self3 (2026) | Reddy | third Patch2Self version; no paper or code found | none | n/a |
| 3 | GLIDE (2026) | Deka | "most optimized tractography algorithm"; almost certainly the GPU tractography of Deka's ISMRM 2025 abstract, published as `cuslines` (`dipy/GPUStreamlines`, CUDA 12/13, Metal, WebGPU backends, pushed Jun 2026) | `pip install cuslines[cu13]` | posterior-sampled tractography (doc 008 §9) is CPU EuDX today; cuslines would make sampling 100 connectomes per subject cheap |
| 4 | BUAN3 | Chandio | third generation of Bundle Analytics; 2026 session also covered BundleWarp (streamline nonrigid registration, single λ) | BUAN and BundleWarp in dipy; SPECTRA statistics (grid LMMs, spatial length scales, hierarchical FDR) arriving in PR #4267 | our scan–rescan ICC tables are ROI-level; SPECTRA/BUAN is where an along-tract p(NEXI) or f_i profile would be analysed |
| 4 | Villalon-Reina guest lecture | — | topic TBA (ENIGMA-DTI / Thompson lab) | — | — |
| 5 | THETAN | Garyfallidis | "new theoretical AI framework for the Era of Data Intelligence"; group page says release aimed at Fall 2026; no paper, no repo as of today | none | watch; the claim is unverifiable until code appears |
| 5 | Unsupervised segmentation | TBA | — | — | — |
| 5 | Tractome | Gor | interactive 3-D tractogram exploration/segmentation by clustering + representatives (FBK-NILab origin, revived by Gor, NIfTI-mask export PR Sep 2026) | github.com/maharshi-gor/tractome | n/a |

2026-only sessions worth knowing: **PUMBA** (Park; purely synthetic skull
stripping, arXiv 2505.07159, code `pjsjongsung/PUMBA`), **Cardiac DWI**
(Coveney; robust/constrained fitting and UQ for DTI/DKI/QTI), **GPU
acceleration** (Rokem).

## 2. Three things that matter for the roadmap (doc 013 §4)

1. **PRISM (Abouagour, Shah, Garyfallidis, arXiv 2604.00250) still has no
   code** ("released upon acceptance"), is not on the 2027 programme, and
   is not in dipy master. FORCE remains the DIPY-shipped method. Our
   positioning in doc 008 is unchanged: the DIPY issue draft
   (`docs/outreach/dipy-lafix-issue.md`) should compare against FORCE
   peaks, which the Skyline tutorial now also renders, and mention PRISM
   only as the related preprint.
2. **DIPY's visible direction is pipelines + viewer (Auto, Skyline, GLIDE)
   and statistics (BUAN3, SPECTRA), not microstructure modelling.** No 2026
   or 2027 session covers compartment models, exchange, or model
   selection. That is the gap `lafix` (fixel posterior) and
   `compore.inference.model_selection` (p(M|S) maps) fill; the issue draft
   should say so in one line.
3. **UQ vocabulary in the DIPY orbit is bootstrap, not Laplace.** Coveney's
   MRM 2026 paper (image-level repetition bootknife over RWLS fits, code on
   request, no comparison with Laplace) is the only UQ talk in two years.
   When the lafix paper (docs/paper/outline.md) discusses related work it
   should cite it and state why a Gauss–Newton Laplace per fixel is the
   right tool for a 288-direction single acquisition where
   image-level resampling is not available.

## 3. Concrete follow-ups (not started)

- Try `cuslines` on the GB10 for the posterior-sampled tractography of
  doc 008 §9 (currently CPU EuDX); record the timing in doc 008.
- When Skyline is released, add a `lafix` example that loads the
  `*_sigmatheta_calibrated.nii.gz` as a layer.
- Add Chigurupati (SuCor, SDC), Deka (cuslines), Chandio (BUAN3/SPECTRA)
  to the people column of doc 012 §4 for the Indiana row.
- Register for the 2027 workshop if the user wants to present `lafix` in
  the Day 2 "Presentations by Participants" slot (15:00 ET, 16 March 2027).

## References

- Chigurupati S, Garyfallidis E. SuCor. arXiv:2603.16758, 2026.
- Abouagour M, Shah A, Garyfallidis E. PRISM. arXiv:2604.00250, 2026.
- Park JS et al. Skull stripping with purely synthetic data (PUMBA). arXiv:2505.07159, 2025.
- Coveney S et al. Uncertainty quantification for cardiac DTI without additional datasets. MRM 96(3):1458–1463, 2026. doi:10.1002/mrm.70414.
- Deka K et al. GPU-accelerated diffusion MRI tractography in DIPY. ISMRM 2025 #4785; `dipy/GPUStreamlines` (cuslines).
- Chandio BQ et al. BundleWarp. bioRxiv 2023; Bundle analytics. Sci Rep 2020.
- DIPY PRs #4258 (SyN SDC), #4267 (SPECTRA statistics), #4291/#4294 (Skyline), #3707 (Cholesky DTI/DKI, Henriques group).

## 4. Execution log

### 4.1 cuslines (GLIDE) for posterior-sampled tractography, 2026-11-07

Scripts: `validation/cuslines_posterior_disco.py` (stage A, main env:
PRISM-plus fit on DiSCo 1 SNR 50, Laplace fixel posterior, 20 direction
samples + MAP, CPU EuDX protocol of `lafix.tracking.track_connectivity`) and
`validation/cuslines_posterior_disco_gpu.py` (stage B, scratch venv with
`cuslines[cu13]` 2026-06 from PyPI, CUDA backend on the GB10). Numbers from
`validation/cuslines_posterior_disco.json` and the stage A log.

| | CPU EuDX (dipy 1.12.1, one core) | cuslines prob getter on GB10 |
|---|---|---|
| per connectome, 35 376 seeds, 40³ volume | 0.84 s median | 1.56 s median (1.41 s first, incl. JIT) |
| 21 connectomes | 17.7 s | 32.5 s |
| streamlines, MAP sample | 79 825 | 55 008 (63 390 median over samples) |
| r vs DiSCo ground truth, MAP / posterior mean | 0.851 / 0.859 | 0.547 / 0.545 |
| r range over the 20 samples | 0.842–0.867 | 0.534–0.551 |
| r GPU connectome vs CPU connectome, same sample | — | 0.46–0.51 |

Facts that decide the question:

- **cuslines has no peaks (EuDX) direction getter**, only bootstrap (CSA/OPDT
  on raw data), PMF-probabilistic and PTT. A fixel set can only be tracked by
  turning it into a PMF on the 362-vertex sphere (Watson kernel
  `exp(κ((v·d)²−1))`, κ = 40). Sweeping κ ∈ {40, 200, 1000} and the
  relative peak threshold ∈ {0.25, 0.5} on the MAP sample moves r only within
  0.55–0.57, so the 0.30 gap to EuDX is tracker semantics (probabilistic
  propagation, cuslines' seeding/merging and stopping rules), not PMF
  sharpness. The GPU connectome is therefore **not a drop-in replacement**
  for the §17.5 protocol; every number in docs 007/008 stays CPU EuDX.
- **At DiSCo size the GPU is slower.** 15 267 voxels and 35 k seeds are
  under the launch/compile overhead; 0.84 s per connectome on one core means
  the 21-connectome posterior sweep is 18 s, which was never the bottleneck
  (the fit is 8 s, the Laplace step 11 s). cuslines pays off at HCP scale
  (741 k voxels, millions of seeds), where doc 008 §9 tracked once per
  subject; a 100-sample posterior connectome per HCP subject is the use case
  to revisit, with a peaks getter.
- **Two API details** recorded for a later attempt: `GPUTracker.__init__`
  creates a CUDA context that `__exit__` does not destroy (stage B pops and
  destroys it after each sample, otherwise the second tracker fails with
  `CUDA_ERROR_OUT_OF_MEMORY`), and streamlines may overshoot the volume by
  half a voxel, which `connectivity_matrix` rejects (stage B clips).
- A peaks/EuDX direction getter in cuslines is the natural upstream
  contribution if the HCP posterior sweep is ever wanted on the GPU; it is a
  CUDA kernel, not a DIPY PR, and is not scheduled.

Artefact: `validation/cuslines_posterior_disco_snr50.npz` (18 MB, direction
samples + CPU connectomes; untracked, listed in docs/artifacts.md).

### 4.2 Skyline example, 2026-11-07

`packages/lafix/examples/viz_skyline_sigma.py` loads the `dipy_refine_peaks`
outputs (`*_peaks.pam5`, `*_sigmatheta_calibrated.nii.gz`, `*_fwm`,
`*_fintra`) and calls `dipy.viz.skyline.app.skyline` with the σ map as the
background image, the refined peaks as a peaks layer, and optionally N
posterior cone samples of the main fixel on the middle slice as short
streamlines (`--samples N`); `--stealth out.png` renders offscreen.
**Untested**: Skyline is only in dipy master (1.13.0.dev0, no wheel on PyPI;
`uv pip install --pre dipy>=1.13.0.dev0` finds nothing) and the source build
from GitHub was not run in this session. It is written against the
`Skyline.__init__` docstring of master on 2026-11-06 (tuple forms
`(data, affine, label)`, `(peak_dirs, affine, label, peak_values)`,
`(sft, label)`; `visualizer_type="stealth"`, `out_stealth_png`). To test:
build dipy master into a venv with `fury>=2.1` and `imgui_bundle`, run the
CLI on Stanford HARDI, then the example with `--stealth`.
