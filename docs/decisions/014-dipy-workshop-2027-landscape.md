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
