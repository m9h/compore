# lafix

Laplace fixel posteriors for diffusion MRI. `lafix` takes any DIPY
`PeaksAndMetrics` (CSD, MSMT-CSD, FORCE), refines it with a differentiable
K-fixel stick + zeppelin + CSF/GM fit under a Rician likelihood, and returns
refined peaks plus a per-fixel Laplace posterior (orientation σ, intra-axonal
fraction σ) with an in-vivo calibration constant from 44 HCP scan–rescan
subjects. Whole brain in about two minutes on CPU.

```bash
uv add lafix           # pulls compore (the physics) and dipy
dipy_refine_peaks dwi.nii.gz dwi.bval dwi.bvec mask.nii.gz peaks.pam5 --out_dir out/
```

Python: `from lafix import dipy_refine as dr; pam_ref, post, fit = dr.refine_peaks(data, gtab, mask, pam)`.

Results and validation: the `compore` repository, `docs/decisions/008-*.md`.
