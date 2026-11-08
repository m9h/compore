#!/usr/bin/env python3
"""End-to-end lafix quick start on DIPY's Stanford HARDI dataset (CPU, ~3 min).

    python stanford_hardi_quickstart.py out/ [--n-fibres 3] [--n-iter 300] [--sigma-inflation 1.0]

1. fetch Stanford HARDI (dipy.data), median-Otsu brain mask
2. CSD (sh_order 8) peaks, 5 per voxel, via dipy.direction.peaks_from_model
3. `dipy_refine_peaks` on those peaks (the same Workflow the CLI runs)
4. report timings and a few summary numbers in out/quickstart.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir"); ap.add_argument("--n-fibres", type=int, default=3); ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--sigma-inflation", type=float, default=1.0); ap.add_argument("--num-cpus", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    from dipy.core.gradients import gradient_table
    from dipy.data import default_sphere, get_fnames
    from dipy.direction import peaks_from_model
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.peaks import save_pam
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
    from dipy.segment.mask import median_otsu
    from lafix.workflow import RefinePeaksFlow

    t = {}
    f_dwi, f_bval, f_bvec = get_fnames(name="stanford_hardi")
    data, affine = load_nifti(f_dwi); bvals, bvecs = read_bvals_bvecs(f_bval, f_bvec); gtab = gradient_table(bvals, bvecs=bvecs)
    t0 = time.time(); _, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3, numpass=1, dilate=2); t["mask_s"] = time.time() - t0
    save_nifti(str(out / "mask.nii.gz"), mask.astype(np.uint8), affine)

    t0 = time.time()
    response, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    csd = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=8)
    pam = peaks_from_model(csd, data, default_sphere, relative_peak_threshold=0.5, min_separation_angle=25, mask=mask,
                           npeaks=5, parallel=a.num_cpus != 1, num_processes=a.num_cpus or None)
    pam.affine = affine
    t["csd_peaks_s"] = time.time() - t0
    save_pam(str(out / "csd_peaks.pam5"), pam)

    t0 = time.time()
    RefinePeaksFlow().run(f_dwi, f_bval, f_bvec, str(out / "mask.nii.gz"), str(out / "csd_peaks.pam5"), n_fibres=a.n_fibres,
                          n_iter=a.n_iter, sigma_inflation=a.sigma_inflation, prefix="stanford_hardi", out_dir=str(out))
    t["refine_and_laplace_s"] = time.time() - t0

    sig, _ = load_nifti(str(out / "stanford_hardi_desc-refined_sigmatheta.nii.gz")); fi, _ = load_nifti(str(out / "stanford_hardi_desc-refined_fintra.nii.gz"))
    m = mask & (sig > 0)
    summary = {"n_mask_voxels": int(mask.sum()), "n_fitted_voxels": int(m.sum()), "sigma_theta_deg_median": float(np.median(sig[m])),
               "sigma_theta_deg_p90": float(np.percentile(sig[m], 90)), "f_intra_median": float(np.median(fi[m])), **t,
               "model": json.loads((out / "stanford_hardi_desc-refined_model.json").read_text())}
    (out / "quickstart.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
