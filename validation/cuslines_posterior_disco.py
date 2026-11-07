#!/usr/bin/env python3
"""Stage A of the cuslines comparison (doc 014 §3): fit PRISM-plus on DiSCo 1,
take the Laplace fixel posterior, draw S direction samples, track every
sample with the CPU EuDX protocol of ``lafix.tracking.track_connectivity``
(doc 004 §17.5) and save everything stage B needs:

    validation/cuslines_posterior_disco_snr{snr}.npz
        dirs      (S+1, N, K, 3)  MAP first, then the S samples
        fracs     (N, K)          per-fixel WM fractions
        idx       (N, 3)          voxel indices of the N mask voxels
        mask, rois, affine, gt, cmats_cpu (S+1,16,16), t_cpu (S+1,)

Stage B (``validation/cuslines_posterior_disco_gpu.py``) runs in a separate
venv that has ``cuslines`` and re-tracks the same samples on the GPU.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, jax, nibabel as nib
from dipy.data import default_sphere

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "validation")]
from validation.lib.force_disco import disco_subject_path, load_disco_subject  # noqa: E402
from validation.lib.force_disco_connectivity import connectivity_pearson, load_gt_connectivity  # noqa: E402
from lafix.prism_jax import fit_prism  # noqa: E402
from lafix import prism_uncertainty as pu  # noqa: E402
from lafix.tracking import peaks_to_pam, track_connectivity  # noqa: E402
import validate_prism_disco_connectivity as drv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--n-fibres", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--peak-frac-min", type=float, default=0.10)
    ap.add_argument("--max-angle", type=float, default=45.0)
    ap.add_argument("--method", default="plus-tort-warm-nores")
    ap.add_argument("--skip-cpu", action="store_true")
    args = ap.parse_args()

    gt = load_gt_connectivity(subject=1)
    affine = nib.load(disco_subject_path(1) / "highRes_DiSCo1_DWI.nii.gz").affine
    out = load_disco_subject(subject=1, snr=args.snr, single_shell_b=None)
    data, mask, rois, gtab = out["data"], out["mask"], out["rois"], out["gtab"]
    bvals = np.asarray(gtab.bvals) * 1e6; bvecs = np.asarray(gtab.bvecs)
    cfg = drv._cfg_for(args.method, args.n_fibres, args.n_iter)
    init_dirs, init_fracs = drv._warm_start(out, bvals, bvecs, args.n_fibres, 500_000)
    t0 = time.time(); fit = fit_prism(data, mask, bvals, bvecs, cfg, init_dirs, init_fracs)
    print(f"fit {time.time()-t0:.0f}s", flush=True)
    t0 = time.time(); post = pu.laplace_fixel_posterior(fit, data, bvals, bvecs)
    print(f"laplace {time.time()-t0:.0f}s", flush=True)

    S = post.sample_dirs(jax.random.PRNGKey(0), args.n_samples)
    dirs = np.concatenate([post.dirs[None], S], axis=0).astype(np.float32)   # (S+1,N,K,3)
    fracs = np.asarray(post.wm_fracs, np.float32)
    idx = np.argwhere(mask)
    K = dirs.shape[2]

    cmats, t_cpu = [], []
    if not args.skip_cpu:
        for s in range(dirs.shape[0]):
            t0 = time.time()
            pdirs = np.zeros(mask.shape + (5, 3)); pvals = np.zeros(mask.shape + (5,))
            for n, (i, j, k) in enumerate(idx):
                for f in range(min(K, 5)):
                    v = fracs[n, f]
                    if v >= args.peak_frac_min:
                        pdirs[i, j, k, f] = dirs[s, n, f]; pvals[i, j, k, f] = v
            pam = peaks_to_pam(pdirs, pvals, mask, default_sphere, affine)
            c = track_connectivity(pam, mask, rois, affine, max_angle=args.max_angle)["connectivity"]
            cmats.append(c); t_cpu.append(time.time() - t0)
            print(f"cpu sample {s}: r={connectivity_pearson(c, gt):.3f}  {t_cpu[-1]:.1f}s", flush=True)
    out_path = ROOT / "validation" / f"cuslines_posterior_disco_snr{args.snr}.npz"
    np.savez_compressed(out_path, dirs=dirs, fracs=fracs, idx=idx, mask=mask, rois=rois, affine=affine, gt=gt,
                        cmats_cpu=np.asarray(cmats), t_cpu=np.asarray(t_cpu),
                        peak_frac_min=args.peak_frac_min, max_angle=args.max_angle)
    if cmats:
        C = np.asarray(cmats)
        print(f"CPU: r(MAP)={connectivity_pearson(C[0], gt):.3f} r(mean)={connectivity_pearson(C[1:].mean(0), gt):.3f} "
              f"total {sum(t_cpu):.0f}s for {len(t_cpu)} connectomes")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
