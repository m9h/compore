#!/usr/bin/env python3
"""Stage B of the cuslines comparison (doc 014 §3): re-track the posterior
direction samples saved by ``cuslines_posterior_disco.py`` on the GPU with
``cuslines`` (dipy/GPUStreamlines, the "GLIDE" of the 2026 DIPY workshop).

cuslines has no peaks (EuDX) direction getter, only bootstrap / PMF /
PTT, so each sampled fixel set is turned into a sharp PMF on the sphere,
    pmf(v) = Σ_f w_f · exp(κ ((v·d_f)² − 1)),   κ = 40  (half-width ≈ 10°),
and tracked with ``ProbDirectionGetter``. That is "EuDX with a 10° jitter",
not EuDX; the connectome Pearson r against the CPU reference says how much
that matters.

Run with the scratch venv that has cuslines, dipy and nibabel only:
    <venv>/bin/python validation/cuslines_posterior_disco_gpu.py --npz validation/cuslines_posterior_disco_snr50.npz
Writes validation/cuslines_posterior_disco.json.
"""
import argparse, json, time
from math import radians
from pathlib import Path
import numpy as np, nibabel as nib
from dipy.data import default_sphere
from dipy.tracking.utils import connectivity_matrix, seeds_from_mask


def pearson_ut(a, b):
    iu = np.triu_indices(a.shape[0], k=1)
    av = np.asarray(a)[iu].astype(float); bv = np.asarray(b)[iu].astype(float)
    if av.std() == 0 or bv.std() == 0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def pmf_from_fixels(dirs_s, fracs, idx, shape, verts, kappa, frac_min):
    """(N,K,3) sampled directions + (N,K) fractions → (X,Y,Z,V) PMF."""
    dots2 = np.einsum("nkc,vc->nkv", dirs_s, verts) ** 2                      # (N,K,V)
    w = np.where(fracs >= frac_min, fracs, 0.0)[..., None]                     # (N,K,1)
    pmf_n = (w * np.exp(kappa * (dots2 - 1.0))).sum(1)                          # (N,V)
    pmf = np.zeros(shape + (verts.shape[0],), np.float32)
    pmf[idx[:, 0], idx[:, 1], idx[:, 2]] = pmf_n.astype(np.float32)
    return pmf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="validation/cuslines_posterior_disco_snr50.npz")
    ap.add_argument("--out", default="validation/cuslines_posterior_disco.json")
    ap.add_argument("--kappa", type=float, default=40.0)
    ap.add_argument("--min-sep-deg", type=float, default=20.0)
    ap.add_argument("--chunk-size", type=int, default=25000)
    ap.add_argument("--max-samples", type=int, default=None, help="only the first N of the saved direction sets (MAP first)")
    ap.add_argument("--rel-peak", type=float, default=0.25)
    args = ap.parse_args()

    import cuslines
    from cuslines.cuda_python.cu_tractography import GPUTracker
    from cuslines.cuda_python.cu_direction_getters import ProbDirectionGetter
    from cuda.bindings import driver as cu_driver

    def _drop_context():
        # GPUTracker.__init__ creates a CUDA context and __exit__ frees the
        # buffers but not the context; one context per sample leaks until
        # CUDA_ERROR_OUT_OF_MEMORY. Pop and destroy it ourselves.
        err, ctx = cu_driver.cuCtxGetCurrent()
        if err == cu_driver.CUresult.CUDA_SUCCESS and int(ctx) != 0:
            cu_driver.cuCtxPopCurrent()
            cu_driver.cuCtxDestroy(ctx)

    z = np.load(args.npz)
    dirs, fracs, idx = z["dirs"], z["fracs"], z["idx"]
    mask, rois, affine, gt = z["mask"].astype(bool), z["rois"], z["affine"], z["gt"]
    cmats_cpu, t_cpu = z["cmats_cpu"], z["t_cpu"]
    if args.max_samples:
        dirs = dirs[: args.max_samples]
    frac_min = float(z["peak_frac_min"]); max_angle = float(z["max_angle"])
    verts = default_sphere.vertices.astype(np.float32); edges = default_sphere.edges.astype(np.int32)

    sc_mask = ((rois > 0) | mask).astype(np.float32)
    seed_mask = (rois > 0) & mask
    seeds = seeds_from_mask(seed_mask, np.eye(4), density=2).astype(np.float32)   # voxel space
    ref_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    print(f"backend {cuslines.BACKEND}; {dirs.shape[0]} samples, {len(seeds)} seeds", flush=True)

    rows, t_gpu, cmats_gpu = [], [], []
    t_first = None
    for s in range(dirs.shape[0]):
        t0 = time.time()
        pmf = pmf_from_fixels(dirs[s], fracs, idx, mask.shape, verts, args.kappa, frac_min)
        t_pmf = time.time() - t0
        t0 = time.time()
        with GPUTracker(ProbDirectionGetter(), pmf, sc_mask, 0.5, verts, edges, sphere_symm=True,
                        max_angle=radians(max_angle), step_size=0.5, min_pts=8, max_pts=600,
                        relative_peak_thresh=args.rel_peak, min_separation_angle=radians(args.min_sep_deg),
                        ngpus=1, rng_seed=s, chunk_size=args.chunk_size) as tracker:
            sft = tracker.generate_sft(seeds, ref_img)
        _drop_context()
        # keep voxel space (DiSCo affine is a pure translation here) and clip the
        # half-voxel overshoot at the box edge that cuslines allows
        hi = np.asarray(mask.shape, float) - 1.0
        sl = [np.clip(np.asarray(x, float), 0.0, hi) for x in sft.streamlines]
        conn_affine = np.eye(4)
        t_track = time.time() - t0
        if t_first is None:
            t_first = t_track           # includes the CUDA JIT compile
        t0 = time.time()
        if len(sl) == 0:
            c = np.zeros((16, 16))
        else:
            cm, _ = connectivity_matrix(sl, conn_affine, rois.astype(np.int32), return_mapping=True,
                                        mapping_as_streamlines=False, symmetric=True)
            c = cm[1:17, 1:17].astype(float)
        t_conn = time.time() - t0
        cmats_gpu.append(c); t_gpu.append(t_track)
        row = {"sample": s, "n_streamlines": int(len(sl)), "r_gt_gpu": pearson_ut(c, gt),
               "r_gt_cpu": pearson_ut(cmats_cpu[s], gt) if len(cmats_cpu) else None,
               "r_gpu_vs_cpu": pearson_ut(c, cmats_cpu[s]) if len(cmats_cpu) else None,
               "t_pmf_s": t_pmf, "t_track_gpu_s": t_track, "t_connectivity_s": t_conn,
               "t_cpu_s": float(t_cpu[s]) if len(t_cpu) else None}
        rows.append(row)
        print(f"sample {s}: {row['n_streamlines']} sl, r_gt gpu {row['r_gt_gpu']:.3f} cpu {row['r_gt_cpu']}, "
              f"gpu~cpu {row['r_gpu_vs_cpu']}, track {t_track:.2f}s (cpu {row['t_cpu_s']})", flush=True)

    C = np.asarray(cmats_gpu)
    summary = {
        "npz": args.npz, "backend": cuslines.BACKEND, "kappa": args.kappa, "min_sep_deg": args.min_sep_deg, "rel_peak": args.rel_peak,
        "n_samples": int(dirs.shape[0] - 1), "n_seeds": int(len(seeds)),
        "r_gt_map_gpu": pearson_ut(C[0], gt), "r_gt_mean_gpu": pearson_ut(C[1:].mean(0), gt) if len(C) > 1 else None,
        "r_gt_map_cpu": pearson_ut(cmats_cpu[0], gt) if len(cmats_cpu) else None,
        "r_gt_mean_cpu": pearson_ut(cmats_cpu[1:].mean(0), gt) if len(cmats_cpu) else None,
        "r_mean_gpu_vs_cpu": pearson_ut(C[1:].mean(0), cmats_cpu[1:].mean(0)) if len(cmats_cpu) and len(C) > 1 else None,
        "t_gpu_first_s_incl_compile": t_first, "t_gpu_median_s": float(np.median(t_gpu[1:])) if len(t_gpu) > 1 else None,
        "t_gpu_total_s": float(np.sum(t_gpu)), "t_cpu_total_s": float(np.sum(t_cpu)) if len(t_cpu) else None,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
