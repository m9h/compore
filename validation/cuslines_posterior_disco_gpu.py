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
Writes validation/cuslines_posterior_disco_chunked.json (doc 014 §4.3; the
§4.1 file cuslines_posterior_disco.json is the run with cuslines' broken
chunking, kept for the record).
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


CUSLINES_HALF_VOXEL = 0.5
"""cuslines samples the stopping map through a CUDA texture (linear filter,
unnormalised coordinates, no +0.5 texel-centre offset: `check_point_d` in
tracking_helpers.cu, `create_texture` in cutils.py), so the stop boundary of
mask voxels i..j lies at [i, j+1) instead of DIPY's [i-0.5, j+0.5); the PMF
trilinear interpolation and the seed voxel lookup use DIPY's centre
convention. Verified on a stop-mask phantom
(tests/validation/test_doc014_tools.py::test_cuslines_output_is_corner_convention).
Consequence: streamlines leave the stopping region half a voxel late on the
+faces and early on the -faces. Labelling endpoints at x - 0.5 (floor(x),
the texture's convention; ``--endpoint-shift``) was tried and LOWERS the
DiSCo connectome r (0.864 → 0.848, doc 014 §4.3), so the default keeps the
raw coordinates; the collapse to r 0.55 in §4.1 was the chunking bug, not
this offset. Seeds and PMF are never shifted (a +0.5 seed shift crashes the
kernel at the volume edge: seed voxel 40 of 40)."""


def generate_sft_chunked(tracker, seeds, ref_img, chunk):
    """Streamlines for ``seeds`` from a cuslines tracker, one ``generate_sft``
    call per ``chunk`` seeds, concatenated. cuslines 2.2.3's own chunking in
    ``generate_sft`` is broken: it keeps lazy generators over the propagator's
    buffers and overwrites them with the next chunk before reading, so every
    chunk returns the LAST chunk's streamlines (two chunks of DiSCo seeds →
    55 008 = 2 × 27 504 streamlines, the first 25 000 seeds lost; this was the
    r 0.55 of doc 014 §4.1). The tracker's own ``chunk_size`` must be ≥ chunk."""
    out = []
    seeds = np.asarray(seeds)
    for i in range(0, len(seeds), chunk):
        out.extend(np.asarray(x) for x in tracker.generate_sft(seeds[i:i + chunk], ref_img).streamlines)
    return out


def clip_to_volume(streamlines, shape):
    """Clip points to [0, shape-1] so ``connectivity_matrix`` never sees a
    negative voxel index."""
    hi = np.asarray(shape, float) - 1.0
    return [np.clip(np.asarray(x, float), 0.0, hi) for x in streamlines]


def to_dipy_vox(streamlines, shape):
    """cuslines output → coordinates whose nearest voxel is the texture's
    floor(x) voxel (undo the stop-map half-voxel offset), clipped."""
    return clip_to_volume([np.asarray(x, float) - CUSLINES_HALF_VOXEL for x in streamlines], shape)


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
    ap.add_argument("--out", default="validation/cuslines_posterior_disco_chunked.json")
    ap.add_argument("--endpoint-shift", action="store_true", help="label endpoints at x-0.5 (stop-map texture convention); lowers r, see doc 014 §4.3")
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
    seeds = seeds_from_mask(seed_mask, np.eye(4), density=2).astype(np.float32)   # DIPY voxel space, unshifted
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
            streamlines = generate_sft_chunked(tracker, seeds, ref_img, args.chunk_size)
        _drop_context()
        # voxel space throughout (identity affine for connectivity_matrix)
        sl = to_dipy_vox(streamlines, mask.shape) if args.endpoint_shift else clip_to_volume(streamlines, mask.shape)
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
        "half_voxel_shift": bool(args.endpoint_shift), "chunk_size": args.chunk_size, "chunked_by_stage_b": True, "n_samples": int(dirs.shape[0] - 1), "n_seeds": int(len(seeds)),
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
