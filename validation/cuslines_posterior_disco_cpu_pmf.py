#!/usr/bin/env python3
"""CPU reference for the cuslines comparison (doc 014 §4.1/§4.3): track the
same sharp-PMF emulation of the posterior fixel samples with DIPY's own
``probabilistic_tracking`` on one core. If this lands at the same r as the
GPU (≈0.55) and not at EuDX's 0.85, the gap is PMF-tracking semantics and
cuslines is a faithful GPU port; if it lands near 0.85, stage B is wrong.

    uv run python validation/cuslines_posterior_disco_cpu_pmf.py --npz validation/cuslines_posterior_disco_snr50.npz --max-samples 4
Writes validation/cuslines_posterior_disco_cpu_pmf.json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from dipy.data import default_sphere
from dipy.tracking.utils import connectivity_matrix, seeds_from_mask

from validation.cuslines_posterior_disco_gpu import clip_to_volume, pearson_ut, pmf_from_fixels


def track_pmf_cpu(pmf, stop_mask, seeds, *, max_angle=45.0, step_size=0.5, min_len=4, max_len=300,
                  pmf_threshold=0.1, random_seed=0):
    """DIPY probabilistic tracking on a (X,Y,Z,V) PMF over ``default_sphere``,
    voxel space (identity affine), BinaryStoppingCriterion; the §17.5 protocol
    with the EuDX getter swapped for the PMF getter."""
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking.tracker import probabilistic_tracking
    sc = BinaryStoppingCriterion(np.asarray(stop_mask).astype(np.float32))
    gen = probabilistic_tracking(np.asarray(seeds, float), sc, np.eye(4), sf=np.asarray(pmf, np.float32),
                                 sphere=default_sphere, max_angle=max_angle, step_size=step_size,
                                 min_len=min_len, max_len=max_len, pmf_threshold=pmf_threshold,
                                 random_seed=random_seed, return_all=True)
    return [s for s in Streamlines(gen) if len(s) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="validation/cuslines_posterior_disco_snr50.npz")
    ap.add_argument("--gpu-json", default="validation/cuslines_posterior_disco.json")
    ap.add_argument("--out", default="validation/cuslines_posterior_disco_cpu_pmf.json")
    ap.add_argument("--kappa", type=float, default=40.0)
    ap.add_argument("--pmf-threshold", type=float, default=0.1)
    ap.add_argument("--max-samples", type=int, default=4, help="MAP + (N-1) posterior samples")
    args = ap.parse_args()

    z = np.load(args.npz)
    dirs, fracs, idx = z["dirs"][: args.max_samples], z["fracs"], z["idx"]
    mask, rois, gt, cmats_cpu = z["mask"].astype(bool), z["rois"], z["gt"], z["cmats_cpu"]
    frac_min = float(z["peak_frac_min"]); max_angle = float(z["max_angle"])
    verts = default_sphere.vertices.astype(np.float32)
    sc_mask = (rois > 0) | mask
    seeds = seeds_from_mask((rois > 0) & mask, np.eye(4), density=2)
    gpu_rows = json.loads(Path(args.gpu_json).read_text())["rows"] if Path(args.gpu_json).exists() else []

    rows, cm = [], []
    for s in range(dirs.shape[0]):
        pmf = pmf_from_fixels(dirs[s], fracs, idx, mask.shape, verts, args.kappa, frac_min)
        t0 = time.time()
        sl = clip_to_volume(track_pmf_cpu(pmf, sc_mask, seeds, max_angle=max_angle, pmf_threshold=args.pmf_threshold, random_seed=s), mask.shape)
        t = time.time() - t0
        c = np.zeros((16, 16))
        if sl:
            c = connectivity_matrix(sl, np.eye(4), rois.astype(np.int32), symmetric=True)[1:17, 1:17].astype(float)
        cm.append(c)
        row = {"sample": s, "n_streamlines": len(sl), "t_cpu_pmf_s": t,
               "r_gt_cpu_pmf": pearson_ut(c, gt), "r_gt_eudx": pearson_ut(cmats_cpu[s], gt),
               "r_gt_gpu": gpu_rows[s]["r_gt_gpu"] if s < len(gpu_rows) else None,
               "r_cpu_pmf_vs_gpu": None, "r_cpu_pmf_vs_eudx": pearson_ut(c, cmats_cpu[s])}
        rows.append(row)
        print(json.dumps(row), flush=True)
    C = np.asarray(cm)
    summary = {"npz": args.npz, "kappa": args.kappa, "pmf_threshold": args.pmf_threshold, "n_seeds": int(len(seeds)),
               "r_gt_map_cpu_pmf": pearson_ut(C[0], gt),
               "r_gt_mean_cpu_pmf": pearson_ut(C[1:].mean(0), gt) if len(C) > 1 else None,
               "r_gt_map_eudx": pearson_ut(cmats_cpu[0], gt),
               "t_cpu_pmf_median_s": float(np.median([r["t_cpu_pmf_s"] for r in rows])), "rows": rows}
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
