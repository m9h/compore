"""
Show `dipy_refine_peaks` output in DIPY Skyline: refined peaks over the
calibrated angular-uncertainty map.

Skyline (``dipy.viz.skyline``, dipy >= 1.13, FURY >= 2.1) is the viewer that
replaces Horizon. It takes already-loaded arrays as tuples, so the lafix
products can be shown without re-saving anything:

    images      (data, affine, label)
    peaks       (peak_dirs, affine, label, peak_values)
    tractograms (StatefulTractogram, label)

Usage (after ``dipy_refine_peaks dwi.nii.gz dwi.bval dwi.bvec mask.nii.gz
peaks.pam5 --out_dir out``):

    python viz_skyline_sigma.py out/<prefix>_desc-refined [--stealth out.png] [--samples 50]

``--samples N`` additionally draws N posterior direction samples of the main
fixel in every voxel of one axial slice as short streamlines, so the width
of each cone is visible next to the σ map. It needs the ``*_fixels``
directory written by the CLI (directions.mif + sigma_theta_deg_calibrated.mif).

Untested against a released dipy: written on 2026-11-07 against the Skyline
API in dipy master (``dipy/viz/skyline/app.py``), see doc 014 §3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load(prefix: Path):
    from dipy.io.image import load_nifti
    from dipy.io.peaks import load_pam

    pam = load_pam(f"{prefix}_peaks.pam5")
    sigma, affine = load_nifti(f"{prefix}_sigmatheta_calibrated.nii.gz")
    fwm, _ = load_nifti(f"{prefix}_fwm.nii.gz")
    fintra, _ = load_nifti(f"{prefix}_fintra.nii.gz")
    return pam, sigma, fwm, fintra, affine


def _cone_samples(pam, sigma_deg, affine, z: int, n: int, seed: int = 0):
    """n unit-vector samples of the main fixel per voxel in slice z, drawn
    from an isotropic tangent Gaussian with sd = σ_θ, rendered as segments
    of length 0.8 voxel centred on the voxel."""
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    import nibabel as nib

    rng = np.random.default_rng(seed)
    d = pam.peak_dirs[:, :, z, 0]; s = np.deg2rad(sigma_deg[:, :, z])
    ii, jj = np.nonzero((np.linalg.norm(d, axis=-1) > 0) & (s > 0))
    lines = []
    for i, j in zip(ii, jj):
        v = d[i, j]; e1 = np.cross(v, [0, 0, 1.0])
        if np.linalg.norm(e1) < 1e-3:
            e1 = np.cross(v, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
        t = rng.normal(0, s[i, j], (n, 2))
        w = v[None] + t[:, :1] * e1[None] + t[:, 1:] * e2[None]
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        c = np.array([i, j, z], float)
        for k in range(n):
            lines.append(np.stack([c - 0.4 * w[k], c + 0.4 * w[k]]))
    ref = nib.Nifti1Image(np.zeros(pam.peak_dirs.shape[:3], np.uint8), affine)
    return StatefulTractogram(lines, ref, Space.VOX)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prefix", help="<out_dir>/<prefix>_desc-refined")
    ap.add_argument("--stealth", metavar="PNG", help="render offscreen to this PNG instead of opening a window")
    ap.add_argument("--samples", type=int, default=0, help="posterior cone samples per voxel on the middle axial slice")
    args = ap.parse_args(argv)

    try:
        from dipy.viz.skyline.app import skyline
    except ImportError:
        sys.exit("Skyline needs dipy >= 1.13 and FURY >= 2.1 (`uv pip install 'dipy>=1.13' 'fury>=2.1'`).")

    prefix = Path(args.prefix)
    pam, sigma, fwm, fintra, affine = _load(prefix)
    sigma_masked = np.where(fwm > 0, sigma, 0.0).astype(np.float32)

    tractograms = []
    if args.samples > 0:
        z = sigma.shape[2] // 2
        tractograms.append((_cone_samples(pam, sigma, affine, z, args.samples), f"posterior cones (slice {z})"))

    kw = {}
    if args.stealth:
        kw = {"visualizer_type": "stealth", "out_stealth_png": args.stealth}
    skyline(
        images=[
            (sigma_masked, affine, "sigma_theta_calibrated [deg]"),
            (fwm.astype(np.float32), affine, "f_wm"),
            (fintra.astype(np.float32), affine, "f_intra"),
        ],
        peaks=[(pam.peak_dirs, affine, "refined peaks", pam.peak_values)],
        tractograms=tractograms,
        tract_colors=(1.0, 0.85, 0.2),
        **kw,
    )


if __name__ == "__main__":
    main()
