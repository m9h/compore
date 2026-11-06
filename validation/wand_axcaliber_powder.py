#!/usr/bin/env python3
"""
doc 013 §4.1 item 3: cortical powder averages of the WAND Connectom
variable-diffusion-time ("AxCaliber") acquisition, in the format of
validation/nexi_c2_powder.py so that validation/nexi_c2_model_selection.py
--dataset wand can run on it.

Per (subject, session) with FSL-preprocessed AxCaliber shells
(derivatives/fsl-dwi/<sub>/<ses>/dwi/AxCaliber{1..4}_eddy.nii.gz) and a
FreeSurfer recon (derivatives/freesurfer/<sub>_<ses>): rigidly register the
FreeSurfer brainmask to the topup-corrected mean b0 (DIPY, mutual
information), bring aparc.DKTatlas+aseg into dwi space (nearest neighbour),
take the cortical ribbon (labels 1000–1035 / 2000–2035), and store per voxel
the mean signal of each (b, Δ) shell normalised by the mean of that shell's
own b0 volumes. Scheme: δ = 7 ms (compore.io.wand), Δ from the sidecar
SeriesDescription (17.3, 30, 42, 55 ms), b ∈ {2200, 4400}, {4000, 8000},
{5800, 11600}, {7750, 15500} s/mm², 30 directions each, 6 b0 per shell.
Output: /data/datasets/compore/wand/powder/<sub>_<ses>.npz with
  S (N, 8), S_b0 (N,), sigma_b0 (N,), label (N,), ijk (N,3),
  features (8,2) float (b s/mm², Δ ms), counts (8,), n_b0, subject, session,
  affine, shape; plus the dwi-space DKT labels as <sub>_<ses>_dseg-dwi.nii.gz.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, nibabel as nib
from compore.io import wand as w

RAW = Path("/data/raw/wand")
CORTEX = lambda lab: ((lab >= 1000) & (lab <= 1035)) | ((lab >= 2000) & (lab <= 2035))


def register_dkt_to_dwi(fs_dir: Path, b0_img, out_path: Path):
    """Rigid FreeSurfer→b0 registration; returns the DKT label volume on the b0 grid."""
    from dipy.align.imaffine import AffineRegistration, MutualInformationMetric, transform_centers_of_mass
    from dipy.align.transforms import RigidTransform3D
    from dipy.align.imaffine import AffineMap
    brain = nib.load(fs_dir / "mri" / "brainmask.mgz"); dkt = nib.load(fs_dir / "mri" / "aparc.DKTatlas+aseg.mgz")
    static = np.asarray(b0_img.dataobj, np.float32); moving = np.asarray(brain.dataobj, np.float32)
    c = transform_centers_of_mass(static, b0_img.affine, moving, brain.affine)
    reg = AffineRegistration(metric=MutualInformationMetric(32, None), level_iters=[10000, 1000, 100], sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1])
    rigid = reg.optimize(static, moving, RigidTransform3D(), None, b0_img.affine, brain.affine, starting_affine=c.affine)
    lab = rigid.transform(np.asarray(dkt.dataobj, np.float32), interpolation="nearest").astype(np.int32)
    nib.save(nib.Nifti1Image(lab, b0_img.affine), out_path)
    return lab, rigid.affine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/datasets/compore/wand/powder")
    ap.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); summary = {}
    for dwi_dir in sorted((RAW / "derivatives" / "fsl-dwi").glob("sub-*/ses-*/dwi")):
        sub, ses = dwi_dir.parts[-3], dwi_dir.parts[-2]; tag = f"{sub}_{ses}"
        if a.subjects and sub not in a.subjects: continue
        shells = sorted(dwi_dir.glob("AxCaliber[1-4]_eddy.nii.gz")); fs_dir = RAW / "derivatives" / "freesurfer" / tag
        if len(shells) != 4 or not (fs_dir / "mri" / "aparc.DKTatlas+aseg.mgz").exists():
            summary[tag] = f"{len(shells)} AxCaliber shells, freesurfer={fs_dir.exists()}"; print(tag, summary[tag], flush=True); continue
        t0 = time.time(); b0_img = nib.load(dwi_dir / "topup_b0_corrected_mean.nii.gz")
        mask = np.asarray(nib.load(dwi_dir / "b0_brain_mask.nii.gz").dataobj) > 0
        lab, _ = register_dkt_to_dwi(fs_dir, b0_img, out / f"{tag}_dseg-dwi.nii.gz")
        cortex = CORTEX(lab) & mask; ijk = np.argwhere(cortex)
        feats, counts, S, S_b0, sig = [], [], [], [], []
        for sh in shells:
            k = sh.name[len("AxCaliber")]
            raw = next((RAW / sub / ses / "dwi").glob(f"*acq-AxCaliber{k}_dir-AP*_dwi.bval")); meta = json.load(open(raw.with_suffix(".json")))
            Delta, delta = w.wand_timings(meta); b = np.loadtxt(raw)
            X = np.asarray(nib.load(sh).dataobj, np.float32)[cortex]           # (N, M)
            b0 = b < 50; ref = X[:, b0].mean(1); S_b0.append(ref); sig.append(X[:, b0].std(1, ddof=1))
            for bb in sorted(set(np.round(b[~b0] / 100) * 100)):
                m = (~b0) & (np.abs(b - bb) <= 100)
                feats.append((float(bb), round(Delta * 1e3, 1))); counts.append(int(m.sum())); S.append(X[:, m].mean(1) / np.maximum(ref, 1e-6))
        S = np.stack(S, 1).astype(np.float32); S_b0 = np.mean(S_b0, 0); sigma_b0 = np.sqrt(np.mean(np.square(sig), 0))
        np.savez_compressed(out / f"{tag}.npz", S=S, S_b0=S_b0, sigma_b0=sigma_b0, label=lab[cortex].astype(np.int32), ijk=ijk.astype(np.int16),
                            features=np.asarray(feats, np.float32), counts=np.asarray(counts, np.int32), n_b0=int(6 * len(shells)),
                            subject=sub, session=ses, affine=b0_img.affine, shape=np.asarray(b0_img.shape), delta_s=delta)
        summary[tag] = {"n_cortex": int(S.shape[0]), "features": feats, "counts": counts, "median_S_b0": float(np.median(S_b0)),
                        "median_sigma_b0": float(np.median(sigma_b0)), "median_SNR_b0": float(np.median(S_b0 / np.maximum(sigma_b0, 1e-6))),
                        "seconds": round(time.time() - t0, 1)}
        print(tag, summary[tag], flush=True)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
