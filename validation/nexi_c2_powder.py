#!/usr/bin/env python3
"""
doc 012 §7.4 step 1: powder averages of ds008771 (reduced NEXI, Connectome 2.0).

For every available (subject, session): load the preprocessed DWI, take the
cortical ribbon from the DKT parcellation in dwi space (FreeSurfer labels
1000–1035 / 2000–2035), and for each voxel compute the mean signal per
(b, Δ) shell normalised by the mean of the b0 volumes of the same Δ block
(b0 of all Δ are pooled when fewer than 2). Also store the per-voxel b0 noise
sd (std over the 13 b0 volumes, which carries TE-matched noise + physiology),
the cortical label per voxel, and the scheme. Output: one .npz per session in
/data/datasets/compore/ds008771/powder/ with
  S (N, 15) powder means in feature order `features` (b s/mm², Δ ms),
  S_b0 (N,), sigma_b0 (N,), label (N,), ijk (N,3), features (15,2), counts (15,),
  n_b0, subject, session.
Sessions without a dwi-space dseg (sub-01 both, sub-02 ses-01) are skipped
with a note.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from compore.io import nexi_c2 as n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/datasets/compore/ds008771/powder")
    ap.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    av = n.available(); summary = {}
    for s, sess in av.items():
        if a.subjects and s not in a.subjects:
            continue
        for ses, ok in sess.items():
            tag = f"{s}_{ses}"
            if not (ok["nii.gz"] and ok["bval"] and ok["delta"]):
                summary[tag] = "no dwi/scheme"; continue
            if not ok["dseg"]:
                summary[tag] = "no dwi-space dseg (public export)"; print(tag, summary[tag], flush=True); continue
            t0 = time.time()
            data, affine, sc = n.load_dwi(s, ses, mmap=True)
            dseg, _ = n.load_dseg(s, ses, space="dwi")
            if dseg.shape != data.shape[:3]:
                summary[tag] = f"dseg shape {dseg.shape} != dwi {data.shape[:3]}"; print(tag, summary[tag], flush=True); continue
            cortex = n.cortex_mask(dseg); ijk = np.argwhere(cortex)
            X = np.asarray(data[cortex], np.float32)                      # (N, M)
            b = np.round(sc.bvals_si / 1e6).astype(int); Dm = np.round(sc.Delta * 1e3).astype(int)
            b0 = b == 0
            S_b0 = X[:, b0].mean(1); sigma_b0 = X[:, b0].std(1, ddof=1)
            feats = [k for k, _ in sc.features]; counts = [c for _, c in sc.features]
            S = np.zeros((X.shape[0], len(feats)), np.float32)
            for j, (bb, dd) in enumerate(feats):
                m = (np.abs(b - bb) <= 100) & (Dm == dd)
                b0_same = b0 & (Dm == dd)
                ref = X[:, b0_same].mean(1) if b0_same.sum() >= 2 else S_b0
                S[:, j] = X[:, m].mean(1) / np.maximum(ref, 1e-6)
            np.savez_compressed(out / f"{tag}.npz", S=S, S_b0=S_b0, sigma_b0=sigma_b0, label=dseg[cortex].astype(np.int32),
                                ijk=ijk.astype(np.int16), features=np.asarray(feats, np.int32), counts=np.asarray(counts, np.int32),
                                n_b0=int(b0.sum()), subject=s, session=ses, affine=affine, shape=np.asarray(data.shape))
            summary[tag] = {"n_cortex": int(X.shape[0]), "median_S_b0": float(np.median(S_b0)), "median_sigma_b0": float(np.median(sigma_b0)),
                            "median_SNR_b0": float(np.median(S_b0 / np.maximum(sigma_b0, 1e-6))), "seconds": round(time.time() - t0, 1)}
            print(tag, summary[tag], flush=True)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
