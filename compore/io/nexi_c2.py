"""
ds008771: reduced-NEXI scan–rescan on the Connectome 2.0 scanner (Uhl et al.,
Imaging Neuroscience 2026, 10.1162/IMAG.a.1380; CC0). Loader for the
preprocessed derivatives (``derivatives/preproc``).

Layout on disk (``/data/raw/openneuro/ds008771`` here):

    derivatives/preproc/sub-XX/ses-YY/dwi/
        sub-XX_ses-YY_desc-preproc_dwi.{nii.gz,bval,bvec,delta}   513 volumes
        sub-XX_ses-YY_desc-noisemap_dwi.{bval,bvec}
        sub-XX_ses-YY_space-dwi_desc-dktaparcaseg_dseg.nii.gz      DKT parcellation in dwi space
    derivatives/preproc/sub-XX/ses-YY/anat/..._space-T1w_desc-dktaparcaseg_dseg.nii.gz

``.bval`` is in s/mm², ``.delta`` is the per-volume big Δ in ms, the gradient
duration is δ = 5 ms for every volume (paper, Methods), TE 66 ms, TR 3.3 s,
2 mm isotropic (sub-08: one 1.6 mm session). Eight subjects, seven with two
sessions. The scheme as measured on sub-01 ses-01 (counts × (b s/mm², Δ ms)):

    3×(0,12) 20×(1000,12) 30×(2500,12) | 2×(0,15) 32×(5000,15) | 2×(0,20) 34×(7350,20)
    3×(0,27) 20×(1000,27) 30×(2500,27) 32×(5000,27) 34×(7500,27) 44×(10000,27)
    3×(0,45) 20×(1000,45) 30×(2500,45) 32×(5000,45) 34×(7500,45) 44×(10000,45) 64×(12500,45)

The small annexed files are not in the OpenNeuro S3 export; they come from
``https://openneuro.org/crn/datasets/ds008771/snapshots/1.0.5/files/<path with ':'>``
(see ``fetch_small_files``). Everything returned is SI: b in s/m², δ/Δ in s.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

DEFAULT_ROOT = Path(os.environ.get("DS008771_ROOT", "/data/raw/openneuro/ds008771"))
SNAPSHOT = "1.0.5"
SMALL_DELTA_S = 5e-3
SUBJECTS = [f"sub-0{i}" for i in range(1, 9)]
SESSIONS = {s: ["ses-01", "ses-02"] for s in SUBJECTS}
SESSIONS["sub-08"] = ["ses-01"]          # single high-resolution session, no rescan
SMALL_FILES = ["dwi/{p}_desc-preproc_dwi.bval", "dwi/{p}_desc-preproc_dwi.bvec", "dwi/{p}_desc-preproc_dwi.delta",
               "dwi/{p}_desc-noisemap_dwi.bval", "dwi/{p}_desc-noisemap_dwi.bvec",
               "dwi/{p}_from-T1w_to-dwi_mode-image_xfm.mat", "dwi/{p}_from-dwi_to-T1w_mode-image_xfm.mat",
               "dwi/{p}_space-dwi_desc-dktaparcaseg_dseg.nii.gz", "anat/{p}_space-T1w_desc-dktaparcaseg_dseg.nii.gz"]


def preproc_dir(subject: str, session: str, root: Path = DEFAULT_ROOT) -> Path:
    return Path(root) / "derivatives" / "preproc" / subject / session


def fetch_small_files(subject: str, session: str, root: Path = DEFAULT_ROOT, snapshot: str = SNAPSHOT) -> Dict[str, bool]:
    """Download the small sidecar files from the OpenNeuro API (idempotent)."""
    d = preproc_dir(subject, session, root); out = {}
    for tmpl in SMALL_FILES:
        rel = tmpl.format(p=f"{subject}_{session}"); dest = d / rel
        if dest.exists() and dest.stat().st_size > 0:
            out[rel] = True; continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        key = ":".join(["derivatives", "preproc", subject, session] + rel.split("/"))
        url = f"https://openneuro.org/crn/datasets/ds008771/snapshots/{snapshot}/files/{key}"
        try:
            urllib.request.urlretrieve(url, dest); out[rel] = dest.stat().st_size > 0
        except Exception:  # noqa: BLE001
            dest.unlink(missing_ok=True); out[rel] = False
    return out


@dataclass
class NEXIScheme:
    bvals_si: np.ndarray      # (M,) s/m²
    bvecs: Optional[np.ndarray]   # (M,3) or None when the public export lacks the file
    Delta: np.ndarray         # (M,) s
    delta: np.ndarray         # (M,) s

    @property
    def n(self):
        return len(self.bvals_si)

    @property
    def features(self):
        """Sorted unique (b s/mm², Δ ms) pairs with counts, b0 excluded."""
        b = np.round(self.bvals_si / 1e6).astype(int); D = np.round(self.Delta * 1e3).astype(int)
        keys, counts = np.unique(np.column_stack([b, D])[b > 0], axis=0, return_counts=True)
        return [((int(k[0]), int(k[1])), int(c)) for k, c in zip(keys, counts)]

    def subset(self, pairs: Sequence[tuple], keep_b0: bool = True) -> np.ndarray:
        """Boolean mask selecting the given (b s/mm², Δ ms) features (+ all b0)."""
        b = np.round(self.bvals_si / 1e6).astype(int); D = np.round(self.Delta * 1e3).astype(int)
        m = np.zeros(self.n, bool)
        for bb, dd in pairs:
            m |= (np.abs(b - bb) <= 100) & (D == dd)
        if keep_b0:
            m |= b == 0
        return m

    def powder_average_groups(self):
        """List of index arrays, one per (b, Δ) shell including each b0 group."""
        b = np.round(self.bvals_si / 1e6).astype(int); D = np.round(self.Delta * 1e3).astype(int)
        keys = np.unique(np.column_stack([b, D]), axis=0)
        return [(tuple(int(x) for x in k), np.flatnonzero((b == k[0]) & (D == k[1]))) for k in keys]


# the eight XAI-selected features of the paper, (b s/mm², Δ ms)
REDUCED_FEATURES = [(1000, 12), (2500, 12), (5000, 15), (7350, 20), (1000, 27), (1000, 45), (2500, 45), (12500, 45)]


def load_scheme(subject: str, session: str, root: Path = DEFAULT_ROOT) -> NEXIScheme:
    d = preproc_dir(subject, session, root); p = f"{subject}_{session}_desc-preproc_dwi"
    bvals = np.loadtxt(d / "dwi" / f"{p}.bval").ravel()
    Delta = np.loadtxt(d / "dwi" / f"{p}.delta").ravel() * 1e-3
    if len(bvals) != len(Delta):
        raise ValueError(f"{subject} {session}: bval {len(bvals)}, delta {len(Delta)}")
    bv = d / "dwi" / f"{p}.bvec"
    if bv.exists():
        bvecs = np.loadtxt(bv); bvecs = bvecs.T if bvecs.shape[0] == 3 else bvecs
        if len(bvecs) != len(bvals):
            raise ValueError(f"{subject} {session}: bvec {len(bvecs)} vs bval {len(bvals)}")
    else:
        # sub-01 and sub-02: the preproc .bvec annex objects are absent from the
        # public S3 export (AccessDenied, 2026-11-05). Powder-averaged models
        # (NEXI, SANDI, SMEX) do not need directions; direction-dependent fits do.
        bvecs = None
    return NEXIScheme(bvals_si=bvals * 1e6, bvecs=bvecs, Delta=Delta, delta=np.full(len(bvals), SMALL_DELTA_S))


def load_dwi(subject: str, session: str, root: Path = DEFAULT_ROOT, mmap: bool = True):
    """(data (X,Y,Z,M) float32 memmap-backed, affine, NEXIScheme)."""
    import nibabel as nib
    d = preproc_dir(subject, session, root); p = f"{subject}_{session}_desc-preproc_dwi"
    img = nib.load(str(d / "dwi" / f"{p}.nii.gz"))
    data = np.asarray(img.dataobj) if not mmap else img.dataobj
    return data, img.affine, load_scheme(subject, session, root)


def load_dseg(subject: str, session: str, root: Path = DEFAULT_ROOT, space: str = "dwi"):
    """DKT aparc+aseg labels (FreeSurfer LUT) in dwi or T1w space."""
    import nibabel as nib
    d = preproc_dir(subject, session, root)
    sub = "dwi" if space == "dwi" else "anat"
    f = d / sub / f"{subject}_{session}_space-{'dwi' if space == 'dwi' else 'T1w'}_desc-dktaparcaseg_dseg.nii.gz"
    img = nib.load(str(f)); return np.asarray(img.dataobj).astype(np.int32), img.affine


# FreeSurfer LUT: cortical ribbon is 1000–1035 (lh) and 2000–2035 (rh) in DKT aparc; WM 2/41
CORTEX_LABEL_RANGES = [(1000, 1035), (2000, 2035)]
WM_LABELS = (2, 41)


def cortex_mask(dseg: np.ndarray) -> np.ndarray:
    m = np.zeros(dseg.shape, bool)
    for lo, hi in CORTEX_LABEL_RANGES:
        m |= (dseg >= lo) & (dseg <= hi)
    return m


def available(root: Path = DEFAULT_ROOT) -> Dict[str, Dict[str, bool]]:
    """Which (subject, session) have the preprocessed NIfTI and the scheme files."""
    out = {}
    for s, sess in SESSIONS.items():
        out[s] = {}
        for ses in sess:
            d = preproc_dir(s, ses, root) / "dwi"; p = f"{s}_{ses}_desc-preproc_dwi"
            out[s][ses] = {e: (d / f"{p}.{e}").exists() for e in ("nii.gz", "bval", "bvec", "delta")}
            out[s][ses]["dseg"] = (d / f"{s}_{ses}_space-dwi_desc-dktaparcaseg_dseg.nii.gz").exists()
    return out
