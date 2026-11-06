"""WAND dataset loader for AxCaliber diffusion MRI.

Provides :class:`WANDLoader` for accessing the
`WAND dataset <https://gin.g-node.org/CUBRIC/WAND>`_ from CUBRIC, which
contains high-gradient-strength (300 mT/m) AxCaliber acquisitions suitable
for axon diameter estimation.

Key methods:

* :meth:`WANDLoader.fetch_data` -- downloads the dataset via DataLad/GIN.
* :meth:`WANDLoader.get_axcaliber_files` -- discovers AxCaliber DWI
  files for a subject/session.
* :meth:`WANDLoader.load_axcaliber_data` -- loads and concatenates all
  AxCaliber acquisitions, extracting pulse timings (delta, Delta) from
  JSON sidecars.

B-values are converted to SI (s/m^2) on load.  Pulse timings are
returned as per-volume JAX arrays for direct use with time-dependent
signal models (e.g. :class:`~compore.signal_models.cylinder_models.RestrictedCylinder`).
"""

import os
import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs

from compore.data.openneuro import fetch_datalad, DEFAULT_DATA_DIR

# WAND GIN Repository
WAND_GIN_URL = "https://gin.g-node.org/CUBRIC/WAND"

# Gradient duration of the WAND AxCaliber/CHARMED scans is not in the sidecars
# (no `t_sdel`); set it from the acquisition documentation before fitting
# anything that depends on δ. None → loaders return NaN for small_delta.
WAND_AXCALIBER_SMALL_DELTA_S = None


def wand_timings(meta: dict):
    """(Delta_s, delta_s) from a WAND dwi sidecar. Δ from SeriesDescription
    'AX_delta17p3_...' (17.3 ms), else from `t_bdel` (ms, string); δ from
    `t_sdel` if present, else WAND_AXCALIBER_SMALL_DELTA_S (None → NaN)."""
    import re, math
    Delta = None
    m = re.search(r"delta(\d+)(?:p(\d+))?", str(meta.get("SeriesDescription", "")))
    if m:
        Delta = float(f"{m.group(1)}.{m.group(2) or 0}") * 1e-3
    elif "t_bdel" in meta:
        Delta = float(meta["t_bdel"]); Delta = Delta * 1e-3 if Delta > 1.0 else Delta
    if "t_sdel" in meta:
        delta = float(meta["t_sdel"]); delta = delta * 1e-3 if delta > 1.0 else delta
    else:
        delta = WAND_AXCALIBER_SMALL_DELTA_S if WAND_AXCALIBER_SMALL_DELTA_S is not None else math.nan
    return (Delta if Delta is not None else math.nan), delta


class WANDLoader:
    def __init__(self, base_path=None, subject="sub-01"):
        if base_path is None:
            self.base_path = DEFAULT_DATA_DIR / "WAND"
        else:
            self.base_path = Path(base_path)
            
        self.subject = subject
        self.bids_root = self.base_path
        
    def fetch_data(self):
        """
        Fetches WAND dataset from GIN if not present.
        """
        if not self.base_path.exists():
            print(f"Fetching WAND dataset to {self.base_path}...")
            fetch_datalad(WAND_GIN_URL, path=self.base_path)
            
    def get_axcaliber_files(self, session="ses-01"):
        """
        Finds AxCaliber acquisitions for the subject.
        Looking for 'acq-AxCaliber1' and 'acq-AxCaliber2'.
        """
        subj_dir = self.bids_root / self.subject / session / "dwi"
        if not subj_dir.exists():
            # Try searching without session if flat?
            # Creating list of possible locations
            candidates = [
                self.bids_root / self.subject / "dwi",
                self.bids_root / "rawdata" / self.subject / session / "dwi"
            ]
            for p in candidates:
                if p.exists():
                    subj_dir = p
                    break
            else:
                 raise FileNotFoundError(f"Could not find DWI directory for {self.subject}")
        
        # Files
        # Pattern: sub-01_ses-01_acq-AxCaliber1_dwi.nii.gz
        # We look for glob *acq-AxCaliber*_dwi.nii.gz
        
        files = sorted(list(subj_dir.glob(f"*{self.subject}*acq-AxCaliber*dwi.nii.gz")))
        # Filter out Ref files
        files = [f for f in files if "Ref" not in f.name]
        print(f"Found {len(files)} AxCaliber files (excluding Ref): {[f.name for f in files]}")
        return files

    def load_axcaliber_data(self, session="ses-01", roi_slice=None):
        """
        Loads and concatenates AxCaliber data.
        Returns:
            data: (X, Y, Z, N)
            bvals: (N,)
            bvecs: (N, 3)
            big_delta: (N,)
            small_delta: (N,)
        """
        files = self.get_axcaliber_files(session)
        if not files:
            raise FileNotFoundError("No AxCaliber files found.")
            
        all_data = []
        all_bvals = []
        all_bvecs = []
        all_big = []
        all_small = []
        
        for fpath in files:
            print(f"Loading {fpath.name}...")
            # Load NIfTI
            data, affine = load_nifti(str(fpath))
            if roi_slice:
                data = data[roi_slice]
                
            # Load bvals/bvecs
            bval_path = str(fpath).replace(".nii.gz", ".bval")
            bvec_path = str(fpath).replace(".nii.gz", ".bvec")
            bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
            
            # Load JSON for Delta/delta
            json_path = str(fpath).replace(".nii.gz", ".json")
            if not os.path.exists(json_path):
                 print(f"Warning: JSON not found for {fpath.name}. Using defaults.")
                 delta = 0.0 # Placeholder
                 Delta = 0.0
            else:
                with open(json_path, 'r') as jf:
                    meta = json.load(jf)
                    
                # WAND stores the big Δ as the aDWI-BIDS key `t_bdel` (string, ms,
                # rounded: AxCaliber1 says '18' for the 17.3 ms scan). The exact value
                # is in SeriesDescription "AX_delta17p3_30xb2200_30xb4400".
                Delta, delta = wand_timings(meta)
            n_vol = len(bvals)
            all_data.append(data)
            all_bvals.append(bvals)
            all_bvecs.append(bvecs)
            all_big.append(jnp.full((n_vol,), Delta))
            all_small.append(jnp.full((n_vol,), delta))
            
        # Concatenate
        full_data = jnp.concatenate(all_data, axis=-1)
        full_bvals = jnp.concatenate(all_bvals, axis=0) * 1e6 # s/mm2 -> s/m2 if needed?
        # Usually bvals are s/mm2 (~1000). SI requires s/m2 (~1e9).
        # We apply conversion here if raw is standard.
        
        full_bvecs = jnp.concatenate(all_bvecs, axis=0)
        full_big = jnp.concatenate(all_big, axis=0)
        full_small = jnp.concatenate(all_small, axis=0)
        
        return {
            'data': full_data,
            'bvals': full_bvals,
            'bvecs': full_bvecs, 
            'big_delta': full_big,
            'small_delta': full_small,
            'affine': affine
        }
