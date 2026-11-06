"""Data loaders for diffusion MRI datasets.

This package provides standardised loaders for reading dMRI data from various
sources and formats into JAX arrays suitable for the SBI pipeline.

Supported data sources:

* :class:`~compore.io.bids.BIDSLoader` -- BIDS-compliant datasets
  (via ``pybids``).
* :mod:`~compore.io.hcp` -- Human Connectome Project data (via S3/boto3).
* :class:`~compore.io.multi_te.MultiTELoader` -- multi-echo-time
  diffusion datasets.
* :class:`~compore.io.wand.WANDLoader` -- WAND (CUBRIC) AxCaliber data
  (via GIN).
* :class:`~compore.io.mica.MicaMICsLoader` -- MICA-MICs multi-shell
  dataset.
* :mod:`~compore.io.lowfield` -- OpenNeuro ultra-low-field brain MRI
  (ds006557).
* :mod:`~compore.io.mesh` -- surface mesh IO (OFF, VTK) for FEM
  simulation.
* :mod:`~compore.io.datasets` -- convenience loaders for standard
  validation datasets (Stanford HARDI, Sherbrooke 3-shell, BigMac).
* :func:`~compore.io.bids.save_bids_derivative` -- save parameter maps
  as BIDS derivatives with JSON sidecars.

All loaders convert b-values to SI units (s/m^2) and gradient directions
to shape ``(N, 3)`` for direct use with :class:`~compore.acquisition.JaxAcquisition`.
"""

from .bids import BIDSLoader, save_bids_derivative
