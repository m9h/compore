"""
dmipy-jax: A JAX-accelerated port of the Diffusion Microstructure Imaging in Python (Dmipy) library.

This package provides GPU-accelerated signal models, fitting algorithms, and tools for 
microstructure imaging. It mirrors the structure of the original dmipy library but leverages
JAX for automatic differentiation, vectorization, and compilation.

Submodules
----------
- `acquisition`: Simplified acquisition scheme handling for JAX.
- `cylinder`: Orientation-dispersed cylinder models (e.g., Stick).
- `gaussian`: Gaussian diffusion models (e.g., Ball).
- `sphere`: Restricted diffusion in spheres.
- `fitting`: Generic voxel-wise fitting using JAXopt.
- `composer`: Utilities for composing multi-compartment models.
- `inference.mcmc`: Bayesian inference using Blackjax (NUTS).
"""

import importlib

__all__ = ['acquisition', 'cylinder', 'gaussian', 'sphere', 'fitting', 'inverse',
           'composer', 'simulation', 'signal_models', 'prism']

# Submodules load on first attribute access, so `import dmipy_jax.prism` pulls in
# jax, optax, dipy and the compartment kernels but not diffrax / scico / flowjax
# (doc 012 §7: the product must import with the core dependencies only).
def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f"dmipy_jax.{name}")
    raise AttributeError(f"module 'dmipy_jax' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
