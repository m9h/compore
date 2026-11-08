"""Deprecated alias of :mod:`lafix.model` (the module was renamed for the
0.1.0 release; the old name collided with the PRISM preprint the model
re-implements). Old names ``PrismConfig``, ``PrismFit``, ``fit_prism`` and
``prism_fit_to_pam`` keep working."""
import warnings as _w
from . import model as _m
from .model import *  # noqa: F401,F403
from .model import FixelConfig as PrismConfig, FixelFit as PrismFit, fit_fixels as fit_prism, fit_to_pam as prism_fit_to_pam  # noqa: F401

globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
_w.warn("lafix.prism_jax is deprecated; import lafix.model (FixelConfig, fit_fixels)", DeprecationWarning, stacklevel=2)
