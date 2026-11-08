"""Deprecated alias of :mod:`lafix.uncertainty` (renamed for the 0.1.0 release)."""
import warnings as _w
from . import uncertainty as _m
from .uncertainty import *  # noqa: F401,F403

globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
_w.warn("lafix.prism_uncertainty is deprecated; import lafix.uncertainty", DeprecationWarning, stacklevel=2)
