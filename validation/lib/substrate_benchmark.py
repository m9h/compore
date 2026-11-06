"""Compatibility shim (2026-11-04, doc 012 §7.3): the sphere-substrate geometry,
SDFs and PGSE walker moved into the wheel as
:mod:`compore.simulation.substrates.spheres` so the oracle protocol can use
them. Validation scripts keep importing ``validation.lib.substrate_benchmark``."""
from compore.simulation.substrates.spheres import *  # noqa: F401,F403
from compore.simulation.substrates import spheres as _s

# private helpers some scripts and tests reach into
for _n in dir(_s):
    if _n.startswith("_") and not _n.startswith("__"):
        globals()[_n] = getattr(_s, _n)
del _n
