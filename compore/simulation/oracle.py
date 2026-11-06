"""
Oracle protocol (doc 012 §7.3): Monte Carlo engines behind one interface.

A *substrate oracle* takes a :class:`~compore.simulation.substrates.spheres.Substrate`
(axons and glia as sphere sets, SI units) and a PGSE scheme and returns the
compartment signals. Three engines are wrapped:

``jax``   the differentiable SDF walker in this package
          (:func:`~compore.simulation.substrates.spheres.simulate_substrate_signal`)
``mcmr``  MCMRSimulator.jl v1.1.0 (Cottaar, FMRIB) through ``julia/mcmr/substrate_pgse.jl``
``mcds``  Permeable_MCDS (CHUV, ``whitematter`` branch) through its CLI binary

The parity of the three is in doc 008 §11.1 (extra-cellular RMS 0.01–0.015
JAX↔MCMR; MCDS freer; intra bracketed MCDS < JAX < MCMR because
overlapping-sphere chains over-restrict axially in MCMR). None of the
external engines is a Python dependency: they are found on the host or
``check_available()`` is False and the oracle raises on use.

Libraries: :meth:`SubstrateOracle.generate_library` runs a substrate sampler
through the oracle and stores (params, descriptors, signals) in a
:class:`~compore.library.storage.SimulationLibrary`, the HDF5 boundary the
SBI pipeline and :mod:`compore.simulation.emulator` consume.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np

from compore.simulation.substrates.spheres import Substrate, simulate_substrate_signal, GAMMA

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PGSEScheme:
    """One PGSE acquisition: b in s/m² (SI), unit gradient directions, δ/Δ in s."""
    bvals_si: np.ndarray
    bvecs: np.ndarray
    delta: float = 6e-3
    Delta: float = 12e-3

    def __post_init__(self):
        self.bvals_si = np.asarray(self.bvals_si, float)
        self.bvecs = np.asarray(self.bvecs, float)
        if self.bvals_si.max() < 1e5:
            raise ValueError("bvals must be SI (s/m²); the largest value looks like s/mm²")

    @property
    def n(self) -> int:
        return len(self.bvals_si)

    @property
    def G(self) -> np.ndarray:
        """Gradient amplitude per measurement, T/m."""
        return np.sqrt(self.bvals_si / (GAMMA ** 2 * self.delta ** 2 * (self.Delta - self.delta / 3.0)))


@dataclass
class OracleResult:
    S: np.ndarray                  # (M,) water-fraction-weighted total, b0 = 1
    S_intra: np.ndarray            # (M,)
    S_extra: np.ndarray            # (M,)
    S_glia: Optional[np.ndarray] = None
    engine: str = ""
    seconds: float = 0.0
    meta: Dict = field(default_factory=dict)


class SubstrateOracle(ABC):
    """Base class. Subclasses implement :meth:`check_available` and :meth:`_simulate`."""
    name: str = "base"

    @abstractmethod
    def check_available(self) -> bool: ...

    @abstractmethod
    def _simulate(self, sub: Substrate, scheme: PGSEScheme, *, D_intra: float, D_extra: float,
                  n_particles: int, seed: int) -> OracleResult: ...

    def simulate(self, sub: Substrate, scheme: PGSEScheme, *, D_intra: float = 2.0e-9,
                 D_extra: float = 2.0e-9, n_particles: int = 4000, seed: int = 0) -> OracleResult:
        if not self.check_available():
            raise RuntimeError(f"oracle '{self.name}' is not available on this host: {self.unavailable_reason()}")
        t0 = time.time()
        r = self._simulate(sub, scheme, D_intra=D_intra, D_extra=D_extra, n_particles=n_particles, seed=seed)
        r.engine = self.name
        r.seconds = time.time() - t0
        return r

    def unavailable_reason(self) -> str:
        return "see check_available()"

    @staticmethod
    def combine(sub: Substrate, S_in, S_ex, S_gl=None) -> np.ndarray:
        """Water-fraction weighting shared by every engine (myelin carries no water)."""
        fr = sub.fractions or {"intra": sub.f_intra, "extra": 1 - sub.f_intra, "glia": 0.0}
        water = fr["intra"] + fr["extra"] + fr["glia"]
        S = fr["intra"] / water * np.asarray(S_in) + fr["extra"] / water * np.asarray(S_ex)
        if fr["glia"] > 0 and S_gl is not None and np.all(np.isfinite(S_gl)):
            S = S + fr["glia"] / water * np.asarray(S_gl)
        return S

    def generate_library(self, sampler: Callable[[np.random.Generator, int], tuple], n: int,
                         scheme: PGSEScheme, *, parameter_names: Sequence[str],
                         descriptor_fn: Optional[Callable[[Substrate], Dict[str, float]]] = None,
                         seed: int = 0, out_path: Optional[str] = None, log=print, **sim_kw):
        """Run ``n`` substrates through the oracle.

        ``sampler(rng, i) -> (params: 1-D array, substrate)``; failures (the
        generators are not seed-stable) are skipped and counted. Descriptors
        are *measured* geometry labels (f_intra, ⟨cos²⟩, …) stored next to the
        requested parameters, because request ≠ realisation for CATERPillar.
        Returns a :class:`SimulationLibrary` whose ``metadata`` carries the
        scheme, engine, descriptors and skip count; saved to ``out_path`` if given.
        """
        from compore.library.storage import SimulationLibrary
        rng = np.random.default_rng(seed)
        P, D, S, Si, Se, skipped = [], [], [], [], [], 0
        desc_names = None
        for i in range(n):
            try:
                params, sub = sampler(rng, i)
                r = self.simulate(sub, scheme, seed=seed + i, **sim_kw)
            except Exception as e:  # noqa: BLE001 — generator / engine failures are data, not bugs
                skipped += 1
                log(f"  [{i}] skipped: {type(e).__name__}: {e}")
                continue
            d = descriptor_fn(sub) if descriptor_fn else {"f_intra": float(sub.f_intra)}
            if desc_names is None:
                desc_names = list(d)
            P.append(np.asarray(params, float)); D.append([d[k] for k in desc_names])
            S.append(r.S); Si.append(r.S_intra); Se.append(r.S_extra)
        if not S:
            raise RuntimeError("every substrate failed; nothing to store")
        import jax.numpy as jnp
        lib = SimulationLibrary(params=jnp.asarray(np.stack(P), jnp.float32), signals=jnp.asarray(np.stack(S), jnp.float32),
                                parameter_names=list(parameter_names),
                                metadata={"engine": self.name, "bvals_si": scheme.bvals_si.tolist(), "bvecs": scheme.bvecs.tolist(),
                                          "delta": scheme.delta, "Delta": scheme.Delta, "descriptor_names": desc_names,
                                          "descriptors": np.asarray(D, float).tolist(), "S_intra": np.stack(Si).tolist(),
                                          "S_extra": np.stack(Se).tolist(), "n_requested": n, "n_skipped": skipped,
                                          **{k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in sim_kw.items()}})
        if out_path:
            lib.save_hdf5(out_path)
        return lib


class JaxWalkerOracle(SubstrateOracle):
    """The package's own SDF walker (chord-periodic intra, box-periodic extra, glia)."""
    name = "jax"

    def __init__(self, dt: float = 1e-5, D_glia: Optional[float] = None):
        self.dt = dt; self.D_glia = D_glia

    def check_available(self) -> bool:
        return True

    def _simulate(self, sub, scheme, *, D_intra, D_extra, n_particles, seed):
        S, S_in, S_ex, S_gl = simulate_substrate_signal(sub, scheme.bvals_si, scheme.bvecs, D_intra=D_intra, D_extra=D_extra,
                                                        D_glia=self.D_glia, delta=scheme.delta, Delta=scheme.Delta, dt=self.dt,
                                                        n_particles=n_particles, seed=seed, return_glia=True)
        return OracleResult(S=S, S_intra=S_in, S_extra=S_ex, S_glia=None if np.all(np.isnan(S_gl)) else S_gl,
                            meta={"dt": self.dt})


class MCMROracle(SubstrateOracle):
    """MCMRSimulator.jl through ``julia/mcmr/substrate_pgse.jl`` (Julia 1.13 env ``julia/mcmr``).

    The script reads ``validation/mcmr/<tag>_spheres.csv`` / ``_ref.csv`` /
    ``meta.json`` relative to its working directory and writes
    ``<tag>_mcmr.csv``; the oracle stages those under ``workdir`` and runs
    Julia with ``cwd=workdir``. Glia are not passed (the script knows axon
    spheres only); ``S_glia`` is None.
    """
    name = "mcmr"

    def __init__(self, workdir: Optional[str] = None, julia: str = "julia", channel: str = "+1.13", threads: int = 16,
                 project: Optional[str] = None, script: Optional[str] = None, n_dirs: int = 0):
        self.workdir = Path(workdir or (Path(os.environ.get("TMPDIR", "/tmp")) / "sbi4dwi_mcmr"))
        self.julia = julia; self.channel = channel; self.threads = threads; self.n_dirs = n_dirs
        self.project = Path(project or REPO_ROOT / "julia" / "mcmr")
        self.script = Path(script or self.project / "substrate_pgse.jl")

    def check_available(self) -> bool:
        return shutil.which(self.julia) is not None and self.script.exists() and (self.project / "Manifest.toml").exists()

    def unavailable_reason(self) -> str:
        return f"julia={shutil.which(self.julia)}, script={self.script.exists()}, manifest={(self.project / 'Manifest.toml').exists()}"

    def _simulate(self, sub, scheme, *, D_intra, D_extra, n_particles, seed):
        if abs(D_intra - D_extra) > 1e-15:
            raise ValueError("substrate_pgse.jl uses one diffusivity; pass D_intra == D_extra")
        import pandas as pd
        tag = f"oracle_{seed}"
        d = self.workdir / "validation" / "mcmr"; d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": sub.centers_m[:, 0] * 1e6, "y": sub.centers_m[:, 1] * 1e6, "z": sub.centers_m[:, 2] * 1e6,
                      "r": sub.radii_m * 1e6, "axon": sub.axon_ids}).to_csv(d / f"{tag}_spheres.csv", index=False)
        zeros = np.zeros(scheme.n)
        np.savetxt(d / f"{tag}_ref.csv", np.column_stack([scheme.bvals_si / 1e6, scheme.bvecs, zeros, zeros, zeros]),
                   delimiter=",", header="b_s_mm2,gx,gy,gz,S,S_intra,S_extra", comments="")
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        meta[tag] = {"box_um": sub.box_m * 1e6, "f_intra": float(sub.f_intra), "axes": np.asarray(sub.axes).tolist(),
                     "n_spheres": int(len(sub.radii_m)), "delta_ms": scheme.delta * 1e3, "Delta_ms": scheme.Delta * 1e3,
                     "D_um2_ms": D_intra * 1e9}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        cmd = [self.julia, self.channel, "-t", str(self.threads), f"--project={self.project}", str(self.script), tag,
               str(n_particles), str(self.n_dirs)]
        r = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"MCMRSimulator failed ({r.returncode}):\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        out = pd.read_csv(d / f"{tag}_mcmr.csv")
        S_in = np.full(scheme.n, np.nan); S_ex = np.full(scheme.n, np.nan)
        S_in[out["i"].values - 1] = out["S_intra_mcmr"].values; S_ex[out["i"].values - 1] = out["S_extra_mcmr"].values
        return OracleResult(S=self.combine(sub, S_in, S_ex), S_intra=S_in, S_extra=S_ex,
                            meta={"n_in": int(out["n_in"].iloc[0]), "n_out": int(out["n_out"].iloc[0]), "stdout_tail": r.stdout[-400:]})


class MCDSOracle(SubstrateOracle):
    """Permeable_MCDS (CHUV) CLI. Walkers are wrapped by MCDS but obstacles are
    not, so spheres crossing a periodic face are replicated at ±L (doc 008 §11.1)."""
    name = "mcds"

    def __init__(self, binary: Optional[str] = None, workdir: Optional[str] = None, threads: int = 16, steps: int = 20000,
                 TE: float = 20e-3, images: bool = True):
        self.binary = Path(binary or os.environ.get("MCDS_BIN", "/home/mhough/dev/_external/Permeable_MCDS/MC-DC_Simulator_built"))
        self.workdir = Path(workdir or (Path(os.environ.get("TMPDIR", "/tmp")) / "sbi4dwi_mcds"))
        self.threads = threads; self.steps = steps; self.TE = TE; self.images = images

    def check_available(self) -> bool:
        return self.binary.exists() and os.access(self.binary, os.X_OK)

    def unavailable_reason(self) -> str:
        return f"binary {self.binary} missing or not executable (set MCDS_BIN)"

    def _geometry(self, sub: Substrate, path: Path):
        import itertools
        L = sub.box_m * 1e6
        x, y, z = (sub.centers_m * 1e6).T; rr = sub.radii_m * 1e6; ids = np.asarray(sub.axon_ids)
        rows = ["id_ax id_sph id_branch Type X Y Z Rin Rout P"]; next_id = int(ids.max()) + 1
        for aid in np.unique(ids):
            w = np.flatnonzero(ids == aid)
            rows += [f"{aid} {j} 0 axon {x[k]:.6f} {y[k]:.6f} {z[k]:.6f} {rr[k]:.6f} {rr[k]:.6f} 0" for j, k in enumerate(w)]
        if self.images:
            for shift in itertools.product((-L, 0.0, L), repeat=3):
                if shift == (0.0, 0.0, 0.0):
                    continue
                for aid in np.unique(ids):
                    w = np.flatnonzero(ids == aid)
                    xs, ys, zs, rs = x[w] + shift[0], y[w] + shift[1], z[w] + shift[2], rr[w]
                    keep = (xs > -rs) & (xs < L + rs) & (ys > -rs) & (ys < L + rs) & (zs > -rs) & (zs < L + rs)
                    if not keep.any():
                        continue
                    rows += [f"{next_id} {j} 0 axon {xs[k]:.6f} {ys[k]:.6f} {zs[k]:.6f} {rs[k]:.6f} {rs[k]:.6f} 0"
                             for j, k in enumerate(np.flatnonzero(keep))]
                    next_id += 1
        path.write_text("\n".join(rows) + "\n")

    def _scheme(self, scheme: PGSEScheme, path: Path):
        lines = ["VERSION: STEJSKALTANNER"]
        for b, g, G in zip(scheme.bvals_si, scheme.bvecs, scheme.G):
            gx, gy, gz = (g if np.linalg.norm(g) > 0 else (1.0, 0.0, 0.0))
            lines.append(f"{gx:.6f} {gy:.6f} {gz:.6f} {G if b > 0 else 0.0:.6f} {scheme.Delta:.6f} {scheme.delta:.6f} {self.TE:.6f}")
        path.write_text("\n".join(lines) + "\n")

    def _run(self, geom: Path, sch: Path, prefix: Path, L_mm: float, init: str, walkers: int, D: float, seed: int) -> np.ndarray:
        import glob
        conf = (f"N {walkers}\nT {self.steps}\nduration {self.TE:.6f}\ndiffusivity_intra {D}\ndiffusivity_extra {D}\n"
                f"scheme_file {sch}\nexp_prefix {prefix}\nscale_from_stu 1\nwrite_txt 1\nwrite_bin 0\nwrite_traj_file 0\n"
                f"num_process {self.threads}\nini_walkers_pos {init}\n<obstacle>\n<axons_list>\n{geom}\npermeability global 0.0\n"
                f"</axons_list>\n</obstacle>\n<voxel>\n0 0 0\n{L_mm} {L_mm} {L_mm}\n</voxel>\n<END>\n")
        cpath = Path(f"{prefix}_{init}.conf"); cpath.write_text(conf)
        r = subprocess.run([str(self.binary), str(cpath)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stdout[-1500:] + r.stderr[-500:])
        reals = sorted(glob.glob(f"{prefix}*_DWI.txt")); imags = sorted(glob.glob(f"{prefix}*_DWI_img.txt"))
        if not reals:
            raise FileNotFoundError(f"no DWI output for {prefix}: {r.stdout[-800:]}")
        re_ = sum(np.loadtxt(f) for f in reals); im = sum(np.loadtxt(f) for f in imags)
        S = np.sqrt(re_ ** 2 + im ** 2)
        return S / S[0]

    def _simulate(self, sub, scheme, *, D_intra, D_extra, n_particles, seed):
        if abs(D_intra - D_extra) > 1e-15:
            raise ValueError("the MCDS config here uses one diffusivity; pass D_intra == D_extra")
        d = self.workdir / f"oracle_{seed}"; d.mkdir(parents=True, exist_ok=True)
        geom, sch = d / "geometry.swc", d / "scheme.scheme"   # MCDS picks the parser by extension
        self._geometry(sub, geom); self._scheme(scheme, sch)
        L_mm = sub.box_m * 1e3
        S_in = self._run(geom, sch, d / "intra", L_mm, "intra", n_particles, D_intra, seed)
        S_ex = self._run(geom, sch, d / "extra", L_mm, "extra", n_particles, D_extra, seed)
        return OracleResult(S=self.combine(sub, S_in, S_ex), S_intra=S_in, S_extra=S_ex, meta={"steps": self.steps})


_REGISTRY: Dict[str, type] = {"jax": JaxWalkerOracle, "mcmr": MCMROracle, "mcds": MCDSOracle}


def get_oracle(name: str, **kw) -> SubstrateOracle:
    """``get_oracle("jax" | "mcmr" | "mcds", **constructor_kwargs)``."""
    try:
        return _REGISTRY[name](**kw)
    except KeyError:
        raise KeyError(f"unknown oracle {name!r}; known: {sorted(_REGISTRY)}") from None


def available_oracles() -> Dict[str, bool]:
    return {k: v().check_available() for k, v in _REGISTRY.items()}


def register_oracle(name: str, cls: type) -> None:
    _REGISTRY[name] = cls
