"""
Neural emulator over an oracle library (doc 008 §11.2, doc 012 §7.3).

An Equinox MLP ensemble ``f(θ, b, ĝ) → log S`` trained on a
:class:`~compore.library.storage.SimulationLibrary` written by a
:class:`~compore.simulation.oracle.SubstrateOracle`. Per-measurement inputs
(b and the antipodally-symmetric features of ĝ) make the emulator valid for
any scheme inside the training b-range and at any voxel orientation when the
gradient directions are rotated into the substrate frame.

Two consumers:

* :meth:`LibraryEmulator.as_model_simulator` wraps the ensemble mean as a
  :class:`~compore.pipeline.simulator.ModelSimulator` forward function, so the
  SBI pipeline (``train_sbi``) can be trained on Monte Carlo physics instead of
  closed-form kernels (the "oracle → emulator → NPE" path of CLAUDE.md).
* :meth:`LibraryEmulator.invert` fits θ and a rotation to one signal with
  Optimistix Levenberg–Marquardt and random restarts (emulator-in-the-loop
  inversion; recovered f_intra r = 0.98 vs MC truth in doc 008 §11.2).

The library's ``params`` are the training targets; use the measured
descriptors as ``params`` when request ≠ realisation (see
``oracle.generate_library``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx


class _MLPEmulator(eqx.Module):
    mlp: eqx.nn.MLP
    theta_dim: int = eqx.field(static=True)

    def __init__(self, key, theta_dim: int, width: int = 256, depth: int = 4):
        self.theta_dim = theta_dim
        self.mlp = eqx.nn.MLP(theta_dim + 1 + 6, 1, width, depth, activation=jax.nn.gelu, key=key)

    def __call__(self, theta_n, b_n, g):
        feat = jnp.array([g[0] ** 2, g[1] ** 2, g[2] ** 2, g[0] * g[1], g[0] * g[2], g[1] * g[2]])
        return self.mlp(jnp.concatenate([theta_n, jnp.array([b_n]), feat]))[0]

    def signal(self, theta_n, b_n, G):
        return jnp.exp(jax.vmap(lambda g, b: self(theta_n, b, g))(G, b_n))


def _rotation(v):
    th = jnp.linalg.norm(v) + 1e-12; k = v / th
    K = jnp.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return jnp.eye(3) + jnp.sin(th) * K + (1 - jnp.cos(th)) * (K @ K)


@dataclass
class LibraryEmulator:
    models: List[_MLPEmulator]
    lo: np.ndarray
    hi: np.ndarray
    b_scale: float
    parameter_names: List[str]
    train_log: dict
    width: int = 256
    depth: int = 4

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, library, *, n_models: int = 5, steps: int = 6000, batch: int = 4096, width: int = 256, depth: int = 4,
            lr: float = 2e-3, key: int = 0, params: Optional[np.ndarray] = None, parameter_names: Optional[Sequence[str]] = None,
            log: Callable = print) -> "LibraryEmulator":
        """Train an ensemble on ``library`` (signals (N,M), b0-normalised, > 0).
        ``params`` overrides ``library.params`` (e.g. measured descriptors)."""
        theta = np.asarray(params if params is not None else library.params, float)
        S = np.asarray(library.signals, float)
        md = library.metadata
        bvals = np.asarray(md["bvals_si"], float); bvecs = np.asarray(md["bvecs"], float)
        ok = np.isfinite(theta).all(1) & np.isfinite(S).all(1) & (S > 0).all(1)
        theta, S = theta[ok], S[ok]
        lo, hi = theta.min(0), theta.max(0); hi = np.where(hi > lo, hi, lo + 1.0)
        b_scale = float(bvals.max())
        tn = (theta - lo) / (hi - lo); bn = bvals / b_scale
        X_theta = jnp.asarray(np.repeat(tn, len(bvals), 0), jnp.float32)
        X_b = jnp.asarray(np.tile(bn, len(theta)), jnp.float32)
        X_g = jnp.asarray(np.tile(bvecs, (len(theta), 1)), jnp.float32)
        Y = jnp.asarray(np.log(np.clip(S, 1e-4, None)).reshape(-1), jnp.float32)
        import optax
        models, losses = [], []
        for m in range(n_models):
            em = _MLPEmulator(jax.random.key(key + m), theta.shape[1], width, depth)
            opt = optax.adam(optax.warmup_cosine_decay_schedule(0, lr, min(300, max(steps // 4, 1)), steps, lr / 200))
            st = opt.init(eqx.filter(em, eqx.is_inexact_array))

            @eqx.filter_jit
            def step(em, st, k):
                idx = jax.random.randint(k, (batch,), 0, Y.shape[0])
                def loss(e):
                    return jnp.mean((jax.vmap(e)(X_theta[idx], X_b[idx], X_g[idx]) - Y[idx]) ** 2)
                l, g = eqx.filter_value_and_grad(loss)(em)
                u, st = opt.update(g, st, eqx.filter(em, eqx.is_inexact_array))
                return eqx.apply_updates(em, u), st, l
            kk = jax.random.key(100 + key + m)
            for _ in range(steps):
                kk, sk = jax.random.split(kk); em, st, l = step(em, st, sk)
            models.append(em); losses.append(float(l))
            log(f"  emulator {m}: final batch MSE(log S) {float(l):.5f}")
        names = list(parameter_names or md.get("descriptor_names") or library.parameter_names)
        return cls(models=models, lo=lo, hi=hi, b_scale=b_scale, parameter_names=names,
                   train_log={"final_mse": losses, "n_train": int(ok.sum()), "steps": steps}, width=width, depth=depth)

    # ------------------------------------------------------------- forward
    def normalise(self, theta):
        return (jnp.asarray(theta) - self.lo) / (self.hi - self.lo)

    def ensemble_signal(self, theta, bvals_si, bvecs):
        """(E, M) one signal per ensemble member; θ in physical units."""
        tn = self.normalise(theta); bn = jnp.asarray(bvals_si) / self.b_scale; G = jnp.asarray(bvecs)
        return jnp.stack([m.signal(tn, bn, G) for m in self.models])

    def __call__(self, theta, bvals_si, bvecs):
        """Ensemble-mean signal (M,)."""
        return jnp.mean(self.ensemble_signal(theta, bvals_si, bvecs), 0)

    def spread(self, theta, bvals_si, bvecs):
        return jnp.std(self.ensemble_signal(theta, bvals_si, bvecs), 0)

    # --------------------------------------------------- SBI pipeline adapter
    def as_model_simulator(self, acquisition, *, ranges: Optional[dict] = None, snr: float = 30.0, **kw):
        """A :class:`~compore.pipeline.simulator.ModelSimulator` whose forward
        model is this emulator. ``acquisition`` needs ``.bvalues`` (SI) and
        ``.gradient_directions`` (JaxAcquisition does). Parameter ranges default
        to the library's hull."""
        from compore.pipeline.simulator import ModelSimulator
        rng = ranges or {n: (float(l), float(h)) for n, l, h in zip(self.parameter_names, self.lo, self.hi)}

        def forward(params, acq):
            return self(params, acq.bvalues, acq.gradient_directions)
        return ModelSimulator(forward, self.parameter_names, rng, acquisition, snr=snr, **kw)

    # ------------------------------------------------------------ inversion
    def invert(self, y, bvals_si, bvecs, *, n_restarts: int = 8, key: int = 0, rotate: bool = True, max_steps: int = 200):
        """Levenberg–Marquardt through the ensemble mean over (θ, optional
        rotation of the gradient directions into the substrate frame).
        Returns dict(theta, axis1, cost, spread)."""
        import optimistix as optx
        b_n = jnp.asarray(bvals_si) / self.b_scale; G0 = jnp.asarray(bvecs); yj = jnp.asarray(y); P = len(self.lo)

        def resid(p, _):
            tn = jax.nn.sigmoid(p[:P]); G = G0 @ _rotation(p[P:]) if rotate else G0
            pred = jnp.mean(jnp.stack([m.signal(tn, b_n, G) for m in self.models]), 0)
            return pred - yj
        solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-8)
        rng = np.random.default_rng(key); best = None
        for _ in range(n_restarts):
            p0 = jnp.asarray(np.concatenate([rng.normal(0, 1.0, P), rng.normal(0, 1.5, 3) if rotate else []]), jnp.float32)
            sol = optx.least_squares(resid, solver, p0, max_steps=max_steps, throw=False)
            c = float(jnp.sum(resid(sol.value, None) ** 2))
            if best is None or c < best[0]:
                best = (c, sol.value)
        p = best[1]; tn = jax.nn.sigmoid(p[:P]); R = _rotation(p[P:]) if rotate else jnp.eye(3)
        theta = np.asarray(tn) * (self.hi - self.lo) + self.lo
        spread = float(jnp.mean(jnp.std(jnp.stack([m.signal(tn, b_n, G0 @ R) for m in self.models]), 0)))
        return {"theta": theta, "axis1": np.asarray(R @ jnp.array([0.0, 0.0, 1.0])), "cost": best[0], "spread": spread}

    # ------------------------------------------------------------- persistence
    def save(self, path: str):
        """``<path>`` (JSON header) + ``<path>.eqx`` (Equinox leaves of the ensemble)."""
        import json
        header = {"theta_dim": int(len(self.lo)), "n_models": len(self.models), "width": self.width, "depth": self.depth,
                  "lo": np.asarray(self.lo).tolist(), "hi": np.asarray(self.hi).tolist(), "b_scale": self.b_scale,
                  "parameter_names": self.parameter_names, "train_log": self.train_log}
        with open(path, "w") as f:
            json.dump(header, f, indent=1)
        with open(str(path) + ".eqx", "wb") as f:
            for m in self.models:
                eqx.tree_serialise_leaves(f, m)

    @classmethod
    def load(cls, path: str) -> "LibraryEmulator":
        import json
        with open(path) as f:
            h = json.load(f)
        models = []
        with open(str(path) + ".eqx", "rb") as f:
            for m in range(h["n_models"]):
                skel = _MLPEmulator(jax.random.key(0), h["theta_dim"], h["width"], h["depth"])
                models.append(eqx.tree_deserialise_leaves(f, skel))
        return cls(models=models, lo=np.asarray(h["lo"]), hi=np.asarray(h["hi"]), b_scale=h["b_scale"],
                   parameter_names=h["parameter_names"], train_log=h["train_log"], width=h["width"], depth=h["depth"])
