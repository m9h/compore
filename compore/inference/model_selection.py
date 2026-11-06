"""
Joint model-structure + parameter inference for powder-averaged GM models
(doc 012 §7.4). Two estimators of p(M, θ | S) over the candidate set
``MODELS = ("nexi", "sandi", "nexi_soma")`` of :mod:`compore.signal_models.gm_powder`:

1. **Laplace evidence**: per model, a bounded Levenberg–Marquardt fit
   (Optimistix, sigmoid-reparameterised to the prior box) and the Gauss–Newton
   Laplace approximation of the marginal likelihood under a uniform prior in
   the box; p(M | S) ∝ exp(log Z_M).
2. **Amortised**: a classifier q(M | S) (MLP, softmax) and one conditional
   normalising flow q(θ | S, M) per model (FlowJAX masked autoregressive,
   spline transformer), trained on simulations from the joint prior
   p(M) p(θ | M) with the measurement noise model below. This is the
   simulation-based joint posterior over structure and parameters.

Noise model: powder averages of ``n_dirs`` Rician-distributed directions with
per-direction σ relative to b0. The shell mean is approximately Gaussian with
mean ``rician_mean(S_shell, σ)`` and sd ``σ / sqrt(n_dirs)`` (b0 shells use
their own counts). σ is sampled per voxel from an SNR range.

Everything is in SI inside; parameters are reported in the box units of
``gm_powder.PRIOR_BOXES``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from compore.signal_models import gm_powder as gm

MODELS: Tuple[str, ...] = ("nexi", "sandi", "nexi_soma")


# ------------------------------------------------------------------ scheme --
@dataclass
class PowderScheme:
    """Powder-average shells: b (s/m²), Δ (s), δ (s), directions per shell."""
    bvals_si: np.ndarray
    Delta: np.ndarray
    delta: np.ndarray
    n_dirs: np.ndarray

    @classmethod
    def from_features(cls, features: Sequence[Tuple[int, int]], counts: Sequence[int], delta_s: float = 5e-3):
        f = np.asarray(features, float); return cls(f[:, 0] * 1e6, f[:, 1] * 1e-3, np.full(len(f), delta_s), np.asarray(counts, float))

    @property
    def n(self):
        return len(self.bvals_si)


# ------------------------------------------------------------- parameters --
def box(model: str) -> Tuple[np.ndarray, np.ndarray]:
    names = gm.MODEL_PARAMS[model]
    lo = np.array([gm.PRIOR_BOXES[k][0] for k in names]); hi = np.array([gm.PRIOR_BOXES[k][1] for k in names])
    return lo, hi


def _valid(model: str, theta: jnp.ndarray) -> jnp.ndarray:
    """Physical constraints inside the box: fractions sum < 1, D_i ≥ D_e."""
    names = gm.MODEL_PARAMS[model]; d = {k: theta[..., i] for i, k in enumerate(names)}
    ok = d["D_i"] >= d["D_e"]
    if "f_s" in d:
        ok = ok & (d["f_n"] + d["f_s"] <= 0.95)
    return ok


def sample_prior(key, model: str, n: int) -> jnp.ndarray:
    """Uniform in the box with the constraints enforced by rejection (vectorised)."""
    lo, hi = box(model); P = len(lo)
    k1, k2 = jax.random.split(key)
    th = lo + (hi - lo) * jax.random.uniform(k1, (2 * n + 64, P))
    ok = _valid(model, th)
    idx = jnp.argsort(~ok)[:n]           # valid ones first (stable enough for a prior sampler)
    th = th[idx]
    # fallback: replace invalid survivors by swapping D_i/D_e and shrinking f_s
    names = gm.MODEL_PARAMS[model]
    i_di, i_de = names.index("D_i"), names.index("D_e")
    swap = th[:, i_di] < th[:, i_de]
    th = th.at[:, i_di].set(jnp.where(swap, th[:, i_de], th[:, i_di])).at[:, i_de].set(jnp.where(swap, th[:, i_di], th[:, i_de]))
    if "f_s" in names:
        i_fn, i_fs = names.index("f_n"), names.index("f_s")
        excess = jnp.maximum(th[:, i_fn] + th[:, i_fs] - 0.95, 0.0)
        th = th.at[:, i_fs].set(th[:, i_fs] - excess)
    return th


def forward(model: str, theta: jnp.ndarray, sch: PowderScheme) -> jnp.ndarray:
    """Noise-free powder signal (n_shells,) for one parameter vector."""
    fn = gm.model_fn(model)
    return fn(jnp.asarray(sch.bvals_si), jnp.asarray(sch.Delta), jnp.asarray(sch.delta), *[theta[i] for i in range(len(gm.MODEL_PARAMS[model]))])


def forward_batch(model: str, theta: jnp.ndarray, sch: PowderScheme) -> jnp.ndarray:
    return jax.vmap(lambda t: forward(model, t, sch))(theta)


# ------------------------------------------------------------------ noise --
def noisy_powder(key, S: jnp.ndarray, sigma: jnp.ndarray, sch: PowderScheme) -> jnp.ndarray:
    """Shell means of Rician directions: mean rician_mean(S, σ), sd σ/√n_dirs."""
    mu = gm.rician_mean(S, sigma[..., None])
    sd = sigma[..., None] / jnp.sqrt(jnp.asarray(sch.n_dirs))
    return mu + sd * jax.random.normal(key, S.shape)


def simulate_joint(key, sch: PowderScheme, n: int, snr_range=(15.0, 60.0), model_probs=None):
    """(model index (n,), θ padded to max P (n, Pmax), σ (n,), noisy powder S (n, n_shells))."""
    Pmax = max(len(gm.MODEL_PARAMS[m]) for m in MODELS)
    k_m, k_s, k_n, *k_t = jax.random.split(key, 3 + len(MODELS))
    probs = jnp.asarray(model_probs if model_probs is not None else np.full(len(MODELS), 1.0 / len(MODELS)))
    m_idx = jax.random.choice(k_m, len(MODELS), (n,), p=probs)
    sigma = 1.0 / jax.random.uniform(k_s, (n,), minval=snr_range[0], maxval=snr_range[1])
    theta = jnp.zeros((n, Pmax)); S = jnp.zeros((n, sch.n))
    for j, m in enumerate(MODELS):
        th = sample_prior(k_t[j], m, n); P = th.shape[1]
        Sm = forward_batch(m, th, sch)
        sel = (m_idx == j)[:, None]
        theta = jnp.where(sel, jnp.pad(th, ((0, 0), (0, Pmax - P))), theta)
        S = jnp.where(sel, Sm, S)
    return m_idx, theta, sigma, noisy_powder(k_n, S, sigma, sch)


# ------------------------------------------------- estimator 1: Laplace ----
def _unit_to_theta(u, lo, hi):
    return lo + (hi - lo) * jax.nn.sigmoid(u)


def fit_model_laplace(model: str, y: jnp.ndarray, sigma: float, sch: PowderScheme, n_restarts: int = 6,
                      key=0, max_steps: int = 200) -> Dict:
    """Bounded LM fit of one model to one powder signal with Gaussian shell
    noise sd σ/√n_dirs, then the Gauss–Newton Laplace log-evidence under the
    uniform box prior: log Z = log L(θ̂) − log|box| + (P/2) log 2π − ½ log|H|,
    with the volume element of the sigmoid reparameterisation accounted for
    by evaluating H in θ units."""
    import optimistix as optx
    lo, hi = box(model); lo_j, hi_j = jnp.asarray(lo), jnp.asarray(hi); P = len(lo)
    sd = jnp.asarray(sigma / np.sqrt(sch.n_dirs))

    def resid_u(u, _):
        th = _unit_to_theta(u, lo_j, hi_j)
        return (gm.rician_mean(forward(model, th, sch), sigma) - y) / sd
    solver = optx.LevenbergMarquardt(rtol=1e-7, atol=1e-9)
    rng = np.random.default_rng(key); best = None
    for r in range(n_restarts):
        u0 = jnp.asarray(rng.normal(0, 1.2, P), jnp.float32)
        sol = optx.least_squares(resid_u, solver, u0, max_steps=max_steps, throw=False)
        c = float(jnp.sum(resid_u(sol.value, None) ** 2))
        if np.isfinite(c) and (best is None or c < best[0]):
            best = (c, sol.value)
    u = best[1]; th = _unit_to_theta(u, lo_j, hi_j)

    # Curvature in unit-box coordinates z ∈ (0,1)^P (θ = lo + (hi − lo) z): the raw
    # θ scales span 1 (fractions) to 1e-9 (D), which float32 cannot invert jointly.
    width = hi_j - lo_j; z_hat = (th - lo_j) / width

    def resid_z(z):
        return (gm.rician_mean(forward(model, lo_j + width * z, sch), sigma) - y) / sd
    Jz = jax.jacfwd(resid_z)(z_hat)                              # (n_shells, P)
    Hz = Jz.T @ Jz; Hz = Hz + 1e-9 * jnp.eye(P) * jnp.maximum(jnp.mean(jnp.diag(Hz)), 1e-30)
    sign, logdet_z = jnp.linalg.slogdet(Hz)
    logdet = logdet_z - 2.0 * jnp.sum(jnp.log(width))           # |H_θ| = |H_z| / Π width²
    chi2 = best[0]
    logL = -0.5 * chi2 - jnp.sum(jnp.log(sd)) - 0.5 * sch.n * jnp.log(2 * jnp.pi)
    log_box = jnp.sum(jnp.log(width))
    logZ = logL - log_box + 0.5 * P * jnp.log(2 * jnp.pi) - 0.5 * logdet
    cov_z = jnp.linalg.inv(Hz)
    sd_theta = jnp.sqrt(jnp.maximum(jnp.diag(cov_z), 0.0)) * width
    return {"theta": np.asarray(th), "theta_sd": np.asarray(sd_theta), "chi2": chi2,
            "logL": float(logL), "logZ": float(logZ), "n_params": P, "at_bound": bool(jnp.any(jnp.abs(u) > 6.0))}


def laplace_model_posterior(y, sigma, sch: PowderScheme, models=MODELS, **kw) -> Dict:
    fits = {m: fit_model_laplace(m, jnp.asarray(y), float(sigma), sch, **kw) for m in models}
    lz = np.array([fits[m]["logZ"] for m in models]); p = np.exp(lz - lz.max()); p /= p.sum()
    return {"fits": fits, "p_model": dict(zip(models, p)), "logZ": dict(zip(models, lz))}


# ------------------------------------------- estimator 2: amortised ---------
class ModelClassifier(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key, n_in: int, n_models: int, width: int = 128, depth: int = 3):
        self.mlp = eqx.nn.MLP(n_in, n_models, width, depth, activation=jax.nn.gelu, key=key)

    def __call__(self, s):
        return jax.nn.log_softmax(self.mlp(s))


def _features(S, sigma):
    """Network inputs: log powder signals and log σ (the noise level is observable from b0 repeats)."""
    return jnp.concatenate([jnp.log(jnp.clip(S, 1e-4, None)), jnp.log(sigma)[..., None]], -1)


@dataclass
class AmortisedPosterior:
    classifier: ModelClassifier
    flows: Dict[str, object]
    sch: PowderScheme
    train_log: dict

    def p_model(self, S, sigma) -> np.ndarray:
        return np.asarray(jnp.exp(jax.vmap(self.classifier)(_features(jnp.asarray(S), jnp.asarray(sigma)))))

    def sample_theta(self, key, model: str, S, sigma, n: int) -> np.ndarray:
        """(n, P) posterior samples in physical units for one voxel."""
        lo, hi = box(model)
        if not jnp.issubdtype(jnp.asarray(key).dtype, jax.dtypes.prng_key):
            key = jax.random.wrap_key_data(key)                  # FlowJAX needs typed keys
        x = _features(jnp.asarray(S), jnp.asarray(sigma))
        u = self.flows[model].sample(key, (n,), condition=x)
        return np.asarray(lo + (hi - lo) * jax.nn.sigmoid(u))

    def log_prob_theta(self, model: str, theta, S, sigma):
        lo, hi = box(model); z = jnp.clip((jnp.asarray(theta) - lo) / (hi - lo), 1e-6, 1 - 1e-6)
        u = jnp.log(z) - jnp.log1p(-z)
        x = _features(jnp.asarray(S), jnp.asarray(sigma))
        return self.flows[model].log_prob(u, condition=x) - jnp.sum(jnp.log((hi - lo) * z * (1 - z)))


def train_amortised(key, sch: PowderScheme, n_sims: int = 200_000, steps_clf: int = 3000, steps_flow: int = 4000,
                    batch: int = 1024, snr_range=(15.0, 60.0), flow_layers: int = 5, nn_width: int = 64, knots: int = 8,
                    lr: float = 1e-3, log: Callable = print) -> AmortisedPosterior:
    import optax
    import flowjax.distributions as dist
    import flowjax.flows as flows
    import flowjax.bijections as bij
    k_sim, k_clf, *k_fl = jax.random.split(key, 2 + len(MODELS))
    m_idx, theta, sigma, S = simulate_joint(k_sim, sch, n_sims, snr_range)
    X = _features(S, sigma)
    n_in = X.shape[1]
    # classifier
    clf = ModelClassifier(k_clf, n_in, len(MODELS))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(lr, steps_clf, 1e-2)))
    st = opt.init(eqx.filter(clf, eqx.is_inexact_array))

    @eqx.filter_jit
    def step_clf(clf, st, k):
        idx = jax.random.randint(k, (batch,), 0, X.shape[0])
        def loss(c):
            lp = jax.vmap(c)(X[idx]); return -jnp.mean(jnp.take_along_axis(lp, m_idx[idx][:, None], 1))
        l, g = eqx.filter_value_and_grad(loss)(clf); u, st = opt.update(g, st, eqx.filter(clf, eqx.is_inexact_array))
        return eqx.apply_updates(clf, u), st, l
    kk = jax.random.key(1)
    for i in range(steps_clf):
        kk, sk = jax.random.split(kk); clf, st, l = step_clf(clf, st, sk)
    acc = float(jnp.mean(jnp.argmax(jax.vmap(clf)(X[:20000]), 1) == m_idx[:20000]))
    log(f"  classifier: final NLL {float(l):.4f}, train accuracy {acc:.3f}")
    # per-model flows on the unit-logit scale
    fl = {}; flow_log = {}
    for j, m in enumerate(MODELS):
        lo, hi = box(m); P = len(lo)
        sel = np.asarray(m_idx == j); th = theta[sel][:, :P]; Xm = X[sel]
        z = jnp.clip((th - lo) / (hi - lo), 1e-5, 1 - 1e-5); U = jnp.log(z) - jnp.log1p(-z)
        flow = flows.masked_autoregressive_flow(key=k_fl[j], base_dist=dist.StandardNormal((P,)), cond_dim=n_in,
                                                transformer=bij.RationalQuadraticSpline(knots=knots, interval=4.0),
                                                flow_layers=flow_layers, nn_width=nn_width, nn_depth=2)
        optf = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(lr, steps_flow, 1e-2)))
        stf = optf.init(eqx.filter(flow, eqx.is_inexact_array))

        @eqx.filter_jit
        def step_flow(flow, stf, k):
            idx = jax.random.randint(k, (batch,), 0, U.shape[0])
            def loss(f): return -jnp.mean(f.log_prob(U[idx], condition=Xm[idx]))
            l, g = eqx.filter_value_and_grad(loss)(flow); u, stf = optf.update(g, stf, eqx.filter(flow, eqx.is_inexact_array))
            return eqx.apply_updates(flow, u), stf, l
        kk = jax.random.key(10 + j)
        for i in range(steps_flow):
            kk, sk = jax.random.split(kk); flow, stf, l = step_flow(flow, stf, sk)
        fl[m] = flow; flow_log[m] = float(l); log(f"  flow {m}: final NLL {float(l):.4f} on {int(sel.sum())} sims")
    return AmortisedPosterior(classifier=clf, flows=fl, sch=sch,
                              train_log={"n_sims": n_sims, "clf_acc": acc, "flow_nll": flow_log, "snr_range": snr_range})


# ---------------------------------------------------------------- scoring --
def sbc_ranks(post: AmortisedPosterior, key, n_test: int = 300, n_post: int = 200) -> Dict[str, np.ndarray]:
    """Simulation-based calibration ranks per model and parameter (uniform if calibrated)."""
    k_sim, k_s = jax.random.split(key)
    m_idx, theta, sigma, S = simulate_joint(k_sim, post.sch, n_test)
    out = {}
    for j, m in enumerate(MODELS):
        sel = np.flatnonzero(np.asarray(m_idx) == j); P = len(box(m)[0]); ranks = []
        for i in sel:
            k_s, kk = jax.random.split(k_s)
            samp = post.sample_theta(kk, m, S[i], sigma[i], n_post)
            ranks.append((samp < np.asarray(theta[i, :P])).sum(0))
        out[m] = np.asarray(ranks)
    return out
