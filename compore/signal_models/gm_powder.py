"""
Powder-averaged grey-matter models on multi-(b, Δ) schemes (doc 012 §7.4).

All kernels take per-measurement ``bvals_si`` (s/m²) and ``Delta``/``delta`` (s)
and return the spherical-mean (powder-averaged) normalised signal, which is
what NEXI / SANDI are fitted to. Narrow-pulse convention: the effective
diffusion time is t_d = Δ − δ/3 and the Kärger system is integrated over t_d
with rate b/t_d·D_c on its diagonal (the same convention as
``validation/lib/prism_exchange.karger_pgse`` and graymatter_swissknife).

Compartments
    stick     intra-neurite, D_i along the neurite, zero across
    ball      extra-neurite, isotropic D_e
    sphere    soma, GPD sphere of radius R_s with intrinsic D_s (SANDI)
    exchange  stick ⇄ ball with exchange time t_ex (NEXI, Jelescu 2022)

Models
    nexi_powder(b, Δ, δ, f, D_i, D_e, t_ex)                       NEXI (Jelescu 2022)
    sandi_powder(b, Δ, δ, f_n, f_s, D_i, D_e, R_s, D_s)             SANDI (Palombo 2020), no exchange
    nexi_soma_powder(b, Δ, δ, f_n, f_s, D_i, D_e, t_ex, R_s, D_s)   stick⇄ball exchange + impermeable soma

The soma pool is the same GPD sphere as the lafix glial compartment; the
exact finite-pulse check against ``matrix_formalism.sphere_mf_pgse`` is in
doc 011 §6 (|GPD − exact| ≤ 0.0014 at the CHUV protocol).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from compore.signal_models.sphere_models import g3_sphere

_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
GL_X = jnp.asarray(0.5 * (_GL_X + 1.0))          # cos θ ∈ [0, 1]
GL_W = jnp.asarray(0.5 * _GL_W)                  # ∫₀¹ dx


def diffusion_time(Delta, delta):
    return jnp.asarray(Delta) - jnp.asarray(delta) / 3.0


def powder_stick(bvals_si, D_i):
    """Spherical mean of exp(−b D_i x²): √π erf(√(bD)) / (2√(bD)), with the b→0 limit."""
    a = jnp.sqrt(jnp.maximum(jnp.asarray(bvals_si) * D_i, 1e-30))
    return jnp.where(a < 1e-6, 1.0 - a ** 2 / 3.0, jnp.sqrt(jnp.pi) * jax.scipy.special.erf(a) / (2.0 * a))


def _karger_2x2(b, td, D_a, D_b, f_a, t_ex):
    """Signal of a two-pool Kärger system after time td under rate b/td·D on each
    pool: M(td) = expm(−td·(diag(b/td·D) + K)) M(0), M(0) = (f_a, 1−f_a),
    K = [[k_ab, −k_ba], [−k_ab, k_ba]] with k_ab = (1−f_a)/t_ex, k_ba = f_a/t_ex.
    Closed form through the eigen-decomposition of the 2×2 matrix."""
    f_b = 1.0 - f_a
    k_ab = f_b / t_ex; k_ba = f_a / t_ex
    a11 = b * D_a + td * k_ab
    a22 = b * D_b + td * k_ba
    a12 = -td * k_ba
    a21 = -td * k_ab
    tr = a11 + a22; det = a11 * a22 - a12 * a21
    disc = jnp.sqrt(jnp.maximum(tr * tr / 4.0 - det, 0.0))
    l1 = tr / 2.0 + disc; l2 = tr / 2.0 - disc
    # expm(−A) = e^{−l2} I + (e^{−l1} − e^{−l2})/(l1 − l2) (A − l2 I)
    e1, e2 = jnp.exp(-l1), jnp.exp(-l2)
    c = jnp.where(jnp.abs(l1 - l2) > 1e-12, (e1 - e2) / jnp.where(jnp.abs(l1 - l2) > 1e-12, l1 - l2, 1.0), -e2)
    m11 = e2 + c * (a11 - l2); m12 = c * a12
    m21 = c * a21; m22 = e2 + c * (a22 - l2)
    return (m11 + m21) * f_a + (m12 + m22) * f_b


def nexi_powder(bvals_si, Delta, delta, f, D_i, D_e, t_ex):
    """NEXI: stick (D_i x²) ⇄ isotropic ball (D_e), exchange time t_ex, powder-averaged
    by Gauss–Legendre over x = cos θ. Shapes broadcast over measurements (M,)."""
    b = jnp.asarray(bvals_si, jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    td = diffusion_time(Delta, delta)

    def at_x(x):
        return _karger_2x2(b, td, D_i * x * x, D_e, f, t_ex)
    S = jnp.einsum("k,km->m", GL_W, jax.vmap(at_x)(GL_X))
    return S


def sphere_powder(bvals_si, Delta, delta, R_s, D_s):
    """GPD sphere attenuation (isotropic, so already the powder average)."""
    b = jnp.asarray(bvals_si)
    zeros = jnp.zeros((b.shape[0], 3)).at[:, 2].set(1.0)
    return g3_sphere(b, zeros, 2.0 * R_s, D_s, jnp.asarray(Delta), jnp.asarray(delta))


def sandi_powder(bvals_si, Delta, delta, f_n, f_s, D_i, D_e, R_s, D_s=3.0e-9):
    """SANDI: f_n stick + f_s sphere + (1 − f_n − f_s) ball, no exchange."""
    f_e = 1.0 - f_n - f_s
    return f_n * powder_stick(bvals_si, D_i) + f_s * sphere_powder(bvals_si, Delta, delta, R_s, D_s) \
        + f_e * jnp.exp(-jnp.asarray(bvals_si) * D_e)


def nexi_soma_powder(bvals_si, Delta, delta, f_n, f_s, D_i, D_e, t_ex, R_s, D_s=3.0e-9):
    """Exchanging stick ⇄ ball (fractions renormalised within the exchanging water)
    plus an impermeable soma pool f_s."""
    w = 1.0 - f_s
    f_in_exch = f_n / jnp.maximum(w, 1e-6)
    return w * nexi_powder(bvals_si, Delta, delta, f_in_exch, D_i, D_e, t_ex) \
        + f_s * sphere_powder(bvals_si, Delta, delta, R_s, D_s)


def rician_mean(S, sigma):
    """E[|S + n|] for complex Gaussian noise of sd σ per channel (Rician mean),
    via the Laguerre form: σ√(π/2) L_{1/2}(−S²/2σ²)."""
    S = jnp.asarray(S); x = S * S / (2.0 * sigma ** 2)
    i0e = jax.scipy.special.i0e(x / 2.0); i1e = jax.scipy.special.i1e(x / 2.0)
    L = jnp.exp(0.0) * ((1.0 + x) * i0e + x * i1e)          # e^{-x/2}·L_{1/2}(−x) with scaled Bessels
    return sigma * jnp.sqrt(jnp.pi / 2.0) * L


MODEL_PARAMS = {
    "nexi": ["f", "D_i", "D_e", "t_ex"],
    "sandi": ["f_n", "f_s", "D_i", "D_e", "R_s"],
    "nexi_soma": ["f_n", "f_s", "D_i", "D_e", "t_ex", "R_s"],
}
# prior boxes (SI): fractions, D in m²/s, t_ex in s, R_s in m — the ranges of Uhl 2026 for NEXI
PRIOR_BOXES = {
    "f": (0.15, 0.80), "f_n": (0.10, 0.80), "f_s": (0.0, 0.50),
    "D_i": (1.7e-9, 3.5e-9), "D_e": (0.5e-9, 1.5e-9), "t_ex": (1e-3, 70e-3), "R_s": (2e-6, 12e-6),
}


def model_fn(name):
    fns = {"nexi": nexi_powder, "sandi": sandi_powder, "nexi_soma": nexi_soma_powder}
    return fns[name]
