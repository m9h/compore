"""
Spectral (Laplacian) matrix formalism for PGSE signals in canonical pores.

Exact finite-pulse PGSE attenuation for diffusion with reflecting boundaries
in three axially symmetric geometries, written as a product of matrix
exponentials in the Laplace eigenbasis (Barzykin 1999, Grebenkov 2007):

* **cylindrical surface** (ring cross-section) — the myelin-water model of
  Canales-Rodríguez et al., Phys. Med. Biol. 2026 (arXiv:2603.23421),
  Eqs. 29–37: eigenvalues n²/r², reduced real (cosine) basis, tridiagonal
  coupling K̃ with √2 in the first off-diagonal;
* **cylinder interior** (disk cross-section), cosine-symmetric eigenmodes
  J_n(α_nk ρ) cos nθ with J_n'(α_nk) = 0;
* **sphere interior**, axisymmetric eigenmodes j_n(α_nk ρ) P_n(μ) with
  j_n'(α_nk) = 0.

For every geometry the pore is scaled to unit radius, so the dimensionless
eigenvalue vector Λ̂ and coupling matrix B̂ (matrix elements of x/R, or z/R)
are computed once on the host and cached; radius, diffusivity and the
sequence timing only enter the three scalars

    a = D δ / R²,   g = D (Δ − δ) / R²,   c = q R,   q = γ G δ = √(b / (Δ − δ/3))

that multiply them, so the kernels are differentiable in radius, D, δ and Δ.
For a uniform initial magnetisation only the ground mode is populated and
the normalised signal is the (0,0) element of the propagator product

    E = Σ_j conj(U_0j) e^{−g λ̂_j} U_j0,    U = expm(−a Λ̂ + i c B̂),

the second gradient lobe being the complex conjugate of the first
(paper §3.4). Only the truncation of the basis is approximate; the paper
reports M = 10 ring modes already at 1e-11 relative error for b ≤ 20 ms/µm².

Units: SI throughout (b in s/m², q in rad/m, lengths in m, times in s,
D in m²/s). ``bvals`` here are the usual full b-values; the perpendicular
projection q⊥ = q sin β is taken inside.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from compore.constants import bessel_derivative_roots, spherical_bessel_derivative_roots

__all__ = [
    "ring_basis", "disk_basis", "sphere_basis",
    "q_from_b",
    "ring_perpendicular_attenuation", "disk_perpendicular_attenuation", "sphere_attenuation",
    "cylinder_surface_pgse", "cylinder_mf_pgse", "sphere_mf_pgse",
    "cylinder_surface_spherical_mean",
]


# ---------------------------------------------------------------------------
# Dimensionless basis matrices (host side, cached)
# ---------------------------------------------------------------------------

def _gauss_legendre(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


@lru_cache(maxsize=None)
def ring_basis(n_modes: int = 10):
    """Λ̂ = diag(0, 1, 4, …, (M)²) and B̂ = K̃/2 in the reduced real basis
    {1/√2π, cos(nθ)/√π} (Canales-Rodríguez 2026 Eqs. 29–30). ``n_modes`` is
    the number of basis functions M + 1."""
    lam = np.arange(n_modes, dtype=float) ** 2
    K = np.zeros((n_modes, n_modes))
    i = np.arange(n_modes - 1)
    K[i, i + 1] = K[i + 1, i] = 1.0
    if n_modes > 1:
        K[0, 1] = K[1, 0] = np.sqrt(2.0)
    return lam, 0.5 * K


@lru_cache(maxsize=None)
def disk_basis(n_ang: int = 8, n_rad: int = 6):
    """Cosine-symmetric Neumann eigenmodes of the unit disk, φ_nk ∝
    J_n(α_nk ρ) cos nθ, n < n_ang, k < n_rad, α_00 = 0 first. Returns the
    eigenvalues α_nk² and the matrix of ⟨φ_n'k'| ρ cos θ |φ_nk⟩, both
    dimensionless; radial integrals by 256-node Gauss–Legendre, angular ones
    by a 512-point trapezoid (exact for these trigonometric polynomials)."""
    import scipy.special as ssp
    alpha = np.asarray(bessel_derivative_roots(n_rad, n_ang))          # alpha[k, n]
    rho, wr = _gauss_legendre(256, 0.0, 1.0)
    th = np.linspace(0.0, 2 * np.pi, 512, endpoint=False); dth = 2 * np.pi / th.size
    modes = [(n, k) for n in range(n_ang) for k in range(n_rad)]
    F = np.array([ssp.jv(n, alpha[k, n] * rho) for n, k in modes])      # (P, Q)
    A = np.array([np.cos(n * th) for n, k in modes])                    # (P, T)
    F /= np.sqrt((F ** 2 * rho * wr).sum(1) * (A ** 2).sum(1) * dth)[:, None]
    rad = (F * rho ** 2 * wr) @ F.T
    ang = (A * np.cos(th) * dth) @ A.T
    B = rad * ang
    lam = np.array([alpha[k, n] ** 2 for n, k in modes])
    return lam, 0.5 * (B + B.T)


@lru_cache(maxsize=None)
def sphere_basis(n_ang: int = 8, n_rad: int = 6):
    """Axisymmetric Neumann eigenmodes of the unit ball, φ_nk ∝
    j_n(α_nk ρ) P_n(μ), with the matrix of ⟨φ_n'k'| ρ μ |φ_nk⟩ (gradient along
    the symmetry axis). Same layout and conventions as :func:`disk_basis`."""
    import scipy.special as ssp
    alpha = np.asarray(spherical_bessel_derivative_roots(n_rad, n_ang))  # alpha[k, n]
    rho, wr = _gauss_legendre(256, 0.0, 1.0)
    mu, wm = _gauss_legendre(64, -1.0, 1.0)
    modes = [(n, k) for n in range(n_ang) for k in range(n_rad)]
    F = np.array([ssp.spherical_jn(n, alpha[k, n] * rho) for n, k in modes])
    A = np.array([ssp.eval_legendre(n, mu) for n, k in modes])
    F /= np.sqrt((F ** 2 * rho ** 2 * wr).sum(1) * 2 * np.pi * (A ** 2 * wm).sum(1))[:, None]
    rad = (F * rho ** 3 * wr) @ F.T
    ang = 2 * np.pi * (A * mu * wm) @ A.T
    B = rad * ang
    lam = np.array([alpha[k, n] ** 2 for n, k in modes])
    return lam, 0.5 * (B + B.T)


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------

def _pgse_attenuation(lam, B, a, c, g):
    """E = Σ_j conj(U_0j) e^{−g λ_j} U_j0 with U = expm(−a Λ̂ + i c B̂)
    (paper Eq. 34 with the conjugate-lobe reduction of §3.4)."""
    # modes with a·λ ≫ 700 are exactly zero in float64 during the pulse; the
    # cap only keeps the Padé scaling-and-squaring inside expm from overflowing
    # (NaN for a·λ ~ 1e6, e.g. a 0.1 µm ring with 50 modes)
    decay = jnp.minimum(a * lam, 2000.0)
    U = jax.scipy.linalg.expm(-jnp.diag(decay) + 1j * c * B)
    return jnp.real(jnp.sum(jnp.conj(U[0]) * jnp.exp(-g * lam) * U[:, 0]))


def _attenuation(basis, q, delta, Delta, radius, D):
    lam, B = (jnp.asarray(x) for x in basis)
    q = jnp.asarray(q)
    delta = jnp.broadcast_to(jnp.asarray(delta, q.dtype), q.shape)
    Delta = jnp.broadcast_to(jnp.asarray(Delta, q.dtype), q.shape)
    a = D * delta / radius ** 2
    g = D * (Delta - delta) / radius ** 2
    c = q * radius
    return jax.vmap(lambda a_, c_, g_: _pgse_attenuation(lam, B, a_, c_, g_))(a, c, g)


def q_from_b(bvals, delta, Delta):
    """q = γ G δ in rad/m from b = q² (Δ − δ/3); zero (with zero gradient)
    where b = 0."""
    bvals = jnp.asarray(bvals)
    tau = jnp.asarray(Delta) - jnp.asarray(delta) / 3.0
    safe = jnp.where(bvals > 0, bvals, 1.0)
    return jnp.where(bvals > 0, jnp.sqrt(safe / tau), 0.0)


def ring_perpendicular_attenuation(q, delta, Delta, radius, D, n_modes: int = 10):
    """E⊥ for diffusion on a cylindrical surface, gradient ⊥ to the axis,
    q = |q⊥| in rad/m (paper Eq. 34)."""
    return _attenuation(ring_basis(n_modes), q, delta, Delta, radius, D)


def disk_perpendicular_attenuation(q, delta, Delta, radius, D, n_ang: int = 8, n_rad: int = 6):
    """E⊥ for diffusion inside a cylinder (disk cross-section), q = |q⊥| in rad/m."""
    return _attenuation(disk_basis(n_ang, n_rad), q, delta, Delta, radius, D)


def sphere_attenuation(q, delta, Delta, radius, D, n_ang: int = 8, n_rad: int = 6):
    """E for diffusion inside a sphere, q = |q| in rad/m."""
    return _attenuation(sphere_basis(n_ang, n_rad), q, delta, Delta, radius, D)


# ---------------------------------------------------------------------------
# Compartment signals (dmipy-style signatures, SI)
# ---------------------------------------------------------------------------

def _split(bvals, bvecs, mu, delta, Delta):
    bvals = jnp.asarray(bvals)
    c2 = jnp.clip((jnp.asarray(bvecs) @ jnp.asarray(mu)) ** 2, 0.0, 1.0)
    q = q_from_b(bvals, delta, Delta)
    return bvals, c2, q * jnp.sqrt(1.0 - c2)


def cylinder_surface_pgse(bvals, bvecs, mu, radius, D, delta, Delta, n_modes: int = 10):
    """Myelin-water (cylindrical-surface) PGSE signal, paper Eq. 37:
    free 1-D Gaussian diffusion along the axis with the same D times the exact
    ring attenuation for q⊥. ``bvals`` (N,) s/m², ``bvecs`` (N,3), ``mu`` (3,)
    unit axis, ``radius`` sheath radius in m, ``delta``/``Delta`` scalars or
    (N,) in s."""
    bvals, c2, q_perp = _split(bvals, bvecs, mu, delta, Delta)
    E_par = jnp.exp(-bvals * D * c2)
    return E_par * ring_perpendicular_attenuation(q_perp, delta, Delta, radius, D, n_modes)


def cylinder_mf_pgse(bvals, bvecs, mu, diameter, D_par, D_perp, delta, Delta,
                     n_ang: int = 8, n_rad: int = 6):
    """Finite-pulse restricted cylinder (matrix formalism): stick along ``mu``
    with ``D_par`` times the exact disk attenuation with ``D_perp``. The
    finite-pulse counterpart of :func:`c3_cylinder_callaghan` (narrow pulse)
    and :func:`c2_cylinder` (Söderman, long time)."""
    bvals, c2, q_perp = _split(bvals, bvecs, mu, delta, Delta)
    E_par = jnp.exp(-bvals * D_par * c2)
    return E_par * disk_perpendicular_attenuation(q_perp, delta, Delta, diameter / 2.0, D_perp, n_ang, n_rad)


def sphere_mf_pgse(bvals, diameter, D, delta, Delta, n_ang: int = 8, n_rad: int = 6):
    """Finite-pulse restricted sphere (matrix formalism); the exact
    counterpart of :func:`g3_sphere` (GPD) and :func:`g3_sphere_callaghan`."""
    q = q_from_b(bvals, delta, Delta)
    return sphere_attenuation(q, delta, Delta, diameter / 2.0, D, n_ang, n_rad)


def cylinder_surface_spherical_mean(bvals, radius, D, delta, Delta, n_modes: int = 10, n_quad: int = 8):
    """Powder average of :func:`cylinder_surface_pgse` by Gauss–Legendre
    quadrature over μ = cos β ∈ [0, 1] (paper Eq. 42; n_q = 8 is at 1e-7
    relative error for b ≤ 6 ms/µm² in their Table IV)."""
    bvals = jnp.asarray(bvals)
    x, w = np.polynomial.legendre.leggauss(n_quad)
    mu = jnp.asarray(0.5 * (x + 1.0)); w = jnp.asarray(0.5 * w)
    q = q_from_b(bvals, delta, Delta)

    def at(m):
        return jnp.exp(-bvals * D * m ** 2) * ring_perpendicular_attenuation(
            q * jnp.sqrt(1.0 - m ** 2), delta, Delta, radius, D, n_modes)
    return jnp.einsum("q,qn->n", w, jax.vmap(at)(mu))
