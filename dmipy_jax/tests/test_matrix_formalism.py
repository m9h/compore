"""Spectral matrix-formalism pores (cylindrical surface, cylinder, sphere)
against closed-form limits, the existing Callaghan / Söderman / GPD kernels
and the convergence figures of Canales-Rodríguez et al. 2026 (Table II)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as ssp

from dmipy_jax.constants import bessel_derivative_roots, spherical_bessel_derivative_roots
from dmipy_jax.signal_models import cylinder_models as cm
from dmipy_jax.signal_models import matrix_formalism as mf
from dmipy_jax.signal_models import sphere_models as sm

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------- ring ----

def test_ring_basis_matches_paper():
    lam, B = mf.ring_basis(5)
    np.testing.assert_array_equal(lam, [0, 1, 4, 9, 16])
    K = 2 * B
    assert K[0, 1] == K[1, 0] == pytest.approx(np.sqrt(2))
    assert K[1, 2] == K[3, 4] == 1.0 and K[0, 0] == 0.0 and K[0, 2] == 0.0


def test_ring_narrow_pulse_is_jacobi_anger_series():
    """δ → 0: E = Σ_n J_n(qR)² e^{−n² D Δ / R²} (Canales-Rodríguez 2025)."""
    R, D, Delta = 2e-6, 0.8e-9, 20e-3
    q = np.linspace(0, 3.0, 7) / R
    E = np.asarray(mf.ring_perpendicular_attenuation(q, 1e-7, Delta, R, D, 30))
    n = np.arange(-40, 41)
    ref = [np.sum(ssp.jv(n, qq * R) ** 2 * np.exp(-n ** 2 * D * Delta / R ** 2)) for qq in q]
    np.testing.assert_allclose(E, ref, atol=1e-5)
    assert E[0] == pytest.approx(1.0)


def test_ring_long_time_is_diffraction_pattern():
    """D Δ ≫ R², δ ≪ R²/D: E → J_0(qR)²."""
    R, D = 1e-6, 0.8e-9
    q = np.array([0.5, 1.5, 2.4048, 3.5]) / R
    E = np.asarray(mf.ring_perpendicular_attenuation(q, 1e-6, 0.5, R, D, 20))
    np.testing.assert_allclose(E, ssp.j0(q * R) ** 2, atol=5e-4)
    assert E[2] < 1e-3                                   # first zero of J_0


def test_ring_short_time_is_tangential_free_diffusion():
    """D t ≪ R²: locally free along the tangent, E⊥ = ⟨e^{−b D sin²θ}⟩ = e^{−bD/2} I_0(bD/2)."""
    R, D, delta, Delta, b = 20e-6, 0.8e-9, 2e-3, 10e-3, 1e9
    q = mf.q_from_b(jnp.array([b]), delta, Delta)
    E = float(mf.ring_perpendicular_attenuation(q, delta, Delta, R, D, 40)[0])
    assert E == pytest.approx(np.exp(-b * D / 2) * ssp.i0(b * D / 2), abs=5e-3)


def test_ring_truncation_convergence_reproduces_table_ii():
    """Paper Table II, b = 3 ms/µm², Δ 9.45 / δ 4.62 ms, D 0.8, radii 0.1–5 µm:
    MRAE of the spherical mean vs truncation order M = 50 is 4.1e-3 % at
    M = 3 and 2.8e-7 % at M = 5 (M + 1 basis functions, n = 0..M)."""
    D, b, delta, Delta = 0.8e-9, 3e9, 4.62e-3, 9.45e-3
    radii = np.arange(0.1, 5.01, 0.1) * 1e-6

    def powder(M):
        return np.array([float(mf.cylinder_surface_spherical_mean(jnp.array([b]), r, D, delta, Delta, M + 1, 20)[0])
                         for r in radii])
    ref = powder(50)
    assert np.all(np.isfinite(ref)) and np.all((ref > 0.2) & (ref < 0.6))
    mrae = {M: 100 * np.mean(np.abs(powder(M) - ref) / ref) for M in (3, 5, 10)}
    assert 5e-4 < mrae[3] < 2e-2
    assert mrae[5] < 1e-5
    assert mrae[10] < 1e-8


def test_cylinder_surface_direction_dependence_and_b0():
    b = jnp.array([0.0, 2e9, 2e9, 2e9])
    g = jnp.array([[0, 0, 1.0], [0, 0, 1.0], [1.0, 0, 0], [0, np.sqrt(0.5), np.sqrt(0.5)]])
    mu = jnp.array([0.0, 0.0, 1.0]); D = 0.8e-9; delta, Delta = 5e-3, 20e-3
    E = mf.cylinder_surface_pgse(b, g, mu, 1.5e-6, D, delta, Delta)
    assert E[0] == pytest.approx(1.0)
    assert E[1] == pytest.approx(np.exp(-2e9 * D))                    # along the axis: free
    assert E[2] > E[1]                                                 # across: restricted
    perp = mf.ring_perpendicular_attenuation(mf.q_from_b(b[3:], delta, Delta) * np.sqrt(0.5), delta, Delta, 1.5e-6, D)
    assert E[3] == pytest.approx(np.exp(-2e9 * D * 0.5) * float(perp[0]), rel=1e-10)


def test_spherical_mean_matches_dense_direction_average():
    b = jnp.array([1e9, 3e9]); D = 0.8e-9; R = 2e-6; delta, Delta = 4e-3, 12e-3
    mu_q = np.linspace(0, 1, 2001)                                      # trapezoid in μ
    E_mu = np.array([np.asarray(mf.cylinder_surface_pgse(
        b, jnp.array([[np.sqrt(1 - m ** 2), 0, m]] * 2), jnp.array([0, 0, 1.0]), R, D, delta, Delta)) for m in mu_q])
    ref = np.trapezoid(E_mu, mu_q, axis=0)
    out = np.asarray(mf.cylinder_surface_spherical_mean(b, R, D, delta, Delta, 10, 8))
    np.testing.assert_allclose(out, ref, rtol=1e-6)


# ---------------------------------------------------------------- disk ----

def test_disk_basis_is_orthonormal_ground_mode_and_symmetric():
    lam, B = mf.disk_basis(6, 4)
    assert lam[0] == 0.0 and np.all(lam[1:] > 0)
    assert np.allclose(B, B.T)
    # ⟨1| x |φ_1k⟩ only couples the ground mode to n = 1 modes
    n_of = [n for n in range(6) for k in range(4)]
    for j, n in enumerate(n_of):
        if n != 1:
            assert abs(B[0, j]) < 1e-12


def test_disk_narrow_pulse_matches_callaghan_cylinder():
    alpha = bessel_derivative_roots(20, 20)
    R, D, Delta = 3e-6, 1.7e-9, 30e-3
    q = np.linspace(0.1, 1.5, 6) / R
    E = np.asarray(mf.disk_perpendicular_attenuation(q, 1e-7, Delta, R, D, 10, 8))
    ref = np.asarray(cm.callaghan_perpendicular_attenuation(jnp.asarray(q / (2 * np.pi)), Delta, 2 * R, D, alpha))
    np.testing.assert_allclose(E, ref, atol=2e-5)


def test_disk_long_time_is_soderman():
    R, D = 3e-6, 1.7e-9
    q = np.linspace(0.1, 1.5, 6) / R
    E = np.asarray(mf.disk_perpendicular_attenuation(q, 1e-7, 2.0, R, D, 10, 8))
    x = q * R
    np.testing.assert_allclose(E, (2 * ssp.jv(1, x) / x) ** 2, atol=5e-5)


def test_cylinder_mf_approaches_free_diffusion_from_above_as_radius_grows():
    """Reflecting walls can only reduce the mean-squared displacement, so
    E⊥ ≥ e^{−bD}, with the gap shrinking with R (Mitra: ∝ (S/V)√(Dt) = 2√(Dt)/R)."""
    b = jnp.array([0.0, 0.2e9]); g = jnp.array([[1.0, 0, 0]] * 2); mu = jnp.array([0, 0, 1.0])
    D, delta, Delta = 1.7e-9, 0.5e-3, 2e-3
    free = np.exp(-0.2e9 * D)
    E = [np.asarray(mf.cylinder_mf_pgse(b, g, mu, 2 * R, D, D, delta, Delta, 12, 8)) for R in (8e-6, 10e-6, 15e-6)]
    assert all(e[0] == pytest.approx(1.0) for e in E)
    gaps = np.array([e[1] for e in E]) - free
    assert np.all(gaps > 0) and np.all(np.diff(gaps) < 0)
    # Mitra leading order at the largest radius: D_eff/D ≈ 1 − (4/3√π)(2/R)√(DΔ)
    mitra = np.exp(-0.2e9 * D * (1 - 4 / (3 * np.sqrt(np.pi)) * 2 / 15e-6 * np.sqrt(D * Delta)))
    assert E[2][1] == pytest.approx(mitra, abs=0.03)


# -------------------------------------------------------------- sphere ----

def test_sphere_narrow_pulse_matches_callaghan_sphere():
    alpha = spherical_bessel_derivative_roots(20, 20)
    R, D, Delta = 5e-6, 2e-9, 30e-3
    q = np.linspace(0.05, 1.5, 6) / R
    E = np.asarray(mf.sphere_attenuation(q, 1e-7, Delta, R, D, 10, 8))
    ref = np.asarray(sm.g3_sphere_callaghan(jnp.asarray(q / (2 * np.pi)), Delta, 2 * R, D, alpha))
    np.testing.assert_allclose(E, ref, atol=2e-5)


def test_sphere_mf_agrees_with_gpd_at_low_attenuation():
    b = jnp.array([0.2e9, 0.5e9, 1e9]); R, D, delta, Delta = 5e-6, 2e-9, 10e-3, 30e-3
    E = np.asarray(mf.sphere_mf_pgse(b, 2 * R, D, delta, Delta, 10, 8))
    gpd = np.asarray(sm.g3_sphere(b, None, 2 * R, D, Delta, delta))
    np.testing.assert_allclose(E, gpd, atol=5e-4)          # GPD is second-cumulant: 3e-4 off at b = 1
    assert np.all(np.diff(E) < 0)


def test_sphere_long_time_is_stejskal_tanner():
    R, D = 3e-6, 2e-9
    q = np.linspace(0.1, 1.5, 6) / R
    E = np.asarray(mf.sphere_attenuation(q, 1e-7, 2.0, R, D, 10, 8))
    x = q * R
    np.testing.assert_allclose(E, (3 * (np.sin(x) - x * np.cos(x)) / x ** 3) ** 2, atol=5e-5)


# ---------------------------------------------------------- numerics ----

def test_tiny_radius_with_many_modes_is_finite():
    q = mf.q_from_b(jnp.array([3e9]), 4.62e-3, 9.45e-3)
    E = mf.ring_perpendicular_attenuation(q, 4.62e-3, 9.45e-3, 0.1e-6, 0.8e-9, 50)
    assert np.isfinite(float(E[0])) and float(E[0]) == pytest.approx(1.0, abs=1e-3)


def test_kernels_are_differentiable_in_radius_and_timing():
    b = jnp.array([2e9, 4e9]); g = jnp.array([[1.0, 0, 0]] * 2); mu = jnp.array([0, 0, 1.0])

    def f(r, delta, Delta):
        return jnp.sum(mf.cylinder_surface_pgse(b, g, mu, r, 0.8e-9, delta, Delta))
    dr, dd, dD = jax.grad(f, argnums=(0, 1, 2))(2e-6, 5e-3, 20e-3)
    assert np.isfinite(dr) and dr < 0                   # larger sheath → more attenuation
    assert np.isfinite(dd) and np.isfinite(dD)
    gs = jax.grad(lambda r: jnp.sum(mf.sphere_mf_pgse(b, 2 * r, 2e-9, 5e-3, 20e-3)))(4e-6)
    gc = jax.grad(lambda r: jnp.sum(mf.cylinder_mf_pgse(b, g, mu, 2 * r, 1.7e-9, 1.7e-9, 5e-3, 20e-3)))(3e-6)
    assert np.isfinite(gs) and gs < 0 and np.isfinite(gc) and gc < 0
    assert np.isfinite(float(jax.jit(f)(2e-6, 5e-3, 20e-3)))
