"""Powder-averaged NEXI / SANDI kernels: limits, consistency with the WM Kärger kernel, gradients."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from compore.signal_models import gm_powder as gm  # noqa: E402


@pytest.fixture(autouse=True)
def x64():
    jax.config.update("jax_enable_x64", True); yield; jax.config.update("jax_enable_x64", False)


def scheme():
    b = np.array([0, 1000, 2500, 5000, 7350, 10000, 12500], float) * 1e6
    D = np.array([12, 12, 12, 15, 20, 27, 45], float) * 1e-3
    return b, D, np.full(len(b), 5e-3)


def test_powder_stick_matches_quadrature():
    b, _, _ = scheme()
    x, w = np.polynomial.legendre.leggauss(200); x = 0.5 * (x + 1); w = 0.5 * w
    ref = np.array([np.sum(w * np.exp(-bb * 2.5e-9 * x ** 2)) for bb in b])
    np.testing.assert_allclose(np.asarray(gm.powder_stick(b, 2.5e-9)), ref, rtol=1e-8)


def test_nexi_limits():
    b, D, d = scheme()
    f, Di, De = 0.5, 2.5e-9, 0.9e-9
    slow = np.asarray(gm.nexi_powder(b, D, d, f, Di, De, t_ex=1e3))
    no_exchange = f * np.asarray(gm.powder_stick(b, Di)) + (1 - f) * np.exp(-b * De)
    np.testing.assert_allclose(slow, no_exchange, rtol=1e-4)   # 64-node GL vs closed form
    fast = np.asarray(gm.nexi_powder(b, D, d, f, Di, De, t_ex=1e-7))
    # fast exchange: one Gaussian pool with the orientation-averaged D at each x, then powder average
    x, w = np.polynomial.legendre.leggauss(64); x = 0.5 * (x + 1); w = 0.5 * w
    ref = np.array([np.sum(w * np.exp(-bb * (f * Di * x ** 2 + (1 - f) * De))) for bb in b])
    np.testing.assert_allclose(fast, ref, rtol=1e-4)
    mid = np.asarray(gm.nexi_powder(b, D, d, f, Di, De, t_ex=20e-3))
    assert np.all(mid[1:] < slow[1:]) and np.all(mid[1:] > fast[1:] - 1e-9)
    assert np.isclose(mid[0], 1.0)


def test_nexi_matches_wm_karger_kernel_on_axis():
    """Along one orientation the NEXI 2×2 system equals the WM Kärger kernel with a
    stick (D⊥ = 0) and an isotropic extra pool (D∥,ex = D⊥,ex = D_e)."""
    from validation.lib.prism_exchange import karger_pgse
    b, D, d = scheme()
    f, Di, De, tex = 0.4, 2.2e-9, 1.0e-9, 15e-3
    dirs = jnp.asarray([0.0, 0.0, 1.0])
    for x in (1.0, 0.6, 0.2):
        gx = np.tile([[np.sqrt(1 - x * x), 0.0, x]], (len(b), 1))
        ref = np.asarray(karger_pgse(jnp.asarray(b), jnp.asarray(gx), dirs, f, Di, De, De, tex, jnp.asarray(d), jnp.asarray(D)))
        ours = np.asarray(gm._karger_2x2(jnp.asarray(b), gm.diffusion_time(D, d), Di * x * x, De, f, tex))
        np.testing.assert_allclose(ours, ref.ravel(), rtol=1e-6, atol=1e-9)


def test_sandi_and_nexi_soma_reduce_correctly():
    b, D, d = scheme()
    s0 = np.asarray(gm.sandi_powder(b, D, d, 0.5, 0.0, 2.5e-9, 0.9e-9, 6e-6))
    np.testing.assert_allclose(s0, 0.5 * np.asarray(gm.powder_stick(b, 2.5e-9)) + 0.5 * np.exp(-b * 0.9e-9), rtol=1e-6)
    ns = np.asarray(gm.nexi_soma_powder(b, D, d, 0.5, 0.0, 2.5e-9, 0.9e-9, 20e-3, 6e-6))
    np.testing.assert_allclose(ns, np.asarray(gm.nexi_powder(b, D, d, 0.5, 2.5e-9, 0.9e-9, 20e-3)), rtol=1e-6)
    sph = np.asarray(gm.sphere_powder(b, D, d, 6e-6, 3e-9))
    assert sph[0] == pytest.approx(1.0) and np.all(np.diff(sph[1:4]) < 0) and sph[-1] > np.exp(-b[-1] * 0.9e-9)


def test_rician_mean_limits():
    S = jnp.asarray([0.0, 0.05, 0.5, 1.0]); sig = 0.02
    m = np.asarray(gm.rician_mean(S, sig))
    assert m[0] == pytest.approx(sig * np.sqrt(np.pi / 2), rel=1e-6)
    assert abs(m[3] - np.sqrt(1.0 + 2 * sig ** 2)) < 5e-4 and m[2] > 0.5   # √(S²+2σ²) is the high-SNR expansion


def test_gradients_finite_and_jit():
    b, D, d = scheme()
    def loss(p):
        return jnp.sum(gm.nexi_soma_powder(b, D, d, p[0], p[1], p[2], p[3], p[4], p[5]))
    g = jax.jit(jax.grad(loss))(jnp.array([0.4, 0.1, 2.5e-9, 1.0e-9, 20e-3, 6e-6]))
    assert np.all(np.isfinite(np.asarray(g))) and np.asarray(g)[4] != 0


def test_nexi_matches_reference_integrand():
    """Reference: the NEXI integrand of Uhl & Jelescu's graymatter_swissknife
    (Apache-2.0; closed-form eigenvalues per orientation, quad over x∈[0,1]),
    re-expressed here in SI. Our Kärger 2×2 + 64-node Gauss–Legendre must agree."""
    import scipy.integrate
    b, D, d = scheme(); td = D - d / 3.0
    f, Di, De, tex = 0.42, 2.6e-9, 0.85e-9, 25e-3

    def M(x):
        D1 = Di * b * x ** 2; D2 = De * b
        disc = np.sqrt(((td / tex) * (1 - 2 * f) + D1 - D2) ** 2 + 4 * (td / tex) ** 2 * f * (1 - f))
        lp = (td / tex + D1 + D2) / 2 + disc / 2; lm = (td / tex + D1 + D2) / 2 - disc / 2
        Pp = (f * D1 + (1 - f) * D2 - lm) / disc
        return Pp * np.exp(-lp) + (1 - Pp) * np.exp(-lm)
    ref = scipy.integrate.quad_vec(M, 0.0, 1.0, epsabs=1e-13)[0]
    ours = np.asarray(gm.nexi_powder(b, D, d, f, Di, De, tex))
    np.testing.assert_allclose(ours, ref, rtol=2e-5, atol=1e-9)
