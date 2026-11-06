"""Joint model-selection machinery on the ds008771 powder scheme: simulation, Laplace evidence, tiny amortised fit."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from compore.inference import model_selection as ms  # noqa: E402


FEATS = [(1000, 12), (2500, 12), (5000, 15), (7350, 20), (1000, 27), (2500, 27), (5000, 27), (7500, 27), (10000, 27),
         (1000, 45), (2500, 45), (5000, 45), (7500, 45), (10000, 45), (12500, 45)]
COUNTS = [20, 30, 32, 34, 20, 30, 32, 34, 44, 20, 30, 32, 34, 44, 64]


@pytest.fixture(scope="module")
def sch():
    return ms.PowderScheme.from_features(FEATS, COUNTS)


def test_prior_respects_constraints_and_simulation_shapes(sch):
    for m in ms.MODELS:
        th = np.asarray(ms.sample_prior(jax.random.PRNGKey(0), m, 500)); lo, hi = ms.box(m)
        assert th.shape == (500, len(lo)) and np.all(th >= lo - 1e-12) and np.all(th <= hi + 1e-12)
        assert np.all(np.asarray(ms._valid(m, jnp.asarray(th))))
    m_idx, theta, sigma, S = ms.simulate_joint(jax.random.PRNGKey(1), sch, 300)
    assert S.shape == (300, 15) and theta.shape == (300, 6) and np.all(np.isfinite(np.asarray(S)))
    assert set(np.unique(np.asarray(m_idx))) <= {0, 1, 2}


def test_laplace_recovers_nexi_at_high_snr(sch):
    truth = jnp.array([0.45, 2.4e-9, 0.9e-9, 30e-3]); sigma = 1 / 80.0
    S = np.asarray(ms.noisy_powder(jax.random.PRNGKey(2), ms.forward("nexi", truth, sch), jnp.asarray(sigma), sch))
    r = ms.laplace_model_posterior(S, sigma, sch, n_restarts=4)
    f = r["fits"]["nexi"]; err = np.abs(f["theta"] - np.asarray(truth))
    assert err[0] < 0.05 and err[3] < 10e-3, (f["theta"], truth)
    assert np.all(err <= 4 * f["theta_sd"] + 1e-12)
    assert r["p_model"]["nexi"] > r["p_model"]["sandi"]      # the exchanging truth beats the non-exchanging model


def test_amortised_trains_and_is_sane(sch):
    post = ms.train_amortised(jax.random.PRNGKey(0), sch, n_sims=6000, steps_clf=200, steps_flow=200, batch=256,
                              flow_layers=2, nn_width=32, log=lambda *a: None)
    m_idx, theta, sigma, S = ms.simulate_joint(jax.random.PRNGKey(9), sch, 60)
    p = post.p_model(np.asarray(S), np.asarray(sigma))
    assert p.shape == (60, 3) and np.allclose(p.sum(1), 1.0, atol=1e-5)
    s = post.sample_theta(jax.random.PRNGKey(3), "nexi", np.asarray(S[0]), float(sigma[0]), 50)
    lo, hi = ms.box("nexi"); assert s.shape == (50, 4) and np.all(s >= lo) and np.all(s <= hi)
    lp = post.log_prob_theta("nexi", s[:5], np.asarray(S[0]), float(sigma[0])); assert np.all(np.isfinite(np.asarray(lp)))


def test_batched_laplace_matches_serial(sch):
    m_idx, theta, sigma, S = ms.simulate_joint(jax.random.PRNGKey(4), sch, 6, (20.0, 60.0))
    S, sigma = np.asarray(S), np.asarray(sigma)
    rb = ms.laplace_model_posterior_batch(S, sigma, sch, n_restarts=8)
    assert rb["p_model"].shape == (6, 3) and np.allclose(rb["p_model"].sum(1), 1.0, atol=1e-5)
    for i in range(3):
        rs = ms.laplace_model_posterior(S[i], float(sigma[i]), sch, n_restarts=8)
        for m in ms.MODELS:                                    # same optimum (χ²) and evidence up to float32 noise
            assert abs(rs["fits"][m]["chi2"] - rb["fits"][m]["chi2"][i]) < 1e-2 * max(1.0, rs["fits"][m]["chi2"]), m
            # log|H| of the Gauss–Newton Hessian is float32-noisy along flat directions (≈1 nat seen for SANDI)
            assert abs(rs["fits"][m]["logZ"] - rb["fits"][m]["logZ"][i]) < 2.0, m


def test_theta_median_batch_matches_single_voxel(sch):
    post = ms.train_amortised(jax.random.PRNGKey(0), sch, n_sims=4000, steps_clf=50, steps_flow=50, batch=256,
                              flow_layers=2, nn_width=32, log=lambda *a: None)
    m_idx, theta, sigma, S = ms.simulate_joint(jax.random.PRNGKey(9), sch, 8)
    S, sigma = np.asarray(S), np.asarray(sigma)
    mb = post.theta_median_batch(jax.random.PRNGKey(3), "sandi", S, sigma, n=400, chunk=5)
    lo, hi = ms.box("sandi"); assert mb.shape == (8, 5) and np.all(mb >= lo) and np.all(mb <= hi)
    one = np.median(post.sample_theta(jax.random.PRNGKey(5), "sandi", S[0], sigma[0], 4000), 0)
    assert np.all(np.abs(mb[0] - one) <= 0.15 * (hi - lo) + 1e-12)      # Monte Carlo medians of the same flow
