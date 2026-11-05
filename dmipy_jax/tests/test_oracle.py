"""Oracle protocol (doc 012 §7.3): registry, the JAX walker oracle on a tiny
substrate, library generation round-trip, and the emulator adapter."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from dmipy_jax.simulation import oracle as oc  # noqa: E402
from dmipy_jax.simulation.substrates.spheres import Substrate  # noqa: E402


def tiny_substrate(seed=0, n_axons=4, box_um=6.0, r_um=0.8):
    """Straight z-axons as sphere chains on a grid; no CATERPillar needed."""
    L = box_um * 1e-6; rng = np.random.default_rng(seed)
    xs = np.linspace(0.2, 0.8, int(np.ceil(np.sqrt(n_axons)))) * L
    centres, ids = [], []
    k = 0
    for x in xs:
        for y in xs:
            if k >= n_axons:
                break
            z = np.arange(0, L + 1e-12, r_um * 1e-6 * 0.8)
            c = np.column_stack([np.full_like(z, x), np.full_like(z, y), z])
            centres.append(c); ids.append(np.full(len(z), k)); k += 1
    C = np.concatenate(centres); I = np.concatenate(ids); R = np.full(len(C), r_um * 1e-6)
    v_in = n_axons * np.pi * (r_um * 1e-6) ** 2 * L; f = v_in / L ** 3
    return Substrate(centers_m=C, radii_m=R, box_m=L, axes=np.array([[0.0, 0.0, 1.0]]), f_intra=f,
                     meta={}, axon_ids=I, fractions={"intra": f, "myelin": 0.0, "glia": 0.0, "extra": 1 - f})


def scheme():
    rng = np.random.default_rng(1)
    g = rng.normal(size=(12, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    g = np.vstack([[0, 0, 0], g]); b = np.r_[0.0, np.full(6, 1e9), np.full(6, 2.5e9)]
    return oc.PGSEScheme(b, g, delta=6e-3, Delta=12e-3)


def test_registry_and_availability():
    assert isinstance(oc.get_oracle("jax"), oc.JaxWalkerOracle)
    av = oc.available_oracles()
    assert av["jax"] is True and set(av) == {"jax", "mcmr", "mcds"}
    with pytest.raises(KeyError):
        oc.get_oracle("nope")
    bad = oc.MCDSOracle(binary="/nonexistent/bin")
    assert not bad.check_available()
    with pytest.raises(RuntimeError, match="not available"):
        bad.simulate(tiny_substrate(), scheme())


def test_scheme_rejects_mm2():
    with pytest.raises(ValueError):
        oc.PGSEScheme(np.array([0, 1000.0, 2000.0]), np.eye(3))


def test_jax_oracle_physics():
    sub, sch = tiny_substrate(), scheme()
    r = oc.get_oracle("jax").simulate(sub, sch, n_particles=600, seed=0)
    assert r.engine == "jax" and r.S.shape == (sch.n,)
    assert np.isclose(r.S[0], 1.0, atol=1e-6)
    # extra-cellular water is nearly free; intra is restricted perpendicular to z
    perp = np.abs(sch.bvecs[:, 2]) < 0.3
    free = np.exp(-sch.bvals_si * 2.0e-9)
    assert np.all(r.S_extra[1:] < 1.0) and np.mean(r.S_extra[perp & (sch.bvals_si > 0)] / free[perp & (sch.bvals_si > 0)]) > 0.7
    assert np.mean(r.S_intra[perp & (sch.bvals_si > 2e9)]) > np.mean(r.S_extra[perp & (sch.bvals_si > 2e9)]) + 0.2
    np.testing.assert_allclose(r.S, oc.SubstrateOracle.combine(sub, r.S_intra, r.S_extra), rtol=1e-6)


def test_generate_library_roundtrip(tmp_path):
    sch = scheme()

    def sampler(rng, i):
        n = int(rng.integers(2, 5)); return np.array([n], float), tiny_substrate(seed=i, n_axons=n)
    lib = oc.get_oracle("jax").generate_library(sampler, 3, sch, parameter_names=["n_axons"],
                                                descriptor_fn=lambda s: {"f_intra": float(s.f_intra)},
                                                out_path=str(tmp_path / "lib.h5"), n_particles=300, log=lambda *a: None)
    assert lib.n_entries == 3 and lib.signal_dim == sch.n and lib.metadata["engine"] == "jax"
    assert lib.metadata["n_skipped"] == 0 and len(lib.metadata["descriptors"]) == 3
    from dmipy_jax.library.storage import SimulationLibrary
    back = SimulationLibrary.load_hdf5(str(tmp_path / "lib.h5"))
    np.testing.assert_allclose(np.asarray(back.signals), np.asarray(lib.signals), rtol=1e-6)
    assert back.metadata["descriptor_names"] == ["f_intra"]


def test_emulator_fits_library_and_adapts_to_model_simulator(tmp_path):
    from dmipy_jax.simulation.emulator import LibraryEmulator
    from dmipy_jax.library.storage import SimulationLibrary
    import jax.numpy as jnp
    sch = scheme()
    # a cheap analytic "library": stick+ball so the emulator has something smooth to learn
    rng = np.random.default_rng(0); N = 200
    f = rng.uniform(0.2, 0.8, N); d = rng.uniform(1.0e-9, 2.5e-9, N)
    S = f[:, None] * np.exp(-sch.bvals_si * 1.7e-9 * sch.bvecs[:, 2] ** 2) + (1 - f)[:, None] * np.exp(-sch.bvals_si * d[:, None])
    lib = SimulationLibrary(params=jnp.asarray(np.column_stack([f, d])), signals=jnp.asarray(S), parameter_names=["f", "d"],
                            metadata={"bvals_si": sch.bvals_si.tolist(), "bvecs": sch.bvecs.tolist()})
    em = LibraryEmulator.fit(lib, n_models=2, steps=300, width=64, depth=2, log=lambda *a: None)
    pred = np.asarray(em(np.array([0.5, 1.5e-9]), sch.bvals_si, sch.bvecs))
    truth = 0.5 * np.exp(-sch.bvals_si * 1.7e-9 * sch.bvecs[:, 2] ** 2) + 0.5 * np.exp(-sch.bvals_si * 1.5e-9)
    assert pred.shape == (sch.n,) and np.sqrt(np.mean((pred - truth) ** 2)) < 0.08
    from dmipy_jax.acquisition import JaxAcquisition
    acq = JaxAcquisition(bvalues=jnp.asarray(sch.bvals_si), gradient_directions=jnp.asarray(sch.bvecs),
                         delta=jnp.full(sch.n, sch.delta), Delta=jnp.full(sch.n, sch.Delta))
    ms = em.as_model_simulator(acq, snr=30.0)
    th, sig = ms.sample_and_simulate(jax.random.PRNGKey(0), 8)
    assert th.shape == (8, 2) and sig.shape == (8, sch.n) and np.all(np.isfinite(np.asarray(sig)))
    em.save(str(tmp_path / "em.json")); em2 = LibraryEmulator.load(str(tmp_path / "em.json"))
    np.testing.assert_allclose(np.asarray(em2(np.array([0.5, 1.5e-9]), sch.bvals_si, sch.bvecs)), pred, rtol=1e-6)
