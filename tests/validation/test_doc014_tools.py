"""Red-green tests for the two doc 014 §4 tools: the cuslines stage-B script
(validation/cuslines_posterior_disco_gpu.py) and the lafix Skyline example
(packages/lafix/examples/viz_skyline_sigma.py).

Everything here runs on the dev venv (dipy 1.12, no cuslines, no FURY). The
CUDA tracker and the real Skyline window are exercised by two gated tests
at the bottom that skip until `cuslines` / `dipy.viz.skyline` are importable.
"""
from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
gpu = pytest.importorskip("validation.cuslines_posterior_disco_gpu")


def _load_example():
    p = ROOT / "packages/lafix/examples/viz_skyline_sigma.py"
    spec = importlib.util.spec_from_file_location("viz_skyline_sigma", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------- stage B --
def test_pmf_from_fixels_peaks_at_fixel_and_is_antipodal():
    from dipy.data import default_sphere
    verts = default_sphere.vertices.astype(np.float32)
    # fixel 1 along a sphere vertex (so the PMF peak is exactly the weight);
    # fixel 2 along x, which default_sphere does not hit exactly (κ=40 → the
    # nearest vertex sees exp(κ(cos²θ−1)) < 1 of the weight)
    k0 = int(np.argmax(verts[:, 2])); d0 = verts[k0].astype(float)
    d = np.array([[d0, [1.0, 0.0, 0.0]]])                          # (N=1, K=2, 3)
    fr = np.array([[0.7, 0.3]])
    idx = np.array([[1, 2, 3]])
    pmf = gpu.pmf_from_fixels(d, fr, idx, (4, 4, 4), verts, kappa=40.0, frac_min=0.1)
    assert pmf.shape == (4, 4, 4, verts.shape[0]) and pmf.dtype == np.float32
    assert np.all(pmf >= 0)
    v = pmf[1, 2, 3]
    assert np.isclose(v.max(), 0.7, atol=1e-5)                     # peak = weight at the fixel direction
    assert np.isclose(v[k0], 0.7, atol=1e-5)
    # antipodal symmetry pmf(v) == pmf(-v): default_sphere is a HemiSphere
    # (repulsion724 upper half, z >= 0), which is exactly what cuslines'
    # ``sphere_symm=True`` expects (it mirrors the vertex list and flips the
    # sampled vertex toward the running direction), so evaluate on -verts.
    assert default_sphere.vertices[:, 2].min() >= 0.0
    pmf_neg = gpu.pmf_from_fixels(d, fr, idx, (4, 4, 4), -verts, kappa=40.0, frac_min=0.1)
    assert np.allclose(v, pmf_neg[1, 2, 3], atol=1e-6)
    # fixel 2 contributes ≤ 0.3 at the vertex nearest x, attenuated by its offset
    ix = np.argmax(np.abs(verts @ [1.0, 0.0, 0.0])); c2 = float(np.abs(verts[ix] @ [1.0, 0.0, 0.0])) ** 2
    assert np.isclose(v[ix], 0.3 * np.exp(40.0 * (c2 - 1.0)), rtol=1e-3)
    # untouched voxels are zero
    assert pmf[0, 0, 0].sum() == 0


def test_pmf_from_fixels_drops_fractions_below_threshold():
    from dipy.data import default_sphere
    verts = default_sphere.vertices.astype(np.float32)
    d = np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
    fr = np.array([[0.9, 0.05]])
    pmf = gpu.pmf_from_fixels(d, fr, np.array([[0, 0, 0]]), (1, 1, 1), verts, 40.0, frac_min=0.1)
    ix = np.argmax(np.abs(verts @ [1.0, 0.0, 0.0]))
    assert pmf[0, 0, 0, ix] < 1e-6


def test_pearson_ut_ignores_diagonal():
    a = np.arange(16, dtype=float).reshape(4, 4); a = a + a.T
    b = 2 * a + 1
    np.fill_diagonal(b, 999)                                         # diagonal must not matter
    assert np.isclose(gpu.pearson_ut(a, b), 1.0)
    assert np.isnan(gpu.pearson_ut(a, np.ones((4, 4))))


def test_clip_to_volume_keeps_points_inside_voxel_box():
    sl = [np.array([[-0.5, 0.0, 1.0], [3.4, 2.0, 2.0]]), np.array([[1.0, 1.0, 1.0]])]
    out = gpu.clip_to_volume(sl, (3, 3, 3))
    assert len(out) == 2
    for s in out:
        assert s.min() >= 0.0 and s.max() <= 2.0
    assert np.allclose(out[0][0], [0.0, 0.0, 1.0]) and np.allclose(out[0][1], [2.0, 2.0, 2.0])
    assert np.allclose(out[1], [[1.0, 1.0, 1.0]])


def test_half_voxel_conversion_of_cuslines_endpoints():
    """cuslines' stop map is sampled half a voxel late (texture without the
    texel-centre offset), so endpoints are labelled at x - 0.5 (= floor(x))
    and clipped to the voxel box; seeds and PMF stay in DIPY's convention."""
    assert not hasattr(gpu, "seeds_to_cuslines")                    # seeds are NOT shifted (voxel-40 crash)
    sl = gpu.to_dipy_vox([np.array([[0.5, 1.0, 3.0], [-0.2, 2.9, 0.0]])], (3, 3, 3))
    assert np.allclose(sl[0][0], [0.0, 0.5, 2.0])          # 3.0 - 0.5 = 2.5 → clipped to shape-1 = 2
    assert np.allclose(sl[0][1], [0.0, 2.0, 0.0])          # -0.7 and -0.5 → 0


def test_generate_sft_chunked_calls_tracker_once_per_chunk_and_concatenates():
    """cuslines 2.2.3 `generate_sft` stores lazy generators over the propagator's
    buffers and re-propagates the next chunk before consuming them, so every
    chunk yields the LAST chunk's streamlines (2 chunks → last chunk twice, the
    first lost). Stage B must call generate_sft once per chunk itself."""
    class FakeSFT:
        def __init__(self, sl): self.streamlines = sl
    class FakeTracker:
        calls = []
        def generate_sft(self, seeds, ref):
            FakeTracker.calls.append(np.asarray(seeds).copy())
            return FakeSFT([np.array([[float(i), 0.0, 0.0]]) for i in seeds[:, 0]])
    seeds = np.arange(7, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
    sl = gpu.generate_sft_chunked(FakeTracker(), seeds, ref_img=None, chunk=3)
    assert [len(c) for c in FakeTracker.calls] == [3, 3, 1]
    assert np.allclose(np.concatenate(FakeTracker.calls)[:, 0], np.arange(7))
    assert [float(x[0, 0]) for x in sl] == list(range(7))


# ------------------------------------------------------------ Skyline example --
@pytest.fixture
def lafix_outputs(tmp_path):
    """Synthetic `<prefix>_desc-refined_*` products of dipy_refine_peaks."""
    from dipy.data import default_sphere
    from dipy.io.image import save_nifti
    from dipy.io.peaks import save_pam
    from lafix.tracking import peaks_to_pam

    shape = (5, 6, 3); affine = np.diag([2.0, 2.0, 2.0, 1.0]); affine[:3, 3] = [-5, -6, -3]
    mask = np.zeros(shape, bool); mask[1:4, 1:5, :] = True
    dirs = np.zeros(shape + (5, 3)); vals = np.zeros(shape + (5,))
    dirs[mask, 0] = [0, 0, 1.0]; vals[mask, 0] = 0.8
    pam = peaks_to_pam(dirs, vals, mask, default_sphere, affine)
    sigma = np.where(mask, 7.5, 0.0).astype(np.float32)
    sigma[0, 0, 0] = 99.0                       # σ outside the WM mask: must be hidden by f_wm
    fwm = np.where(mask, 0.9, 0.0).astype(np.float32)
    fintra = np.where(mask, 0.6, 0.0).astype(np.float32)
    prefix = tmp_path / "sub-01_desc-refined"
    save_pam(f"{prefix}_peaks.pam5", pam)
    save_nifti(f"{prefix}_sigmatheta_calibrated.nii.gz", sigma, affine)
    save_nifti(f"{prefix}_fwm.nii.gz", fwm, affine)
    save_nifti(f"{prefix}_fintra.nii.gz", fintra, affine)
    return prefix, mask, affine


def test_cone_samples_geometry(lafix_outputs):
    from dipy.io.stateful_tractogram import Space
    from dipy.io.peaks import load_pam
    from dipy.io.image import load_nifti
    ex = _load_example()
    prefix, mask, affine = lafix_outputs
    pam = load_pam(f"{prefix}_peaks.pam5"); sigma, _ = load_nifti(f"{prefix}_sigmatheta_calibrated.nii.gz")
    z, n = 1, 200
    sft = ex._cone_samples(pam, sigma, affine, z, n, seed=0)
    assert sft.space == Space.VOX
    n_vox = int(mask[:, :, z].sum())
    assert len(sft.streamlines) == n_vox * n
    seg = np.asarray(sft.streamlines[0])
    assert seg.shape == (2, 3)
    assert np.isclose(np.linalg.norm(seg[1] - seg[0]), 0.8)       # 0.8 voxel long
    assert np.allclose(seg.mean(0)[2], z)                            # centred on the slice
    # angular spread ≈ σ_θ (isotropic tangent Gaussian, sd 7.5° per axis)
    w = np.stack([s[1] - s[0] for s in sft.streamlines]); w /= np.linalg.norm(w, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(np.abs(w[:, 2]), 0, 1)))
    assert 5.0 < np.sqrt(np.mean(ang**2) / 2) < 10.0


def test_build_skyline_kwargs_from_cli_outputs(lafix_outputs):
    ex = _load_example()
    prefix, mask, affine = lafix_outputs
    kw = ex.build_skyline_kwargs(prefix, samples=0, stealth=None)
    labels = [t[2] for t in kw["images"]]
    assert labels == ["sigma_theta_calibrated [deg]", "f_wm", "f_intra"]
    sig = kw["images"][0][0]
    assert sig.dtype == np.float32 and sig[0, 0, 0] == 0.0 and np.isclose(sig[2, 2, 1], 7.5)
    assert all(np.allclose(t[1], affine) for t in kw["images"])
    pd, paff, plabel, pv = kw["peaks"][0]
    assert pd.shape == mask.shape + (5, 3) and pv.shape == mask.shape + (5,) and np.allclose(paff, affine)
    assert kw["tractograms"] == []
    assert "visualizer_type" not in kw
    kw2 = ex.build_skyline_kwargs(prefix, samples=3, stealth="out.png")
    assert kw2["visualizer_type"] == "stealth" and kw2["out_stealth_png"] == "out.png"
    assert len(kw2["tractograms"]) == 1 and kw2["tractograms"][0][1].startswith("posterior cones")


def test_main_calls_skyline_with_cli_products(lafix_outputs, monkeypatch, tmp_path):
    """Wire-up test with a stand-in `dipy.viz.skyline.app.skyline`: proves the
    CLI products reach the viewer in the tuple shapes Skyline documents."""
    ex = _load_example()
    prefix, mask, affine = lafix_outputs
    seen = {}
    fake_pkg = types.ModuleType("dipy.viz.skyline"); fake_app = types.ModuleType("dipy.viz.skyline.app")
    fake_app.skyline = lambda **kw: seen.update(kw)
    monkeypatch.setitem(sys.modules, "dipy.viz.skyline", fake_pkg)
    monkeypatch.setitem(sys.modules, "dipy.viz.skyline.app", fake_app)
    png = str(tmp_path / "shot.png")
    ex.main([str(prefix), "--stealth", png, "--samples", "2"])
    assert seen["visualizer_type"] == "stealth" and seen["out_stealth_png"] == png
    assert len(seen["images"]) == 3 and len(seen["peaks"]) == 1 and len(seen["tractograms"]) == 1
    assert seen["tract_colors"] == (1.0, 0.85, 0.2)
    assert len(seen["tractograms"][0][0].streamlines) == int(mask[:, :, 1].sum()) * 2


def test_main_explains_missing_skyline(lafix_outputs, monkeypatch):
    ex = _load_example()
    prefix, _, _ = lafix_outputs
    monkeypatch.setitem(sys.modules, "dipy.viz.skyline.app", None)   # forces ImportError
    with pytest.raises(SystemExit) as e:
        ex.main([str(prefix)])
    assert "dipy >= 1.13" in str(e.value)


# ------------------------------------------------- gated on the real tools --
def test_skyline_stealth_renders_png(lafix_outputs, tmp_path):
    pytest.importorskip("dipy.viz.skyline.app")
    ex = _load_example()
    prefix, _, _ = lafix_outputs
    png = tmp_path / "skyline.png"
    ex.main([str(prefix), "--stealth", str(png), "--samples", "5"])
    assert png.exists() and png.stat().st_size > 1000


@pytest.mark.gpu
def test_cuslines_tracks_single_fixel_phantom():
    pytest.importorskip("cuslines")
    from math import radians
    import nibabel as nib
    from dipy.data import default_sphere
    from cuslines.cuda_python.cu_tractography import GPUTracker
    from cuslines.cuda_python.cu_direction_getters import ProbDirectionGetter
    shape = (10, 10, 10); idx = np.argwhere(np.ones(shape, bool))
    d = np.tile([[0.0, 0.0, 1.0], [0, 0, 0]], (idx.shape[0], 1, 1)); fr = np.tile([1.0, 0.0], (idx.shape[0], 1))
    verts = default_sphere.vertices.astype(np.float32); edges = default_sphere.edges.astype(np.int32)
    pmf = gpu.pmf_from_fixels(d, fr, idx, shape, verts, 40.0, 0.1)
    seeds = np.array([[5.0, 5.0, 5.0]], np.float32)
    with GPUTracker(ProbDirectionGetter(), pmf, np.ones(shape, np.float32), 0.5, verts, edges, sphere_symm=True,
                    max_angle=radians(45), step_size=0.5, min_pts=4, max_pts=100, ngpus=1, rng_seed=0) as tr:
        sft = tr.generate_sft(seeds, nib.Nifti1Image(np.zeros(shape, np.uint8), np.eye(4)))
    sl = gpu.to_dipy_vox(list(sft.streamlines), shape)[0]
    assert np.ptp(sl[:, 2]) > 5 and np.ptp(sl[:, 0]) < 1.5 and np.ptp(sl[:, 1]) < 1.5


def _phantom_pmf(shape):
    from dipy.data import default_sphere
    idx = np.argwhere(np.ones(shape, bool))
    d = np.tile([[0.0, 0.0, 1.0], [0, 0, 0]], (idx.shape[0], 1, 1)); fr = np.tile([1.0, 0.0], (idx.shape[0], 1))
    verts = default_sphere.vertices.astype(np.float32); edges = default_sphere.edges.astype(np.int32)
    return gpu.pmf_from_fixels(d, fr, idx, shape, verts, 40.0, 0.1), verts, edges


def _phantom_tracker(pmf, verts, edges, shape, chunk_size, stop=None):
    from math import radians
    from cuslines.cuda_python.cu_tractography import GPUTracker
    from cuslines.cuda_python.cu_direction_getters import ProbDirectionGetter
    return GPUTracker(ProbDirectionGetter(), pmf, np.ones(shape, np.float32) if stop is None else stop, 0.5, verts, edges,
                      sphere_symm=True, max_angle=radians(45), step_size=0.5, min_pts=2, max_pts=100, ngpus=1,
                      rng_seed=0, chunk_size=chunk_size)


@pytest.mark.gpu
@pytest.mark.xfail(strict=True, reason="cuslines 2.2.3 generate_sft returns the last chunk once per chunk (upstream bug, docs/outreach/cuslines-issues.md)")
def test_cuslines_generate_sft_is_chunk_invariant_upstream():
    pytest.importorskip("cuslines")
    import nibabel as nib
    shape = (12, 12, 12); pmf, verts, edges = _phantom_pmf(shape)
    ref = nib.Nifti1Image(np.zeros(shape, np.uint8), np.eye(4))
    seeds = np.stack(np.meshgrid(*[np.arange(2, 10, dtype=np.float32)] * 3, indexing="ij"), -1).reshape(-1, 3)   # 512 seeds
    with _phantom_tracker(pmf, verts, edges, shape, 1000) as tr:
        n_one = len(tr.generate_sft(seeds, ref).streamlines)
    with _phantom_tracker(pmf, verts, edges, shape, 300) as tr:
        n_two = len(tr.generate_sft(seeds, ref).streamlines)
    assert n_one == n_two


@pytest.mark.gpu
def test_generate_sft_chunked_matches_single_call_on_gpu():
    pytest.importorskip("cuslines")
    import nibabel as nib
    shape = (12, 12, 12); pmf, verts, edges = _phantom_pmf(shape)
    ref = nib.Nifti1Image(np.zeros(shape, np.uint8), np.eye(4))
    seeds = np.stack(np.meshgrid(*[np.arange(2, 10, dtype=np.float32)] * 3, indexing="ij"), -1).reshape(-1, 3)
    with _phantom_tracker(pmf, verts, edges, shape, 1000) as tr:
        n_one = len(tr.generate_sft(seeds, ref).streamlines)
        n_chunked = len(gpu.generate_sft_chunked(tr, seeds, ref, chunk=300))
    assert n_one == n_chunked == len(seeds)                      # one straight streamline per seed


@pytest.mark.gpu
def test_cuslines_output_is_corner_convention():
    """Stop mask on voxels z = 3..7 (DIPY extent [2.5, 7.5)): raw cuslines
    points run 3..8 (texture convention), converted points stay inside the
    DIPY extent. The PMF/seed side is centre convention: the seed at z = 5
    is tracked, which the §4.3 seed-shift attempt broke at the volume edge."""
    pytest.importorskip("cuslines")
    from math import radians
    import nibabel as nib
    from dipy.data import default_sphere
    from cuslines.cuda_python.cu_tractography import GPUTracker
    from cuslines.cuda_python.cu_direction_getters import ProbDirectionGetter
    shape = (12, 12, 12); idx = np.argwhere(np.ones(shape, bool))
    d = np.tile([[0.0, 0.0, 1.0], [0, 0, 0]], (idx.shape[0], 1, 1)); fr = np.tile([1.0, 0.0], (idx.shape[0], 1))
    verts = default_sphere.vertices.astype(np.float32); edges = default_sphere.edges.astype(np.int32)
    pmf = gpu.pmf_from_fixels(d, fr, idx, shape, verts, 40.0, 0.1)
    stop = np.zeros(shape, np.float32); stop[:, :, 3:8] = 1
    seeds = np.array([[5.0, 5.0, 5.0]], np.float32)
    with GPUTracker(ProbDirectionGetter(), pmf, stop, 0.5, verts, edges, sphere_symm=True, max_angle=radians(45),
                    step_size=0.5, min_pts=2, max_pts=100, ngpus=1, rng_seed=0) as tr:
        sft = tr.generate_sft(seeds, nib.Nifti1Image(np.zeros(shape, np.uint8), np.eye(4)))
    raw = np.asarray(sft.streamlines[0])
    assert raw[:, 2].min() > 2.9 and raw[:, 2].max() > 7.6          # corner convention, as shipped
    fixed = gpu.to_dipy_vox([raw], shape)[0]
    assert fixed[:, 2].min() >= 2.4 and fixed[:, 2].max() <= 7.5


# ------------------------------------- CPU reference for the PMF emulation --
def test_cpu_pmf_tracker_follows_single_fixel_phantom():
    """Same phantom as the cuslines test, DIPY's own probabilistic tracker on
    the same PMF: the CPU reference that says whether r 0.55 (doc 014 §4.1)
    is PMF-tracking semantics or a cuslines artefact."""
    cpu = pytest.importorskip("validation.cuslines_posterior_disco_cpu_pmf")
    from dipy.data import default_sphere
    shape = (10, 10, 10); idx = np.argwhere(np.ones(shape, bool))
    d = np.tile([[0.0, 0.0, 1.0], [0, 0, 0]], (idx.shape[0], 1, 1)); fr = np.tile([1.0, 0.0], (idx.shape[0], 1))
    verts = default_sphere.vertices.astype(np.float32)
    pmf = gpu.pmf_from_fixels(d, fr, idx, shape, verts, 40.0, 0.1)
    sl = cpu.track_pmf_cpu(pmf, np.ones(shape, bool), np.array([[5.0, 5.0, 5.0]]), max_angle=45.0, step_size=0.5, min_len=4)
    assert len(sl) == 1
    s = np.asarray(sl[0])
    assert np.ptp(s[:, 2]) > 5 and np.ptp(s[:, 0]) < 1.5 and np.ptp(s[:, 1]) < 1.5
