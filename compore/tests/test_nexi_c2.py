"""ds008771 (reduced NEXI, Connectome 2.0) loader: scheme parsing and feature selection.
Runs against the local copy when present, else on a synthetic preproc directory."""
import numpy as np
import pytest

from compore.io import nexi_c2 as n


def _fake_session(tmp_path):
    d = tmp_path / "derivatives" / "preproc" / "sub-01" / "ses-01" / "dwi"; d.mkdir(parents=True)
    p = "sub-01_ses-01_desc-preproc_dwi"
    b = [0, 1000, 2500, 0, 5000, 0, 7350, 0, 1000, 10000, 0, 12500]
    D = [12, 12, 12, 15, 15, 20, 20, 27, 27, 27, 45, 45]
    (d / f"{p}.bval").write_text(" ".join(f"{x:.6f}" for x in b))
    (d / f"{p}.delta").write_text(" ".join(f"{x:.6f}" for x in D))
    return tmp_path


def test_scheme_without_bvecs_is_si_and_selects_reduced_features(tmp_path):
    root = _fake_session(tmp_path)
    sc = n.load_scheme("sub-01", "ses-01", root)
    assert sc.n == 12 and sc.bvecs is None
    assert sc.bvals_si.max() == 12.5e9 and np.allclose(sc.delta, 5e-3) and sc.Delta.max() == 45e-3
    assert ((1000, 12), 1) in sc.features and ((7350, 20), 1) in sc.features
    m = sc.subset(n.REDUCED_FEATURES)
    # all b0 + the reduced pairs present in this fake: (1000,12),(2500,12),(5000,15),(7350,20),(1000,27),(12500,45)
    assert int(m.sum()) == 5 + 6
    groups = sc.powder_average_groups()
    assert sum(len(ix) for _, ix in groups) == 12


@pytest.mark.skipif(not n.DEFAULT_ROOT.exists(), reason="ds008771 not on this host")
def test_local_copy_scheme_matches_paper():
    av = n.available()
    ready = [(s, ses) for s, d in av.items() for ses, f in d.items() if f["bval"] and f["delta"]]
    assert ready, "no session with bval + delta"
    sc = n.load_scheme(*ready[0])
    assert sc.n == 513
    feats = dict(sc.features)
    assert feats[(12500, 45)] == 64 and feats[(1000, 12)] == 20 and feats[(7350, 20)] == 34
    assert int(sc.subset(n.REDUCED_FEATURES).sum()) == 263      # 250 directions + 13 b0
    assert len(sc.features) == 15                                # the paper's 15 (b, Δ) features
