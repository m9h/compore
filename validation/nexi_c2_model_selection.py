#!/usr/bin/env python3
"""
doc 012 §7.4 step 2: joint model-structure + parameter inference on ds008771.

Phase A (simulation, the calibration half)
  * train the amortised posterior (classifier + per-model flows) on the
    15-shell powder scheme at the measured SNR range;
  * on held-out simulations: classification accuracy and calibration of
    q(M | S) vs the Laplace evidence p(M | S); parameter recovery (RMSE,
    coverage of the 90 % interval) for both; SBC ranks for the flows.
Phase B (data)
  * per session: q(M | S) and posterior medians of θ for every cortical
    voxel (amortised), and for the 62 DKT-label median signals both
    estimators; held-out posterior predictive check: fit the 8 reduced
    features, predict the 7 others (amortised: posterior-mean prediction;
    Laplace: at the MAP);
  * scan–rescan: per DKT label, p(M) and θ in ses-01 vs ses-02 over the
    rescanned subjects → Bland–Altman bias / LoA and ICC.
Writes validation/nexi_c2_model_selection.json and the per-session npz maps
to /data/datasets/compore/ds008771/model_selection/.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
from compore.inference import model_selection as ms
from compore.signal_models import gm_powder as gm
from compore.io import nexi_c2 as n

POWDER = Path("/data/datasets/compore/ds008771/powder")
OUT = Path("/data/datasets/compore/ds008771/model_selection")


def icc(a, b):
    """ICC(2,1)-style agreement between two sessions (a, b over units)."""
    x = np.column_stack([a, b]); x = x[np.isfinite(x).all(1)]
    if len(x) < 3: return float("nan")
    k, nn = 2, len(x); mean = x.mean(); ms_r = k * np.sum((x.mean(1) - mean) ** 2) / (nn - 1)
    ms_c = nn * np.sum((x.mean(0) - mean) ** 2) / (k - 1)
    ms_e = (np.sum((x - mean) ** 2) - (nn - 1) * ms_r - (k - 1) * ms_c) / ((nn - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / nn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=300_000); ap.add_argument("--steps-clf", type=int, default=4000)
    ap.add_argument("--steps-flow", type=int, default=6000); ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-laplace-test", type=int, default=120); ap.add_argument("--voxels-per-session", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-data", action="store_true"); ap.add_argument("--out", default="validation/nexi_c2_model_selection.json")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    res = {"args": vars(a), "device": str(jax.devices()[0])}
    sessions = sorted(POWDER.glob("sub-*_ses-*.npz"))
    first = np.load(sessions[0], allow_pickle=True)
    feats = [tuple(int(v) for v in f) for f in first["features"]]; counts = [int(c) for c in first["counts"]]
    sch = ms.PowderScheme.from_features(feats, counts)
    # measured relative b0 noise across sessions → SNR range for training
    snr_med = [float(np.median(np.load(p)["S_b0"] / np.maximum(np.load(p)["sigma_b0"], 1e-6))) for p in sessions]
    snr_range = (max(8.0, 0.5 * min(snr_med)), 2.5 * max(snr_med)); res["snr_median_per_session"] = dict(zip([p.stem for p in sessions], snr_med)); res["snr_range_train"] = snr_range
    print("sessions", len(sessions), "SNR medians", np.round(snr_med, 1), "train range", np.round(snr_range, 1), flush=True)

    # ---------------- Phase A
    t0 = time.time()
    post = ms.train_amortised(jax.random.PRNGKey(0), sch, n_sims=a.n_sims, steps_clf=a.steps_clf, steps_flow=a.steps_flow, snr_range=snr_range)
    res["train"] = {**post.train_log, "seconds": time.time() - t0}; print("trained", res["train"], flush=True)
    m_idx, theta, sigma, S = ms.simulate_joint(jax.random.PRNGKey(123), sch, a.n_test, snr_range)
    m_idx, theta, sigma, S = np.asarray(m_idx), np.asarray(theta), np.asarray(sigma), np.asarray(S)
    pm = post.p_model(S, sigma)
    A = {"amortised_accuracy": float(np.mean(pm.argmax(1) == m_idx)), "confusion_amortised": np.zeros((3, 3), int).tolist()}
    for t, p in zip(m_idx, pm.argmax(1)): A["confusion_amortised"][t][p] += 1
    # calibration of q(M|S): reliability over bins of max prob
    conf = pm.max(1); hit = pm.argmax(1) == m_idx; bins = np.linspace(0.33, 1.0, 6)
    A["reliability_amortised"] = [{"p_lo": float(bins[i]), "p_hi": float(bins[i + 1]), "n": int(((conf >= bins[i]) & (conf < bins[i + 1])).sum()),
                                   "acc": float(hit[(conf >= bins[i]) & (conf < bins[i + 1])].mean()) if ((conf >= bins[i]) & (conf < bins[i + 1])).any() else None} for i in range(5)]
    # parameter recovery (amortised) on the true model
    rec = {}
    for j, m in enumerate(ms.MODELS):
        sel = np.flatnonzero(m_idx == j)[:150]; P = len(ms.box(m)[0]); err = []; cov = []
        for i in sel:
            s = post.sample_theta(jax.random.PRNGKey(int(i)), m, S[i], sigma[i], 400)
            med = np.median(s, 0); lo, hi = np.percentile(s, [5, 95], 0)
            err.append(med - theta[i, :P]); cov.append((theta[i, :P] >= lo) & (theta[i, :P] <= hi))
        err = np.asarray(err); cov = np.asarray(cov)
        rec[m] = {"params": gm.MODEL_PARAMS[m], "rmse": np.sqrt((err ** 2).mean(0)).tolist(), "coverage90": cov.mean(0).tolist(), "n": int(len(sel))}
    A["recovery_amortised"] = rec
    # Laplace on a subset (slow)
    t0 = time.time(); sel = np.arange(min(a.n_laplace_test, a.n_test)); lp_hit = []; lp_conf = []; lp_rec = {m: [] for m in ms.MODELS}; lp_cov = {m: [] for m in ms.MODELS}
    for i in sel:
        r = ms.laplace_model_posterior(S[i], float(sigma[i]), sch, n_restarts=4)
        p = np.array([r["p_model"][m] for m in ms.MODELS]); lp_hit.append(p.argmax() == m_idx[i]); lp_conf.append(p.max())
        m = ms.MODELS[m_idx[i]]; P = len(ms.box(m)[0]); f = r["fits"][m]
        lp_rec[m].append(f["theta"] - theta[i, :P]); lp_cov[m].append(np.abs(f["theta"] - theta[i, :P]) <= 1.645 * f["theta_sd"])
    A["laplace_accuracy"] = float(np.mean(lp_hit)); A["laplace_n"] = int(len(sel)); A["laplace_seconds_per_voxel"] = (time.time() - t0) / len(sel)
    A["recovery_laplace"] = {m: {"rmse": np.sqrt((np.asarray(v) ** 2).mean(0)).tolist(), "coverage90": np.asarray(lp_cov[m]).mean(0).tolist(), "n": len(v)} for m, v in lp_rec.items() if v}
    A["amortised_accuracy_same_subset"] = float(np.mean(pm[sel].argmax(1) == m_idx[sel]))
    # SBC
    ranks = ms.sbc_ranks(post, jax.random.PRNGKey(7), n_test=240, n_post=200)
    A["sbc"] = {m: {"n": int(len(r)), "ks_p_per_param": [float(__import__("scipy.stats", fromlist=["kstest"]).kstest(r[:, k] / 200.0, "uniform").pvalue) for k in range(r.shape[1])]} for m, r in ranks.items() if len(r)}
    res["phaseA"] = A; print("phase A", json.dumps({k: v for k, v in A.items() if "confusion" not in k and "reliab" not in k}, indent=None)[:1500], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.skip_data:
        return

    # ---------------- Phase B
    red = np.array([f in n.REDUCED_FEATURES for f in feats]); B = {"sessions": {}}
    labels_all = None
    for p in sessions:
        d = np.load(p, allow_pickle=True); tag = p.stem; t0 = time.time()
        S_v = d["S"].astype(np.float32); sig_v = (d["sigma_b0"] / np.maximum(d["S_b0"], 1e-6)).astype(np.float32); lab = d["label"]
        if a.voxels_per_session and len(S_v) > a.voxels_per_session:
            idx = np.random.default_rng(0).choice(len(S_v), a.voxels_per_session, replace=False); S_v, sig_v, lab = S_v[idx], sig_v[idx], lab[idx]
        sig_v = np.clip(sig_v, 1.0 / snr_range[1], 1.0 / snr_range[0])
        pm_v = post.p_model(S_v, sig_v)
        # amortised θ medians for the arg-max model, batched by model
        th_med = np.full((len(S_v), 6), np.nan, np.float32)
        for j, m in enumerate(ms.MODELS):
            sel = np.flatnonzero(pm_v.argmax(1) == j); P = len(ms.box(m)[0])
            for i in sel:
                s = post.sample_theta(jax.random.PRNGKey(int(i)), m, S_v[i], sig_v[i], 100); th_med[i, :P] = np.median(s, 0)
        # ROI medians: both estimators + PPC on the 7 held-out shells
        rois = {}
        for L in np.unique(lab):
            sel = lab == L
            if sel.sum() < 20: continue
            S_roi = np.median(S_v[sel], 0); sig_roi = float(np.median(sig_v[sel]) / np.sqrt(sel.sum()))
            sig_roi = max(sig_roi, 1.0 / snr_range[1])
            pm_roi = post.p_model(S_roi[None], np.array([sig_roi]))[0]
            # amortised on reduced features only: train-time scheme is the full one, so PPC uses the Laplace fit on reduced shells
            sch_red = ms.PowderScheme.from_features([f for f, r in zip(feats, red) if r], [c for c, r in zip(counts, red) if r])
            lap_red = ms.laplace_model_posterior(S_roi[red], sig_roi, sch_red, n_restarts=4)
            ppc = {}
            for m in ms.MODELS:
                pred = np.asarray(gm.rician_mean(ms.forward(m, jnp.asarray(lap_red["fits"][m]["theta"]), sch), sig_roi))
                ppc[m] = float(np.sqrt(np.mean((pred[~red] - S_roi[~red]) ** 2)))
            lap_full = ms.laplace_model_posterior(S_roi, sig_roi, sch, n_restarts=4)
            rois[int(L)] = {"n_vox": int(sel.sum()), "p_amortised": pm_roi.tolist(), "p_laplace_full": [lap_full["p_model"][m] for m in ms.MODELS],
                            "p_laplace_reduced": [lap_red["p_model"][m] for m in ms.MODELS], "ppc_rmse_heldout": ppc,
                            "theta_laplace": {m: lap_full["fits"][m]["theta"].tolist() for m in ms.MODELS},
                            "theta_amortised_nexi": np.median(post.sample_theta(jax.random.PRNGKey(1), "nexi", S_roi, sig_roi, 400), 0).tolist()}
        np.savez_compressed(OUT / f"{tag}.npz", p_model=pm_v, theta_med=th_med, label=lab, ijk=d["ijk"] if not a.voxels_per_session else d["ijk"][idx], shape=d["shape"], affine=d["affine"])
        B["sessions"][tag] = {"n_vox": int(len(S_v)), "frac_argmax": np.bincount(pm_v.argmax(1), minlength=3).tolist(), "mean_p": pm_v.mean(0).tolist(),
                              "rois": rois, "seconds": time.time() - t0}
        print(tag, B["sessions"][tag]["frac_argmax"], np.round(B["sessions"][tag]["mean_p"], 3), f"{time.time()-t0:.0f}s", flush=True)
        res["phaseB"] = B; json.dump(res, open(a.out, "w"), indent=1)
    # scan–rescan over subjects with both sessions, per ROI
    subs = sorted({t.split("_")[0] for t in B["sessions"]}); pairs = [(s, f"{s}_ses-01", f"{s}_ses-02") for s in subs if f"{s}_ses-01" in B["sessions"] and f"{s}_ses-02" in B["sessions"]]
    rr = {}
    for key, getter in [("p_nexi_amortised", lambda r: r["p_amortised"][0]), ("p_nexi_laplace", lambda r: r["p_laplace_full"][0]),
                        ("tex_laplace_ms", lambda r: r["theta_laplace"]["nexi"][3] * 1e3), ("tex_amortised_ms", lambda r: r["theta_amortised_nexi"][3] * 1e3),
                        ("f_laplace", lambda r: r["theta_laplace"]["nexi"][0]), ("f_amortised", lambda r: r["theta_amortised_nexi"][0]),
                        ("Di_laplace", lambda r: r["theta_laplace"]["nexi"][1] * 1e9), ("Di_amortised", lambda r: r["theta_amortised_nexi"][1] * 1e9),
                        ("De_laplace", lambda r: r["theta_laplace"]["nexi"][2] * 1e9), ("De_amortised", lambda r: r["theta_amortised_nexi"][2] * 1e9)]:
        a1, a2 = [], []
        for s, t1, t2 in pairs:
            r1, r2 = B["sessions"][t1]["rois"], B["sessions"][t2]["rois"]
            for L in set(r1) & set(r2):
                a1.append(getter(r1[L])); a2.append(getter(r2[L]))
        a1, a2 = np.asarray(a1), np.asarray(a2); diff = a2 - a1
        rr[key] = {"n": int(len(a1)), "bias": float(diff.mean()), "loa": float(1.96 * diff.std()), "icc": icc(a1, a2), "mean": float(np.mean((a1 + a2) / 2))}
    res["scan_rescan"] = {"pairs": [p[0] for p in pairs], "metrics": rr}; json.dump(res, open(a.out, "w"), indent=1)
    print("scan-rescan", json.dumps(rr, indent=None), flush=True)


if __name__ == "__main__":
    main()
