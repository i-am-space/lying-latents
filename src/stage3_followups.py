"""Stage 3 follow-ups (config/preregistration.yaml stage3_amendment_3, fixed before either ran).

--experiment stress   mechanism of the FDR rise and the failure boundary, on the v2 data (p = 512):
                      real_mvr / real_equi / gauss_mvr reproduce v2 at amplitudes 1-8 and extend the
                      sweep to 12 and 20; real_hurdle adds Stage 4's SCIP hurdle knockoffs on the same
                      labels. Every fit also records, per null latent, whether W > 0, binned by the
                      latent's max |corr| with the planted set (valid knockoffs: a fair coin per bin).
--experiment dims     what causes the power collapse at p = 2048: latents (512, 2048) x rows per
                      latent (n/p 9.77, 32.88) x S (equicorrelated, MVR) x real / Gaussian data.

Stages:  --stage diagnose  build, timing, sampler checks and a small pilot, in minutes
         --stage full      the whole grid (checkpointed after every replicate; rerun to resume)

Usage: python src/stage3_followups.py --config config/default.yaml --device cuda --experiment stress --stage diagnose
       python src/stage3_followups.py --config config/default.yaml --device cuda --experiment dims --stage full
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import cache_path, load_config, results_dir, rng_for
from knockoff_audit import estimate_cov, standardise, swap_sweep
from planted_fdr import generate_planted_labels
from planted_fdr_controls import (METRICS, build_datasets, cell_rng, fit_lasso_checked, knockoff_threshold,
                                  mean_se, score_fit, wy_marginal)
from stage4_repairs import SCIPSampler, mean_corr

ARM_LABEL = {"real_mvr": "real, Gaussian knockoffs (MVR)", "real_equi": "real, Gaussian knockoffs (equicorr.)",
             "gauss_mvr": "Gaussian control (MVR)", "real_hurdle": "real, hurdle knockoffs"}
ARM_COLOR = {"real_mvr": "#c0392b", "real_equi": "#e07b39", "gauss_mvr": "#3b6ea5", "real_hurdle": "#2e8b57"}
GAUSS_ARMS = ("real_mvr", "real_equi", "gauss_mvr")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def load_cache(args, cfg):
    cp = Path(args.cache) if args.cache else cache_path(cfg)
    if not cp.exists():
        raise SystemExit(f"Cache not found at {cp}. Run src/cache_activations.py first.")
    d = np.load(cp, allow_pickle=True)
    return d, d[f"X_{cfg['aggregation']['primary']}"].astype(np.float32, copy=False)


def save_checkpoint(path: Path, arrays: dict, meta: dict, warm=None) -> None:
    np.savez(path.with_suffix(".npz"), **arrays)
    if warm is not None:
        torch.save(warm, path.with_suffix(".pt"))
    path.write_text(json.dumps(meta, default=float))


def load_checkpoint(path: Path, meta_key: dict):
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    if any(meta.get(k) != v for k, v in meta_key.items()):
        print("  checkpoint does not match this design; starting afresh", flush=True)
        return None
    return meta, dict(np.load(path.with_suffix(".npz")))


def drop_checkpoint(path: Path) -> None:
    for f in (path, path.with_suffix(".npz"), path.with_suffix(".pt")):
        f.unlink(missing_ok=True)


def paired(a: np.ndarray, b: np.ndarray):
    """Mean and SE (ddof = 1) of a - b over replicates."""
    return mean_se(np.asarray(a, np.float64) - np.asarray(b, np.float64))


def unpaired(a: np.ndarray, b: np.ndarray):
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    ma, sa = mean_se(a)
    mb, sb = mean_se(b)
    return ma - mb, math.sqrt(sa ** 2 + sb ** 2)


def cell_rows(M: dict, names, amps, forms, ks, qs, extra=None):
    """Per-cell means and SEs over replicates; NaN-only metrics are skipped."""
    conds = []
    for d in names:
        for ia, amp in enumerate(amps):
            for jf, form in enumerate(forms):
                for kk, k in enumerate(ks):
                    for iq, q in enumerate(qs):
                        row = {"arm": d, "amplitude": amp, "form": form, "k": k, "q": q,
                               "power_capped_k_lt_1_over_q": bool(k < math.ceil(1 / q))}
                        for m in M:
                            v = M[m][d][ia, jf, kk, :, iq]
                            if np.isnan(v).all():
                                continue
                            row[m], row[m + "_se"] = mean_se(v)
                        row["ko_inflated"] = bool(row["ko_fdr"] - 1.96 * row["ko_fdr_se"] > q)
                        if extra:
                            row.update(extra(d))
                        conds.append(row)
    return conds


# ---------------------------------------------------------------------------
# stress: mechanism and boundary (p = 512, the v2 data)
# ---------------------------------------------------------------------------

# columns of a mechanism record (per link bin)
C_N, C_NZ, C_POS, C_SUMW, C_MPOS, C_MSUM, C_FD0 = range(7)


def marginal_w(Phi: torch.Tensor, y_t: torch.Tensor, p: int) -> np.ndarray:
    """|corr(y, X_j)| - |corr(y, X~_j)| (columns are standardised). Antisymmetric under swapping X_j
    and X~_j, so for a null latent with valid knockoffs its sign is a fair coin, like the lasso W's;
    unlike the lasso W it is never exactly 0, so every null contributes."""
    r = (Phi.t() @ (y_t - y_t.mean())).abs().cpu().numpy()
    return r[:p] - r[p:]


def mech_stats(W: np.ndarray, Wm: np.ndarray, S_idx: np.ndarray, absC: np.ndarray, edges, qs) -> np.ndarray:
    """Per link bin: [#nulls, #nonzero W, #W > 0, sum W, #marginal W > 0, sum marginal W,
    Knockoff+ false discoveries at each q]."""
    p = W.size
    null = np.ones(p, dtype=bool)
    null[S_idx] = False
    link = absC[:, S_idx].max(axis=1)
    nb = len(edges) - 1
    b = np.clip(np.digitize(link, edges[1:-1]), 0, nb - 1)
    taus = [knockoff_threshold(W, q, 1) for q in qs]
    out = np.zeros((nb, C_FD0 + len(qs)), dtype=np.float32)
    for bi in range(nb):
        m = null & (b == bi)
        w, wm = W[m], Wm[m]
        out[bi, :C_FD0] = (m.sum(), (w != 0).sum(), (w > 0).sum(), w.sum(), (wm > 0).sum(), wm.sum())
        for iq, tau in enumerate(taus):
            out[bi, C_FD0 + iq] = 0 if math.isinf(tau) else (w >= tau).sum()
    return out


def gaussian_fit(dsd: dict, y_np: np.ndarray, seed: int, s3: dict, device, true_set: set):
    """planted_fdr_controls.run_fit, step for step (so v2 reproduces), also returning W."""
    qs = s3["nominal_fdr_targets"]
    Z = dsd["Z"]
    p = Z.shape[1]
    np.random.seed(seed)
    Zk = dsd["sampler"].sample_knockoffs()
    Phi = torch.from_numpy(np.hstack([Z, Zk]).astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
    w, conv, res, _ = fit_lasso_checked(Phi, y_t, s3["lasso"]["lambda"], s3["lasso"]["max_iter"], s3["lasso"]["tol"])
    w_np = w.cpu().numpy()
    W = np.abs(w_np[:p]) - np.abs(w_np[p:])
    r, thr = wy_marginal(Phi[:, :p], y_t, qs, s3["wy_baseline"]["n_permutations"], seed)
    return score_fit(W, r, thr, qs, true_set), conv, res, W, marginal_w(Phi, y_t, p)


def build_stress(cfg: dict, sec: dict, X_all: np.ndarray, device) -> dict:
    s3 = cfg["stage3_v2"]
    ds, info = build_datasets(cfg, X_all)
    # the raw real latents behind ds["real_mvr"]: the first two draws of s3v2_data, as in build_datasets
    rng = rng_for(cfg, "s3v2_data")
    n_all, p_all = X_all.shape
    cols = np.sort(rng.choice(p_all, s3["p"], replace=False))
    rows = np.sort(rng.choice(n_all, s3["n_rows"], replace=False))
    X = X_all[np.ix_(rows, cols)].astype(np.float64)
    Z, mu, sd = standardise(X)
    if not np.array_equal(Z, ds["real_mvr"]["Z"]):
        raise SystemExit("raw latents do not match build_datasets' real data")
    absC = {}
    for name in ("real_mvr", "gauss_mvr"):
        C = np.abs(np.corrcoef(ds[name]["Z"], rowvar=False)).astype(np.float32)
        np.fill_diagonal(C, 0.0)
        absC[name] = C
    t = time.time()
    scip = SCIPSampler(X, "hurdle", sec["scip"], device)
    info["scip_setup_seconds"] = time.time() - t
    return {"ds": ds, "info": info, "X": X, "mu": mu, "sd": sd, "scip": scip,
            "absC": {"real_mvr": absC["real_mvr"], "real_equi": absC["real_mvr"],
                     "real_hurdle": absC["real_mvr"], "gauss_mvr": absC["gauss_mvr"]}}


def hurdle_draw(cfg: dict, B: dict, rep: int, stream_idx=()):
    seed = int(cell_rng(cfg, "s3s_hurdle", *stream_idx, rep).integers(2**31))
    t = time.time()
    Xk, info = B["scip"].sample(seed)
    info["zero_mass_error_max"] = float(np.abs((B["X"] == 0).mean(0) - (Xk == 0).mean(0)).max())
    info["draw_seconds"] = time.time() - t
    Zk = (Xk - B["mu"]) / B["sd"]
    info["mean_corr_X_Xk"] = mean_corr(B["ds"]["real_mvr"]["Z"], Zk)
    return Zk, info


def stress_exchangeability(cfg: dict, sec: dict, B: dict, arms) -> dict:
    ex = sec["exchangeability"]
    rng = rng_for(cfg, "s3s_diagnostics")
    out = {}
    for ai, name in enumerate(arms):
        print(f"  [{name}]", flush=True)
        if name == "real_hurdle":
            Zk, info = hurdle_draw(cfg, B, 0, stream_idx=(999,))
            Zd = B["ds"]["real_mvr"]["Z"]
        else:
            np.random.seed(int(cell_rng(cfg, "s3s_diagnostics", ai).integers(2**31)))
            Zd = B["ds"][name]["Z"]
            Zk, info = B["ds"][name]["sampler"].sample_knockoffs(), {}
        rows = swap_sweep(Zd, Zk, cfg, rng, ex["swap_sizes"], ex["swap_replicates"])
        out[name] = {"swap": rows, "mean_corr_X_Xk": mean_corr(Zd, Zk), **info}
    p = B["ds"]["real_mvr"]["Z"].shape[1]
    return {"arms": out, "full_swap_auc": {n: next(r["auc"] for r in v["swap"] if r["swap_size"] == p)
                                          for n, v in out.items()}}


def amp_index(amp: float, v2_amps: list, amps: list) -> int:
    """v2's amplitude index for amplitudes v2 ran (so seeds reproduce v2); 6, 7, ... for new ones."""
    if amp in v2_amps:
        return v2_amps.index(amp)
    return len(v2_amps) + [a for a in amps if a not in v2_amps].index(amp)


def run_stress(cfg: dict, sec: dict, B: dict, device, R: int, amps, forms, ks, arms, checkpoint=None):
    s3 = cfg["stage3_v2"]
    qs = sec["nominal_fdr_targets"]
    edges = sec["link_bins"]
    nb, nq = len(edges) - 1, len(qs)
    p = B["ds"]["real_mvr"]["Z"].shape[1]
    shape = (len(amps), len(forms), len(ks), R)
    M = {m: {a: np.full(shape + (nq,), np.nan, dtype=np.float32) for a in arms} for m in METRICS}
    MECH = {a: np.zeros(shape + (nb, C_FD0 + nq), dtype=np.float32) for a in arms}
    CONV = {a: np.zeros(shape, dtype=bool) for a in arms}
    hinfo, start = [], 0
    key = {"shape": list(shape), "arms": list(arms), "amps": list(amps)}
    if checkpoint is not None:
        ck = load_checkpoint(checkpoint, key)
        if ck:
            meta, z = ck
            for a in arms:
                for m in METRICS:
                    M[m][a] = z[f"{m}__{a}"]
                MECH[a], CONV[a] = z[f"mech__{a}"], z[f"conv__{a}"]
            hinfo, start = meta["hurdle_info"], meta["reps_done"]
            if checkpoint.with_suffix(".pt").exists():
                B["scip"].warm = torch.load(checkpoint.with_suffix(".pt"), map_location=device)
            print(f"  resuming from checkpoint: {start}/{R} replicates done", flush=True)
    no_r, no_thr = np.zeros(p), {q: float("inf") for q in qs}
    v2_amps = s3["signal_amplitudes"]
    t0 = time.time()
    for rep in range(start, R):
        Phi_h = None
        if "real_hurdle" in arms:
            Zk_h, info = hurdle_draw(cfg, B, rep)
            hinfo.append(info)
            Phi_h = torch.from_numpy(np.hstack([B["ds"]["real_mvr"]["Z"], Zk_h]).astype(np.float32)).to(device)
            del Zk_h
        for ia, amp in enumerate(amps):
            iv = amp_index(amp, v2_amps, amps)
            for jf, form in enumerate(forms):
                for kk, k in enumerate(ks):
                    seed = int(cell_rng(cfg, "s3v2_knockoff", iv, jf, kk, rep).integers(2**31))
                    y_real, S_real = None, None
                    for name in arms:
                        rng = cell_rng(cfg, "s3v2_planted", iv, jf, kk, rep)
                        S_idx = np.sort(rng.choice(p, size=k, replace=False))
                        truth = set(S_idx.tolist())
                        if name == "real_hurdle":
                            if y_real is None:
                                y_real, _ = generate_planted_labels(B["ds"]["real_mvr"]["Z"], S_idx, form, amp, rng)
                            y_t = torch.from_numpy(y_real.astype(np.float32)).to(device)
                            w, cv, _, _ = fit_lasso_checked(Phi_h, y_t, s3["lasso"]["lambda"],
                                                            s3["lasso"]["max_iter"], s3["lasso"]["tol"])
                            w_np = w.cpu().numpy()
                            W = np.abs(w_np[:p]) - np.abs(w_np[p:])
                            Wm = marginal_w(Phi_h, y_t, p)
                            sc = score_fit(W, no_r, no_thr, qs, truth)
                            for m in ("wy_fdr", "wy_pow", "wy_nd"):
                                sc[m][:] = np.nan
                        else:
                            dsd = B["ds"][name]
                            y_np, _ = generate_planted_labels(dsd["Z"], S_idx, form, amp, rng)
                            if name in ("real_mvr", "real_equi"):
                                y_real = y_np
                            sc, cv, _, W, Wm = gaussian_fit(dsd, y_np, seed, s3, device, truth)
                        for m in METRICS:
                            M[m][name][ia, jf, kk, rep] = sc[m]
                        CONV[name][ia, jf, kk, rep] = cv
                        MECH[name][ia, jf, kk, rep] = mech_stats(W, Wm, S_idx, B["absC"][name], edges, qs)
        del Phi_h
        if checkpoint is not None:
            arrays = {f"{m}__{a}": M[m][a] for m in METRICS for a in arms}
            arrays.update({f"mech__{a}": MECH[a] for a in arms})
            arrays.update({f"conv__{a}": CONV[a] for a in arms})
            save_checkpoint(checkpoint, arrays, {**key, "reps_done": rep + 1, "hurdle_info": hinfo},
                            warm=B["scip"].warm if "real_hurdle" in arms else None)
        el = time.time() - t0
        done = rep + 1 - start
        print(f"  rep {rep + 1:>2}/{R}  elapsed {el / 60:.1f}m  ETA {el / done * (R - rep - 1) / 60:.1f}m"
              + (f"  hurdle draw {hinfo[-1]['draw_seconds']:.0f}s zero-mass err {hinfo[-1]['zero_mass_error_max']:.4f}"
                 if hinfo else ""), flush=True)
    return M, MECH, CONV, hinfo


def frac_pos(mech: np.ndarray, amp_sel, bin_sel=None, stat: str = "lasso") -> np.ndarray:
    """Per replicate: null P(W > 0 | W != 0), pooled over forms, k and the selected amplitudes/bins.
    stat = lasso (the filter's W) or marginal (the dense marginal-correlation W)."""
    x = mech[amp_sel]                                   # (A', F, K, R, nb, c)
    if bin_sel is not None:
        x = x[..., bin_sel, :]
    den_c, pos_c = (C_NZ, C_POS) if stat == "lasso" else (C_N, C_MPOS)
    nz = x[..., den_c].sum(axis=(0, 1, 2, 4))
    pos = x[..., pos_c].sum(axis=(0, 1, 2, 4))
    return np.where(nz > 0, pos / np.maximum(nz, 1), np.nan)


def ms_nan(v):
    v = np.asarray(v, np.float64)
    v = v[~np.isnan(v)]
    return mean_se(v) if v.size else (float("nan"), float("nan"))


def aggregate_stress(cfg, sec, M, MECH, CONV, amps, forms, ks, arms, R):
    qs = sec["nominal_fdr_targets"]
    edges = sec["link_bins"]
    nb = len(edges) - 1
    conds = cell_rows(M, arms, amps, forms, ks, qs)
    idx = {(c["arm"], c["amplitude"], c["form"], c["k"], c["q"]): c for c in conds}

    # contrasts: hurdle vs Gaussian on the same labels are paired; vs the control, unpaired (as in v2)
    contrasts = []
    pairs = [("real_mvr", "gauss_mvr", False), ("real_hurdle", "real_equi", True), ("real_hurdle", "real_mvr", True)]
    for a, b, is_paired in pairs:
        if a not in arms or b not in arms:
            continue
        for ia, amp in enumerate(amps):
            for jf, form in enumerate(forms):
                for kk, k in enumerate(ks):
                    for iq, q in enumerate(qs):
                        row = {"a": a, "b": b, "paired": is_paired, "amplitude": amp, "form": form, "k": k, "q": q}
                        for m in ("ko_fdr", "ko_pow"):
                            va, vb = M[m][a][ia, jf, kk, :, iq], M[m][b][ia, jf, kk, :, iq]
                            row[m + "_diff"], row[m + "_diff_se"] = paired(va, vb) if is_paired else unpaired(va, vb)
                            row[m + "_sig"] = bool(abs(row[m + "_diff"]) > 1.96 * row[m + "_diff_se"] > 0)
                        contrasts.append(row)

    # mechanism summaries
    mech = {}
    for a in arms:
        per_amp = {}
        for ia, amp in enumerate(amps):
            x = MECH[a][ia].sum(axis=(0, 1, 2))           # (nb, c) over forms, k, reps
            iq = qs.index(0.10)
            per_amp[str(amp)] = {
                "pooled_frac_pos": ms_nan(frac_pos(MECH[a], [ia])),
                "bin_frac_pos": [ms_nan(frac_pos(MECH[a], [ia], [bi])) for bi in range(nb)],
                "pooled_frac_pos_marginal": ms_nan(frac_pos(MECH[a], [ia], stat="marginal")),
                "bin_frac_pos_marginal": [ms_nan(frac_pos(MECH[a], [ia], [bi], "marginal")) for bi in range(nb)],
                "bin_n_nulls": x[:, C_N].tolist(),
                "bin_n_nonzero_W": x[:, C_NZ].tolist(),
                "bin_mean_W": (x[:, C_SUMW] / np.maximum(x[:, C_N], 1)).tolist(),
                "bin_mean_marginal_W": (x[:, C_MSUM] / np.maximum(x[:, C_N], 1)).tolist(),
                "bin_false_disc_per_null_q0.10": (x[:, C_FD0 + iq] / np.maximum(x[:, C_N], 1)).tolist()}
        mech[a] = per_amp

    hi_amp = [ia for ia, a in enumerate(amps) if a >= 5]
    hi_bins = [bi for bi in range(nb) if edges[bi] >= 0.3]
    rules = {}
    # R1-R3 on the filter's lasso W (pre-registered); repeated on the dense marginal W as "<id>_marginal"
    for stat, sfx in (("lasso", ""), ("marginal", "_marginal")):
        def fp(a, amp_sel, bin_sel=None):
            return frac_pos(MECH[a], amp_sel, bin_sel, stat)
        if "gauss_mvr" in arms:
            m, s = ms_nan(fp("gauss_mvr", list(range(len(amps)))))
            infl = [c for c in conds if c["arm"] == "gauss_mvr" and c["ko_inflated"]]
            rules["R1" + sfx] = {"control_pooled_frac_pos": [m, s], "n_control_inflated": len(infl),
                                 "holds": bool(not (m - 0.5 > 1.96 * s) and not infl)}
        if "real_mvr" in arms and "gauss_mvr" in arms and hi_amp:
            f_hi, f_lo = fp("real_mvr", hi_amp, hi_bins), fp("real_mvr", hi_amp, [0])
            g_hi = fp("gauss_mvr", hi_amp, hi_bins)
            a1 = ms_nan(fp("real_mvr", hi_amp))
            a2 = ms_nan(f_hi - f_lo)
            a3 = unpaired(f_hi[~np.isnan(f_hi)], g_hi[~np.isnan(g_hi)])
            rules["R2" + sfx] = {"real_pooled_frac_pos_amp_ge5": a1, "high_minus_low_link": a2,
                                 "real_minus_control_high_link": a3,
                                 "supported": bool(a1[0] - 0.5 > 1.96 * a1[1] and a2[0] > 1.96 * a2[1]
                                                   and a3[0] > 1.96 * a3[1])}
        if "real_hurdle" in arms and "real_equi" in arms and hi_amp:
            cs = [c for c in contrasts if c["a"] == "real_hurdle" and c["b"] == "real_equi" and c["amplitude"] >= 5]
            n_lower = sum(c["ko_fdr_sig"] and c["ko_fdr_diff"] < 0 for c in cs)
            fh, fe = ms_nan(fp("real_hurdle", hi_amp)), ms_nan(fp("real_equi", hi_amp))
            rules["R3" + sfx] = {"n_cells_amp_ge5": len(cs), "n_hurdle_fdr_sig_lower": n_lower,
                                 "hurdle_frac_pos": fh, "equi_frac_pos": fe,
                                 "supported": bool(n_lower > len(cs) / 2 and abs(fh[0] - 0.5) < abs(fe[0] - 0.5))}
    rules["R4"] = {}
    for a in arms:
        bad = sorted({c["amplitude"] for c in conds if c["arm"] == a and c["ko_inflated"]})
        cells = [{k: c[k] for k in ("amplitude", "form", "k", "q", "ko_fdr", "ko_fdr_se")}
                 for c in conds if c["arm"] == a and c["ko_inflated"]]
        rules["R4"][a] = {"first_inflated_amplitude": bad[0] if bad else None, "inflated_cells": cells,
                          "max_fdr": max(c["ko_fdr"] for c in conds if c["arm"] == a)}

    # R0: reproduction of v2
    v2p = results_dir(cfg) / "stage3_v2_records.npz"
    if v2p.exists():
        v2 = np.load(v2p)
        v2s = cfg["stage3_v2"]
        v2_amps = v2s["signal_amplitudes"]
        fk = np.ix_([v2s["functional_forms"].index(f) for f in forms], [v2s["signal_sizes"].index(k) for k in ks])
        r0 = {}
        for a in GAUSS_ARMS:
            if a not in arms:
                continue
            dmax, same, tot = 0.0, 0, 0
            for ia, amp in enumerate(amps):
                if amp not in v2_amps:
                    continue
                iv = v2_amps.index(amp)
                for m in ("ko_fdr", "ko_pow"):
                    new = M[m][a][ia][..., :R, :]
                    old = v2[f"{m}__{a}"][iv][fk][..., :R, :]
                    dmax = max(dmax, float(np.abs(new.mean(axis=2) - old.mean(axis=2)).max()))
                    same += int((new == old).sum())
                    tot += new.size
            r0[a] = {"max_abs_cell_mean_diff": dmax, "share_identical_replicate_values": same / max(tot, 1)}
        rules["R0"] = r0
    conv = {a: float(CONV[a].mean()) for a in arms}
    return conds, contrasts, mech, rules, conv


def fig_boundary(conds, amps, ks, arms, q_ref, rd: Path) -> None:
    forms = ("linear", "interaction")
    fig, axes = plt.subplots(2, len(ks) + 1, figsize=(4.0 * (len(ks) + 1), 7.0), sharex=True, squeeze=False)
    for i, form in enumerate(forms):
        for j, k in enumerate(ks):
            ax = axes[i, j]
            for a in arms:
                rows = sorted([c for c in conds if c["arm"] == a and c["form"] == form and c["k"] == k
                               and c["q"] == q_ref], key=lambda c: c["amplitude"])
                ax.errorbar([c["amplitude"] for c in rows], [c["ko_fdr"] for c in rows],
                            yerr=[1.96 * c["ko_fdr_se"] for c in rows], marker="o", ms=3, capsize=2,
                            color=ARM_COLOR[a], label=ARM_LABEL[a])
            ax.axhline(q_ref, color="k", ls="--", lw=1)
            ax.set_xscale("log"); ax.set_title(f"FDR — {form}, {k} real latents", fontsize=9)
            if j == 0: ax.set_ylabel(f"realised FDR (target q = {q_ref})")
            if i == 1: ax.set_xlabel("signal strength")
        ax = axes[i, -1]
        for a in arms:
            pw = [np.mean([c["ko_pow"] for c in conds if c["arm"] == a and c["form"] == form
                           and c["amplitude"] == amp and c["q"] == q_ref]) for amp in amps]
            ax.plot(amps, pw, marker="o", ms=3, color=ARM_COLOR[a])
        ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_title(f"power — {form} (mean over k)", fontsize=9)
        if i == 1: ax.set_xlabel("signal strength")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Stage 3: does FDR ever cross the target as the signal grows? (95% intervals)", y=1.0)
    fig.tight_layout()
    fig.savefig(rd / "fig19_stage3_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_mechanism(mech, amps, arms, edges, rd: Path) -> None:
    order = [a for a in ("real_mvr", "real_equi", "real_hurdle", "gauss_mvr") if a in arms]
    labels = [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    x = np.arange(len(labels))
    cmap = plt.get_cmap("viridis")
    rows = (("bin_frac_pos", "P(null beats its knockoff)\nlasso W (valid: 0.5)"),
            ("bin_frac_pos_marginal", "P(null beats its knockoff)\nmarginal W (valid: 0.5)"),
            ("bin_false_disc_per_null_q0.10", "false discoveries per null\n(q = 0.10)"))
    fig, axes = plt.subplots(len(rows), len(order), figsize=(4.0 * len(order), 3.3 * len(rows)),
                             sharey="row", squeeze=False)
    for j, a in enumerate(order):
        for ia, amp in enumerate(amps):
            col = cmap(ia / max(1, len(amps) - 1))
            b = mech[a][str(amp)]
            for r, (key, _) in enumerate(rows):
                if key == "bin_false_disc_per_null_q0.10":
                    axes[r, j].plot(x, b[key], marker="o", ms=3, color=col)
                else:
                    m = np.array([v[0] for v in b[key]], dtype=float)
                    s = np.array([v[1] for v in b[key]], dtype=float)
                    axes[r, j].errorbar(x, m, yerr=1.96 * s, marker="o", ms=3, capsize=2, color=col,
                                        label=f"strength {amp:g}")
        for r in (0, 1):
            axes[r, j].axhline(0.5, color="k", ls="--", lw=1)
            axes[r, j].set_ylim(0, 1)
        axes[2, j].set_yscale("symlog", linthresh=1e-4)
        axes[0, j].set_title(ARM_LABEL[a], fontsize=9)
        for r in range(len(rows)):
            axes[r, j].set_xticks(x); axes[r, j].set_xticklabels(labels, fontsize=7, rotation=30)
        axes[-1, j].set_xlabel("null latent's max |corr| with the real latents")
    for r, (_, lab) in enumerate(rows):
        axes[r, 0].set_ylabel(lab, fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Stage 3: which null latents beat their knockoffs, and when? (95% intervals)", y=1.0)
    fig.tight_layout()
    fig.savefig(rd / "fig20_stage3_mechanism.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main_stress(args, cfg, d, X_all, device, rd: Path) -> None:
    sec = cfg["stage3_stress"]
    arms = list(sec["arms"])
    t0 = time.time()
    print("\n=== data (the stage3_v2 datasets) and the hurdle sampler ===")
    B = build_stress(cfg, sec, X_all, device)
    del X_all
    exch = None
    if not args.skip_exchangeability:
        print("\n=== exchangeability swap tests at p = 512 ===")
        exch = stress_exchangeability(cfg, sec, B, arms)
        B["scip"].warm = {}            # the grid's hurdle draws do not depend on whether this block ran
        for n, auc in exch["full_swap_auc"].items():
            a = exch["arms"][n]
            print(f"  {n:12s} full-swap AUC {auc:.3f}  corr(X,Xk) {a['mean_corr_X_Xk']:.3f}"
                  + (f"  zero-mass err {a['zero_mass_error_max']:.4f}  draw {a['draw_seconds']:.0f}s"
                     if "zero_mass_error_max" in a else ""), flush=True)

    if args.stage == "diagnose":
        amp, form, k = sec["diagnose"]["pilot_cell"]
        R = args.limit_reps or sec["diagnose"]["pilot_replicates"]
        print(f"\n=== pilot: amplitude {amp}, {form}, k = {k}, {R} replicate(s), all arms ===")
        t = time.time()
        M, MECH, CONV, hinfo = run_stress(cfg, sec, B, device, R, [amp], [form], [k], arms)
        secs = time.time() - t
        conds, contrasts, mech, rules, conv = aggregate_stress(cfg, sec, M, MECH, CONV, [amp], [form], [k], arms, R)
        iq = sec["nominal_fdr_targets"].index(0.10)
        for a in arms:
            print(f"  {a:12s} FDR {np.nanmean(M['ko_fdr'][a][0, 0, 0, :, iq]):.3f}  power "
                  f"{np.nanmean(M['ko_pow'][a][0, 0, 0, :, iq]):.3f}  null P(W>0) lasso "
                  f"{mech[a][str(amp)]['pooled_frac_pos'][0]:.3f} (n={sum(mech[a][str(amp)]['bin_n_nonzero_W']):.0f}) "
                  f"marginal {mech[a][str(amp)]['pooled_frac_pos_marginal'][0]:.3f}  converged {conv[a]:.0%}")
        print(f"  R0 vs v2 records: {json.dumps(rules.get('R0'), default=float)}")
        n_cells = len(sec["signal_amplitudes"]) * len(sec["functional_forms"]) * len(sec["signal_sizes"])
        draw = np.mean([h["draw_seconds"] for h in hinfo]) if hinfo else 0.0
        per_cell = (secs - draw * R) / R
        est = sec["replicates"] * (draw + n_cells * per_cell) / 60
        print(f"  hurdle draw {draw:.0f}s, {per_cell:.1f}s per cell (all arms) -> projected full run ~{est:.0f} min")
        (rd / "stage3_stress_diagnose.json").write_text(json.dumps(
            {"info": B["info"], "exchangeability": exch, "pilot_rules": rules, "pilot_mechanism": mech,
             "hurdle_info": hinfo, "projected_full_minutes": est, "minutes": (time.time() - t0) / 60},
            indent=2, default=float))
        print(f"\nwrote {rd}/stage3_stress_diagnose.json ({(time.time() - t0) / 60:.1f} min)")
        return

    amps, forms, ks = sec["signal_amplitudes"], sec["functional_forms"], sec["signal_sizes"]
    R = args.limit_reps or sec["replicates"]
    print(f"\n=== full: {len(arms)} arms x {len(amps)} amplitudes x {len(forms)} forms x {len(ks)} k x {R} reps ===")
    ckpt = rd / "stage3_stress_checkpoint.json"
    M, MECH, CONV, hinfo = run_stress(cfg, sec, B, device, R, amps, forms, ks, arms, checkpoint=ckpt)
    conds, contrasts, mech, rules, conv = aggregate_stress(cfg, sec, M, MECH, CONV, amps, forms, ks, arms, R)
    print("\n=== results ===")
    print(f"  R0 reproduction: {json.dumps(rules.get('R0'), default=float)}")
    for rid in ("R1", "R2", "R3", "R1_marginal", "R2_marginal", "R3_marginal"):
        print(f"  {rid}: {json.dumps(rules.get(rid), default=float)}")
    for a, v in rules["R4"].items():
        print(f"  R4 [{a:12s}] first inflated amplitude {v['first_inflated_amplitude']}  max FDR {v['max_fdr']:.3f}  "
              f"inflated cells {len(v['inflated_cells'])}")
    for a in arms:
        print(f"  {a:12s} null P(W>0) lasso / marginal by strength: " + "  ".join(
            f"{amp:g}:{mech[a][str(amp)]['pooled_frac_pos'][0]:.3f}/{mech[a][str(amp)]['pooled_frac_pos_marginal'][0]:.3f}"
            for amp in amps))
    out = {"config_hash": str(d["config_hash"]), "master_seed": cfg["master_seed"], "replicates": R,
           "amplitudes": amps, "forms": forms, "signal_sizes": ks, "arms": arms,
           "nominal_fdr_targets": sec["nominal_fdr_targets"], "link_bins": sec["link_bins"], "info": B["info"],
           "exchangeability": exch, "hurdle_info": hinfo, "convergence": conv, "rules": rules,
           "mechanism": mech, "conditions": conds, "contrasts": contrasts,
           "minutes": round((time.time() - t0) / 60, 1)}
    (rd / "stage3_stress.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(rd / "stage3_stress_records.npz", **{f"{m}__{a}": M[m][a] for m in METRICS for a in arms},
             **{f"mech__{a}": MECH[a] for a in arms})
    fig_boundary(conds, amps, ks, arms, 0.10, rd)
    fig_mechanism(mech, amps, arms, sec["link_bins"], rd)
    drop_checkpoint(ckpt)
    print(f"\nwrote {rd}/stage3_stress.json, stage3_stress_records.npz, fig19, fig20 ({out['minutes']} min)")


# ---------------------------------------------------------------------------
# dims: latents x rows per latent x S x real/Gaussian
# ---------------------------------------------------------------------------

class GaussKnockoffGPU:
    """Second-order Gaussian knockoffs, the law of knockpy's GaussianSampler, drawn on the GPU:
    Xk = mu + (X - mu)(I - Sigma^-1 S) + E V^(1/2),  V = 2S - S Sigma^-1 S."""

    def __init__(self, Z_t: torch.Tensor, Sigma: np.ndarray, S: np.ndarray):
        p = Sigma.shape[0]
        s = np.diag(S)
        SinvS = np.linalg.solve(Sigma, np.diag(s))                # Sigma^-1 S
        A = np.eye(p) - SinvS
        V = 2 * np.diag(s) - np.diag(s) @ SinvS
        V = 0.5 * (V + V.T)
        ev, U = np.linalg.eigh(V)
        self.min_eig_V = float(ev.min())
        L = U * np.sqrt(np.clip(ev, 0.0, None))
        dev = Z_t.device
        self.Z = Z_t
        self.mu = Z_t.mean(0, keepdim=True)
        self.A = torch.from_numpy(A.astype(np.float32)).to(dev)
        self.Lt = torch.from_numpy(L.T.astype(np.float32)).to(dev)

    def sample(self, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=self.Z.device)
        gen.manual_seed(int(seed))
        E = torch.randn(self.Z.shape, generator=gen, device=self.Z.device)
        return self.mu + (self.Z - self.mu) @ self.A + E @ self.Lt


def corr_gpu(A: torch.Tensor, B: torch.Tensor) -> float:
    ac, bc = A - A.mean(0), B - B.mean(0)
    den = torch.sqrt((ac ** 2).sum(0) * (bc ** 2).sum(0))
    return float(torch.where(den > 0, (ac * bc).sum(0) / den.clamp(min=1e-30), torch.zeros_like(den)).mean())


def covariance_for(Z: np.ndarray, rule: str, cfg: dict):
    """rule: ledoit_wolf, sample, or v2_rule (sample unless ill-conditioned, then Ledoit-Wolf)."""
    if rule in ("ledoit_wolf", "sample"):
        return estimate_cov(Z, rule), rule
    S = estimate_cov(Z, "sample")
    ev = np.linalg.eigvalsh(S)
    if ev[-1] / max(ev[0], 1e-300) > float(cfg["covariance"]["ill_conditioned_cond_number"]):
        return estimate_cov(Z, "ledoit_wolf"), "ledoit_wolf"
    return S, "sample"


def s_matrix(Sigma: np.ndarray, method: str, sec: dict):
    from knockpy import smatrix
    t = time.time()
    if method == "mvr":
        S = smatrix.compute_smatrix(Sigma, method="mvr", how_approx="blockdiag", max_block=sec["mvr_max_block"])
    else:
        S = smatrix.compute_smatrix(Sigma, method=method)
    S = np.asarray(S)
    return S, time.time() - t, float(np.linalg.eigvalsh(2 * Sigma - S).min())


def build_dims(cfg: dict, sec: dict, X_all: np.ndarray, device) -> dict:
    n_all, p_all = X_all.shape
    v2 = cfg["stage3_v2"]
    cols = {512: np.sort(rng_for(cfg, "s3v2_data").choice(p_all, v2["p"], replace=False)),
            2048: np.arange(p_all)}
    if p_all != 2048 or v2["p"] != 512:
        raise SystemExit(f"dims expects 2048 cached latents and stage3_v2.p = 512 (got {p_all}, {v2['p']})")
    rng = rng_for(cfg, "s3d_data")
    hi512 = rng.choice(n_all, sec["rows"][512][1], replace=False)
    rows = {(512, 0): np.sort(hi512[: sec["rows"][512][0]]), (512, 1): np.sort(hi512),
            (2048, 0): np.sort(rng_for(cfg, "s3p2048_data").choice(n_all, sec["rows"][2048][0], replace=False)),
            (2048, 1): np.arange(n_all)}
    D, info = {}, {}
    for p in sec["p_levels"]:
        for lvl in (0, 1):
            r = rows[(p, lvl)]
            t = time.time()
            X = X_all[np.ix_(r, cols[p])].astype(np.float64)
            p0 = float(np.median((X == 0).mean(0)))
            Z, _, _ = standardise(X)
            del X
            Sigma, est = covariance_for(Z, sec["covariance"][p], cfg)
            Lc = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
            Zg, _, _ = standardise(rng.standard_normal((len(r), p)) @ Lc.T)
            del Lc
            Sigma_g, _ = covariance_for(Zg, est, cfg)
            for data, Zd, Sig in (("real", Z, Sigma), ("gauss", Zg, Sigma_g)):
                if data not in sec["data"]:
                    continue
                key = f"p{p}_{'lo' if lvl == 0 else 'hi'}_{data}"
                Z_t = torch.from_numpy(Zd.astype(np.float32)).to(device)
                D[key] = {"p": p, "lvl": lvl, "data": data, "n": len(r), "Z_np": Zd.astype(np.float32),
                          "Z_t": Z_t, "samplers": {}, "Sigma": Sig, "S": {}}
                info[key] = {"p": p, "n": len(r), "n_over_p": len(r) / p, "covariance": est,
                             "median_zero_mass": p0 if data == "real" else 0.0, "S": {}}
                for sm in sec["s_methods"]:
                    S, secs, mineig = s_matrix(Sig, sm, sec)
                    smp = GaussKnockoffGPU(Z_t, Sig, S)
                    D[key]["samplers"][sm] = smp
                    D[key]["S"][sm] = S
                    Zk = smp.sample(12345)
                    info[key]["S"][sm] = {"mean_s": float(np.diag(S).mean()), "seconds": secs,
                                          "min_eig_2Sigma_minus_S": mineig, "min_eig_V": smp.min_eig_V,
                                          "mean_corr_X_Xk": corr_gpu(Z_t, Zk)}
                    del Zk
                    print(f"  [{key:14s} {sm:14s}] n={len(r)} n/p={len(r) / p:.2f} cov={est}  mean s="
                          f"{np.diag(S).mean():.4f}  corr(X,Xk)={info[key]['S'][sm]['mean_corr_X_Xk']:.3f}  "
                          f"S {secs:.0f}s  min eig(2Sigma-S)={mineig:.2e}", flush=True)
            del Z, Zg
            print(f"  built p={p} n={len(r)} in {time.time() - t:.0f}s", flush=True)
    return {"D": D, "info": info, "rows": rows, "cols": cols}


def sampler_check(cfg, sec, Bd: dict) -> dict:
    """GPU sampler vs knockpy's GaussianSampler on one dataset: same law, so the knockoff-variable
    correlation and the error against the target cross-covariance should agree up to noise."""
    from knockpy.knockoffs import GaussianSampler
    key = "p512_lo_real"
    d = Bd["D"][key]
    Z = d["Z_np"].astype(np.float64)
    out = {}
    Sig = d["Sigma"]
    for sm, smp in d["samplers"].items():
        res = {}
        S = d["S"][sm]
        Zk_gpu = smp.sample(777).double().cpu().numpy()
        np.random.seed(777)
        Zk_kp = GaussianSampler(Z, mu=Z.mean(0), Sigma=Sig, S=S).sample_knockoffs()
        n = Z.shape[0]
        target_cross = Sig - S
        for tag, Zk in (("gpu", Zk_gpu), ("knockpy", Zk_kp)):
            Zc, Kc = Z - Z.mean(0), Zk - Zk.mean(0)
            cross = Zc.T @ Kc / (n - 1)
            kk = Kc.T @ Kc / (n - 1)
            res[tag] = {"mean_corr_X_Xk": mean_corr(Z, Zk),
                        "max_abs_cross_cov_error": float(np.abs(cross - target_cross).max()),
                        "max_abs_knockoff_cov_error": float(np.abs(kk - Sig).max())}
        out[sm] = res
        print(f"  sampler check [{key} {sm}]: " + "  ".join(
            f"{t}: corr {v['mean_corr_X_Xk']:.3f} cross-cov err {v['max_abs_cross_cov_error']:.3f} "
            f"cov err {v['max_abs_knockoff_cov_error']:.3f}" for t, v in res.items()), flush=True)
    return out


def run_dims(cfg: dict, sec: dict, Bd: dict, device, R: int, amps, forms, ks, checkpoint=None):
    qs, lam = sec["nominal_fdr_targets"], sec["lasso"]["lambda"]
    D = Bd["D"]
    names = [f"{k}_{sm}" for k in D for sm in sec["s_methods"]]
    shape = (len(amps), len(forms), len(ks), R)
    M = {m: {n: np.full(shape + (len(qs),), np.nan, dtype=np.float32) for n in names} for m in METRICS}
    CONV = {n: np.zeros(shape, dtype=bool) for n in names}
    key = {"shape": list(shape), "names": names}
    start = 0
    if checkpoint is not None:
        ck = load_checkpoint(checkpoint, key)
        if ck:
            meta, z = ck
            for n in names:
                for m in METRICS:
                    M[m][n] = z[f"{m}__{n}"]
                CONV[n] = z[f"conv__{n}"]
            start = meta["reps_done"]
            print(f"  resuming from checkpoint: {start}/{R} replicates done", flush=True)
    t_fit = {p: 0.0 for p in sec["p_levels"]}
    n_fit = {p: 0 for p in sec["p_levels"]}
    t0 = time.time()
    for rep in range(start, R):
        for ip, p in enumerate(sec["p_levels"]):
            no_r, no_thr = np.zeros(p), {q: float("inf") for q in qs}
            for ia, amp in enumerate(amps):
                for jf, form in enumerate(forms):
                    for kk, k in enumerate(ks):
                        kseed = int(cell_rng(cfg, "s3d_knockoff", ip, ia, jf, kk, rep).integers(2**31))
                        for dk, d in D.items():
                            if d["p"] != p:
                                continue
                            rng = cell_rng(cfg, "s3d_planted", ip, ia, jf, kk, rep)
                            S_idx = np.sort(rng.choice(p, size=k, replace=False))
                            y_np, _ = generate_planted_labels(d["Z_np"], S_idx, form, amp, rng)
                            y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
                            truth = set(S_idx.tolist())
                            for sm, smp in d["samplers"].items():
                                t = time.time()
                                Phi = torch.cat([d["Z_t"], smp.sample(kseed)], dim=1)
                                w, cv, _, _ = fit_lasso_checked(Phi, y_t, lam, sec["lasso"]["max_iter"], sec["lasso"]["tol"])
                                del Phi
                                w_np = w.cpu().numpy()
                                sc = score_fit(np.abs(w_np[:p]) - np.abs(w_np[p:]), no_r, no_thr, qs, truth)
                                nm = f"{dk}_{sm}"
                                for m in METRICS:
                                    M[m][nm][ia, jf, kk, rep] = np.nan if m.startswith("wy") else sc[m]
                                CONV[nm][ia, jf, kk, rep] = cv
                                t_fit[p] += time.time() - t
                                n_fit[p] += 1
        if checkpoint is not None:
            arrays = {f"{m}__{n}": M[m][n] for m in METRICS for n in names}
            arrays.update({f"conv__{n}": CONV[n] for n in names})
            save_checkpoint(checkpoint, arrays, {**key, "reps_done": rep + 1})
        el = time.time() - t0
        done = rep + 1 - start
        print(f"  rep {rep + 1:>2}/{R}  elapsed {el / 60:.1f}m  ETA {el / done * (R - rep - 1) / 60:.1f}m", flush=True)
    timing = {"seconds_per_fit": {str(p): t_fit[p] / max(1, n_fit[p]) for p in t_fit}}
    return M, CONV, names, timing


def aggregate_dims(sec, M, CONV, names, amps, forms, ks, Bd):
    qs = sec["nominal_fdr_targets"]
    split = {n: dict(zip(("p", "lvl", "data", "s"), (int(n.split("_")[0][1:]), n.split("_")[1],
                                                       n.split("_")[2], n.split("_", 3)[3]))) for n in names}
    conds = cell_rows({m: M[m] for m in ("ko_fdr", "ko_pow", "ko_nd", "k0_fdr", "k0_pow", "k0_nd")},
                      names, amps, forms, ks, qs, extra=lambda d: split[d])
    summary = {}
    for n in names:
        cs = [c for c in conds if c["arm"] == n]
        inf = Bd["info"][n.rsplit("_", 1)[0]]
        summary[n] = {**split[n], "n": inf["n"], "n_over_p": inf["n_over_p"],
                      "mean_corr_X_Xk": inf["S"][split[n]["s"]]["mean_corr_X_Xk"],
                      "mean_power": float(np.mean([c["ko_pow"] for c in cs])),
                      "mean_power_q0.10": float(np.mean([c["ko_pow"] for c in cs if c["q"] == 0.10])),
                      "mean_fdr": float(np.mean([c["ko_fdr"] for c in cs])),
                      "max_fdr": float(max(c["ko_fdr"] for c in cs)),
                      "n_fdr_inflated": int(sum(c["ko_inflated"] for c in cs)),
                      "fraction_converged": float(CONV[n].mean())}

    def nm(p, lvl, data, s):
        return f"p{p}_{lvl}_{data}_{s}"

    contrasts, effects = [], {}
    defs = []
    for p in sec["p_levels"]:
        for data in sec["data"]:
            for s in sec["s_methods"]:
                defs.append((f"rows|p{p}|{data}|{s}", nm(p, "hi", data, s), nm(p, "lo", data, s), False))
            for lvl in ("lo", "hi"):
                defs.append((f"S|p{p}|{lvl}|{data}", nm(p, lvl, data, "mvr"), nm(p, lvl, data, "equicorrelated"), True))
        for lvl in ("lo", "hi"):
            for s in sec["s_methods"]:
                if "real" in sec["data"] and "gauss" in sec["data"]:
                    defs.append((f"atom|p{p}|{lvl}|{s}", nm(p, lvl, "gauss", s), nm(p, lvl, "real", s), False))
    for label, a, b, is_paired in defs:
        if a not in M["ko_pow"] or b not in M["ko_pow"]:
            continue
        rows = []
        for ia, amp in enumerate(amps):
            for jf, form in enumerate(forms):
                for kk, k in enumerate(ks):
                    for iq, q in enumerate(qs):
                        row = {"contrast": label, "a": a, "b": b, "paired": is_paired,
                               "amplitude": amp, "form": form, "k": k, "q": q}
                        for m in ("ko_pow", "ko_fdr"):
                            va, vb = M[m][a][ia, jf, kk, :, iq], M[m][b][ia, jf, kk, :, iq]
                            row[m + "_diff"], row[m + "_diff_se"] = paired(va, vb) if is_paired else unpaired(va, vb)
                        rows.append(row)
        contrasts += rows
        effects[label] = {"a": a, "b": b, "paired": is_paired,
                          "mean_power_diff": float(np.mean([r["ko_pow_diff"] for r in rows])),
                          "n_power_sig_pos": int(sum(r["ko_pow_diff"] > 1.96 * r["ko_pow_diff_se"] > 0 for r in rows)),
                          "n_power_sig_neg": int(sum(-r["ko_pow_diff"] > 1.96 * r["ko_pow_diff_se"] > 0 for r in rows)),
                          "mean_fdr_diff": float(np.mean([r["ko_fdr_diff"] for r in rows])),
                          "n_cells": len(rows)}

    def attribution(data):
        rows_e = [effects.get(f"rows|p2048|{data}|{s}") for s in sec["s_methods"]]
        s_e = [effects.get(f"S|p2048|{lvl}|{data}") for lvl in ("lo", "hi")]
        if None in rows_e or None in s_e:
            return None
        r_sig, s_sig = sum(e["n_power_sig_pos"] for e in rows_e), sum(e["n_power_sig_pos"] for e in s_e)
        r_mean, s_mean = np.mean([e["mean_power_diff"] for e in rows_e]), np.mean([e["mean_power_diff"] for e in s_e])
        if r_sig > s_sig and r_mean > s_mean:
            driver = "rows per latent"
        elif s_sig > r_sig and s_mean > r_mean:
            driver = "S matrix (near-copy knockoffs)"
        else:
            driver = "both"
        return {"rows_effect_sig_cells": r_sig, "rows_effect_mean": float(r_mean),
                "S_effect_sig_cells": s_sig, "S_effect_mean": float(s_mean), "main_driver": driver}

    rules = {"R5": {d: attribution(d) for d in sec["data"]}}
    return conds, contrasts, summary, effects, rules


def fig_dims(summary, sec, amps, conds, rd: Path) -> None:
    settings = [(p, lvl) for p in sec["p_levels"] for lvl in ("lo", "hi")]
    bars = [(data, s) for data in ("real", "gauss") for s in ("equicorrelated", "mvr") if data in sec["data"]]
    colors = {("real", "equicorrelated"): "#e07b39", ("real", "mvr"): "#c0392b",
              ("gauss", "equicorrelated"): "#8fb3de", ("gauss", "mvr"): "#3b6ea5"}
    fig, axes = plt.subplots(1, len(amps) + 1, figsize=(5.0 * (len(amps) + 1), 4.6))
    w = 0.8 / len(bars)
    x = np.arange(len(settings))
    xl = []
    for p, lvl in settings:
        nn = summary[f"p{p}_{lvl}_{bars[0][0]}_{bars[0][1]}"]["n_over_p"]
        xl.append(f"{p} latents\nn/p = {nn:.1f}")
    for ai, amp in enumerate(amps):
        ax = axes[ai]
        for bi, (data, s) in enumerate(bars):
            vals, errs = [], []
            for p, lvl in settings:
                cs = [c for c in conds if c["arm"] == f"p{p}_{lvl}_{data}_{s}" and c["amplitude"] == amp and c["q"] == 0.10]
                vals.append(np.mean([c["ko_pow"] for c in cs]))
                errs.append(1.96 * math.sqrt(sum(c["ko_pow_se"] ** 2 for c in cs)) / len(cs))
            ax.bar(x + (bi - (len(bars) - 1) / 2) * w, vals, w, yerr=errs, capsize=2, color=colors[(data, s)],
                   label=f"{'real latents' if data == 'real' else 'Gaussian control'}, {'MVR' if s == 'mvr' else 'equicorr.'} S")
        ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=8); ax.set_ylim(0, 1.02)
        ax.set_ylabel("power at q = 0.10 (mean over forms and k)")
        ax.set_title(f"signal strength {amp:g}")
    axes[0].legend(fontsize=7, loc="upper left")
    ax = axes[-1]
    for bi, (data, s) in enumerate(bars):
        ax.bar(x + (bi - (len(bars) - 1) / 2) * w,
               [summary[f"p{p}_{lvl}_{data}_{s}"]["mean_corr_X_Xk"] for p, lvl in settings], w, color=colors[(data, s)])
    ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_ylabel("mean corr(latent, its knockoff)"); ax.set_title("how close to a copy the knockoffs are")
    fig.suptitle("Stage 3: where does the power go? latents x rows per latent x S matrix x zero atom", y=1.02)
    fig.tight_layout()
    fig.savefig(rd / "fig21_stage3_dims.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main_dims(args, cfg, d, X_all, device, rd: Path) -> None:
    sec = cfg["stage3_dims"]
    t0 = time.time()
    print("\n=== datasets and S matrices ===")
    Bd = build_dims(cfg, sec, X_all, device)
    del X_all
    if device.type == "cuda":
        print(f"  GPU memory after build: {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)

    if args.stage == "diagnose":
        print("\n=== GPU sampler vs knockpy ===")
        chk = sampler_check(cfg, sec, Bd)
        amp, form, k = sec["diagnose"]["pilot_cell"]
        R = args.limit_reps or sec["diagnose"]["pilot_replicates"]
        print(f"\n=== pilot: amplitude {amp}, {form}, k = {k}, {R} replicate(s), all datasets ===")
        t = time.time()
        M, CONV, names, timing = run_dims(cfg, sec, Bd, device, R, [amp], [form], [k])
        secs = time.time() - t
        iq = sec["nominal_fdr_targets"].index(0.10)
        pilot = {}
        for n in names:
            pilot[n] = {"power_q0.10": float(np.nanmean(M["ko_pow"][n][0, 0, 0, :, iq])),
                        "fdr_q0.10": float(np.nanmean(M["ko_fdr"][n][0, 0, 0, :, iq])),
                        "converged": float(CONV[n].mean())}
            print(f"  {n:30s} power {pilot[n]['power_q0.10']:.3f}  FDR {pilot[n]['fdr_q0.10']:.3f}  "
                  f"converged {pilot[n]['converged']:.0%}")
        n_cells = len(sec["signal_amplitudes"]) * len(sec["functional_forms"]) * len(sec["signal_sizes"])
        est = sec["replicates"] * n_cells * secs / R / 60
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        print(f"  timing {json.dumps(timing)}  peak GPU {peak:.2f} GB -> projected full run ~{est:.0f} min")
        (rd / "stage3_dims_diagnose.json").write_text(json.dumps(
            {"info": Bd["info"], "sampler_check": chk, "pilot": pilot, "timing": timing, "peak_gpu_gb": peak,
             "projected_full_minutes": est, "minutes": (time.time() - t0) / 60}, indent=2, default=float))
        print(f"\nwrote {rd}/stage3_dims_diagnose.json ({(time.time() - t0) / 60:.1f} min)")
        return

    amps, forms, ks = sec["signal_amplitudes"], sec["functional_forms"], sec["signal_sizes"]
    R = args.limit_reps or sec["replicates"]
    print(f"\n=== full: {len(Bd['D']) * len(sec['s_methods'])} datasets x {len(amps)} amplitudes x {len(forms)} forms "
          f"x {len(ks)} k x {R} reps ===")
    ckpt = rd / "stage3_dims_checkpoint.json"
    M, CONV, names, timing = run_dims(cfg, sec, Bd, device, R, amps, forms, ks, checkpoint=ckpt)
    conds, contrasts, summary, effects, rules = aggregate_dims(sec, M, CONV, names, amps, forms, ks, Bd)
    print("\n=== results ===")
    for n, s in summary.items():
        print(f"  {n:30s} n/p {s['n_over_p']:5.1f}  corr {s['mean_corr_X_Xk']:.3f}  power(q=.1) "
              f"{s['mean_power_q0.10']:.3f}  max FDR {s['max_fdr']:.3f}  inflated {s['n_fdr_inflated']}")
    for lab, e in effects.items():
        print(f"  {lab:28s} power {e['mean_power_diff']:+.3f}  sig +{e['n_power_sig_pos']}/-{e['n_power_sig_neg']} "
              f"of {e['n_cells']}")
    print(f"  R5: {json.dumps(rules['R5'], default=float)}")
    out = {"config_hash": str(d["config_hash"]), "master_seed": cfg["master_seed"], "replicates": R,
           "amplitudes": amps, "forms": forms, "signal_sizes": ks, "nominal_fdr_targets": sec["nominal_fdr_targets"],
           "info": Bd["info"], "timing": timing, "summary": summary, "effects": effects, "rules": rules,
           "conditions": conds, "contrasts": contrasts, "minutes": round((time.time() - t0) / 60, 1)}
    (rd / "stage3_dims.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(rd / "stage3_dims_records.npz", **{f"{m}__{n}": M[m][n] for m in METRICS for n in names if not m.startswith("wy")})
    fig_dims(summary, sec, amps, conds, rd)
    drop_checkpoint(ckpt)
    print(f"\nwrote {rd}/stage3_dims.json, stage3_dims_records.npz, fig21 ({out['minutes']} min)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 follow-ups: mechanism, boundary, dimension")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--experiment", required=True, choices=["stress", "dims"])
    ap.add_argument("--stage", default="diagnose", choices=["diagnose", "full"])
    ap.add_argument("--limit-reps", type=int, default=None)
    ap.add_argument("--skip-exchangeability", action="store_true", help="stress only")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(args.config)
    rd = results_dir(cfg)
    d, X_all = load_cache(args, cfg)
    print(f"device: {device} | cache {d['config_hash']} | experiment: {args.experiment} | stage: {args.stage}")
    (main_stress if args.experiment == "stress" else main_dims)(args, cfg, d, X_all, device, rd)


if __name__ == "__main__":
    main()
