#!/usr/bin/env python3
"""Aggregate per-network diagnostics for each experiment directory (v3 or v4 format).
Diagnostics are recomputed from pred/true with diag.py, so v3 and v4 runs are scored
identically.  Prints mean +/- sd, median, and sign counts; writes aggregate_B_<dirs>.json to $PDROM_OUT_DIR.
v2 (2026-11-24): proper argument parsing (--help no longer writes a file); stops with an error if a
directory contains no networks instead of skipping it."""
import os, sys, glob, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag import diagnostics
_ap = argparse.ArgumentParser(description="per-ensemble summaries"); _ap.add_argument("dirs", nargs="+")
dirs = _ap.parse_args().dirs
_empty = [D for D in dirs if not glob.glob(os.path.join(D, "seed_*_art.npz"))]
if _empty: sys.exit("aggregate_B_v2: no networks (seed_*_art.npz) in: " + ", ".join(_empty))
out = {}
for D in dirs:
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "seed_*_art.npz"))):
        a = np.load(f); pred = a["pred"].astype(float)
        if "true" in a: true = a["true"].astype(float)
        else:
            ln = (a["latents"] - a["lmean"]) / a["lstd"]; ts = int(a["test_start"]); true = ln[ts:ts + len(pred)].astype(float)
        dt = float(a["dt"]) if "dt" in a else 0.03
        d = diagnostics(pred, true, dt=dt)
        meta = f.replace("_art.npz", ".npy")
        if os.path.exists(meta):
            m = np.load(meta, allow_pickle=True).item()
            for k in ("lstm_val", "cae_val", "pod_energy"):
                if k in m: d[k] = m[k]
        rows.append(d)
    keys = ["share_phase", "share_amp", "share_tan", "growth_tan", "growth_perp", "r2_cal", "omega_mean",
            "I_phase", "err_ext", "clock_bias", "amp_ratio", "lstm_val", "cae_val", "pod_energy"]
    summ = {"n": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and r[k] is not None], float)
        v = v[np.isfinite(v)]
        if len(v): summ[k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, median=float(np.median(v)),
                                  min=float(v.min()), max=float(v.max()))
    om = np.array([r["omega_mean"] for r in rows])
    summ["n_slow"] = int((om < 0).sum()); summ["n_fast"] = int((om > 0).sum())
    summ["n_gate_pass"] = int(sum(r["r2_cal"] > 0.3 for r in rows))
    summ["I_phase_gated_mean"] = float(np.mean([r["I_phase"] for r in rows if r["r2_cal"] > 0.3])) if summ["n_gate_pass"] else None
    summ["abs_omega_mean"] = float(np.abs(om).mean()); summ["abs_omega_cv"] = float(np.abs(om).std(ddof=1) / np.abs(om).mean()) if len(om) > 1 else None
    out[os.path.basename(os.path.normpath(D))] = dict(summary=summ, per_seed=rows)
    s = summ; g = lambda k: f"{s[k]['mean']:.3f}±{s[k]['sd']:.3f}" if k in s else "-"
    print(f"{os.path.basename(os.path.normpath(D)):24s} n={s['n']:2d} phase={g('share_phase')} tan={g('share_tan')} "
          f"|w|={s['abs_omega_mean']:.2e} (CV {s['abs_omega_cv'] or 0:.2f}) slow/fast={s['n_slow']}/{s['n_fast']} "
          f"r2cal={g('r2_cal')} pass={s['n_gate_pass']} I={g('I_phase')} err={g('err_ext')} gperp={g('growth_perp')} "
          f"val={s['lstm_val']['mean']:.2e}" if 'lstm_val' in s else "", flush=True)
from paths import OUT_ENS
json.dump(out, open(os.path.join(OUT_ENS, "aggregate_B_" +
          "_".join(os.path.basename(os.path.normpath(d)) for d in dirs)[:80] + ".json"), "w"), indent=1)
