#!/usr/bin/env python3
"""v3 (2026-11-24): "lateral boundaries" with footnote c (inlet and outlet also moved); tangential share
shown as n/a for the lift-only model (n_z = 1, where it is 1 by construction); stops if an ensemble
directory has no networks instead of skipping the row; outputs *_v3.  Numbers unchanged.
v2 (2026-11-22): \\Rey macro, "baseline" instead of "production", Re1000 baseline row shown as no limit cycle, p<1e-3 formatting. Build the LaTeX rows of the sensitivity table (new Sec. III, tab:sensitivity) from
the per-network diagnostics of each ensemble (recomputed with diag.py), and the
Mann-Whitney p-value of |omega_d| against the reference ensemble of each block.
Rows whose directory has no finished networks are skipped.  Writes
sensitivity_table_rows.tex and sensitivity_table.json."""
import os, sys, glob, json, numpy as np
from scipy.stats import mannwhitneyu
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from diag import diagnostics
from paths import REV as W, ENS, OUT_ENS
# (block, label, directory, reference directory or None)
ROWS = [
 ("Solver mesh ($\\Rey=400$, $n_z=4$)", "baseline mesh$^{a}$", f"{W}/gridconv_Re400_coarse", None),
 ("", "refined mesh ($\\sqrt2$)", f"{W}/gridconv_Re400_fine", f"{W}/gridconv_Re400_coarse"),
 ("Domain ($\\Rey=400$)", "baseline, $9.3D$, $\\pm5D$", f"{ENS}/S7b_base_v4", None),
 ("", "outlet at $25D$", f"{ENS}/S7_long25D", f"{ENS}/S7b_base_v4"),
 ("", "lateral boundaries at $\\pm10D^{c}$", f"{ENS}/S11_wide10D", f"{ENS}/S7b_base_v4"),
 ("Autoencoder data", "all snapshots (baseline)", f"{ENS}/S7b_base_v4", None),
 ("", "first $60\\%$ only", f"{ENS}/S6_causalCAE", f"{ENS}/S7b_base_v4"),
 ("Compression", "CAE (fixed), LSTM seeds", f"{ENS}/S1_fixedCAE", None),
 ("", "POD, LSTM seeds", f"{ENS}/S3_pod", f"{ENS}/S1_fixedCAE"),
 ("Initialisation", "CAE fixed, LSTM varied", f"{ENS}/S1_fixedCAE", None),
 ("", "LSTM fixed, CAE varied", f"{ENS}/S2_fixedLSTMseed", f"{ENS}/S1_fixedCAE"),
 ("Input window $\\tau$", "$\\tau=5$", f"{ENS}/S4_tau5", f"{ENS}/S1_fixedCAE"),
 ("", "$\\tau=10$", f"{ENS}/S4_tau10", f"{ENS}/S1_fixedCAE"),
 ("", "$\\tau=20$ (baseline)", f"{ENS}/S1_fixedCAE", None),
 ("", "$\\tau=40$", f"{ENS}/S4_tau40", f"{ENS}/S1_fixedCAE"),
 ("", "$\\tau=80$", f"{ENS}/S4_tau80", f"{ENS}/S1_fixedCAE"),
 ("Training loss", "one-step (baseline)", f"{ENS}/S1_fixedCAE", None),
 ("", "unrolled, $K=5$", f"{ENS}/S5_ms5", f"{ENS}/S1_fixedCAE"),
 ("", "unrolled, $K=20$", f"{ENS}/S5_ms20", f"{ENS}/S1_fixedCAE"),
 ("Model input", "velocity and pressure fields", f"{ENS}/S7b_base_v4", None),
 ("", "lift coefficient only", f"{ENS}/S9_lift", f"{ENS}/S7b_base_v4"),
 ("Solver mesh ($\\Rey=800$, $n_z=8$)", "baseline mesh", f"{ENS}/S12a_Re800_coarse_L8", None),
 ("", "refined mesh", f"{ENS}/S12b_Re800_fine_L8", f"{ENS}/S12a_Re800_coarse_L8"),
 ("Solver mesh ($\\Rey=1000$, $n_z=4$)", "baseline mesh", f"{ENS}/S13a_Re1000_coarse", None),
 ("", "refined mesh", f"{ENS}/S13b_Re1000_fine", f"{ENS}/S13a_Re1000_coarse"),
]
cache = {}
def load(D):
    if D in cache: return cache[D]
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "seed_*_art.npz"))):
        a = np.load(f); pred = a["pred"].astype(float)
        if "true" in a: true = a["true"].astype(float)
        else:
            ln = (a["latents"] - a["lmean"]) / a["lstd"]; ts = int(a["test_start"]); true = ln[ts:ts + len(pred)].astype(float)
        rows.append(diagnostics(pred, true, dt=float(a["dt"]) if "dt" in a else 0.03))
    cache[D] = rows; return rows
out, tex = [], []
for blk, lab, D, ref in ROWS:
    r = load(D)
    if not r: sys.exit(f"sensitivity_table_v3: no networks in {D}")
    w = np.array([x["omega_mean"] for x in r]); aw = np.abs(w)
    p = mannwhitneyu(aw, np.abs([x["omega_mean"] for x in load(ref)])).pvalue if ref and load(ref) else None
    s = dict(block=blk, label=lab, dir=os.path.basename(D), n=len(r), phase=float(np.median([x["share_phase"] for x in r])),
             tan=float(np.median([x["share_tan"] for x in r])), w_med=float(np.median(aw)), w_cv=float(aw.std(ddof=1) / aw.mean()) if len(r) > 1 else 0.0,
             slow=int((w < 0).sum()), fast=int((w > 0).sum()), gate=int(sum(x["r2_cal"] > 0.3 for x in r)),
             I_med=float(np.median([x["I_phase"] for x in r])), err=float(np.median([x["err_ext"] for x in r])),
             gperp=float(np.median([x["growth_perp"] for x in r])), p=p)
    out.append(s)
    ps = "--" if p is None else ("$<10^{-3}$" if p < 1e-3 else f"{p:.2f}")
    if s['phase'] > 1.0:   # prediction has left the limit cycle: phase/amplitude split undefined
        tex.append(f"{blk} & {lab}$^{{b}}$ & {s['n']} & \\multicolumn{{7}}{{c}}{{prediction does not stay on a limit cycle}} & {ps} \\\\")
    else:
        tex.append(f"{blk} & {lab} & {s['n']} & {100*s['phase']:.0f} & {'n/a' if 'lift' in lab else f"{100*s['tan']:.0f}"} & {1e3*s['w_med']:.2f} & {s['w_cv']:.2f} & "
                   f"{s['slow']}/{s['fast']} & {s['gate']} & {s['I_med']:.0f} & {ps} \\\\")
open(os.path.join(OUT_ENS, "sensitivity_table_rows_v3.tex"), "w").write("\n".join(tex) + "\n")
json.dump(out, open(os.path.join(OUT_ENS, "sensitivity_table_v3.json"), "w"), indent=1)
print("\n".join(tex))
