#!/usr/bin/env python3
"""
audit_numbers.py
================
One table holding every quantity the paper reports, with its provenance, and an
automatic check for sources that disagree.

WHY
---
Five transcription-class errors have been found by hand so far, and every one of
them understated the result:

  * a sinusoid entry (47.6%) quoted where a phase entry (71.6%) belonged
  * bar-chart values plotting the sinusoid while labelled phase correction
  * a per-file minimum (+16.1%) reported as the minimum over all records (-40.3%)
  * a field-space value differing between figure (14.25%) and tables (15.5, 15.7)
  * a phasecorr file silently overwritten at a different calibration window, so
    that the figure and the table described different experiments

The last one is the reason a list of values is not enough. Two files can hold the
same quantity for the same case and still disagree legitimately, because they were
produced with different settings. Every number therefore needs its provenance
recorded beside it: case, latent dimension, calibration window, timestep, and the
file it came from.

WHAT THIS DOES
--------------
Walks the data directory, reads every analysis output it recognises, and writes a
single CSV with one row per (quantity, case, source). It then groups rows by
quantity and case and reports any group whose values disagree by more than a
tolerance, listing the settings of each so that a genuine difference of
configuration can be told apart from an error.

Recognised sources:
    phasecorr_Re{RE}_L{LD}*.npy        I_pha, I_sin, omega, r2, ncal, dt
    pipeline_summary_Re{RE}_L{LD}.npy  improvement_pct (SINUSOID), rho, St, errors
    decompose_Re{RE}_L{LD}.npy         amplitude / phase variance shares
    fieldval_Re{RE}_L{LD}.npy          field RMSEs, G
    dmd_baseline_Re{RE}_L{LD}*.json    latent and field ratios per method
    seed_*_summary.json                per-seed ensembles, with trim
    *.json from the newer analyses     tangential, hankel, heldout, omega sweeps

A NOTE ON TWO KNOWN TRAPS, both encoded as warnings below:
    pipeline_summary improvement_pct is the SINUSOID correction, not the phase one
    dmd_baseline latent_scores.lstm_phase_corrected is likewise the SINUSOID

Usage
-----
    python3 audit_numbers.py --data-dir <DATA>
    python3 audit_numbers.py --data-dir <DATA> --out audit.csv --tol 0.02
"""

import os, re, csv, json, glob, argparse
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--out", default="audit_numbers.csv")
ap.add_argument("--tol", type=float, default=0.01,
                help="relative disagreement above which a conflict is reported")
ap.add_argument("--quiet", action="store_true", help="only print conflicts")
args = ap.parse_args()
DATA = os.path.expanduser(args.data_dir)

ROWS = []           # dicts: quantity, case, latent, ncal, dt, trim, value, source, note


def add(quantity, case, value, source, latent=None, ncal=None, dt=None,
        trim=None, note=""):
    if value is None:
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(v):
        return
    ROWS.append(dict(quantity=quantity, case=str(case), latent=latent, ncal=ncal,
                     dt=dt, trim=trim, value=v,
                     source=os.path.basename(source), note=note))


def case_of(re_, ld=None):
    name = {101: "Square Re=100", 998: "Modulated eps=0.05",
            999: "Modulated eps=0.15"}.get(re_, f"Circular Re={re_}")
    return f"{name}" + (f" (nz={ld})" if ld is not None else "")


def parse_tag(fn):
    m = re.search(r"Re(\d+)(?:_L(\d+))?", fn)
    if not m:
        return None, None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)


# ── phasecorr: the authoritative source for the correction efficacies ───────
for p in sorted(glob.glob(os.path.join(DATA, "phasecorr_Re*_L*.npy"))):
    re_, ld = parse_tag(os.path.basename(p))
    try:
        d = np.load(p, allow_pickle=True).item()
    except Exception:
        continue
    ncal, dt = d.get("ncal"), d.get("dt")
    c = case_of(re_, ld)
    add("efficacy_phase_pct", c, d.get("I_pha"), p, ld, ncal, dt)
    add("efficacy_sinusoid_pct", c, d.get("I_sin"), p, ld, ncal, dt)
    if "omega" in d:
        om = np.atleast_1d(d["omega"])
        add("omega_mean_rad_per_t", c, om.mean(), p, ld, ncal, dt)
        n = len(d.get("e_unc", []))
        if n and dt:
            add("phi_total_rad", c, om.mean() * n * dt, p, ld, ncal, dt,
                note="omega_mean * n * dt")
    if "r2" in d:
        add("r2_phase_fit_mean", c, np.atleast_1d(d["r2"]).mean(), p, ld, ncal, dt)

# ── pipeline summaries: rho, St, and the SINUSOID improvement ───────────────
for p in sorted(glob.glob(os.path.join(DATA, "pipeline_summary_Re*.npy"))):
    re_, ld = parse_tag(os.path.basename(p))
    try:
        d = np.load(p, allow_pickle=True).item()
    except Exception:
        continue
    c = case_of(re_, ld)
    dt = d.get("dt")
    add("efficacy_sinusoid_pct", c, d.get("improvement_pct"), p, ld,
        d.get("calib_len"), dt,
        note="run_pipeline improvement_pct is the SINUSOID correction")
    add("rho", c, d.get("rho"), p, ld, d.get("calib_len"), dt)
    add("St_lstm", c, d.get("St_lstm"), p, ld, None, dt)
    add("cae_floor_rmse", c, d.get("cae_floor_rmse"), p, ld, None, dt)
    add("relerr_unc_pct", c, d.get("relerr_unc"), p, ld, d.get("calib_len"), dt)

# ── decomposition: amplitude / phase shares ─────────────────────────────────
for p in sorted(glob.glob(os.path.join(DATA, "decompose_Re*_L*.npy"))):
    re_, ld = parse_tag(os.path.basename(p))
    try:
        d = np.load(p, allow_pickle=True).item()
    except Exception:
        continue
    c = case_of(re_, ld)
    for k, q in (("phase_share", "phase_share_pct"), ("amp_share", "amp_share_pct"),
                 ("amp_rms_pct", "amplitude_rms_pct")):
        if k in d:
            add(q, c, np.atleast_1d(d[k]).mean(), p, ld)

# ── field validation ────────────────────────────────────────────────────────
for p in sorted(glob.glob(os.path.join(DATA, "fieldval_Re*_L*.npy"))):
    re_, ld = parse_tag(os.path.basename(p))
    try:
        d = np.load(p, allow_pickle=True).item()
    except Exception:
        continue
    c = case_of(re_, ld)
    add("field_G_pct", c, d.get("closed"), p, ld, d.get("ncal"))
    for k in ("r_floor", "r_unc", "r_sin", "r_pha"):
        if k in d:
            add(f"field_rmse_{k[2:]}", c, np.mean(d[k]), p, ld, d.get("ncal"))

# ── dmd baselines ───────────────────────────────────────────────────────────
for p in sorted(glob.glob(os.path.join(DATA, "dmd_baseline_Re*.json"))):
    re_, ld = parse_tag(os.path.basename(p))
    fixed = p.endswith("_fixed.json")
    try:
        J = json.load(open(p))
    except Exception:
        continue
    c = case_of(re_, ld)
    dt = J.get("dt"); ncal = J.get("calib")
    S = J.get("latent_scores", {})
    for k, v in S.items():
        if not isinstance(v, dict) or "ratio_to_lstm" not in v:
            continue
        note = ""
        q = f"latent_ratio_{k}"
        if k == "lstm_phase_corrected" and not fixed:
            note = "MISLABELLED: holds the sinusoid result"
            q = "latent_ratio_lstm_sinusoid_MISLABELLED"
        if k == "lstm_phase_true":
            q = "latent_ratio_lstm_phase"
        add(q, c, v["ratio_to_lstm"], p, ld, ncal, dt, note=note)
    F = J.get("field", {})
    base = F.get("lstm_uncorrected", {}).get("rmse")
    for k, v in F.items():
        if isinstance(v, dict) and "rmse" in v and base:
            note = "provenance unverified" if k == "lstm_phase_corrected" else ""
            add(f"field_ratio_{k}", c, v["rmse"] / base, p, ld, ncal, dt, note=note)

# ── seed ensembles ──────────────────────────────────────────────────────────
for p in sorted(glob.glob(os.path.join(DATA, "**", "seed_*summary.json"), recursive=True)):
    low = p.lower()
    if "_dry" in low:
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if "per_seed" not in d:
        continue
    re_, ld = d.get("re"), d.get("latent")
    c = case_of(re_, ld)
    trim = d.get("trim")
    kind = "field" if "field" in os.path.basename(p) else "latent"
    I = [s["I_pha"] for s in d["per_seed"] if "I_pha" in s]
    if I:
        add(f"ensemble_{kind}_I_pha_mean", c, np.mean(I), p, ld, d.get("calib"),
            d.get("dt"), trim)
        add(f"ensemble_{kind}_I_pha_min", c, np.min(I), p, ld, d.get("calib"),
            d.get("dt"), trim)
        add(f"ensemble_{kind}_I_pha_std", c, np.std(I, ddof=1), p, ld,
            d.get("calib"), d.get("dt"), trim)
    G = [s["G"] for s in d["per_seed"] if s.get("G") is not None]
    if G:
        add("ensemble_field_G_mean", c, np.mean(G), p, ld, d.get("calib"),
            d.get("dt"), trim)
    # horizon block -> the ensemble column of the horizon table. The single-run
    # columns of that table are printed by phase_correction.py but never saved,
    # so they cannot be traced from any file; see the note printed at the end.
    H = d.get("horizon")
    if isinstance(H, dict) and "mean" in H:
        fr = H.get("fracs") or list(range(len(H["mean"])))
        for f_, m_ in zip(fr, H["mean"]):
            add(f"horizon_{kind}_ensemble_mean_pct", f"{c} @ frac={f_:g}", m_, p,
                ld, d.get("calib"), d.get("dt"), trim)
        for f_, sd_ in zip(fr, H.get("std", [])):
            add(f"horizon_{kind}_ensemble_std_pct", f"{c} @ frac={f_:g}", sd_, p,
                ld, d.get("calib"), d.get("dt"), trim)

# ── newer analyses ──────────────────────────────────────────────────────────
for p in sorted(glob.glob(os.path.join(DATA, "tangential_transverse.json"))):
    for k, v in json.load(open(p)).items():
        add("tangential_share", k, v.get("share_par", 0) * 100, p)
        add("tangential_r2", k, v.get("r2_par"), p)
        add("transverse_r2", k, v.get("r2_perp"), p)

for p in sorted(glob.glob(os.path.join(DATA, "heldout_threshold_results.json"))):
    J = json.load(open(p))
    R = J.get("results", {})
    for key, v in R.items():
        if key.startswith("A_") and isinstance(v, dict):
            met = key.split("_", 1)[1]
            add(f"heldout_accepted_{met}", "pooled", v.get("mean"), p,
                note="stratified split, held-out half")
            add(f"heldout_all_{met}", "pooled", v.get("mean_all"), p)
        if key.startswith("C_curve_") and isinstance(v, list):
            met = key.split("C_curve_", 1)[1]
            for row in v:
                thr = row.get("threshold")
                add(f"threshold_coverage_{met}", f"thr={thr:g}",
                    100 * row.get("coverage", 0), p)
                add(f"threshold_efficacy_{met}", f"thr={thr:g}",
                    row.get("mean_accepted"), p)

for p in sorted(glob.glob(os.path.join(DATA, "omega_sensitivity.json"))):
    for case, v in json.load(open(p)).items():
        add("omega_sens_efficacy_at_fit", case, v.get("efficacy_at_fit"), p)
        add("omega_sens_band_spread", case, v.get("band_spread"), p,
            note="efficacy spread over the +/-11% systematic")

for p in sorted(glob.glob(os.path.join(DATA, "spectral_robustness.json"))):
    for case, v in json.load(open(p)).items():
        for row in v.get("rows", []):
            # the label is like "periodogram (unwindowed, published)" or
            # "periodogram (Hann)"; splitting on "(" collapses two different
            # estimators onto one key and fabricates a conflict
            lab = row.get("estimator", "?")
            est = re.sub(r"[^0-9a-zA-Z]+", "_", lab).strip("_")
            add(f"spec_inband_{est}", case, 100 * row.get("in_band_fraction", 0), p)
            add(f"spec_decades_{est}", case, row.get("log10_ratio"), p)

for p in sorted(glob.glob(os.path.join(DATA, "hankel_dmd.json"))):
    for k, v in json.load(open(p)).items():
        ok = [s for s in v.get("sweep", []) if not s.get("diverged")]
        if ok:
            best = min(ok, key=lambda s: s["mean_err"])
            add("hankel_best_err", k, best["mean_err"], p,
                note=f"delays={best['delays']} rank={best['rank']}")
            add("hankel_omega", k, best["omega_mean"], p)
            add("hankel_r2_phi", k, best["r2_phi"], p)
        add("lstm_err_for_hankel_cmp", k, v.get("lstm_mean_err"), p)


# ── verify_mechanism.py output ────────────────────────────────────────────────
# The rollout-wide phi(t) fit lives here, NOT in phasecorr_*.npy, whose r2 is the
# calibration-window value. Methods quotes both and they must not be conflated.
for p in sorted(glob.glob(os.path.join(DATA, "mechanism_Re*_L*.npy"))):
    d = np.load(p, allow_pickle=True).item()
    re_ = int(d.get("re", 0)); ld = int(d.get("latent", 0))
    case = case_of(re_, ld)
    rows = d.get("results", [])
    if rows:
        add("phi_r2_rollout_wide", case,
            float(np.mean([r["r2"] for r in rows])), p, latent=ld,
            note="mean over coordinates; NOT the calibration-window r2")
        add("mech_slope_measured", case,
            float(np.mean([r["slope_measured"] for r in rows])), p, latent=ld)
        add("mech_slope_theory", case,
            float(np.mean([r["slope_theory"] for r in rows])), p, latent=ld,
            note="2*pi*(f_pred - f_true); the frequency-difference route")
        add("mech_envelope_r", case,
            float(np.mean([r["r_env"] for r in rows])), p, latent=ld)

# ── extended delay sweep on the modulated cases ───────────────────────────────
# Every modulated row of the Hankel table comes from here, not from
# hankel_dmd.json. Best is taken over the FULL rank sweep at each delay count so
# that rows differ only in q.
for p in sorted(glob.glob(os.path.join(DATA, "hankel_modulated_longq.json"))):
    for k, v in json.load(open(p)).items():
        lstm = v.get("lstm_mean_err")
        add("longq_lstm_err", k, lstm, p)
        for q in sorted({e["delays"] for e in v.get("sweep", [])}):
            ok = [e for e in v["sweep"]
                  if e["delays"] == q and not e.get("diverged")]
            if not ok:
                continue
            best = min(ok, key=lambda e: e["mean_err"])
            add(f"longq_err_q{q}", k, best["mean_err"], p,
                note=f"rank={best['rank']}")
            if lstm:
                add(f"longq_ratio_q{q}", k, best["mean_err"] / lstm, p)
            add(f"longq_omega_q{q}", k, best["omega_mean"], p)
            add(f"longq_r2_phi_q{q}", k, best["r2_phi"], p)

# ── slope t-statistic ─────────────────────────────────────────────────────────
for p in sorted(glob.glob(os.path.join(DATA, "slope_tstat.json"))):
    for k, v in json.load(open(p)).items():
        add("tstat_mean_abs", k, v.get("t_mean"), p,
            ncal=v.get("ncal"), note="mean |t| over coordinates")
        add("tstat_min_abs", k, v.get("t_min"), p, ncal=v.get("ncal"))
        add("tstat_r2_mean", k, v.get("r2_mean"), p, ncal=v.get("ncal"),
            note="calibration-window r2; cross-check against phasecorr_*")

# ── write ───────────────────────────────────────────────────────────────────
out = os.path.join(DATA, args.out)
fields = ["quantity", "case", "latent", "ncal", "dt", "trim", "value", "source", "note"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in sorted(ROWS, key=lambda r: (r["quantity"], r["case"], r["source"])):
        w.writerow(r)

print("=" * 78)
print(f"AUDIT: {len(ROWS)} values from "
      f"{len({r['source'] for r in ROWS})} files -> {out}")
print("=" * 78)

# ── conflicts ───────────────────────────────────────────────────────────────
groups = defaultdict(list)
for r in ROWS:
    groups[(r["quantity"], r["case"])].append(r)

conflicts = []
for (q, c), rs in sorted(groups.items()):
    if len(rs) < 2:
        continue
    vals = [r["value"] for r in rs]
    lo, hi = min(vals), max(vals)
    scale = max(abs(lo), abs(hi), 1e-12)
    if (hi - lo) / scale > args.tol:
        conflicts.append((q, c, rs, (hi - lo) / scale))

if conflicts:
    print(f"\n{len(conflicts)} quantities have disagreeing sources "
          f"(relative spread > {100*args.tol:.0f}%):\n")
    for q, c, rs, spread in conflicts:
        print(f"  {q}  |  {c}   spread {100*spread:.1f}%")
        for r in sorted(rs, key=lambda r: r["value"]):
            setting = " ".join(f"{k}={r[k]}" for k in ("ncal", "dt", "trim")
                               if r[k] not in (None, ""))
            print(f"      {r['value']:>12.5f}   {r['source']:<44} {setting}"
                  + (f"   [{r['note']}]" if r["note"] else ""))
        print()
    print("  A spread is not automatically an error: two files may legitimately")
    print("  differ if produced with different settings. Compare the settings")
    print("  column. Where they match and the values do not, one is wrong.")
else:
    print("\nNo disagreements above tolerance.")

if not args.quiet:
    print(f"\nQuantities collected ({len(groups)} case-quantity pairs):")
    for q in sorted({r["quantity"] for r in ROWS}):
        n = sum(1 for r in ROWS if r["quantity"] == q)
        print(f"    {q:<44} {n:>3} values")

print("""
NOT COVERED, and worth knowing: the SINGLE-RUN columns of the horizon table are
printed by phase_correction.py but not written to disk, so no file can confirm
them. Either save that block, or regenerate the table from the ensemble means,
which this audit does read.

Use the CSV as the single source for every number in the manuscript. When a value
in the text or a table cannot be traced to a row here, that is itself the finding.
""")
