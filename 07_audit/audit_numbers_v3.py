#!/usr/bin/env python3
"""
audit_numbers_v2.py
===================
v2 (PoF revision R1): v1 unchanged, plus (a) the revision outputs under
<DATA>/revision as sources, and (b) a CLAIMS check: every number quoted in the
revised manuscript/response, recomputed from its source and compared within its
rounding. v1 docstring follows.

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
ap.add_argument("--out", default="audit_numbers_v3.csv")
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

# ── PoF revision R1 sources (inserted into audit_numbers_v2.py) ───────────────
REV = os.path.join(DATA, "revision"); REVB = os.path.join(REV, "B")
def jload(p):
    try: return json.load(open(p))
    except Exception: return None

# saturation (wake probe) and force-based Strouhal numbers -> conflicts reveal disagreement
d = jload(os.path.join(REV, "saturation_check_v3.json"))
for re_, v in (d or {}).items():
    c = case_of(int(re_))
    add("St_flow_baseline_mesh", c, v["St"], "saturation_check_v3.json", note="probe u_y(2D,0) zero crossings")
    add("sat_dT_halves_pct", c, 100 * v["dT_halves"], "saturation_check_v3.json")
    add("sat_dA_halves_pct", c, 100 * v["dA_halves"], "saturation_check_v3.json")
    add("sat_dphi_true_end_rad", c, v["dphi_roll_end_rad"], "saturation_check_v3.json")
    add("sat_clock_noise_rad", c, v["clock_noise_rad"], "saturation_check_v3.json")
fv = jload(os.path.join(REV, "forces_vs_re.json"))
for re_, v in ((fv or {}).get("base") or {}).items():
    c = case_of(int(re_))
    add("St_flow_baseline_mesh", c, v["St"], "forces_vs_re.json", note="lift zero crossings")
    add("Cd_mean_baseline_mesh", c, v["Cd_mean"], "forces_vs_re.json")
    add("Cl_rms_baseline_mesh", c, v["Cl_rms"], "forces_vs_re.json")
    add("relax_per_period", c, v.get("relax_per_period"), "forces_vs_re.json")
for lab, dd in ((fv or {}).get("extra") or {}).items():
    for re_, v in dd.items():
        c = case_of(int(re_)) + f" [{lab}]"
        add("St_flow", c, v["St"], "forces_vs_re.json"); add("Cd_mean", c, v["Cd_mean"], "forces_vs_re.json")
        add("relax_per_period", c, v.get("relax_per_period"), "forces_vs_re.json")
for f in glob.glob(os.path.join(REV, "forces_*.json")):
    v = jload(f); nm = os.path.basename(f)
    if not v or "St" not in v: continue
    tag = nm.replace("forces_", "").replace(".json", "")
    add("St_flow_by_file", tag, v["St"], nm); add("Cd_mean_by_file", tag, v["Cd_mean"], nm)
# near-wall resolution
d = jload(os.path.join(REV, "nearwall_resolution_v2.json"))
for mesh, v in (d or {}).items():
    add("first_cell_height_over_D", mesh, v["h1_over_D_mean"], "nearwall_resolution_v2.json")
    add("wall_faces", mesh, v["n_wall_faces"], "nearwall_resolution_v2.json")
    for re_, cc in v["cells_in_delta"].items():
        add("cells_within_D_Re^-1/2", f"{mesh} Re={re_}", cc["min"], "nearwall_resolution_v2.json")
# periodic continuation with sub-sample interpolation
for fn in ("periodic_tiling_interp_v2.json",) + tuple(os.path.basename(x) for x in glob.glob(os.path.join(REV, "periodic_tiling_interp_v3_*.json"))):
    d = jload(os.path.join(REV, fn))
    for tag, v in (d or {}).items():
        re_, ld = parse_tag(tag); c = case_of(re_, ld)
        for k in ("latent_pub_linear", "latent_pub_cubic", "latent_pub_fourier", "incal_linear", "incal_cubic",
                  "field_pub_linear", "field_pub_cubic", "lstm_uncorrected", "lstm_phase_published"):
            if k in v: add(f"tiling_{k}", c, v[k], fn)
        if "latent_pub_cubic" in v: add("tiling_cubic_latent_ratio", c, v["latent_pub_cubic"] / v["lstm_uncorrected"], fn)
        if "tiling_published" in v: add("tiling_linear_latent_err", c, v["tiling_published"], fn, note="periodic_baseline JSON")
        if "latent_pub_linear" in v: add("tiling_linear_latent_err", c, v["latent_pub_linear"], fn, note="recomputed")
# field-space timing, signed phase, long rollout, learned map
d = jload(os.path.join(REV, "field_timing_check_v2.json"))
if d:
    add("field_timing_explained_pct", "Circular Re=400", 100 * d["explained_fraction_last10pct"], "field_timing_check_v2.json")
    add("field_timing_lag_corr", "Circular Re=400", d["corr_phase_vs_ls_lag"], "field_timing_check_v2.json")
    add("field_timing_dt_over_T_pct", "Circular Re=400", 100 * abs(d["delta_t_end_over_T"]), "field_timing_check_v2.json")
d = jload(os.path.join(REV, "fig_signed_phase_v2.json"))
for tag, v in (d or {}).items():
    re_, ld = parse_tag(tag); c = case_of(re_, ld); w = np.mean(v["omega"])
    add("clock_rel_St_error", c, w / (2 * np.pi * v["fs"]), "fig_signed_phase_v2.json")
    add("omega_spread_across_coords_pct", c, 100 * np.std(v["omega"]) / abs(w), "fig_signed_phase_v2.json")
d = jload(os.path.join(REVB, "long_rollout_v2.json"))
if d:
    for r in d:
        add("long_r2_linear", f"S8 {r['seed']}", r["r2_linear"], "long_rollout_v2.json")
        add("long_phi_end_rad", f"S8 {r['seed']}", r["phi_end"], "long_rollout_v2.json")
        add("long_I_free_pct", f"S8 {r['seed']}", r["I_free_long"], "long_rollout_v2.json")
d = jload(os.path.join(REVB, "clock_perturb_v2_Re500_40seeds.json"))
if d:
    wp = np.array([r["w_pred"] for r in d]); wr = np.array([r["w_roll"] for r in d]); wc = np.array([r["w_cal_mean"] for r in d])
    Ic = np.array([r["I_cal"] for r in d]); If = np.array([r["I_free"] for r in d])
    add("freerun_corr_wpred_wroll", "Re=500 x40", np.corrcoef(wp, wr)[0, 1], "clock_perturb_v2_Re500_40seeds.json")
    add("freerun_median_rel_err", "Re=500 x40", np.median(np.abs(wp - wr) / np.abs(wr)), "clock_perturb_v2_Re500_40seeds.json")
    add("calwin_median_rel_err", "Re=500 x40", np.median(np.abs(wc - wr) / np.abs(wr)), "clock_perturb_v2_Re500_40seeds.json")
    add("freerun_sign_agree", "Re=500 x40", np.sum(np.sign(wp) == np.sign(wr)), "clock_perturb_v2_Re500_40seeds.json")
    add("I_free_median", "Re=500 x40", np.median(If), "clock_perturb_v2_Re500_40seeds.json")
    add("I_cal_median", "Re=500 x40", np.median(Ic), "clock_perturb_v2_Re500_40seeds.json")
    add("free_better_count", "Re=500 x40", np.sum(If > Ic), "clock_perturb_v2_Re500_40seeds.json")
    add("free_harmful_count", "Re=500 x40", np.sum(If < 0), "clock_perturb_v2_Re500_40seeds.json")
    add("cal_harmful_count", "Re=500 x40", np.sum(Ic < 0), "clock_perturb_v2_Re500_40seeds.json")
d = jload(os.path.join(REVB, "learned_map_panel_c_v5.json"))
if d:
    add("ampkick_remaining_cycle0_median", "Re=500 x40", d["amp_dev_per_cycle_median"][0], "learned_map_panel_c_v5.json")
    add("ampkick_remaining_cycle0_max", "Re=500 x40", d["amp_dev_per_cycle_max"][0], "learned_map_panel_c_v5.json")
    add("phasekick_persist_min", "Re=500 x40", min(d["phase_persist_min"]), "learned_map_panel_c_v5.json")
    add("phasekick_persist_max", "Re=500 x40", max(d["phase_persist_max"]), "learned_map_panel_c_v5.json")
# sensitivity ensembles
d = jload(os.path.join(REVB, "sensitivity_table_v2.json"))
for r in (d or []):
    c = f"{r['dir']} [{r['label']}]"
    for k in ("phase", "tan", "w_med", "w_cv", "gate", "I_med", "p", "slow", "fast", "n"):
        if r.get(k) is not None: add(f"sens_{k}", c, r[k], "sensitivity_table_v2.json")

# ── CLAIMS: numbers as written in the revised manuscript / response ──────────
# Each claim: (where, quoted value, (quantity, case), tolerance as absolute rounding half-width).
def val(q, c):
    vs = [r["value"] for r in ROWS if r["quantity"] == q and r["case"] == c]
    return vs[0] if vs else None
def sens(dirn, k):
    vs = [r["value"] for r in ROWS if r["quantity"] == f"sens_{k}" and r["case"].startswith(dirn + " [")]
    return vs[0] if vs else None
CLAIMS = [
 ("Methods/sat: max true-clock departure Re300-800 (rad)", 0.0024, lambda: max(abs(val("sat_dphi_true_end_rad", case_of(r))) for r in (300, 400, 500, 600, 700, 800)), 0.00005),
 ("Methods/sat: Re=1000 clock departure (rad)", 4.3, lambda: val("sat_dphi_true_end_rad", case_of(1000)), 0.05),
 ("Methods/sat: max |dT halves| Re300-800 (%)", 0.005, lambda: max(abs(val("sat_dT_halves_pct", case_of(r))) for r in (300, 400, 500, 600, 700, 800)), 0.0005),
 ("Methods/sat: max |dA halves| Re300-800 (%) [upper bound]", 0.06, lambda: (lambda a: 0.06 if a <= 0.06 else a)(max(abs(val("sat_dA_halves_pct", case_of(r))) for r in (300, 400, 500, 600, 700, 800))), 0.0),
 ("Methods/nearwall: first cell baseline (D)", 0.034, lambda: val("first_cell_height_over_D", "circular_baseline"), 0.0005),
 ("Methods/nearwall: first cell fine (D)", 0.024, lambda: val("first_cell_height_over_D", "circular_fine"), 0.0005),
 ("Methods/nearwall: first cell square (D)", 0.042, lambda: val("first_cell_height_over_D", "square"), 0.0005),
 ("Methods/conv: Re800 St fine", 0.236, lambda: val("St_flow", "Circular Re=800 [fine mesh]"), 0.0005),
 ("Methods/conv: Re1000 St fine", 0.235, lambda: val("St_flow", "Circular Re=1000 [fine mesh]"), 0.0005),
 ("Methods/forces: relax Re=100", 0.47, lambda: val("relax_per_period", case_of(100)), 0.005),
 ("Methods/forces: relax Re300-800 min", 0.33, lambda: min(val("relax_per_period", case_of(r)) for r in (300, 400, 500, 600, 700, 800)), 0.005),
 ("Methods/forces: relax Re300-800 max", 0.38, lambda: max(val("relax_per_period", case_of(r)) for r in (300, 400, 500, 600, 700, 800)), 0.005),
 ("Sec III H: cubic tiling latent ratio min", 0.10, lambda: min(val("tiling_cubic_latent_ratio", case_of(*t)) for t in ((300, 4), (400, 4), (800, 8), (101, 4))), 0.005),
 ("Sec III H: cubic tiling latent ratio max", 0.23, lambda: max(val("tiling_cubic_latent_ratio", case_of(*t)) for t in ((300, 4), (400, 4), (800, 8), (101, 4))), 0.005),
 ("Sec III D: clock rel St error Re400", -8.3e-5, lambda: val("clock_rel_St_error", case_of(400, 4)), 0.05e-5),
 ("Sec III E: field timing explained (%)", 97.8, lambda: val("field_timing_explained_pct", "Circular Re=400"), 0.05),
 ("Sec III E: lag correlation", 0.97, lambda: val("field_timing_lag_corr", "Circular Re=400"), 0.005),
 ("Sec III E: lag as % of period", 0.27, lambda: val("field_timing_dt_over_T_pct", "Circular Re=400"), 0.005),
 ("Sec III O: free-run corr", 0.993, lambda: val("freerun_corr_wpred_wroll", "Re=500 x40"), 0.0005),
 ("Sec III O: free-run median rel err", 0.10, lambda: val("freerun_median_rel_err", "Re=500 x40"), 0.005),
 ("Sec III O: cal-window median rel err", 0.33, lambda: val("calwin_median_rel_err", "Re=500 x40"), 0.005),
 ("Sec III O: sign agreement (of 40)", 38, lambda: val("freerun_sign_agree", "Re=500 x40"), 0.5),
 ("Sec III O: I_free median (%)", 57, lambda: val("I_free_median", "Re=500 x40"), 0.5),
 ("Sec III O: I_cal median (%)", 37, lambda: val("I_cal_median", "Re=500 x40"), 0.5),
 ("Sec III O: free better (of 40)", 37, lambda: val("free_better_count", "Re=500 x40"), 0.5),
 ("Sec III O: free harmful", 3, lambda: val("free_harmful_count", "Re=500 x40"), 0.5),
 ("Sec III O: cal harmful", 10, lambda: val("cal_harmful_count", "Re=500 x40"), 0.5),
 ("Sec III O: amp kick after 1 period median", 0.012, lambda: val("ampkick_remaining_cycle0_median", "Re=500 x40"), 0.0005),
 ("Sec III O: amp kick after 1 period max", 0.026, lambda: val("ampkick_remaining_cycle0_max", "Re=500 x40"), 0.0005),
 ("Sec III O: phase kick persist min", 0.995, lambda: val("phasekick_persist_min", "Re=500 x40"), 0.0005),
 ("Sec III O: phase kick persist max", 1.003, lambda: val("phasekick_persist_max", "Re=500 x40"), 0.0005),
 ("Sec III D long: r2 min", 0.965, lambda: min(r["value"] for r in ROWS if r["quantity"] == "long_r2_linear"), 0.0005),
 ("Sec III D long: r2 max", 0.998, lambda: max(r["value"] for r in ROWS if r["quantity"] == "long_r2_linear"), 0.0005),
 ("Sec III D long: |phi_end| max (rad)", 0.27, lambda: max(abs(r["value"]) for r in ROWS if r["quantity"] == "long_phi_end_rad"), 0.005),
 ("Sec III N: Re400 mesh p", 0.34, lambda: sens("gridconv_Re400_fine", "p"), 0.005),
 ("Sec III N: Re800 mesh p", 0.79, lambda: sens("S12b_Re800_fine_L8", "p"), 0.005),
 ("Sec III N: 25D p", 0.34, lambda: sens("S7_long25D", "p"), 0.005),
 ("Sec III N: blockage p", 0.85, lambda: sens("S11_wide10D", "p"), 0.005),
 ("Sec III N: causal CAE p", 0.79, lambda: sens("S6_causalCAE", "p"), 0.005),
 ("Sec III N: POD phase share", 0.97, lambda: sens("S3_pod", "phase"), 0.005),
 ("Sec III N: init p", 0.97, lambda: sens("S2_fixedLSTMseed", "p"), 0.005),
 ("Sec III N: lift-only p", 0.57, lambda: sens("S9_lift", "p"), 0.005),
 ("Sec III N: lift-only phase share", 0.95, lambda: sens("S9_lift", "phase"), 0.005),
 ("Sec III N: multistep K=5 p", 0.16, lambda: sens("S5_ms5", "p"), 0.005),
 ("Sec III N: multistep K=20 p", 0.68, lambda: sens("S5_ms20", "p"), 0.005),
 ("Sec III N: multistep reduction K=5 (%)", 27, lambda: 100 * (1 - sens("S5_ms5", "w_med") / sens("S1_fixedCAE", "w_med")), 0.5),
 ("Sec III N: multistep reduction K=20 (%)", 15, lambda: 100 * (1 - sens("S5_ms20", "w_med") / sens("S1_fixedCAE", "w_med")), 0.5),
 ("Sec III N: Re1000 fine phase share", 0.89, lambda: sens("S13b_Re1000_fine", "phase"), 0.005),
 ("Sec III N: tau p min", 0.59, lambda: min(sens(t, "p") for t in ("S4_tau5", "S4_tau10", "S4_tau40", "S4_tau80")), 0.005),
]
print("\n" + "=" * 78 + "\nCLAIMS CHECK (numbers as quoted in the revised text)\n" + "=" * 78)
nbad = 0
for where, quoted, fn, tol in CLAIMS:
    try: actual = fn()
    except Exception as e: actual = None
    if actual is None:
        print(f"  [NO SOURCE] {where}: quoted {quoted}"); nbad += 1; continue
    ok = abs(actual - quoted) <= tol
    nbad += (not ok)
    print(f"  [{'OK' if ok else 'MISMATCH'}] {where}: quoted {quoted}  actual {actual:.6g}")
print(f"\n{len(CLAIMS) - nbad} of {len(CLAIMS)} claims verified; {nbad} need attention.")
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
