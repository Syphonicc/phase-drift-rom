#!/usr/bin/env python3
"""
spectral_robustness.py
======================
Does the error-spectrum result depend on the spectral estimator?

Figures 3-4 use a single unwindowed periodogram of the rollout-error time series.
That choice is deliberate: an earlier Hann-windowed estimate broadened the peak
and inflated the in-band energy fraction, and the unwindowed periodogram is the
unbiased estimator of the line spectrum we are actually claiming. But a single
periodogram is a high-variance estimator -- which is why the noise floor in those
figures looks jagged -- and the paper has never shown that the result survives
variance reduction.

This checks it. For each case, the rollout error is analysed with

    * the unwindowed periodogram              (as published)
    * Welch's method at several segment counts (variance reduced, resolution lost)
    * a Hann-windowed periodogram             (the estimator we moved away from)

and for each we report the peak location relative to the shedding frequency, the
peak-to-floor ratio, and the fraction of energy within +/-10% of the peak.

The claims at stake are (i) the peak sits at f_s, (ii) it stands three to four
orders of magnitude above the floor, and (iii) the latent dynamics are narrowband
to about one part in a hundred. If (i) and (ii) are estimator-independent, the
figures are sound; (iii) is expected to MOVE with the estimator, because averaging
trades frequency resolution for variance, and that trade should be reported rather
than hidden.

Usage
-----
    python3 spectral_robustness.py --dry-run
    python3 spectral_robustness.py --data-dir <DATA>
    python3 spectral_robustness.py --data-dir <DATA> --cases 400:4
"""

import os, sys, json, argparse
import numpy as np
from scipy.signal import periodogram, welch, get_window

ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="+", default=["300:4", "400:4", "800:8", "101:4"])
ap.add_argument("--data-dir", default=None,
                help="directory holding the rollout_*.npy files (required unless --dry-run)")
ap.add_argument("--dt", type=float, default=0.03)
ap.add_argument("--dt-square", type=float, default=0.0438)
ap.add_argument("--D", type=float, default=0.12, help="cylinder diameter")
ap.add_argument("--U", type=float, default=1.0, help="free-stream velocity")
ap.add_argument("--nseg", type=int, nargs="+", default=[2, 4, 8],
                help="Welch segment counts to try")
ap.add_argument("--band", type=float, default=0.10,
                help="in-band fraction is measured within +/- this fraction of the peak")
ap.add_argument("--out", default="spectral_robustness.json")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
if args.data_dir is None and not args.dry_run:
    ap.error("--data-dir is required (or use --dry-run)")
DATA = os.path.expanduser(args.data_dir) if args.data_dir else "."


def load_error(re_, nz):
    for name in (f"rollout_Re{re_}_L{nz}.npy", f"rollout_Re{re_}.npy"):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True).item()
            pred = d["raw_pred"] if "raw_pred" in d else d["predicted"]
            true = d["true"]
            n = min(len(pred), len(true))
            return np.asarray(pred[:n], float) - np.asarray(true[:n], float), name
    return None, None


def synth_error(nz=4, n=534, dt=0.03, f=1.8, seed=0):
    """A pure phase error on a limit cycle: coherent at f, plus a small floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    e = np.zeros((n, nz))
    for d in range(nz):
        e[:, d] = 0.02 * t * np.sin(2 * np.pi * f * t + 2 * np.pi * d / nz)
    return e + rng.normal(0, 1e-4, e.shape)


def describe(f, P, f_ref, band):
    """Peak location, peak/floor ratio, and in-band energy fraction."""
    m = f > 0                                   # drop DC
    f, P = f[m], P[m]
    k = int(np.argmax(P))
    f_pk = float(f[k])
    inb = np.abs(f - f_pk) <= band * f_pk
    frac = float(P[inb].sum() / P.sum())
    # floor: median of the out-of-band power, a robust estimate
    floor = float(np.median(P[~inb])) if (~inb).any() else float("nan")
    ratio = float(P[k] / floor) if floor > 0 else float("inf")
    return dict(f_peak=f_pk, f_peak_rel=float(f_pk / f_ref) if f_ref else float("nan"),
                peak_over_floor=ratio, log10_ratio=float(np.log10(ratio)) if ratio > 0 else float("nan"),
                in_band_fraction=frac, n_freq=int(len(f)),
                df=float(f[1] - f[0]) if len(f) > 1 else float("nan"))


print("=" * 78, flush=True)
print("SPECTRAL-ESTIMATOR ROBUSTNESS OF THE ERROR-SPECTRUM RESULT", flush=True)
print("=" * 78, flush=True)

results = {}
for spec in args.cases:
    try:
        re_s, nz_s = spec.split(":"); re_, nz = int(re_s), int(nz_s)
    except Exception:
        print(f"  malformed case '{spec}'", flush=True); continue
    dt = args.dt_square if re_ == 101 else args.dt
    fs = 1.0 / dt

    if args.dry_run:
        E, src, f_ref = synth_error(nz=nz, dt=dt), "synthetic", 1.8
    else:
        E, src = load_error(re_, nz)
        if E is None:
            print(f"  Re={re_} L={nz}: no rollout file, skipping", flush=True); continue
        f_ref = None                            # inferred below from the periodogram

    n = len(E)
    print(f"\n  Re={re_} n_z={nz}  ({src}, {n} samples, dt={dt})", flush=True)

    # aggregate the per-coordinate spectra, as the figures do
    def agg(fun, **kw):
        Ps = []
        for d in range(E.shape[1]):
            f, P = fun(E[:, d] - E[:, d].mean(), fs=fs, **kw)
            Ps.append(P)
        return f, np.mean(Ps, axis=0)

    f0, P0 = agg(periodogram, window="boxcar", scaling="density")
    if f_ref is None:
        mm = f0 > 0
        f_ref = float(f0[mm][np.argmax(P0[mm])])   # the published peak defines f_s
    st = f_ref * args.D / args.U

    rows = []
    r = describe(f0, P0, f_ref, args.band)
    r["estimator"] = "periodogram (unwindowed, published)"
    rows.append(r)

    fh, Ph = agg(periodogram, window="hann", scaling="density")
    r = describe(fh, Ph, f_ref, args.band); r["estimator"] = "periodogram (Hann)"
    rows.append(r)

    for k in args.nseg:
        nper = max(16, n // k)
        fw, Pw = agg(welch, nperseg=nper, noverlap=nper // 2, window="hann",
                     scaling="density")
        r = describe(fw, Pw, f_ref, args.band)
        r["estimator"] = f"Welch, {k} segments (nperseg={nper})"
        rows.append(r)

    print(f"    shedding frequency from the published estimator: "
          f"f_s = {f_ref:.4f} (St = {st:.4f})", flush=True)
    print(f"    {'estimator':<38} {'f_pk/f_s':>9} {'peak/floor':>12} "
          f"{'in-band':>8} {'df':>8}", flush=True)
    for r in rows:
        print(f"    {r['estimator']:<38} {r['f_peak_rel']:>9.4f} "
              f"{r['log10_ratio']:>9.1f} dec {100*r['in_band_fraction']:>7.1f}% "
              f"{r['df']:>8.4f}", flush=True)

    results[f"Re{re_}_L{nz}"] = dict(f_s=f_ref, St=st, n=n, dt=dt, rows=rows)

if not results:
    sys.exit("No cases processed.")

json.dump(results, open(os.path.join(DATA, args.out), "w"), indent=2, default=float)
print(f"\nSaved -> {os.path.join(DATA, args.out)}", flush=True)

# ── summary across cases, which is what the paper needs ─────────────────────
print("\n" + "=" * 78, flush=True)
print("SUMMARY (what to quote)", flush=True)
print("=" * 78, flush=True)
for key, tag in [("f_peak_rel", "peak location, relative to the published f_s"),
                 ("log10_ratio", "peak above floor (decades)"),
                 ("in_band_fraction", "energy within +/-10% of the peak")]:
    print(f"\n  {tag}:", flush=True)
    ests = [r["estimator"] for r in next(iter(results.values()))["rows"]]
    for i, est in enumerate(ests):
        vals = [res["rows"][i][key] for res in results.values()]
        if key == "in_band_fraction":
            print(f"    {est:<38} {100*min(vals):5.1f}–{100*max(vals):5.1f}%", flush=True)
        else:
            print(f"    {est:<38} {min(vals):6.3f}–{max(vals):6.3f}", flush=True)
print("""
Reading this:
  * If the peak stays at f_pk/f_s = 1 across estimators, the location claim is
    estimator-independent and Figs. 3-4 are sound.
  * The peak-to-floor ratio should stay large; Welch reduces the variance of the
    floor, so the ratio is usually easier to read there, not smaller.
  * The in-band fraction is EXPECTED to change: Welch trades frequency resolution
    for variance, so a fixed +/-10% band contains a different share. That is a
    property of the estimator, not of the flow, and the paper should say so.
""", flush=True)
