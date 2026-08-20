#!/usr/bin/env python3
"""
omega_sensitivity.py
====================
How much does the reported efficacy depend on the drift rate being right?

The drift rate omega_d carries a systematic uncertainty of about +/-11%, measured
in phase_bias.py by comparing three independent estimators against a signal with an
imposed drift. That uncertainty has never been propagated into the efficacy figures
the paper reports.

Rather than linearise, this re-applies the correction with omega_d deliberately
perturbed and measures what happens. Two things are reported:

  1. SENSITIVITY. Efficacy as a function of the multiplicative error on omega_d,
     over a range wide enough to see the shape (default +/-30%), with the +/-11%
     systematic marked. If efficacy is flat there, the published figures are robust
     to the estimator uncertainty and the error bars are small.

  2. BREAK-EVEN. The multiplicative error at which the correction stops helping
     at all. This is the more useful robustness statement: it tells a practitioner
     how accurately the drift must be known for the method to be worth applying.

A note on what the +/-11% is. It bounds how well an ESTIMATOR recovers a known
drift. The correction does not use that estimator's output blindly -- omega is
fitted by regression on the calibration window, against the actual phase data --
so the two are related but not identical. Treating the estimator uncertainty as
though it were uncertainty in the fitted value is therefore conservative, which is
the direction to err.

Inputs: rollout_Re{RE}_L{NZ}.npy (keys 'raw_pred', 'true'), as written by
run_pipeline.py. No retraining.

Usage
-----
    python3 omega_sensitivity.py --dry-run
    python3 omega_sensitivity.py --data-dir <DATA>
    python3 omega_sensitivity.py --data-dir <DATA> --cases 400:4 --systematic 0.11
"""

import os, sys, json, argparse
import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress

ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="+", default=["300:4", "400:4", "800:8", "101:4"])
ap.add_argument("--data-dir", default=None,
                help="directory holding rollout_*.npy (required unless --dry-run)")
ap.add_argument("--dt", type=float, default=0.03)
ap.add_argument("--dt-square", type=float, default=0.0438)
ap.add_argument("--ncal", type=int, default=150)
ap.add_argument("--systematic", type=float, default=0.11,
                help="systematic uncertainty on omega_d, as a fraction")
ap.add_argument("--span", type=float, default=0.60,
                help="sweep the multiplicative error over +/- this fraction")
ap.add_argument("--npts", type=int, default=25)
ap.add_argument("--out", default="omega_sensitivity.json")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
if args.data_dir is None and not args.dry_run:
    ap.error("--data-dir is required (or use --dry-run)")
DATA = os.path.expanduser(args.data_dir) if args.data_dir else "."


def load_rollout(re_, nz):
    for name in (f"rollout_Re{re_}_L{nz}.npy", f"rollout_Re{re_}.npy"):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True).item()
            pred = d["raw_pred"] if "raw_pred" in d else d["predicted"]
            true = d["true"]
            n = min(len(pred), len(true))
            return np.asarray(pred[:n], float), np.asarray(true[:n], float), name
    return None, None, None


def synth(nz=4, n=535, dt=0.03, f=1.8, drift=0.004, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt; w = 2 * np.pi * f
    true = np.zeros((n, nz)); pred = np.zeros((n, nz))
    for d in range(nz):
        ph = 2 * np.pi * d / nz
        true[:, d] = np.cos(w * t + ph)
        pred[:, d] = np.cos((w + drift) * t + ph)
    return pred + rng.normal(0, 1e-3, pred.shape), true


def iphase(x):
    return np.unwrap(np.angle(hilbert(x - x.mean())))


def fit_omega(pred, true, dt, ncal):
    """The drift rate as the paper fits it: regression of the accumulated phase
    difference over the calibration window."""
    n, nz = pred.shape
    t = np.arange(n) * dt
    om = np.zeros(nz)
    for d in range(nz):
        phi = iphase(pred[:, d]) - iphase(true[:, d]); phi -= phi[0]
        m = min(ncal, n - 1)
        om[d] = linregress(t[:m], phi[:m]).slope
    return om


def apply_correction(pred, dt, omega):
    """Re-reference the predicted phase by omega*t, per latent coordinate,
    preserving the instantaneous amplitude and the coordinate mean."""
    n, nz = pred.shape
    t = np.arange(n) * dt
    out = np.empty_like(pred)
    for d in range(nz):
        x = pred[:, d]; mu = x.mean()
        a = hilbert(x - mu)
        amp = np.abs(a)
        ph = np.unwrap(np.angle(a))
        out[:, d] = amp * np.cos(ph - omega[d] * t) + mu
    return out


def efficacy(pred, corr, true, ncal):
    """Percentage reduction in mean L2 error over the extrapolation region."""
    e_u = np.linalg.norm(pred - true, axis=1)
    e_c = np.linalg.norm(corr - true, axis=1)
    sl = slice(ncal, len(true))
    return float((1 - e_c[sl].mean() / e_u[sl].mean()) * 100)


print("=" * 78, flush=True)
print("SENSITIVITY OF THE EFFICACY TO THE DRIFT RATE", flush=True)
print("=" * 78, flush=True)
print(f"systematic on omega_d taken as +/-{100*args.systematic:.0f}%\n", flush=True)

scales = 1.0 + np.linspace(-args.span, args.span, args.npts)
results = {}

for spec in args.cases:
    try:
        re_s, nz_s = spec.split(":"); re_, nz = int(re_s), int(nz_s)
    except Exception:
        print(f"  malformed case '{spec}'", flush=True); continue
    dt = args.dt_square if re_ == 101 else args.dt

    if args.dry_run:
        pred, true = synth(nz=nz, dt=dt); src = "synthetic"
    else:
        pred, true, src = load_rollout(re_, nz)
        if pred is None:
            print(f"  Re={re_} L={nz}: no rollout file, skipping", flush=True); continue

    om0 = fit_omega(pred, true, dt, args.ncal)
    base = efficacy(pred, apply_correction(pred, dt, om0), true, args.ncal)

    curve = []
    for s in scales:
        eff = efficacy(pred, apply_correction(pred, dt, om0 * s), true, args.ncal)
        curve.append(dict(scale=float(s), efficacy=eff))

    effs = np.array([c["efficacy"] for c in curve])
    # efficacy within the systematic band
    band = np.abs(scales - 1.0) <= args.systematic + 1e-9
    lo, hi = float(effs[band].min()), float(effs[band].max())
    # break-even: where the correction stops helping
    neg = [c["scale"] for c in curve if c["efficacy"] <= 0]
    below = [s for s in neg if s < 1]; above = [s for s in neg if s > 1]
    be_lo = f"{100*(max(below)-1):+.0f}%" if below else "beyond sweep"
    be_hi = f"{100*(min(above)-1):+.0f}%" if above else "beyond sweep"

    print(f"  Re={re_} n_z={nz}  ({src})", flush=True)
    print(f"    fitted omega_d (mean over coords): {om0.mean():+.6f} rad/t", flush=True)
    print(f"    efficacy at the fitted value     : {base:.1f}%", flush=True)
    print(f"    over +/-{100*args.systematic:.0f}% on omega_d          : "
          f"{lo:.1f}% to {hi:.1f}%   (spread {hi-lo:.1f} points)", flush=True)
    print(f"    correction stops helping at      : {be_lo} / {be_hi}", flush=True)
    print(f"    {'scale':>7} {'efficacy':>10}", flush=True)
    for c in curve[::max(1, len(curve)//9)]:
        mark = ""
        if abs(c["scale"] - 1) < 1e-9: mark = "  <- fitted"
        elif abs(abs(c["scale"] - 1) - args.systematic) < 0.02: mark = "  <- systematic"
        print(f"    {c['scale']:>7.2f} {c['efficacy']:>9.1f}%{mark}", flush=True)
    print("", flush=True)

    results[f"Re{re_}_L{nz}"] = dict(omega_fitted=float(om0.mean()),
                                     efficacy_at_fit=base,
                                     band_lo=lo, band_hi=hi,
                                     band_spread=hi - lo,
                                     breakeven_low=be_lo, breakeven_high=be_hi,
                                     curve=curve)

if not results:
    sys.exit("No cases processed.")

json.dump(results, open(os.path.join(DATA, args.out), "w"), indent=2, default=float)
print(f"Saved -> {os.path.join(DATA, args.out)}", flush=True)

spreads = [r["band_spread"] for r in results.values()]
print("\n" + "=" * 78, flush=True)
print(f"SUMMARY: over the +/-{100*args.systematic:.0f}% systematic on omega_d, the "
      f"efficacy varies by", flush=True)
print(f"         {min(spreads):.1f} to {max(spreads):.1f} percentage points across cases.", flush=True)
print("""
If that spread is small relative to the reported efficacy, the headline figures are
robust to the uncertainty in the drift rate, and the break-even columns give the
accuracy a drift estimate actually needs for the correction to be worth applying.
""", flush=True)
