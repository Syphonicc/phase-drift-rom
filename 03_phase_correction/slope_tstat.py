#!/usr/bin/env python3
"""
slope_tstat.py
==============
Computes the slope t-statistic  t = omega_hat / SE(omega_hat)  alongside the
R^2 of the same regression, for the four production cases.

WHY
---
R^2 measures how much of the variance in phi(t) the linear model explains. It
depends on the record length and on the noise level of the phase estimator, so a
threshold fixed on one calibration length does not obviously transfer to another.
The t-statistic asks a different and better-posed question: is the fitted drift
rate distinguishable from zero? It is dimensionless and, unlike R^2, its
interpretation does not change with N.

SCOPE -- READ THIS
------------------
This runs on the four production cases only, because those are the ones whose
rollouts survive on disk. The per-seed ensemble records (seed_corr_*/seed_*.npy)
store scalars only -- I_pha, I_sin, r2_mean, drift_ratio -- with no omega, no
phi(t) and no trajectories. Refitting the regressions across the ensembles
therefore requires RETRAINING the networks, not re-reading the outputs. That is a
real constraint and the manuscript should say so plainly rather than implying the
standard errors were merely not written down.

WHAT TO LOOK FOR
----------------
The question is not what t equals. It is whether ranking the cases by t gives the
same accept/reject verdicts as ranking them by R^2. If it does, the choice of
statistic is immaterial and the objection dissolves. If it does not, that is a
finding, and the case that moves should be discussed.

USAGE
    python3 slope_tstat.py
    python3 slope_tstat.py --ncal 150 --r2-threshold 0.3
"""
import argparse, os
import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress, t as tdist

DATA = os.path.expanduser("~/cylinder_lstm_data")

CASES = [(300, 4, 0.03), (400, 4, 0.03), (800, 8, 0.03), (101, 4, 0.0438)]
LABEL = {(300, 4): "Circular Re=300", (400, 4): "Circular Re=400",
         (800, 8): "Circular Re=800", (101, 4): "Square   Re=100"}

ap = argparse.ArgumentParser()
ap.add_argument("--ncal", type=int, default=150)
ap.add_argument("--r2-threshold", type=float, default=0.3)
ap.add_argument("--out", default="slope_tstat.json")
args = ap.parse_args()


def find(names):
    for n in names:
        p = os.path.join(DATA, n)
        if os.path.exists(p):
            return p, n
    return None, None


def analytic_phase(x):
    """Unwrapped instantaneous phase of the mean-removed signal."""
    return np.unwrap(np.angle(hilbert(x - x.mean())))


results = {}
print(f"{'case':17s} {'d':>2} {'omega_d':>12} {'SE':>11} {'t':>9} "
      f"{'p':>10} {'R^2':>7}", flush=True)
print("-" * 74, flush=True)

for re_, nz, dt in CASES:
    tag = f"Re{re_}_L{nz}"
    path, used = find([f"rollout_{tag}.npy", f"rollout_Re{re_}.npy"])
    if path is None:
        print(f"{LABEL[(re_,nz)]:17s}  MISSING rollout file -- skipped", flush=True)
        continue

    d = np.load(path, allow_pickle=True).item()
    # Be explicit about which arrays are used. The file also contains
    # 'corrected', which is the SINUSOID baseline, not the phase correction
    # (footgun 1), and 'inst_phase', which is the predicted phase only. Neither
    # is wanted here: we need the raw uncorrected prediction and the truth, and
    # we recompute both analytic phases ourselves so the regression is the same
    # one the paper reports.
    for k in ("raw_pred", "true"):
        if k not in d:
            raise KeyError(f"{used} has no '{k}'; keys are {sorted(d)}")
    pred = np.asarray(d["raw_pred"], float)
    true = np.asarray(d["true"], float)
    print(f"  [{LABEL[(re_,nz)]}] source: {used}", flush=True)
    assert pred.shape == true.shape, (pred.shape, true.shape)
    n = pred.shape[0]
    t_ax = np.arange(n) * dt

    per_coord = []
    for c in range(pred.shape[1]):
        phi = analytic_phase(pred[:, c]) - analytic_phase(true[:, c])
        phi = phi - phi[0]
        lr = linregress(t_ax[:args.ncal], phi[:args.ncal])
        se = lr.stderr
        tstat = lr.slope / se if se > 0 else np.inf
        dof = args.ncal - 2
        pval = 2 * tdist.sf(abs(tstat), dof)
        r2 = lr.rvalue ** 2
        per_coord.append(dict(coord=c, omega=lr.slope, se=se, t=tstat,
                              p=pval, r2=r2))
        print(f"{LABEL[(re_,nz)] if c==0 else '':17s} {c:>2} "
              f"{lr.slope:>+12.6f} {se:>11.6f} {tstat:>+9.2f} "
              f"{pval:>10.2e} {r2:>7.4f}", flush=True)

    r2_mean = float(np.mean([q["r2"] for q in per_coord]))
    t_mean = float(np.mean([abs(q["t"]) for q in per_coord]))
    t_min = float(np.min([abs(q["t"]) for q in per_coord]))
    verdict_r2 = "ACCEPT" if r2_mean >= args.r2_threshold else "reject"
    print(f"{'':17s}    mean R^2 = {r2_mean:.4f} -> {verdict_r2}"
          f"    mean |t| = {t_mean:.2f}   min |t| = {t_min:.2f}\n", flush=True)

    results[tag] = dict(source=used, ncal=args.ncal, dt=dt,
                        per_coord=per_coord, r2_mean=r2_mean,
                        t_mean=t_mean, t_min=t_min,
                        accepted_by_r2=bool(r2_mean >= args.r2_threshold))

import json
out = os.path.join(DATA, args.out)
json.dump(results, open(out, "w"), indent=2, default=float)
print(f"Saved -> {out}\n", flush=True)

# ------------------------------------------------------------- does it matter?
if results:
    order_r2 = sorted(results, key=lambda k: results[k]["r2_mean"])
    order_t = sorted(results, key=lambda k: results[k]["t_mean"])
    print("Ranking by mean R^2 :", " < ".join(order_r2), flush=True)
    print("Ranking by mean |t| :", " < ".join(order_t), flush=True)
    print("\nRankings AGREE -- the choice of statistic does not change the "
          "ordering." if order_r2 == order_t else
          "\nRankings DIFFER -- this is a finding and belongs in Results.",
          flush=True)

print("""
Note on scope
-------------
Four production cases only. The per-seed ensembles cannot be re-fitted: their
records hold summary scalars, not trajectories. Reporting t for these four while
stating that constraint is honest and is a stronger position than declining the
statistic altogether.
""", flush=True)
