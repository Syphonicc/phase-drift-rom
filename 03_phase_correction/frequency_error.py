#!/usr/bin/env python3
"""
p0s4c_frequency_error.py

The tangential growth RATIO is not robust: it spans 2.5x to 428x across
reasonable windowing conventions. Growth is clearly linear (R2 0.93-0.98 vs
0.57-0.72 exponential), so the right reportable quantity is a RATE, not a
ratio -- and the physically meaningful rate is a frequency error.

If the mechanism is "correct attractor, wrong traversal rate", then the
network's effective shedding frequency differs from the truth by some small
percentage, and everything else follows. That number is:

  - dimensionless
  - independent of any windowing convention
  - reproducible by any reader
  - a direct statement of the mechanism rather than a symptom of it

TWO INDEPENDENT ROUTES, computed here and cross-checked:

  A. SPECTRAL/GEOMETRIC PHASE
     Project onto the leading 2-D POD plane of the reference (whitened, so an
     elliptical orbit becomes circular and the phase advances uniformly).
     Unwrap the phase angle for truth and prediction, fit both, compare slopes.
        dw/w = (w_pred - w_true) / w_true

  B. TANGENTIAL DISPLACEMENT
     On a near-circular orbit of radius R, a tangential error of magnitude
     e_par corresponds to a phase offset of e_par / R radians. Fit that
     against time and convert to a frequency error the same way.

Route A uses phase angles; route B uses the rolldrift decomposition. They
share no machinery. If they agree, the result is not an artefact of either.

Run from the repo root:
    python p0s4c_frequency_error.py
"""

import os
import sys

import numpy as np

try:
    import rolldrift as rd
except ImportError:
    sys.exit("rolldrift not importable. Run `pip install -e .` first.")

DATA = os.path.expanduser("~/cylinder_lstm_data")
SKIP = 150      # calibration window; the fit starts after it

CASES = [
    ("circular Re=300", 300, 4, 0.03,   None),
    ("circular Re=400", 400, 4, 0.03,   0.548),
    ("circular Re=800", 800, 8, 0.03,   None),
    ("square   Re=101", 101, 4, 0.0438, 0.876),
]


def load(re_, nz):
    for name in (f"rollout_Re{re_}_L{nz}.npy", f"rollout_Re{re_}.npy"):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True).item()
            return (np.asarray(d["raw_pred"], float),
                    np.asarray(d["true"], float), name)
    raise FileNotFoundError(f"Re={re_}, nz={nz}")


def pod_plane(true):
    """Leading 2-D POD plane of the reference, with a whitening scale that
    circularises an elliptical orbit so the phase angle advances uniformly."""
    mu = true.mean(axis=0)
    X = true - mu
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    basis = Vt[:2]                      # (2, n)
    scale = s[:2] / s[0]                # whiten the second axis
    return mu, basis, scale, s


def phase_of(traj, mu, basis, scale):
    p = (traj - mu) @ basis.T           # (T, 2)
    p = p / scale                       # circularise
    return np.unwrap(np.arctan2(p[:, 1], p[:, 0])), np.linalg.norm(p, axis=1)


def robust_slope(t, y):
    """Least-squares slope plus R^2."""
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    r2 = 1.0 - np.var(resid) / np.var(y) if np.var(y) > 0 else np.nan
    return coef[0], coef[1], r2


print("=" * 80, flush=True)
print("Phase drift expressed as a frequency error", flush=True)
print("=" * 80, flush=True)

rows = []

for label, re_, nz, dt, T_shed_known in CASES:
    try:
        pred, true, fname = load(re_, nz)
    except FileNotFoundError as e:
        print(f"\n{label}: SKIPPED ({e})", flush=True)
        continue

    T = len(true)
    t = np.arange(T) * dt
    mu, basis, scale, sv = pod_plane(true)

    th_true, r_true = phase_of(true, mu, basis, scale)
    th_pred, _ = phase_of(pred, mu, basis, scale)

    sl = slice(SKIP, T)
    tt = t[sl]

    # ---- route A: phase angle slopes -----------------------------------
    w_true, _, r2_t = robust_slope(tt, th_true[sl])
    w_pred, _, r2_p = robust_slope(tt, th_pred[sl])
    T_meas = 2 * np.pi / abs(w_true)
    dw_over_w_A = (w_pred - w_true) / w_true

    # phase error directly, and its drift rate
    dth = th_pred - th_true
    slope_dth, _, r2_dth = robust_slope(tt, dth[sl])
    cycles_per_shed = slope_dth * T_meas / (2 * np.pi)

    # ---- route B: tangential displacement / orbit radius ----------------
    rep = rd.diagnose(pred, true, dt)
    R = float(np.median(np.linalg.norm(true - mu, axis=1)))
    dth_B = rep.tangential_t / R
    sign = np.sign(slope_dth) if slope_dth != 0 else 1.0
    slope_B, _, r2_B = robust_slope(tt, sign * dth_B[sl])
    dw_over_w_B = slope_B / w_true

    print(f"\n{label}   ({fname})", flush=True)
    print(f"  measured shedding period : {T_meas:.4f}"
          + (f"   [stated {T_shed_known}]" if T_shed_known else "   [not stated]"),
          flush=True)
    print(f"  reference angular rate   : {w_true:+.6f} rad/time  (R2 {r2_t:.5f})",
          flush=True)
    print(f"  predicted angular rate   : {w_pred:+.6f} rad/time  (R2 {r2_p:.5f})",
          flush=True)
    print(f"  phase error drift        : {slope_dth:+.6f} rad/time  "
          f"(R2 {r2_dth:.4f})", flush=True)
    print(f"  -> per shedding period   : {cycles_per_shed:+.5f} cycles "
          f"({cycles_per_shed*360:+.3f} deg)", flush=True)
    print(f"", flush=True)
    print(f"  ROUTE A  frequency error : {dw_over_w_A*100:+.4f} %", flush=True)
    print(f"  ROUTE B  frequency error : {dw_over_w_B*100:+.4f} %   "
          f"(R2 {r2_B:.4f})", flush=True)
    agree = abs(abs(dw_over_w_A) - abs(dw_over_w_B)) / max(abs(dw_over_w_A), 1e-12)
    verdict = ("AGREE" if agree < 0.25 else
               "PARTIAL" if agree < 0.6 else "DISAGREE -- investigate")
    print(f"  cross-check              : {agree*100:.1f}% relative gap  [{verdict}]",
          flush=True)

    # how many cycles until the phase error reaches half a period
    if slope_dth != 0:
        t_half = abs(np.pi / slope_dth)
        print(f"  time to antiphase        : {t_half:.1f} time units "
              f"({t_half/T_meas:.0f} shedding periods)", flush=True)

    rows.append((label, T_meas, dw_over_w_A * 100, dw_over_w_B * 100,
                 cycles_per_shed * 360, verdict))

print("\n" + "=" * 80, flush=True)
print("SUMMARY", flush=True)
print("=" * 80, flush=True)
hdr = f"{'case':<18}{'T_shed':>9}{'df/f A %':>11}{'df/f B %':>11}{'deg/cycle':>11}  check"
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
for r in rows:
    print(f"{r[0]:<18}{r[1]:>9.4f}{r[2]:>+11.4f}{r[3]:>+11.4f}{r[4]:>+11.3f}  {r[5]}",
          flush=True)

print("\nIf routes A and B agree, the sentence for the paper is:", flush=True)
print('  "The network reproduces the attractor geometry but traverses it at a', flush=True)
print('   frequency X% below the reference, accumulating Y degrees of phase lag', flush=True)
print('   per shedding period."', flush=True)
print("\nThat number is convention-free, unlike the growth ratio.", flush=True)
