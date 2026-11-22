#!/usr/bin/env python3
"""R3.3, v3: v2's Hilbert phase of a non-sinusoidal probe wobbles within each
cycle (+/-0.3 rad), which swamps any drift.  Here the clock is read only at the
same point of every cycle: upward zero crossings t_i of u_y(2D,0).  A line
t_i = a + T*i is fitted to the pre-rollout crossings (first 60% of the window)
and extrapolated; the rollout-segment residuals, times 2*pi/T, are the phase by
which the TRUE flow departs from a constant-period clock.  The same fit is also
done over the whole window with a quadratic term (period drift).  Crossing
times are linearly interpolated (18 samples/cycle); the scatter of residuals in
the fitted segment measures this interpolation noise."""
import json, os, numpy as np
D = 0.12
from paths import DATA as HOME, REV, OUT_REV
HERE = os.path.dirname(os.path.abspath(__file__))
v1 = json.load(open(os.path.join(REV, "saturation_check.json")))
def crossings(t, s):
    i = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    return t[i] - s[i] * (t[i + 1] - t[i]) / (s[i + 1] - s[i])
out = {}
for re in [300, 400, 500, 600, 700, 800, 1000]:
    snap = np.load(f"{HOME}/snapshots_Re{re}.npy", mmap_mode="r")
    meta = np.load(f"{HOME}/metadata_Re{re}.npy", allow_pickle=True).item()
    t = np.asarray(meta["times"], float)
    ix = int(np.argmin(np.abs(meta["x_grid"] - 2 * D))); iy = int(np.argmin(np.abs(meta["y_grid"])))
    s = np.asarray(snap[:, 1, iy, ix], float); s -= s.mean()
    tc = crossings(t, s); i = np.arange(len(tc))
    t_roll = t[0] + 0.6 * (t[-1] - t[0]); pre = tc < t_roll
    a = np.polyfit(i[pre], tc[pre], 1); T = a[0]
    res_pre = (tc[pre] - np.polyval(a, i[pre])) * 2 * np.pi / T
    res_roll = (tc[~pre] - np.polyval(a, i[~pre])) * 2 * np.pi / T
    q = np.polyfit(i - i.mean(), tc, 2)
    dT_frac = 2 * q[0] * (i[-1] - i[0]) / q[1]           # period change across window / T
    d = dict(v1[str(re)])
    d.update(clock_noise_rad=float(res_pre.std()), dphi_roll_end_rad=float(res_roll[-1]),
             dphi_roll_mean_rad=float(res_roll.mean()), period_drift_frac=float(dT_frac),
             n_roll_cycles=int((~pre).sum()))
    out[re] = d
json.dump(out, open(os.path.join(OUT_REV, "saturation_check_v3.json"), "w"), indent=1)
print(f"{'Re':>5} {'St':>6} {'dT_h%':>7} {'dA_h%':>7} {'dT/T%':>8} {'noise':>7} {'dphi_end':>9} {'dphi_mean':>9} {'ncyc':>5}")
for re, d in out.items():
    print(f"{re:>5} {d['St']:6.4f} {100*d['dT_halves']:7.3f} {100*d['dA_halves']:7.3f} {100*d['period_drift_frac']:8.4f} "
          f"{d['clock_noise_rad']:7.4f} {d['dphi_roll_end_rad']:9.4f} {d['dphi_roll_mean_rad']:9.4f} {d['n_roll_cycles']:5d}")
