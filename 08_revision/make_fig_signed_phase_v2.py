#!/usr/bin/env python3
"""v2: phase referenced to the calibration-fit intercept (not sample 0, which carries
Hilbert end effects); smooth Eq.(4) envelope from the fitted linear drift; one hue
for all coordinates (the message is the common sign, not coordinate identity).
R2.7: signed error and signed accumulated phase for the four representative
cases.  Top row: signed latent error e_d(t) of the dominant coordinate with the
envelope -A_d*phi_d(t)*sin(...) bound +/-A_d|phi_d(t)| predicted by Eq. (4).
Bottom row: signed accumulated phase phi_d(t) of each latent coordinate (for
n_z=8 the four coordinates of largest amplitude), with the calibration window
shaded.  Phases from the analytic signal exactly as in phase_correction.py."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from paths import DATA, OUT_REV as OUT
CASES = [("Re300_L4", "rollout_Re300_L4.npy", 0.03, r"circular, $Re=300$, $n_z=4$"),
         ("Re400_L4", "rollout_Re400.npy", 0.03, r"circular, $Re=400$, $n_z=4$"),
         ("Re800_L8", "rollout_Re800_L8.npy", 0.03, r"circular, $Re=800$, $n_z=8$"),
         ("Re101_L4", "rollout_Re101_L4.npy", 0.0438, r"square, $Re=100$, $n_z=4$")]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; LS = ["-", "--", "-.", ":"]
NCAL, TRIM = 150, 10
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "font.family": "serif", "mathtext.fontset": "cm"})
fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.6), sharex="col", gridspec_kw=dict(hspace=0.12, wspace=0.42))
summary = {}
for j, (tag, fn, dt, title) in enumerate(CASES):
    r = np.load(f"{DATA}/{fn}", allow_pickle=True).item()
    zt = r["true"].astype(float); zp = (r["raw_pred"] if "raw_pred" in r else r["predicted"]).astype(float)
    n = min(len(zt), len(zp)); zt, zp = zt[:n], zp[:n]
    at = hilbert(zt - zt.mean(0), axis=0); ap = hilbert(zp - zp.mean(0), axis=0)
    A = np.abs(at).mean(0)
    phi = np.unwrap(np.angle(ap), axis=0) - np.unwrap(np.angle(at), axis=0)
    tt = np.arange(n) * dt; cal = slice(TRIM, NCAL)
    fit = np.array([np.polyfit(tt[cal], phi[cal, d], 1) for d in range(phi.shape[1])])   # (slope, intercept)
    phi = phi - fit[:, 1]
    fs = np.median(np.diff(np.unwrap(np.angle(at), axis=0), axis=0).mean(0)) / (2 * np.pi * dt)
    cyc = np.arange(n) * dt * fs
    order = np.argsort(-A)[:4]
    d0 = order[0]; e = zp[:, d0] - zt[:, d0]
    k = slice(TRIM, n - TRIM)
    a0 = ax[0, j]
    a0.plot(cyc, e, color=COL[0], lw=0.7)
    env = A[d0] * np.abs(fit[d0, 0] * tt)
    a0.plot(cyc[k], env[k], color=INK, lw=1.0, ls="--"); a0.plot(cyc[k], -env[k], color=INK, lw=1.0, ls="--")
    a0.set_title(title, fontsize=8, color=INK)
    a1 = ax[1, j]
    a1.axvspan(0, NCAL * dt * fs, color=GRID, lw=0)
    a1.axhline(0, color=MUTED, lw=0.5)
    for i, d in enumerate(order):
        a1.plot(cyc[k], phi[k, d], color=COL[0], lw=0.5, alpha=0.55)
        a1.plot(cyc[k], fit[d, 0] * tt[k], color=INK, lw=0.9, ls="--")
    a1.set_xlabel("shedding cycles")
    for a in (a0, a1):
        a.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        a.set_xlim(0, cyc[-1] * 1.12)
    summary[tag] = dict(fs=float(fs), n_cycles=float(cyc[-1]), omega=[float(fit[d,0]) for d in order], phi_end=[float(phi[n - TRIM - 1, d]) for d in order],
                        A=[float(A[d]) for d in order])
ax[0, 0].set_ylabel(r"signed error $e_d$")
ax[1, 0].set_ylabel(r"$\phi_d(t)$ [rad]")
fig.legend([plt.Line2D([], [], color=COL[0], lw=0.8), plt.Line2D([], [], color=INK, ls="--", lw=0.9)],
           [r"top: signed error $e_d$ of the dominant coordinate;  bottom: $\phi_d(t)$ of each latent coordinate",
            r"top: $\pm A_d\,|\omega_d t|$ from Eq. (4);  bottom: linear fit on the calibration window (shaded), extrapolated"],
           loc="upper center", ncol=1, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.savefig(f"{OUT}/fig_signed_phase_v2.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_signed_phase_v2.png", dpi=200, bbox_inches="tight")
import json; json.dump(summary, open(f"{OUT}/fig_signed_phase_v2.json", "w"), indent=1)
for k_, v in summary.items(): print(k_, {a: (np.round(b, 4).tolist() if isinstance(b, list) else round(b, 3)) for a, b in v.items()})
