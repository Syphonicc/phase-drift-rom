#!/usr/bin/env python3
"""
make_paper_figures_v2.py  (2026-11-24, presentation-only revision of make_paper_figures.py)
Changes: only Figs 5, 6, 7 are made (spectrum figures 3-4 are not called); all sized for one
column (3.35 in) with 7-8 pt fonts; Fig 5(c) legend moved above the curves (y-range extended upward);
Fig 6 restacked as two rows with a shared legend below and "calibration ends" moved off the data;
"error-fitted" renamed "sinusoid" (as in the tables). Output files get a _v2 suffix.
ALL outputs go to the --out directory.
=====================
Produces publication-quality versions of Figures 3-7, from the saved analysis
data. Consistent styling, vector PDF output, sized for a single journal column.

Figures produced:
    fig03_spectrum_cylinder.pdf   error spectrum, circular cylinder Re=400
    fig04_spectrum_square.pdf     error spectrum, square cylinder Re=100
    fig05_decomposition.pdf       amplitude-phase decomposition (ONE latent coord)
    fig06_correction.pdf          error trajectory: uncorrected / sinusoid / phase
    fig07_field.pdf               field RMSE with the CAE floor

Figures 1 (flow configurations) and 2 (pipeline schematic) must be made by hand.

Usage:   python3 make_paper_figures.py
Output:  ~/cylinder_lstm_data/paper_figs/
"""

import os, numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--out", required=True)
DATA = os.path.expanduser("~/cylinder_lstm_data")
OUT  = _ap.parse_args().out
os.makedirs(OUT, exist_ok=True)

# ── consistent style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size'        : 8,
    'axes.labelsize'   : 8,
    'axes.titlesize'   : 8,
    'legend.fontsize'  : 7.5,
    'xtick.labelsize'  : 7.5,
    'ytick.labelsize'  : 7.5,
    'axes.linewidth'   : 0.7,
    'lines.linewidth'  : 1.1,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'font.family'      : 'serif',
    'mathtext.fontset' : 'dejavuserif',
})

# journal column widths (inches)
W1 = 3.35    # single column
W2 = 7.0     # double column

C_TRUE  = 'k'
C_UNC   = '#d1495b'    # red
C_SIN   = '#edae49'    # amber
C_PHASE = '#2a9d8f'    # teal
C_FLOOR = '#8d99ae'    # grey

def find(cands):
    for c in cands:
        p = os.path.join(DATA, c)
        if os.path.exists(p): return p
    raise FileNotFoundError(cands)

def load_rollout(re, ld):
    d = np.load(find([f"rollout_Re{re}_L{ld}.npy", f"rollout_Re{re}.npy"]),
                allow_pickle=True).item()
    pred = d['raw_pred'] if 'raw_pred' in d else d['predicted']
    true = d['true']
    n = min(len(pred), len(true))
    return pred[:n].astype(float), true[:n].astype(float)

def analytic(s):
    m = s.mean(); z = hilbert(s - m)
    return m, np.abs(z), np.unwrap(np.angle(z))

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 3 & 4 — error spectra
# ══════════════════════════════════════════════════════════════════════════════
def spectrum_fig(re, ld, dt, D, U, st_flow, fname, title):
    pred, true = load_rollout(re, ld)
    n, nz = pred.shape
    err = pred - true
    freqs = np.fft.rfftfreq(n, d=dt)
    St = freqs * D / U

    fig, ax = plt.subplots(figsize=(W1, 2.6))
    cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for d in range(nz):
        e = err[:, d] - err[:, d].mean()
        P = np.abs(np.fft.rfft(e))**2
        m = St <= 0.55
        ax.semilogy(St[m], P[m] + 1e-20, color=cols[d % len(cols)],
                    lw=0.9, label=f'$z_{d}$')
    ax.axvline(st_flow, color='k', ls='--', lw=0.9,
               label=f'$\\mathrm{{St}}_{{\\mathrm{{flow}}}}={st_flow:.3f}$')
    # The +1e-20 guard above keeps the DC null finite on a log axis, but left
    # to autoscale it drags the y-limits down some fifteen empty decades and
    # hides the peak-to-floor separation that is the point of the figure.
    ax.set_ylim(1e-6, 2e0)
    ax.set_xlabel(r'$\mathrm{St} = f D / U_\infty$')
    ax.set_ylabel('error power')
    ax.set_title(title, pad=4)
    ax.legend(frameon=False, ncol=2, columnspacing=0.9, handlelength=1.4)
    ax.grid(alpha=0.25, which='both', lw=0.4)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"  {fname}", flush=True)

if False:  # v2: spectrum figures 3-4 not regenerated
  spectrum_fig(400, 4, 0.03,   0.12, 1.0, 0.219,
             "fig03_spectrum_cylinder.pdf",
             r'Circular cylinder, $\mathrm{Re}=400$')
  spectrum_fig(101, 4, 0.0438, 0.12, 1.0, 0.148,
             "fig04_spectrum_square.pdf",
             r'Square cylinder, $\mathrm{Re}=100$')

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — decomposition (ONE latent coordinate, 3 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5: amplitude-phase decomposition", flush=True)
RE, LD, DT, DIM = 400, 4, 0.03, 0
pred, true = load_rollout(RE, LD)
n = len(pred); t = np.arange(n)*DT

m_t, A_t, th_t = analytic(true[:, DIM])
m_p, A_p, th_p = analytic(pred[:, DIM])
e = pred[:, DIM] - true[:, DIM]
phi = (th_p - th_t); phi -= phi[0]
e_phase = A_t*(np.cos(th_p) - np.cos(th_t))
e_amp   = (A_p - A_t)*np.cos(th_t)

fig, axes = plt.subplots(3, 1, figsize=(W1, 5.0), sharex=True)

ax = axes[0]
ax.plot(t, A_t, color=C_TRUE,  lw=1.3, label=r'true, $A$')
ax.plot(t, A_p, color=C_UNC, lw=0.9, ls='--', label=r'predicted, $\hat{A}$')
ax.set_ylabel('amplitude')
ax.set_title('(a) the amplitude is correct', loc='left', pad=3)
ax.legend(frameon=False, loc='lower right')
ax.grid(alpha=0.25, lw=0.4)

ax = axes[1]
ax.plot(t, phi, color=C_PHASE, lw=1.0)
lr = linregress(t, phi)
ax.plot(t, lr.intercept + lr.slope*t, 'k--', lw=0.8,
        label=fr'linear, $R^2={lr.rvalue**2:.3f}$')
ax.set_ylabel(r'$\phi$ [rad]')
ax.set_title('(b) the phase drifts, linearly', loc='left', pad=3)
ax.legend(frameon=False, loc='lower left')
ax.grid(alpha=0.25, lw=0.4)

ax = axes[2]
ax.plot(t, e,       color=C_TRUE,  lw=1.2, label='actual error')
ax.plot(t, e_phase, color=C_PHASE, lw=0.8, ls='--', label='phase term')
ax.plot(t, e_amp,   color=C_UNC,   lw=0.8, label='amplitude term')
ax.set_ylabel('error')
ax.set_xlabel(r'$t$')
ax.set_title('(c) the error is the phase term', loc='left', pad=3)
# v2: legend in one row above the curves; extend the y-range upward to make room
_lo, _hi = ax.get_ylim(); ax.set_ylim(_lo, _hi + 0.6*(_hi - _lo))
ax.legend(frameon=False, loc='upper left', ncol=2, columnspacing=1.2,
          handlelength=1.8, handletextpad=0.4, borderaxespad=0.2, labelspacing=0.2)
ax.grid(alpha=0.25, lw=0.4)

fig.savefig(os.path.join(OUT, "fig05_decomposition_v2.pdf"))
plt.close(fig)
print("  fig05_decomposition_v2.pdf", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 6 — correction comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 6: correction comparison", flush=True)
# the unsuffixed file on disk is the Ncal=200 run; the manuscript reports
# Ncal=150, so name that window explicitly and fall back only if absent
pc = np.load(find([f"phasecorr_Re{RE}_L{LD}_ncal150.npy",
                   f"phasecorr_Re{RE}_L{LD}.npy"]), allow_pickle=True).item()
if int(pc.get("ncal", -1)) != 150:
    print(f"  WARNING: correction figure is using Ncal={pc.get('ncal')}, "
          f"not the 150 reported in the manuscript", flush=True)
e_unc, e_sin, e_pha = pc['e_unc'], pc['e_sin'], pc['e_pha']
NCAL = pc['ncal']
t = np.arange(len(e_unc))*DT

fig, axes = plt.subplots(2, 1, figsize=(W1, 3.9), sharex=True)

for ax, logscale in zip(axes, [False, True]):
    plot = ax.semilogy if logscale else ax.plot
    plot(t, e_unc, color=C_UNC,   lw=1.0, label='uncorrected')
    plot(t, e_sin, color=C_SIN,   lw=1.0,
         label=f"sinusoid ({pc['I_sin']:+.0f}%)")
    plot(t, e_pha, color=C_PHASE, lw=1.1,
         label=f"phase correction ({pc['I_pha']:+.0f}%)")
    ax.axvline(NCAL*DT, color='k', ls=':', lw=0.8)
    ax.grid(alpha=0.25, which='both', lw=0.4)
axes[1].set_xlabel(r'$t$')
axes[0].set_ylabel(r'$\|\hat{\mathbf{z}}-\mathbf{z}\|$')
axes[1].set_ylabel(r'$\|\hat{\mathbf{z}}-\mathbf{z}\|$')
axes[0].set_title('(a) linear scale', loc='left', pad=3)
axes[1].set_title('(b) logarithmic scale', loc='left', pad=3)
# v2: shared legend below the panels; label placed in the empty upper part of (a)
_h, _l = axes[0].get_legend_handles_labels()
fig.tight_layout(pad=0.2, h_pad=0.5, rect=(0, 0.10, 1, 1))
fig.legend(_h, _l, frameon=False, loc='lower center', ncol=2, columnspacing=1.0,
           handlelength=1.6, handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
_y0, _y1 = axes[0].get_ylim()
axes[0].text(NCAL*DT, _y1 - 0.03*(_y1 - _y0), ' calibration ends',
             fontsize=7.5, va='top', ha='left', color='k')

fig.savefig(os.path.join(OUT, "fig06_correction_v2.pdf"))
plt.close(fig)
print("  fig06_correction_v2.pdf", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 7 — field validation
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 7: field validation", flush=True)
fv = np.load(find([f"fieldval_Re{RE}_L{LD}.npy"]), allow_pickle=True).item()
r_floor, r_unc = fv['r_floor'], fv['r_unc']
r_sin, r_pha   = fv['r_sin'],   fv['r_pha']
NCAL = fv['ncal']
t = np.arange(len(r_unc))*DT

fig, ax = plt.subplots(figsize=(W1, 2.5))
ax.plot(t, r_unc,   color=C_UNC,   lw=1.0, label='uncorrected')
ax.plot(t, r_sin,   color=C_SIN,   lw=1.0, label='sinusoid')
ax.plot(t, r_pha,   color=C_PHASE, lw=1.1, label='phase correction')
ax.plot(t, r_floor, color=C_FLOOR, lw=1.0, ls=':', label='autoencoder floor')
ax.axvline(NCAL*DT, color='k', ls=':', lw=0.8)
ax.set_xlabel(r'$t$')
ax.set_ylabel('field RMSE')
ax.set_title(fr"$G = {fv['closed']:.0f}\%$ of the correctable error removed",
             pad=4)
ax.legend(frameon=False, loc='upper left')
ax.grid(alpha=0.25, lw=0.4)
fig.savefig(os.path.join(OUT, "fig07_field_v2.pdf"))
plt.close(fig)
print("  fig07_field_v2.pdf", flush=True)

print(f"\nAll figures written to {OUT}", flush=True)
print("\nStill to make BY HAND:", flush=True)
print("  fig01_configurations.pdf  — mesh + instantaneous wake, both bodies", flush=True)
print("  fig02_pipeline.pdf        — CAE -> LSTM -> rollout -> correction schematic",
      flush=True)
