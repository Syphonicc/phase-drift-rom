#!/usr/bin/env python3
"""
make_fig01.py
=============
Builds Figure 1: the two flow configurations.

A 2x2 panel:
    (a) circular cylinder — instantaneous streamwise velocity
    (b) square cylinder   — instantaneous streamwise velocity
    (c) circular cylinder — instantaneous vorticity (shows the vortex street)
    (d) square cylinder   — instantaneous vorticity

Vorticity is computed from the velocity field by finite differences, which is
sufficient for a qualitative figure.

Usage:  python3 make_fig01.py
Output: ~/cylinder_lstm_data/paper_figs/fig01_configurations.pdf
"""

import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = os.path.expanduser("~/cylinder_lstm_data")
OUT  = os.path.join(DATA, "paper_figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

def load(tag, meta_tag=None):
    s = np.load(os.path.join(DATA, f"snapshots_{tag}.npy"))
    m = np.load(os.path.join(DATA, f"metadata_{meta_tag or tag}.npy"),
                allow_pickle=True).item()
    return s, m

def vorticity(ux, uy, x, y):
    """omega_z = duy/dx - dux/dy"""
    dx = x[1]-x[0]; dy = y[1]-y[0]
    duy_dx = np.gradient(uy, dx, axis=1)
    dux_dy = np.gradient(ux, dy, axis=0)
    return duy_dx - dux_dy

# ── load both cases ──────────────────────────────────────────────────────────
print("loading ...", flush=True)
sc, mc = load("Re400")     # circular cylinder
ss, ms = load("SQ100")     # square cylinder

# pick a snapshot well into the saturated limit cycle
ic = len(sc)//2
is_ = len(ss)//2

xc, yc = mc['x_grid'], mc['y_grid']
xs, ys = ms['x_grid'], ms['y_grid']

uxc, uyc = sc[ic,0], sc[ic,1]
uxs, uys = ss[is_,0], ss[is_,1]

wc = vorticity(uxc, uyc, xc, yc)
ws = vorticity(uxs, uys, xs, ys)

# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6))

D = 0.12          # body dimension, used for the scale bar
U_INF = 1.0       # freestream speed, the natural centre for u_x

def panel(ax, F, x, y, cmap, title, vlim=None, body=None,
          centre=None, scalebar=False):
    """centre: value the diverging colourmap is centred on.

    u_x is NOT a signed field about zero -- it runs from a small reverse-flow
    minimum to slightly above the freestream. Centring RdBu_r on zero (the
    previous behaviour) puts the freestream at ~90% of the scale, saturating the
    whole panel and compressing the wake into a few colour levels. Centring on
    U_inf instead makes the map read as deficit/excess about the freestream,
    which is what the panel is meant to show. Vorticity is genuinely signed and
    keeps centre = 0.
    """
    ext = [x[0], x[-1], y[0], y[-1]]
    if vlim is None:
        c = 0.0 if centre is None else centre
        v = np.percentile(np.abs(F - c), 99)
        vlim = (c - v, c + v)
    im = ax.imshow(F, extent=ext, origin='lower', cmap=cmap,
                   vmin=vlim[0], vmax=vlim[1], aspect='equal',
                   interpolation='bilinear')
    if body == 'circle':
        ax.add_patch(plt.Circle((0,0), 0.06, fc='w', ec='k', lw=0.8, zorder=5))
    elif body == 'square':
        ax.add_patch(plt.Rectangle((-0.06,-0.06), 0.12, 0.12,
                                   fc='w', ec='k', lw=0.8, zorder=5))
    if scalebar:
        x0 = x[0] + 0.06*(x[-1]-x[0])
        y0 = y[0] + 0.10*(y[-1]-y[0])
        ax.plot([x0, x0 + D], [y0, y0], color='k', lw=1.6,
                solid_capstyle='butt', zorder=6)
        ax.text(x0 + D/2, y0 + 0.035*(y[-1]-y[0]), r'$D$', ha='center',
                va='bottom', fontsize=7.5, zorder=6)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    return im

# row 1: streamwise velocity, centred on the freestream
im1 = panel(axes[0,0], uxc, xc, yc, 'RdBu_r',
            r'(a) circular cylinder, $\mathrm{Re}=400$', body='circle',
            centre=U_INF, scalebar=True)
im2 = panel(axes[0,1], uxs, xs, ys, 'RdBu_r',
            r'(b) square cylinder, $\mathrm{Re}=100$', body='square',
            centre=U_INF, scalebar=True)

# row 2: vorticity — this is what shows the street
im3 = panel(axes[1,0], wc, xc, yc, 'RdBu_r',
            r'(c) vorticity, $\omega_z$', body='circle', centre=0.0)
im4 = panel(axes[1,1], ws, xs, ys, 'RdBu_r',
            r'(d) vorticity, $\omega_z$', body='square', centre=0.0)

# a single shared label
axes[0,0].set_ylabel(r'$u_x$', fontsize=9)
axes[1,0].set_ylabel(r'$\omega_z$', fontsize=9)

fig.tight_layout(pad=0.5, h_pad=0.9, w_pad=0.9)

# colourbars: one per panel, so every field carries its own numerical scale.
# Both referees asked for quantitative annotation on this figure.
for ax, im, lab in ((axes[0,0], im1, r'$u_x/U_\infty$'),
                    (axes[0,1], im2, r'$u_x/U_\infty$'),
                    (axes[1,0], im3, r'$\omega_z D/U_\infty$'),
                    (axes[1,1], im4, r'$\omega_z D/U_\infty$')):
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label(lab, fontsize=7.5, labelpad=1.5)
fp = os.path.join(OUT, "fig01_configurations.pdf")
fig.savefig(fp)
print(f"\n  written -> {fp}", flush=True)
print("\n  NOTE: this shows the FLOW. If you also want mesh detail, take two", flush=True)
print("  ParaView screenshots (surface-with-edges, zoomed on each body) and", flush=True)
print("  add them as a second row, or as a separate figure.", flush=True)
