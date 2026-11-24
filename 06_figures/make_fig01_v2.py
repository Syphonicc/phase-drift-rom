#!/usr/bin/env python3
# make_fig01_v2.py (2026-11-24) -- presentation-only revision of make_fig01.py.
# Changes: one-column size (3.35 in), 7-8 pt fonts; one shared colour bar per row
# (u_x/U_inf and omega_z D/U_inf; vorticity scaled by D/U_inf so the bar label is
# correct); short panel titles; no scale-bar text over the field. Same snapshots
# (index len//2 of snapshots_Re400 / snapshots_SQ100).
# All output goes to the --out directory (default: current directory).
import os, argparse, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='.')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
DATA = os.path.expanduser("~/cylinder_lstm_data")
plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif', 'savefig.dpi': 300})
def load(tag):
    s = np.load(os.path.join(DATA, f"snapshots_{tag}.npy"), mmap_mode='r')
    m = np.load(os.path.join(DATA, f"metadata_{tag}.npy"), allow_pickle=True).item()
    return s, m
def vorticity(ux, uy, x, y):
    return np.gradient(uy, x[1]-x[0], axis=1) - np.gradient(ux, y[1]-y[0], axis=0)
sc, mc = load("Re400"); ss, ms = load("SQ100")
ic, is_ = len(sc)//2, len(ss)//2
xc, yc, xs, ys = mc['x_grid'], mc['y_grid'], ms['x_grid'], ms['y_grid']
uxc, uyc = np.array(sc[ic, 0]), np.array(sc[ic, 1])
uxs, uys = np.array(ss[is_, 0]), np.array(ss[is_, 1])
D, U_INF = 0.12, 1.0
wc = vorticity(uxc, uyc, xc, yc) * D / U_INF
ws = vorticity(uxs, uys, xs, ys) * D / U_INF
uxc, uxs = uxc / U_INF, uxs / U_INF
# shared colour limits per row (centre: U_inf for u_x, 0 for vorticity)
vu = max(np.percentile(np.abs(uxc-1), 99), np.percentile(np.abs(uxs-1), 99))
vw = max(np.percentile(np.abs(wc), 99), np.percentile(np.abs(ws), 99))
ac = (xc[-1]-xc[0])/(yc[-1]-yc[0]); as_ = (xs[-1]-xs[0])/(ys[-1]-ys[0])
fig = plt.figure(figsize=(3.35, 1.95))
gs = fig.add_gridspec(2, 3, width_ratios=[ac, as_, 0.08], left=0.01, right=0.86,
                      bottom=0.02, top=0.92, wspace=0.06, hspace=0.22)
def panel(ax, F, x, y, vmin, vmax, body, title):
    im = ax.imshow(F, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='RdBu_r',
                   vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear')
    if body == 'circle':
        ax.add_patch(plt.Circle((0, 0), 0.06, fc='w', ec='k', lw=0.6, zorder=5))
    else:
        ax.add_patch(plt.Rectangle((-0.06, -0.06), 0.12, 0.12, fc='w', ec='k', lw=0.6, zorder=5))
    ax.set_title(title, pad=2); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.5)
    return im
ax = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]
im1 = panel(ax[0][0], uxc, xc, yc, 1-vu, 1+vu, 'circle', r'(a) circular, $\mathrm{Re}=400$')
panel(ax[0][1], uxs, xs, ys, 1-vu, 1+vu, 'square', r'(b) square, $\mathrm{Re}=100$')
im3 = panel(ax[1][0], wc, xc, yc, -vw, vw, 'circle', r'(c) circular, $\mathrm{Re}=400$')
panel(ax[1][1], ws, xs, ys, -vw, vw, 'square', r'(d) square, $\mathrm{Re}=100$')
fig.canvas.draw()
for r, im, lab in ((0, im1, r'$u_x/U_\infty$'), (1, im3, r'$\omega_z D/U_\infty$')):
    p = ax[r][1].get_position()
    cax = fig.add_axes([p.x1 + 0.02, p.y0, 0.025, p.height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, length=2, width=0.5); cb.outline.set_linewidth(0.5)
    cb.set_label(lab, fontsize=7.5, labelpad=2)
fp = os.path.join(args.out, "fig01_configurations_v2.pdf")
fig.savefig(fp, bbox_inches='tight', pad_inches=0.02); print("written", fp)
