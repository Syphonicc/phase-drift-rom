#!/usr/bin/env python3
"""
plot_dmd_baseline.py
====================
Builds the linear-baseline comparison figure (reviewer item 1).

Four panels:
  (a) latent error vs time for one showcase case -- the headline gap
  (b) DMD eigenvalues on the complex plane, all cases -- proves DMD is fitted
      correctly (its neutral mode sits at the true Strouhal number)
  (c) latent-space error ratio, all cases, log axis
  (d) field-space error ratio, all cases, log axis

LAYOUT NOTE: every axes rectangle is positioned explicitly with fig.add_axes()
in figure-fraction coordinates computed from the margins below. tight_layout()
and bbox_inches='tight' are deliberately NOT used -- they reflow axes after the
fact and are what causes overlapping/misaligned panels. If you need to nudge
something, change the margin constants in LAYOUT and nothing else moves.

Usage:
    python3 plot_dmd_baseline.py
    python3 plot_dmd_baseline.py --showcase 400 --outfile fig08_dmd.pdf
    python3 plot_dmd_baseline.py --data-dir ~/cylinder_lstm_data
"""

import os, sys, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--showcase", type=int, default=400,
                help="Reynolds number for the time-series panel (a)")
ap.add_argument("--outfile", default="fig08_dmd_baseline.pdf")
ap.add_argument("--dpi", type=int, default=300)
ap.add_argument("--width", type=float, default=7.0, help="inches (double column)")
ap.add_argument("--height", type=float, default=5.5, help="inches")
args = ap.parse_args()

DATA_DIR = os.path.expanduser(args.data_dir)

# ----------------------------------------------------------------- style
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
})

C_UNC  = "#c1443c"   # LSTM uncorrected
C_COR  = "#2a8a7d"   # LSTM + phase correction
C_DMD  = "#35618f"   # DMD
C_HAR  = "#d99a2b"   # harmonic
C_GREY = "#8a8a8a"

# ----------------------------------------------------------------- load
CASES = [(300, 4), (400, 4), (800, 8), (101, 4)]
LABEL = {(300,4): "Circ.\n$Re$=300", (400,4): "Circ.\n$Re$=400",
         (800,8): "Circ.\n$Re$=800", (101,4): "Square\n$Re$=100"}

def load_case(re_, ld):
    tag = f"Re{re_}_L{ld}"
    # prefer the corrected file written by patch_dmd_figure.py; the original
    # mislabels the sinusoid result as the phase correction
    j_fixed = os.path.join(DATA_DIR, f"dmd_baseline_{tag}_fixed.json")
    j = j_fixed if os.path.exists(j_fixed) else os.path.join(DATA_DIR, f"dmd_baseline_{tag}.json")
    n = os.path.join(DATA_DIR, f"dmd_baseline_{tag}.npy")
    J = json.load(open(j)) if os.path.exists(j) else None
    N = np.load(n, allow_pickle=True).item() if os.path.exists(n) else None
    return J, N

DATA = {}
for c in CASES:
    J, N = load_case(*c)
    if J is None:
        print(f"  [warn] missing JSON for {c}; skipping", flush=True)
        continue
    DATA[c] = dict(json=J, npy=N)
if not DATA:
    sys.exit(f"No dmd_baseline_*.json found in {DATA_DIR}")
print(f"  loaded {len(DATA)} cases", flush=True)

def best_dmd_key(J, space="latent"):
    """the strongest of the three DMD variants -- we always report DMD at its best"""
    keys = ["dmd_train", "dmd_unitcircle", "dmd_calib"]
    if space == "latent":
        vals = {k: J["latent_scores"][k]["ratio_to_lstm"] for k in keys}
    else:
        base = J["field"]["lstm_uncorrected"]["rmse"]
        vals = {k: J["field"][k]["rmse"]/base for k in keys}
    return min(vals, key=vals.get)

# ----------------------------------------------------------------- layout
# All positions are computed here. Nothing else in the script moves axes.
W, H = args.width, args.height
LEFT, RIGHT = 0.098, 0.975
BOT,  TOP   = 0.082, 0.915
WGAP, HGAP  = 0.115, 0.165

col_w = (RIGHT - LEFT - WGAP) / 2.0
row_h = (TOP  - BOT  - HGAP) / 2.0
row2_y = BOT + row_h + HGAP          # y of the TOP row
col2_x = LEFT + col_w + WGAP         # x of the RIGHT column

fig = plt.figure(figsize=(W, H))

# panel (b) is a complex plane -> must be square in INCHES, so convert:
sq_h_frac = row_h
sq_w_frac = sq_h_frac * H / W                       # equal physical size
b_x = col2_x + (col_w - sq_w_frac) / 2.0            # centred in its column

ax_a = fig.add_axes([LEFT,   row2_y, col_w,     row_h])
ax_b = fig.add_axes([b_x,    row2_y, sq_w_frac, sq_h_frac])
ax_c = fig.add_axes([LEFT,   BOT,    col_w,     row_h])
ax_d = fig.add_axes([col2_x, BOT,    col_w,     row_h])

def panel_tag(ax, s):
    """place the tag ABOVE-LEFT of the axes in FIGURE coords -> never collides
    with tick labels, whatever the axis limits turn out to be."""
    r = ax.get_position()
    fig.text(max(r.x0 - 0.072, 0.004), min(r.y1 + 0.052, 0.995), s,
             fontsize=9.5, fontweight="bold", va="top", ha="left")

# ============================================================ (a) error vs time
show = (args.showcase, dict(CASES)[args.showcase] if args.showcase in dict(CASES) else 4)
if show not in DATA:
    show = sorted(DATA)[0]
J, N = DATA[show]["json"], DATA[show]["npy"]

if N is not None:
    true = N["true"]; dt = J["dt"]; ncal = J["calib"]
    nstep = len(true); t = np.arange(nstep) * dt

    def emag(P):
        P = np.asarray(P, float)
        return np.linalg.norm(P[:nstep] - true, axis=1)

    # smooth over one shedding period so the trend is legible under the
    # within-cycle oscillation; raw curves are drawn faintly underneath
    f_shed = max(abs(J["dmd_eigs"]["freq_hz"][int(np.argmax(J["dmd_eigs"]["abs"]))]), 1e-9)
    win = max(int(round((1.0 / f_shed) / dt)), 3)
    def smooth(y, w):
        y = np.asarray(y, float)
        if np.all(np.isnan(y)): return y
        k = np.ones(w) / w
        pad = np.concatenate([np.full(w, y[np.isfinite(y)][0] if np.any(np.isfinite(y)) else 0),
                              y,
                              np.full(w, y[np.isfinite(y)][-1] if np.any(np.isfinite(y)) else 0)])
        out = np.convolve(np.nan_to_num(pad, nan=np.nanmean(pad)), k, mode="same")
        return out[w:-w]

    bd = best_dmd_key(J, "latent")
    series = [
        ("CAE--LSTM, uncorrected",      emag(N["lstm_raw"]),                C_UNC, "-",  1.5),
        ("CAE--LSTM + phase correction", emag(N["lstm_cor"]),               C_COR, "-",  1.5),
        (f"CAE + DMD",                   emag(N["preds"][bd]),              C_DMD, "--", 1.3),
        ("CAE + harmonic fit",           emag(N["preds"]["harmonic"]),      C_HAR, ":",  1.4),
    ]
    for lab, e, col, ls, lw in series:
        ax_a.plot(t, e, color=col, lw=0.5, alpha=0.22, ls="-")
        ax_a.plot(t, smooth(e, win), color=col, lw=lw, ls=ls, label=lab)

    ax_a.axvline(ncal * dt, color=C_GREY, lw=0.8, ls="-.")
    ymin, ymax = ax_a.get_ylim()
    ax_a.text(ncal*dt, ax_a.get_ylim()[1], " calibration\n ends",
              fontsize=6.5, color=C_GREY, va="top", ha="left")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"$t$")
    ax_a.set_ylabel(r"latent error $\Vert \mathbf{e}_n \Vert$")
    ax_a.set_xlim(t[0], t[-1])
    ax_a.legend(loc="lower right", frameon=False, handlelength=2.0, borderpad=0.2,
                labelspacing=0.28)
    ax_a.set_title(f"{LABEL[show].replace(chr(10),' ')}", pad=4)
else:
    ax_a.text(0.5, 0.5, "trajectory .npy not found", ha="center", va="center",
              transform=ax_a.transAxes, color=C_GREY)
panel_tag(ax_a, "(a)")

# ============================================================ (b) DMD eigenvalues
th = np.linspace(0, 2*np.pi, 400)
ax_b.plot(np.cos(th), np.sin(th), color="k", lw=0.8, alpha=0.55)
ax_b.axhline(0, color=C_GREY, lw=0.5, alpha=0.5)
ax_b.axvline(0, color=C_GREY, lw=0.5, alpha=0.5)

mk = {(300,4):"o", (400,4):"s", (800,8):"^", (101,4):"D"}
for c in CASES:
    if c not in DATA: continue
    e = DATA[c]["json"]["dmd_eigs"]
    absl = np.array(e["abs"]); frq = np.array(e["freq_hz"])
    # reconstruct lambda from |lambda| and frequency: arg = 2*pi*f*dt
    dt = DATA[c]["json"]["dt"]
    ang = 2*np.pi*frq*dt
    x, y = absl*np.cos(ang), absl*np.sin(ang)
    neutral = absl > 0.995
    ax_b.scatter(x[~neutral], y[~neutral], s=16, marker=mk[c], facecolors="none",
                 edgecolors=C_GREY, linewidths=0.8, zorder=3)
    ax_b.scatter(x[neutral], y[neutral], s=26, marker=mk[c], color=C_DMD,
                 zorder=4, label=LABEL[c].replace("\n", " "))

ax_b.set_xlim(-1.25, 1.25); ax_b.set_ylim(-1.25, 1.25)
ax_b.set_aspect("equal", adjustable="box")
ax_b.set_xlabel(r"$\mathrm{Re}\,\lambda$")
ax_b.set_ylabel(r"$\mathrm{Im}\,\lambda$")
ax_b.set_xticks([-1, 0, 1]); ax_b.set_yticks([-1, 0, 1])
ax_b.spines["top"].set_visible(True); ax_b.spines["right"].set_visible(True)
leg_b = [Line2D([], [], marker=mk[c], color=C_DMD, ls="none", ms=4,
                label=LABEL[c].replace("\n", " ")) for c in CASES if c in DATA]
leg_b.append(Line2D([], [], marker="o", color=C_GREY, ls="none", ms=4,
                    markerfacecolor="none", label="decaying modes"))
ax_b.legend(handles=leg_b, loc="lower left", frameon=False, fontsize=5.9,
            handletextpad=0.3, labelspacing=0.22, borderpad=0.15,
            borderaxespad=0.25)
ax_b.set_title("DMD spectrum", pad=4)
panel_tag(ax_b, "(b)")

# ============================================================ (c)/(d) bar panels
def bars(ax, space, ylabel, title):
    cs = [c for c in CASES if c in DATA]
    n = len(cs); w = 0.26
    xs = np.arange(n)
    vals = {"cor": [], "dmd": [], "har": []}
    for c in cs:
        J = DATA[c]["json"]
        if space == "latent":
            S = J["latent_scores"]
            # lstm_phase_true is the genuine phase correction (from phasecorr
            # I_pha); lstm_phase_corrected in the original JSON is the sinusoid
            key = "lstm_phase_true" if "lstm_phase_true" in S else "lstm_phase_corrected"
            vals["cor"].append(S[key]["ratio_to_lstm"])
            vals["dmd"].append(S[best_dmd_key(J, "latent")]["ratio_to_lstm"])
            vals["har"].append(S["harmonic"]["ratio_to_lstm"])
        else:
            F = J["field"]; base = F["lstm_uncorrected"]["rmse"]
            # F["lstm_phase_corrected"] is the SINUSOID baseline, not the phase
            # correction (same mislabel as latent space). The genuine field-space
            # phase result lives in fieldval_{tag}.npy as i_pha_f, a percentage
            # improvement over the uncorrected rollout; the bar wants a ratio.
            fv = np.load(os.path.join(DATA_DIR, f"fieldval_Re{c[0]}_L{c[1]}.npy"),
                         allow_pickle=True).item()
            assert int(fv["ncal"]) == 150, f"fieldval ncal={fv['ncal']}, expected 150"
            vals["cor"].append(1.0 - float(fv["i_pha_f"])/100.0)
            vals["dmd"].append(F[best_dmd_key(J, "field")]["rmse"]/base)
            vals["har"].append(F["harmonic"]["rmse"]/base)

    ax.bar(xs - w, vals["cor"], w, color=C_COR, label="CAE--LSTM + phase corr.")
    ax.bar(xs,     vals["dmd"], w, color=C_DMD, label="CAE + DMD (best variant)")
    ax.bar(xs + w, vals["har"], w, color=C_HAR, label="CAE + harmonic fit")
    ax.axhline(1.0, color=C_UNC, lw=1.1, ls="--", zorder=1)

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[c] for c in cs], fontsize=6.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    top = max(max(vals["har"]), 1.0)
    ax.set_ylim(0.18, top*5.5)
    # value labels above each bar, in error-ratio units
    for xarr, v in [(xs-w, vals["cor"]), (xs, vals["dmd"]), (xs+w, vals["har"])]:
        for xi, vi in zip(xarr, v):
            ax.text(xi, vi*1.10, f"{vi:.2f}" if vi < 10 else f"{vi:.0f}",
                    ha="center", va="bottom", fontsize=5.9, rotation=90)
    return ax

bars(ax_c, "latent", r"error / uncorrected", "latent space")
panel_tag(ax_c, "(c)")
bars(ax_d, "field",  r"field RMSE / uncorrected", "physical space")
panel_tag(ax_d, "(d)")
ref = Line2D([], [], color=C_UNC, ls="--", lw=1.1, label="uncorrected CAE--LSTM")
h, l = ax_c.get_legend_handles_labels()
ax_c.legend(handles=h + [ref], loc="upper left", frameon=False, handlelength=1.4,
            borderpad=0.2, labelspacing=0.25, handletextpad=0.5, fontsize=6.2)

# ----------------------------------------------------------------- save
out = args.outfile if os.path.isabs(args.outfile) else os.path.join(DATA_DIR, args.outfile)
fig.savefig(out, dpi=args.dpi)                       # NOTE: no bbox_inches='tight'
fig.savefig(out.replace(".pdf", ".png"), dpi=args.dpi)
print(f"  wrote {out}", flush=True)
print(f"  wrote {out.replace('.pdf','.png')}", flush=True)

# ----------------------------------------------------------------- numbers for the caption
print("\n  numbers for the caption / table:", flush=True)
print(f"  {'case':<12}{'corr':>7}{'DMD':>8}{'harm':>8}   {'St_DMD':>8}{'St_paper':>9}", flush=True)
paper_St = {300: 0.213, 400: 0.219, 800: 0.232, 101: 0.148}
for c in CASES:
    if c not in DATA: continue
    J = DATA[c]["json"]; S = J["latent_scores"]; e = J["dmd_eigs"]
    i = int(np.argmax(e["abs"]))
    print(f"  {LABEL[c].replace(chr(10),' '):<12}"
          f"{(S['lstm_phase_true'] if 'lstm_phase_true' in S else S['lstm_phase_corrected'])['ratio_to_lstm']:>7.2f}"
          f"{S[best_dmd_key(J,'latent')]['ratio_to_lstm']:>8.2f}"
          f"{S['harmonic']['ratio_to_lstm']:>8.2f}   "
          f"{e['St'][i]:>8.4f}{paper_St.get(c[0], float('nan')):>9.3f}", flush=True)
