#!/usr/bin/env python3
"""R2.2: the wake physics behind the clock.  Force coefficients of every circular-cylinder
case (production set-up, t=50-90 unless stated), compared with the two-dimensional DNS
relation St = 0.2698 - 1.0271 Re^-1/2 (Jiang & Cheng) already used in the paper, plus the
start-up relaxation of the lift amplitude onto the limit cycle (factor per shedding period
by which the amplitude deficit decreases, fitted while it falls from 35% to 0.5%).
Writes forces_vs_re.json and fig_forces_vs_re_v2.{pdf,png} into $PDROM_OUT_DIR (default: $PDROM_REVISION_DIR).

v2 (2026-11-24): case directories use the names of 01_generate_data/circular_cylinder/revision
(Re100, forces_only/ReNNN, forces_prod/Re400|Re800, Re1000, Re800_finemesh, Re1000_finemesh,
Re400_outlet25D, Re400_walls10D), all under $PDROM_CASES_DIR; the script stops with an error
if any case is missing instead of skipping it.  Figure: upright Re, fonts >= 7 pt at 7.0 in,
legend of (b) moved off the data.  Data and numbers are unchanged from v1."""
import os, sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from forces import load, stats
from paths import CASES as R, OUT_REV
BASE = {100: f"{R}/Re100", 300: f"{R}/forces_only/Re300", 400: f"{R}/forces_prod/Re400", 500: f"{R}/forces_only/Re500",
        600: f"{R}/forces_only/Re600", 700: f"{R}/forces_only/Re700", 800: f"{R}/forces_prod/Re800", 1000: f"{R}/Re1000"}
EXTRA = {"fine mesh": {800: f"{R}/Re800_finemesh", 1000: f"{R}/Re1000_finemesh"},
         "outlet at 25D": {400: f"{R}/Re400_outlet25D"}, "5% blockage": {400: f"{R}/Re400_walls10D"}}
_missing = [c for c in list(BASE.values()) + [c for d in EXTRA.values() for c in d.values()] if not os.path.isdir(c)]
if _missing:
    sys.exit("make_fig_forces_v2: missing solved case directories (set PDROM_CASES_DIR):\n  " + "\n  ".join(_missing))
WIN = {100: (210, 250)}

def relax(case):
    t, cd, cl = load(case); s = cl - np.median(cl[t > t.max() - 20])
    i = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]; tc = t[i] - s[i] * (t[i + 1] - t[i]) / (s[i + 1] - s[i])
    A = np.array([0.5 * (s[(t >= a) & (t < b)].max() - s[(t >= a) & (t < b)].min()) for a, b in zip(tc[:-1], tc[1:])])
    Ai = np.median(A[-20:]); g = (Ai - A) / Ai; j0 = np.argmax(A > 0.5 * Ai)
    k = [j for j in range(j0, len(g)) if 0.005 < g[j] < 0.35]
    k = [j for j in k if j - k[0] == k.index(j)]          # contiguous run only
    if len(k) < 3: return None, t, cl
    return float(np.exp(np.polyfit(np.arange(len(k)), np.log(g[k]), 1)[0])), t, cl

out = {"base": {}, "extra": {}}
for re, c in BASE.items():
    t0, t1 = WIN.get(re, (50, 90)); t, cd, cl = load(c); st = stats(t, cd, cl, t0, t1)
    q, *_ = relax(c); st["relax_per_period"] = q; out["base"][re] = st
for lab, d in EXTRA.items():
    for re, c in d.items():
        t, cd, cl = load(c); st = stats(t, cd, cl, 50, 90); st["relax_per_period"] = relax(c)[0]; out["extra"].setdefault(lab, {})[re] = st
json.dump(out, open(os.path.join(OUT_REV, "forces_vs_re.json"), "w"), indent=1)
for re, s in out["base"].items(): print(re, {k: round(s[k], 4) for k in ("St", "Cd_mean", "Cl_rms", "T_cv")}, "relax", s["relax_per_period"])
for lab, d in out["extra"].items():
    for re, s in d.items(): print(lab, re, {k: round(s[k], 4) for k in ("St", "Cd_mean", "Cl_rms", "T_cv")}, "relax", s["relax_per_period"])

INK, MUT, GRID, C1, C2, C3 = "#1f1f1e", "#6b6a64", "#e4e3dd", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif", "mathtext.fontset": "cm", "axes.edgecolor": MUT, "xtick.color": MUT, "ytick.color": MUT, "axes.linewidth": 0.6})
fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.1), gridspec_kw=dict(wspace=0.55))
t, cd, cl = load(BASE[400]); k = (t > 0) & (t < 12)
ax[0].plot(t[k], cl[k], color=C1, lw=0.6, label="$C_L$"); ax[0].plot(t[k], cd[k], color=C2, lw=0.8, label="$C_D$")
ax[0].set_xlabel("$t$"); ax[0].set_ylabel("force coefficient"); ax[0].set_title(r"(a) start-up, $\mathrm{Re}=400$", fontsize=8); ax[0].set_ylim(-1.6, 2.6); ax[0].legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
re = np.array(sorted(out["base"])); per = lambda key: np.array([out["base"][r][key] for r in re])
rr = np.linspace(250, 1050, 200); ax[1].plot(rr, 0.2698 - 1.0271 / np.sqrt(rr), color=MUT, lw=0.9, ls="--", label="2D DNS fit")
reg = re != 1000
ax[1].plot(re[reg], per("St")[reg], "o", color=C1, ms=4, label="baseline mesh"); ax[1].plot(re[~reg], per("St")[~reg], "o", mfc="white", color=C1, ms=4)
mk = {"fine mesh": ("s", C2), "outlet at 25D": ("^", C3), "5% blockage": ("v", INK)}
for lab, d in out["extra"].items():
    m, c = mk[lab]; ax[1].plot(list(d), [d[r]["St"] for r in d], m, color=c, ms=4, label=lab)
ax[1].set_xlabel(r"$\mathrm{Re}$"); ax[1].set_ylabel("St"); ax[1].set_title("(b) Strouhal number", fontsize=8); _h, _l = ax[1].get_legend_handles_labels(); ax[1].legend(_h[:1], _l[:1], fontsize=7, frameon=False, loc="lower right")
ax[3].legend(_h[1:], _l[1:], fontsize=7, frameon=False, loc="upper right", handletextpad=0.2, labelspacing=0.3)  # marker key for (b)-(d), in empty area of (d)
ax[2].plot(re, per("Cd_mean"), "o-", color=C2, ms=3.5, lw=0.8, label=r"$\overline{C_D}$"); ax[2].plot(re, per("Cl_rms"), "o-", color=C1, ms=3.5, lw=0.8, label=r"$C_L'$")
for lab, d in out["extra"].items():
    m, c = mk[lab]; ax[2].plot(list(d), [d[r]["Cd_mean"] for r in d], m, color=c, ms=4, mfc="white"); ax[2].plot(list(d), [d[r]["Cl_rms"] for r in d], m, color=c, ms=4, mfc="white")
ax[2].set_xlabel(r"$\mathrm{Re}$"); ax[2].set_title("(c) force coefficients", fontsize=8); ax[2].legend(fontsize=7, frameon=False)
q = [(r, out["base"][r]["relax_per_period"]) for r in re if out["base"][r]["relax_per_period"]]
ax[3].plot([a for a, b in q], [b for a, b in q], "o-", color=C1, ms=3.5, lw=0.8)
for lab, d in out["extra"].items():
    m, c = mk[lab]; ax[3].plot([r for r in d if d[r]["relax_per_period"]], [d[r]["relax_per_period"] for r in d if d[r]["relax_per_period"]], m, color=c, ms=4)
ax[3].set_xlabel(r"$\mathrm{Re}$"); ax[3].set_ylabel("factor per period"); ax[3].set_title("(d) amplitude relaxation", fontsize=8); ax[3].set_ylim(0, 1)
for a in ax:
    a.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.savefig(os.path.join(OUT_REV, "fig_forces_vs_re_v2.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT_REV, "fig_forces_vs_re_v2.png"), dpi=200, bbox_inches="tight")
