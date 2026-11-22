#!/usr/bin/env python3
"""v2: figure uses per-cycle RMS of the error components and trims one period at each
end of phi (Hilbert end effects); numbers unchanged.
R4.2: phase drift over a ~260-cycle rollout (S8: fixed CAE, LSTM seeds 0-4, trained on the
standard window, rolled out against the continued CFD to t=290).
Per network: coordinate-mean phase phi(t) (Hilbert, referenced to its value at the start);
linear fit over the whole rollout (slope, R^2); drift rate in the first and last 30 cycles;
transverse/tangential magnitudes in the first and last tenth; error norm vs cycles;
free-run clock prediction 2*pi*(f_free - f_s) (training data + network only) and the
efficacy of the phase correction using it over the whole long horizon.
Writes long_rollout.json and fig_long_rollout.{pdf,png}."""
import os, sys, glob, json, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import hilbert
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x", "none", "--out", "/dev/null"]
exec(open(os.path.join(HERE, "clock_freerun_perturb_v2.py")).read().split("res = []")[0])
from paths import ENS, OUT_ENS
D = os.path.join(ENS, "S8_longroll"); DT = 0.03; res = []; curves = []
for f in sorted(glob.glob(os.path.join(D, "seed_*_art.npz"))):
    a = np.load(f); pred = a["pred"].astype(float); true = a["true"].astype(float); n, ld = true.shape
    ln = ((a["latents"] - a["lmean"]) / a["lstd"]).astype(np.float32); ntr = int(a["ntrain"]); tau = int(a["tau"])
    t = np.arange(n) * DT
    th_t = np.stack([phase(true[:, d]) for d in range(ld)], 1); th_p = np.stack([phase(pred[:, d]) for d in range(ld)], 1)
    fs = np.mean([np.polyfit(t, th_t[:, d], 1)[0] for d in range(ld)]) / (2 * np.pi); per = 1 / (fs * DT); cyc = t * fs
    phi = (th_p - th_t); phi = (phi - phi[:int(per)].mean(0)).mean(1)
    k = slice(int(per), n - int(per))
    c = np.polyfit(t[k], phi[k], 1); r2 = 1 - np.var(phi[k] - np.polyval(c, t[k])) / np.var(phi[k])
    q = np.polyfit(t[k] - t[k].mean(), phi[k], 2)
    w30 = lambda a0, a1: np.polyfit(t[(cyc >= a0) & (cyc < a1)], phi[(cyc >= a0) & (cyc < a1)], 1)[0]
    w_first, w_last = w30(1, 31), w30(cyc[-1] - 31, cyc[-1] - 1)
    # tangential / transverse
    tang = np.gradient(true, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True); e = pred - true
    ep = np.abs(np.einsum("ij,ij->i", e, tang)); eq = np.linalg.norm(e - np.einsum("ij,ij->i", e, tang)[:, None] * tang, axis=1)
    m10 = n // 10
    # free-run clock (network + training data only)
    sd = torch.load(f.replace("_art.npz", "_lstm.pt"), map_location="cpu"); m = LSTMP(ld); m.load_state_dict(sd); m.eval()
    f_tr = np.mean([np.polyfit(np.arange(ntr)[int(per):-int(per)] * DT, phase(ln[:ntr, d].astype(float))[int(per):-int(per)], 1)[0] for d in range(ld)]) / (2 * np.pi)
    free = rollout(m, ln[ntr - tau:ntr].astype(np.float32), 4000)[200:].astype(float)
    f_free = np.mean([np.polyfit(np.arange(len(free))[int(per):-int(per)] * DT, phase(free[:, d])[int(per):-int(per)], 1)[0] for d in range(ld)]) / (2 * np.pi)
    w_pred = 2 * np.pi * (f_free - f_tr)
    Ap = np.abs(hilbert(pred - pred.mean(0), axis=0)); mp = pred.mean(0)
    corr = np.stack([Ap[:, d] * np.cos(th_p[:, d] - w_pred * t) + mp[d] for d in range(ld)], 1)
    eu = np.linalg.norm(e, axis=1)[150:].mean(); ec = np.linalg.norm(corr - true, axis=1)[150:].mean()
    r = dict(seed=os.path.basename(f)[:8], n_cycles=float(cyc[-1]), slope=float(c[0]), r2_linear=float(r2),
             quad_change_frac=float(2 * q[0] * (t[k][-1] - t[k][0]) / c[0]) if c[0] != 0 else None,
             w_first30=float(w_first), w_last30=float(w_last), phi_end=float(phi[-int(per)]),
             share_tan=float(np.sum(ep[5:-5] ** 2) / np.sum(ep[5:-5] ** 2 + eq[5:-5] ** 2)),
             tan_first=float(ep[:m10].mean()), tan_last=float(ep[-m10:].mean()), perp_first=float(eq[:m10].mean()), perp_last=float(eq[-m10:].mean()),
             w_pred_free=float(w_pred), I_free_long=float(100 * (1 - ec / eu)))
    res.append(r); curves.append((cyc, phi, np.linalg.norm(e, axis=1), eq, c))
    print({k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in r.items()}, flush=True)
json.dump(res, open(os.path.join(OUT_ENS, "long_rollout_v2.json"), "w"), indent=1)
# figure
INK, MUT, GRID, C1, C2 = "#1f1f1e", "#6b6a64", "#e4e3dd", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm", "axes.edgecolor": MUT, "xtick.color": MUT, "ytick.color": MUT, "axes.linewidth": 0.6})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.3))
for i, (cyc, phi, en, eq, c) in enumerate(curves):
    P = int(round(1 / (cyc[1]))) ; kk = slice(P, len(cyc) - P)
    ax[0].plot(cyc[kk], phi[kk], color=C1, lw=0.6, alpha=0.8); ax[0].plot(cyc[kk], np.polyval(c, cyc[kk] * DT / cyc[1]), color=INK, lw=0.6, ls="--")
    nc = int(cyc[-1]); cc = np.arange(nc) + 0.5
    rms = lambda x: np.array([np.sqrt(np.mean(x[(cyc >= j) & (cyc < j + 1)] ** 2)) for j in range(nc)])
    ax[1].semilogy(cc, rms(en), color=C1, lw=0.9, label="total error" if i == 0 else None)
    ax[1].semilogy(cc, rms(eq), color=C2, lw=0.9, label="transverse component" if i == 0 else None)
ax[0].axhline(0, color=MUT, lw=0.5); ax[0].set_xlabel("shedding cycles"); ax[0].set_ylabel(r"$\phi(t)$ [rad]")
ax[0].set_title("(a) accumulated phase, five networks", fontsize=8)
ax[1].set_xlabel("shedding cycles"); ax[1].set_ylabel("latent error"); ax[1].set_title("(b) error magnitude", fontsize=8); ax[1].legend(fontsize=6.5, frameon=False)
for a_ in ax:
    a_.grid(True, color=GRID, lw=0.5); a_.set_axisbelow(True)
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
fig.savefig(os.path.join(OUT_ENS, "fig_long_rollout_v2.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT_ENS, "fig_long_rollout_v2.png"), dpi=200, bbox_inches="tight")
