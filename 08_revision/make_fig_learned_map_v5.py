#!/usr/bin/env python3
"""v5: CFD reference lines use the production-mesh relaxation factors from
make_fig_forces.py (Re=100: 0.47, Re=400: 0.375); v4 used 0.46 and 0.32 (the latter from the
25D-domain case).
v4: cycle windows of fixed length round(period) samples (v3 rounded both ends, so
some windows were one sample short of a period, giving a spike at cycle 4).
v3: panel (c) amplitude recovery measured as the cycle-RMS amplitude (removes the
intra-cycle wobble of the Hilbert envelope that produced a spurious 1% floor in v2),
median over all 40 networks; CFD start-up relaxation at Re=100 and 400 as references.
v2: panel (c) uses whole-window kicks (amplitude kick: Hilbert-envelope deviation;
time-shift kick: phase offset), with the CFD start-up amplitude relaxation at Re=400
(factor 0.32 per period) as a reference.
R3.8 + R4.1 figure from clock_perturb_Re500_40seeds.json, plus one network's
perturbation time series (seed_000, Re=500) recomputed here.
(a) drift rate predicted from the free-running LSTM vs drift measured in the rollout
(b) efficacy of the correction using the free-run rate vs the calibration-window rate
(c) |transverse| and |tangential| difference after a kick (eps=0.1) across / along the cycle"""
import os, sys, json, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x", "none", "--out", "/dev/null"]
exec(open(os.path.join(HERE, "clock_freerun_perturb_v2.py")).read().split("res = []")[0])
from paths import DATA, ENS, OUT_ENS
r = json.load(open(os.path.join(ENS, "clock_perturb_v2_Re500_40seeds.json")))
wp = np.array([x["w_pred"] for x in r]) * 1e3; wr = np.array([x["w_roll"] for x in r]) * 1e3
Ic = np.array([x["I_cal"] for x in r]); If = np.array([x["I_free"] for x in r])
pre = os.path.join(DATA, "seed_sweep_v3_Re500", "seed_000"); a = np.load(pre + "_art.npz")
m = LSTMP(4); m.load_state_dict(torch.load(pre + "_lstm.pt", map_location="cpu")); m.eval()
ln = ((a["latents"] - a["lmean"]) / a["lstd"]).astype(np.float32); ts = int(a["test_start"]); pred = a["pred"].astype(np.float32)
s0 = 60; win0 = np.concatenate([ln[ts - 20:ts], pred])[s0:s0 + 20].copy(); ref = pred[s0:]; n = len(ref)
tang = np.gradient(ref, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True); per = 1 / (r[0]["f_true_train"] * 0.03)
import glob
eps = 0.1; DEV, PH = [], []
for lp in sorted(glob.glob(os.path.join(DATA, "seed_sweep_v3_Re500", "seed_*_lstm.pt"))):
    pre_ = lp[:-8]; a_ = np.load(pre_ + "_art.npz"); m_ = LSTMP(4); m_.load_state_dict(torch.load(lp, map_location="cpu")); m_.eval()
    ln_ = ((a_["latents"] - a_["lmean"]) / a_["lstd"]).astype(np.float32); ts_ = int(a_["test_start"]); pr_ = a_["pred"].astype(np.float32)
    ntr_ = int(a_["ntrain"]); full_ = np.concatenate([ln_[ts_ - 20:ts_], pr_]); w0 = full_[s0:s0 + 20].copy(); rf = pr_[s0:]; nn2 = len(rf)
    mu_ = ln_[:ntr_].mean(0); dw = np.gradient(full_, axis=0)[s0:s0 + 20]
    per_ = 2 * np.pi / (np.mean([np.polyfit(np.arange(ntr_) * 0.03, phase(ln_[:ntr_, d]), 1)[0] for d in range(4)]) * 0.03)
    pa = rollout(m_, (mu_ + (1 + eps) * (w0 - mu_)).astype(np.float32), nn2)
    pt = rollout(m_, (w0 + (eps / np.linalg.norm(dw[-1])) * dw).astype(np.float32), nn2)
    L_ = int(round(per_)); crms = lambda z, c: np.sqrt(((z[int(round(c * per_)):int(round(c * per_)) + L_] - rf.mean(0)) ** 2).mean(0))
    DEV.append([np.mean(np.abs(crms(pa, c) / crms(rf, c) - 1)) / eps for c in range(12)])
    thr_ = np.stack([phase(rf[:, d]) for d in range(4)], 1); dph = (np.stack([phase(pt[:, d]) for d in range(4)], 1) - thr_).mean(1)
    cm = lambda c: dph[int(round(c * per_)):int(round(c * per_)) + L_].mean()
    PH.append([cm(c) / cm(1) for c in range(1, 12)])
DEV = np.array(DEV); PH = np.array(PH)
json.dump(dict(amp_dev_per_cycle_median=np.median(DEV, 0).tolist(), amp_dev_per_cycle_max=DEV.max(0).tolist(),
               phase_persist_median=np.median(PH, 0).tolist(), phase_persist_min=PH.min(0).tolist(), phase_persist_max=PH.max(0).tolist()),
          open(os.path.join(OUT_ENS, "learned_map_panel_c_v5.json"), "w"), indent=1)
cyc = np.arange(n) / per
INK, MUT, GRID, C1, C2 = "#1f1f1e", "#6b6a64", "#e4e3dd", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm", "axes.edgecolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.linewidth": 0.6})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(wspace=0.42))
lim = 1.1 * max(abs(wp).max(), abs(wr).max())
ax[0].plot([-lim, lim], [-lim, lim], color=MUT, lw=0.7, ls="--"); ax[0].scatter(wr, wp, s=14, color=C1, edgecolor="white", lw=0.5, zorder=3)
ax[0].set_xlabel(r"$\omega_d$ measured in rollout [$10^{-3}$ rad/$t$]"); ax[0].set_ylabel(r"$2\pi(f_{\rm free}-f_s)$ [$10^{-3}$ rad/$t$]")
ax[0].set_title("(a) drift predicted from free run", fontsize=8)
lo = min(Ic.min(), If.min()) - 10
ax[1].plot([lo, 100], [lo, 100], color=MUT, lw=0.7, ls="--"); ax[1].axhline(0, color=GRID, lw=0.6); ax[1].axvline(0, color=GRID, lw=0.6)
ax[1].scatter(Ic, If, s=14, color=C1, edgecolor="white", lw=0.5, zorder=3)
ax[1].set_xlabel("calibration-window correction [%]"); ax[1].set_ylabel("free-run correction [%]"); ax[1].set_title("(b) error removed", fontsize=8)
cc = np.arange(12)
ax[2].semilogy(cc, np.median(DEV, 0), "o-", color=C1, ms=3, lw=1.2, label="LSTM, amplitude kick: amplitude deviation")
ax[2].fill_between(cc, DEV.min(0), DEV.max(0), color=C1, alpha=0.15, lw=0)
ax[2].semilogy(np.arange(1, 12), np.median(PH, 0), "s-", color=C2, ms=3, lw=1.2, label="LSTM, time-shift kick: phase offset")
for q, lab, ls in [(0.47, "CFD amplitude relaxation, $Re=100$", ":"), (0.375, "CFD amplitude relaxation, $Re=400$", "--")]:
    ax[2].semilogy(cc, q ** cc, color=MUT, lw=0.9, ls=ls, label=lab)
ax[2].set_ylim(1e-4, 3); ax[2].set_xlim(0, 11); ax[2].set_xlabel("shedding cycles after the kick"); ax[2].set_ylabel("normalised response")
ax[2].set_title("(c) response to a perturbation", fontsize=8); ax[2].legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
for a_ in ax:
    a_.grid(True, color=GRID, lw=0.5); a_.set_axisbelow(True)
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
fig.savefig(os.path.join(OUT_ENS, "fig_learned_map_v5.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT_ENS, "fig_learned_map_v5.png"), dpi=200, bbox_inches="tight")
print("ok")
