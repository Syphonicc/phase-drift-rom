#!/usr/bin/env python3
"""v3 (2026-11-24): presentation only -- en dash in "CAE–LSTM", U_inf in labels, sized to the printed
width (0.85 textwidth = 5.95 in) with 7-8 pt fonts. ALL outputs (json, pdf) go to the --out directory.
v2: colour bar for the top row added.
R2.2/R2.3: the rollout error in PHYSICAL space is a timing error of the wake.
Representative circular-cylinder Re=400 network (rollout_Re400.npy, cae_Re400.pt).
  e_dec(x,t)  = D(z_pred) - D(z_true)                 (decoder-consistent; CAE floor removed)
  delta_t(t)  = phi(t) / (2 pi f_s)                    phi: coordinate-mean accumulated latent phase,
                                                       referenced to its calibration-fit intercept
  e_tim(x,t)  = delta_t(t) * dD(z_true)/dt             (central difference)
Reports, over the last tenth of the rollout, the fraction of ||e_dec||^2 explained by e_tim,
and draws vorticity: CFD truth, prediction, e_dec and e_tim at the final step.
"""
import argparse, os, json, numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import hilbert
H = os.path.expanduser("~/cylinder_lstm_data"); _ap = argparse.ArgumentParser(); _ap.add_argument("--out", required=True); OUT = _ap.parse_args().out; os.makedirs(OUT, exist_ok=True)
DT, NCAL, D = 0.03, 150, 0.12
FLAT = 64 * 8 * 16
class Decoder(nn.Module):
    def __init__(s, ld):
        super().__init__(); s.f = nn.Linear(ld, FLAT)
        s.d = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1))
    def forward(s, z): return s.d(s.f(z).view(-1, 64, 8, 16))
sd = torch.load(f"{H}/cae_Re400.pt", map_location="cpu")
sd = {k.replace("decoder.fc.", "decoder.f.").replace("decoder.deconv.", "decoder.d."): v for k, v in sd.items() if k.startswith("decoder")}
sd = {k[len("decoder."):]: v for k, v in sd.items()}
dec = Decoder(4); dec.load_state_dict(sd); dec.eval()
cn = np.load(f"{H}/norm_Re400.npy", allow_pickle=True).item(); lnm = np.load(f"{H}/latent_norm_Re400.npy", allow_pickle=True).item()
r = np.load(f"{H}/rollout_Re400.npy", allow_pickle=True).item(); zp, zt = r["raw_pred"].astype(float), r["true"].astype(float)
n = len(zt)
def decode(z):
    zz = torch.tensor(z * lnm["std"] + lnm["mean"], dtype=torch.float32)
    with torch.no_grad(): return np.concatenate([dec(zz[i:i + 64]).numpy() for i in range(0, len(zz), 64)]) * cn["std"] + cn["mean"]
Fp, Ft = decode(zp), decode(zt)
snaps = np.load(f"{H}/snapshots_Re400.npy", mmap_mode="r"); N = len(snaps); ts = int(N * 0.52) + int(N * 0.08)
Fcfd = np.asarray(snaps[ts:ts + n], float)
meta = np.load(f"{H}/metadata_Re400.npy", allow_pickle=True).item(); x, y = meta["x_grid"] / D, meta["y_grid"] / D
# timing from the latent phase
t = np.arange(n) * DT
ph = lambda s: np.unwrap(np.angle(hilbert(s - s.mean())))
phi = np.stack([ph(zp[:, d]) - ph(zt[:, d]) for d in range(4)], 1)
phi = np.stack([phi[:, d] - np.polyfit(t[:NCAL], phi[:NCAL, d], 1)[1] for d in range(4)], 1).mean(1)
fs = np.mean([np.polyfit(t, ph(zt[:, d]), 1)[0] for d in range(4)]) / (2 * np.pi)
dtl = phi / (2 * np.pi * fs)                                    # >0: prediction ahead of the flow
dF = np.gradient(Ft, DT, axis=0)
Etim = dtl[:, None, None, None] * dF; Edec = Fp - Ft
k = slice(int(0.9 * n), n - 2)
expl = 1 - np.sum((Edec[k] - Etim[k]) ** 2) / np.sum(Edec[k] ** 2)
# a lag fitted by least squares at each step, for comparison with the phase-derived one
dt_ls = np.array([np.sum(Edec[i] * dF[i]) / np.sum(dF[i] ** 2) for i in range(n)])
corr_dt = np.corrcoef(dt_ls[NCAL:n - 2], dtl[NCAL:n - 2])[0, 1]
floor = np.sqrt(np.mean((Ft[k] - Fcfd[k]) ** 2)); edec_rms = np.sqrt(np.mean(Edec[k] ** 2))
res = dict(f_s=float(fs), St=float(fs * D), delta_t_end=float(dtl[n - 3]), delta_t_end_over_T=float(dtl[n - 3] * fs),
           explained_fraction_last10pct=float(expl), corr_phase_vs_ls_lag=float(corr_dt),
           edec_rms_last10pct=float(edec_rms), cae_floor_rms_last10pct=float(floor))
json.dump(res, open(f"{OUT}/field_timing_check_v3.json", "w"), indent=1); print(json.dumps(res, indent=1))
# ---- figure: vorticity
def vort(F): dx = (x[1] - x[0]); dy = (y[1] - y[0]); return np.gradient(F[1], dx, axis=1) - np.gradient(F[0], dy, axis=0)
i = n - 3
W = [vort(Fcfd[i]), vort(Fp[i]), vort(Edec[i]), vort(Etim[i])]
X, Y = np.meshgrid(x, y)
plt.rcParams.update({"font.size": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif", "mathtext.fontset": "cm"})
fig, ax = plt.subplots(2, 2, figsize=(5.95, 2.95), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.28, wspace=0.08))
titles = [r"(a) CFD, $\omega_z D/U_\infty$", "(b) CAE–LSTM rollout, " + r"$\omega_z D/U_\infty$",
          r"(c) error of the rollout, $\omega_z(\hat{q})-\omega_z(q)$", r"(d) timing model, $\delta t\,\partial_t\omega_z$"]
vm = np.percentile(np.abs(W[0]), 99); ve = np.percentile(np.abs(W[2]), 99.5)
for a, w, tt, v in zip(ax.flat, W, titles, [vm, vm, ve, ve]):
    m = a.pcolormesh(X, Y, w, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto", rasterized=True)
    a.add_patch(plt.Circle((0, 0), 0.5, color="#bdbcb5", zorder=3)); a.set_aspect("equal"); a.set_title(tt, fontsize=7.5, pad=3)
    a.set_xlim(x[0], x[-1]); a.set_ylim(y[0], y[-1])
for a in ax[1]: a.set_xlabel(r"$x/D$")
for a in ax[:, 0]: a.set_ylabel(r"$y/D$")
fig.colorbar(ax[0,1].collections[0], ax=ax[0, :], shrink=0.8, pad=0.01, label=r"$\omega_z D/U_\infty$")
fig.colorbar(m, ax=ax[1, :], shrink=0.8, pad=0.01, label=r"$\omega_z D/U_\infty$ (error)")
fig.savefig(f"{OUT}/fig_field_timing_v3.pdf", bbox_inches="tight", dpi=300)
