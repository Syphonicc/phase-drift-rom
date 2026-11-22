"""Highlight (table-of-contents) image for the PoF revision, built from the same data
# v2: writes only highlight_image.{png,pdf}; does not rewrite field_timing_check_v2.json
and decoding as field_timing_check_v2.py: CFD vorticity at the end of the Re=400 rollout,
with the vorticity of the network's decoded error overlaid as contours."""

import os, json, numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import hilbert
from paths import DATA as H, OUT_REV as OUT
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
print(json.dumps(res, indent=1))  # v2: no results file written

# ---- highlight (TOC) image: CFD vorticity street with the network's timing error as contours
import matplotlib.colors as mcolors
def vort(F): dx=(x[1]-x[0]); dy=(y[1]-y[0]); return np.gradient(F[1],dx,axis=1)-np.gradient(F[0],dy,axis=0)
i=n-3; Wc=vort(Fcfd[i]); We=vort(Edec[i]); X,Y=np.meshgrid(x,y)
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm"})
fig,ax=plt.subplots(figsize=(8,3.2))
vm=np.percentile(np.abs(Wc),99)
ax.imshow(Wc,cmap="RdBu_r",vmin=-vm,vmax=vm,origin="lower",extent=[x[0],x[-1],y[0],y[-1]],interpolation="bicubic",alpha=0.9)
lev=np.percentile(np.abs(We),97)*np.array([-1,-0.5,0.5,1])
ax.contour(X,Y,We,levels=lev,colors=["#1f1f1e"],linewidths=0.8,linestyles=["--","--","-","-"])
ax.add_patch(plt.Circle((0,0),0.5,color="#bdbcb5",zorder=3))
ax.set_xlim(-2,9.3); ax.set_ylim(-2.2,2.2); ax.set_aspect("equal"); ax.axis("off")
ax.text(-1.9,1.85,"vortex street (CFD)",fontsize=10,color="#1f1f1e")
ax.text(-1.9,-2.05,"contours: error of the learned model $\\approx\\ \\delta t\\,\\partial_t\\omega$, a time lag of 0.3% of a shedding period",fontsize=9,color="#1f1f1e")
fig.savefig(f"{OUT}/highlight_image.png",dpi=300,bbox_inches="tight",pad_inches=0.05); fig.savefig(f"{OUT}/highlight_image.pdf",bbox_inches="tight",pad_inches=0.05)
print("highlight written")
