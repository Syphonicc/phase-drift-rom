#!/usr/bin/env python3
"""
field_validation.py
===================
THE LAST SCRIPT.

Does the phase correction's latent-space improvement survive decoding to the
physical flow field? A fluids reviewer cares only about this.

Decodes three latent trajectories through the trained CAE and compares each
against the original OpenFOAM snapshots:

    - true latents          -> the CAE FLOOR (best achievable; compression limit)
    - uncorrected rollout   -> the baseline
    - sinusoid-corrected    -> the old method
    - phase-corrected       -> the new method

Reports field RMSE, relative field error, and recomputes rho against each.

Usage:
    python3 field_validation.py --re 400 --latent 4
    python3 field_validation.py --re 101 --latent 4 --dt 0.0438
    python3 field_validation.py --re 800 --latent 8
"""

import os, argparse, numpy as np
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--re",     type=int,   required=True)
ap.add_argument("--latent", type=int,   default=4)
ap.add_argument("--dt",     type=float, default=0.03)
ap.add_argument("--calib",  type=int,   default=150)
args = ap.parse_args()

RE, LD, DT, NCAL = args.re, args.latent, args.dt, args.calib
DATA_DIR = os.path.expanduser("~/cylinder_lstm_data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find(c):
    for x in c:
        p = os.path.join(DATA_DIR, x)
        if os.path.exists(p): return p
    raise FileNotFoundError(f"none of {c}")

# square cylinder snapshots are tagged SQ100, not Re101
SNAP = f"snapshots_SQ100.npy" if RE == 101 else f"snapshots_Re{RE}.npy"

roll  = np.load(find([f"rollout_Re{RE}_L{LD}.npy", f"rollout_Re{RE}.npy"]),
                allow_pickle=True).item()
snaps = np.load(find([SNAP]))
cnorm = np.load(find([f"norm_Re{RE}_L{LD}.npy", f"norm_Re{RE}.npy"]),
                allow_pickle=True).item()
lnorm = np.load(find([f"latent_norm_Re{RE}_L{LD}.npy", f"latent_norm_Re{RE}.npy"]),
                allow_pickle=True).item()
cae_path = find([f"cae_Re{RE}_L{LD}.pt", f"cae_Re{RE}.pt"])

pred = roll['raw_pred'] if 'raw_pred' in roll else roll['predicted']
true = roll['true']
n = min(len(pred), len(true))
pred, true = pred[:n].astype(float), true[:n].astype(float)
LD = pred.shape[1]
t = np.arange(n)*DT

print("="*78, flush=True)
print(f"FIELD VALIDATION — Re={RE}  latent={LD}", flush=True)
print("  Does the latent improvement survive decoding to the flow field?", flush=True)
print("="*78, flush=True)

# ── rebuild the two corrections (same as before) ─────────────────────────────
def analytic(s):
    m = s.mean(); z = hilbert(s-m)
    return m, np.abs(z), np.unwrap(np.angle(z))

A_p = np.zeros((n,LD)); th_p = np.zeros((n,LD)); m_p = np.zeros(LD)
th_t = np.zeros((n,LD))
omega = np.zeros(LD); r2 = np.zeros(LD)
for d in range(LD):
    m_p[d], A_p[:,d], th_p[:,d] = analytic(pred[:,d])
    _,     _,        th_t[:,d] = analytic(true[:,d])
    phi = th_p[:,d]-th_t[:,d]; phi -= phi[0]
    lr = linregress(t[:NCAL], phi[:NCAL])
    omega[d] = lr.slope; r2[d] = lr.rvalue**2

# phase correction
corr_phase = np.zeros_like(pred)
for d in range(LD):
    corr_phase[:,d] = A_p[:,d]*np.cos(th_p[:,d] - omega[d]*t) + m_p[d]

# sinusoid correction
dim_err = pred - true; t_cal = t[:NCAL]
sig = np.zeros_like(pred)
for d in range(LD):
    e = dim_err[:NCAL,d]; th = th_p[:NCAL,d]
    def mf(idx,A,g,psi):
        idx=idx.astype(int)
        return A*np.exp(g*t_cal[idx])*np.sin(th[idx]+psi)
    try:
        popt,_=curve_fit(mf,np.arange(NCAL,dtype=float),e,p0=[e.std(),.1,0.],
                         bounds=([-1,0.,-2*np.pi],[1,1.,2*np.pi]),maxfev=100000)
    except Exception:
        popt=[e.std(),.1,0.]
    A,g,psi=popt; cap=np.abs(e).max()*1.5
    sig[:,d]=np.clip(A*np.exp(g*t)*np.sin(th_p[:,d]+psi),-cap,cap)
corr_sin = pred - sig

print(f"\n  mean |omega| = {np.mean(np.abs(omega)):.6f} rad/t   "
      f"mean R^2 = {np.mean(r2):.3f}", flush=True)
if np.mean(r2) < 0.3:
    print(f"  WARNING: R^2 < 0.3 — the linear phase model does NOT hold here.",
          flush=True)
    print(f"           The phase correction should NOT be applied to this case.",
          flush=True)

# ── CAE decoder ──────────────────────────────────────────────────────────────
FLAT = 64*8*16
class Encoder(nn.Module):
    def __init__(s,ld):
        super().__init__()
        s.c=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(64,64,3,stride=2,padding=1),nn.ReLU())
        s.f=nn.Linear(FLAT,ld)
    def forward(s,x): return s.f(s.c(x).flatten(1))
class Decoder(nn.Module):
    def __init__(s,ld):
        super().__init__(); s.f=nn.Linear(ld,FLAT)
        s.d=nn.Sequential(
            nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=1),nn.ReLU(),
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),nn.ReLU(),
            nn.Conv2d(16,3,3,padding=1))
    def forward(s,z): return s.d(s.f(z).view(-1,64,8,16))
class CAE(nn.Module):
    def __init__(s,ld):
        super().__init__(); s.encoder=Encoder(ld); s.decoder=Decoder(ld)
    def decode(s,z): return s.decoder(z)

cae = CAE(LD).to(device)
_sd = torch.load(cae_path, map_location=device)

# Older checkpoints (Re=300, trained with the original scripts) name the layers
# conv/fc/deconv; the current pipeline names them c/f/d. Remap if needed.
if any(k.startswith("encoder.conv") for k in _sd):
    _remap = {}
    for k, v in _sd.items():
        nk = (k.replace("encoder.conv.", "encoder.c.")
                .replace("encoder.fc.",   "encoder.f.")
                .replace("decoder.fc.",   "decoder.f.")
                .replace("decoder.deconv.", "decoder.d."))
        _remap[nk] = v
    _sd = _remap
    print("  (remapped legacy CAE layer names)", flush=True)

cae.load_state_dict(_sd); cae.eval()

ch_mean, ch_std = cnorm['mean'], cnorm['std']
lmean, lstd = lnorm['mean'], lnorm['std']

def decode(latn):
    lp = latn*lstd + lmean
    out=[]
    with torch.no_grad():
        for i in range(0,len(lp),64):
            b=torch.tensor(lp[i:i+64],dtype=torch.float32).to(device)
            out.append(cae.decode(b).cpu().numpy())
    return np.concatenate(out,0)*ch_std + ch_mean

N = len(snaps); N_TRAIN=int(N*0.52); N_VAL=int(N*0.08)
TEST_START = N_TRAIN + N_VAL
ref = snaps[TEST_START:TEST_START+n]      # OpenFOAM ground truth fields

print("\n  decoding four trajectories ...", flush=True)
f_floor = decode(true)          # CAE floor
f_unc   = decode(pred)
f_sin   = decode(corr_sin)
f_pha   = decode(corr_phase)

# ── metrics ──────────────────────────────────────────────────────────────────
def rmse_t(P): return np.sqrt(np.mean((P-ref)**2, axis=(1,2,3)))
r_floor = rmse_t(f_floor); r_unc = rmse_t(f_unc)
r_sin   = rmse_t(f_sin);   r_pha = rmse_t(f_pha)

ext = slice(NCAL, n)
def relerr(P):
    num = np.sqrt(np.sum((P[ext]-ref[ext])**2))
    den = np.sqrt(np.sum(ref[ext]**2))
    return num/den*100

def rho_of(P):
    e = np.sqrt(np.mean((P[ext]-ref[ext])**2))
    fl = np.sqrt(np.mean((f_floor[ext]-ref[ext])**2))
    lstm_only = max(e**2 - fl**2, 0.)**0.5
    return lstm_only/fl if fl>0 else np.inf

print("\n" + "="*78, flush=True)
print("FIELD-SPACE RESULTS  (extrapolation region only)", flush=True)
print("="*78, flush=True)
print(f"  {'':<32} {'field RMSE':>11} {'rel err':>9} {'vs unc':>9}", flush=True)
print("  " + "-"*66, flush=True)
print(f"  {'CAE floor (best possible)':<32} {r_floor[ext].mean():>11.6f} "
      f"{relerr(f_floor):>8.2f}% {'--':>9}", flush=True)
print(f"  {'uncorrected':<32} {r_unc[ext].mean():>11.6f} "
      f"{relerr(f_unc):>8.2f}% {'--':>9}", flush=True)
i_sin_f = (1-r_sin[ext].mean()/r_unc[ext].mean())*100
i_pha_f = (1-r_pha[ext].mean()/r_unc[ext].mean())*100
print(f"  {'sinusoid corrected':<32} {r_sin[ext].mean():>11.6f} "
      f"{relerr(f_sin):>8.2f}% {i_sin_f:>+8.1f}%", flush=True)
print(f"  {'PHASE corrected':<32} {r_pha[ext].mean():>11.6f} "
      f"{relerr(f_pha):>8.2f}% {i_pha_f:>+8.1f}%", flush=True)
print("  " + "-"*66, flush=True)
print(f"\n  rho (uncorrected)  = {rho_of(f_unc):.3f}", flush=True)
print(f"  rho after sinusoid = {rho_of(f_sin):.3f}", flush=True)
print(f"  rho after phase    = {rho_of(f_pha):.3f}   "
      f"(closer to 0 = closer to the CAE floor)", flush=True)

gap_unc = r_unc[ext].mean()  - r_floor[ext].mean()
gap_pha = r_pha[ext].mean()  - r_floor[ext].mean()
closed  = (1 - gap_pha/gap_unc)*100 if gap_unc>0 else np.nan
print(f"\n  Gap to the CAE floor closed by the phase correction: {closed:.1f}%",
      flush=True)
print(f"  (100% would mean the correction reaches the compression limit)", flush=True)

# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle(f"Field-level validation — Re={RE}, latent={LD}",
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(t, r_unc,   color='tomato',   lw=1.2, label=f'uncorrected', alpha=.85)
ax.plot(t, r_sin,   color='orange',   lw=1.2, label=f'sinusoid ({i_sin_f:+.0f}%)', alpha=.85)
ax.plot(t, r_pha,   color='seagreen', lw=1.3, label=f'phase ({i_pha_f:+.0f}%)', alpha=.9)
ax.plot(t, r_floor, color='grey', lw=1.0, ls=':', label='CAE floor')
ax.axvline(NCAL*DT, color='k', ls='--', lw=.9, label='calib end')
ax.set_xlabel('t'); ax.set_ylabel('field RMSE vs OpenFOAM')
ax.set_title('Field RMSE'); ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axes[1]
ax.semilogy(t, r_unc-r_floor+1e-9,   color='tomato',   lw=1.2, label='uncorrected')
ax.semilogy(t, r_sin-r_floor+1e-9,   color='orange',   lw=1.2, label='sinusoid')
ax.semilogy(t, np.abs(r_pha-r_floor)+1e-9, color='seagreen', lw=1.3, label='phase')
ax.axvline(NCAL*DT, color='k', ls='--', lw=.9)
ax.set_xlabel('t'); ax.set_ylabel('excess over CAE floor (log)')
ax.set_title('How far above the compression limit?\n(lower = better; this is what the correction can act on)')
ax.legend(fontsize=8); ax.grid(alpha=.3, which='both')

plt.tight_layout()
fp = os.path.join(DATA_DIR, f"fieldval_Re{RE}_L{LD}.png")
plt.savefig(fp, dpi=150, bbox_inches='tight')
print(f"\n  Figure -> {fp}", flush=True)

np.save(os.path.join(DATA_DIR, f"fieldval_Re{RE}_L{LD}.npy"),
        {'r_floor':r_floor,'r_unc':r_unc,'r_sin':r_sin,'r_pha':r_pha,
         'i_sin_f':i_sin_f,'i_pha_f':i_pha_f,'omega':omega,'r2':r2,
         'closed':closed,'ncal':NCAL}, allow_pickle=True)
print("="*78, flush=True)
