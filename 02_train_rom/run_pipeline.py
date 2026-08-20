#!/usr/bin/env python3
"""
run_pipeline.py  —  Full Re-parameterized ROM pipeline
======================================================
One command runs the entire chain for a given Reynolds number:

   CAE training → encode latents → LSTM training → autoregressive rollout
   → phase-locked correction (v6) → relative-error metrics → save everything

Usage:
   python3 run_pipeline.py --re 400
   python3 run_pipeline.py --re 600 --cae-epochs 300 --lstm-epochs 500

Requires (already produced by extract_snapshots_v2.py):
   ~/cylinder_lstm_data/snapshots_Re{RE}.npy
   ~/cylinder_lstm_data/metadata_Re{RE}.npy

Produces (all tagged by Re):
   cae_Re{RE}.pt, latents_Re{RE}.npy, norm_Re{RE}.npy
   lstm_Re{RE}.pt, latent_norm_Re{RE}.npy
   rollout_Re{RE}.npy, correction_v6_Re{RE}.npy
   pipeline_summary_Re{RE}.npy   ← key numbers for the cross-Re comparison
"""

import os, sys, argparse, numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import hilbert

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Arguments ─────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--re",          type=int,   required=True)
ap.add_argument("--latent",      type=int,   default=4)
ap.add_argument("--cae-epochs",  type=int,   default=300)
ap.add_argument("--lstm-epochs", type=int,   default=500)
ap.add_argument("--tau",         type=int,   default=20)
ap.add_argument("--calib",       type=int,   default=150)
ap.add_argument("--dt",          type=float, default=0.03)
ap.add_argument("--diameter",    type=float, default=0.12)  # cylinder D
ap.add_argument("--velocity",    type=float, default=1.0)   # inlet U
ap.add_argument("--seed",        type=int,   default=42)
args = ap.parse_args()

RE          = args.re
LATENT_DIM  = args.latent
DT          = args.dt
TAU         = args.tau
CALIB_LEN   = args.calib
D, U        = args.diameter, args.velocity

DATA_DIR = os.path.expanduser("~/cylinder_lstm_data")
TAG = f"Re{RE}_L{LATENT_DIM}"   # unique tag per (Re, latent) run
torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 64, flush=True)
print(f"PIPELINE  Re={RE}  latent={LATENT_DIM}  device={device}", flush=True)
print("=" * 64, flush=True)

snap_path = os.path.join(DATA_DIR, f"snapshots_Re{RE}.npy")
if not os.path.exists(snap_path):
    print(f"ERROR: {snap_path} not found. Run extract_snapshots_v2.py first.")
    sys.exit(1)

raw = np.load(snap_path)
N   = len(raw)
print(f"Snapshots : {raw.shape}", flush=True)

# Split sizes scale with available data (keeps 70/10/20 feel, leaves test region)
N_TRAIN = int(N * 0.52)
N_VAL   = int(N * 0.08)
# remaining ~40% is the test/rollout region

# ══════════════════════════════════════════════════════════════════════════════
#  CAE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[CAE] normalising + training ...", flush=True)
ch_mean = raw.mean(axis=(0,2,3), keepdims=True)
ch_std  = raw.std(axis=(0,2,3), keepdims=True)
ch_std  = np.where(ch_std < 1e-8, 1.0, ch_std)
norm    = ((raw - ch_mean) / ch_std).astype(np.float32)
np.save(os.path.join(DATA_DIR, f"norm_{TAG}.npy"),
        {"mean": ch_mean, "std": ch_std}, allow_pickle=True)

X = torch.tensor(norm)
FLAT = 64 * 8 * 16

class Encoder(nn.Module):
    def __init__(s, ld):
        super().__init__()
        s.c = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,stride=2,padding=1), nn.ReLU())
        s.f = nn.Linear(FLAT, ld)
    def forward(s,x): return s.f(s.c(x).flatten(1))

class Decoder(nn.Module):
    def __init__(s, ld):
        super().__init__()
        s.f = nn.Linear(ld, FLAT)
        s.d = nn.Sequential(
            nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.Conv2d(16,3,3,padding=1))
    def forward(s,z): return s.d(s.f(z).view(-1,64,8,16))

class CAE(nn.Module):
    def __init__(s, ld):
        super().__init__(); s.encoder=Encoder(ld); s.decoder=Decoder(ld)
    def forward(s,x): return s.decoder(s.encoder(x))
    def encode(s,x): return s.encoder(x)
    def decode(s,z): return s.decoder(z)

cae = CAE(LATENT_DIM).to(device)
opt = torch.optim.Adam(cae.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5, min_lr=1e-5)
crit = nn.MSELoss()

nval = int(len(X)*0.15); ntr = len(X)-nval
tr, va = random_split(TensorDataset(X), [ntr, nval],
                      generator=torch.Generator().manual_seed(args.seed))
trl = DataLoader(tr, batch_size=32, shuffle=True)
val = DataLoader(va, batch_size=32)

best = float('inf'); cae_path = os.path.join(DATA_DIR, f"cae_{TAG}.pt")
for ep in range(args.cae_epochs):
    cae.train()
    for (b,) in trl:
        b=b.to(device); opt.zero_grad()
        l=crit(cae(b),b); l.backward(); opt.step()
    cae.eval(); vl=0
    with torch.no_grad():
        for (b,) in val:
            b=b.to(device); vl+=crit(cae(b),b).item()*len(b)
    vl/=nval; sch.step(vl)
    if vl<best: best=vl; torch.save(cae.state_dict(), cae_path)
    if (ep+1)%50==0 or ep==0:
        print(f"  CAE ep{ep+1}/{args.cae_epochs} val={vl:.6f}", flush=True)
print(f"  CAE best val: {best:.6f}", flush=True)

# encode all
cae.load_state_dict(torch.load(cae_path, map_location=device)); cae.eval()
lat=[]
with torch.no_grad():
    for i in range(0,len(X),64):
        lat.append(cae.encode(X[i:i+64].to(device)).cpu().numpy())
latents=np.concatenate(lat,0)
np.save(os.path.join(DATA_DIR, f"latents_{TAG}.npy"), latents)
print(f"  latents: {latents.shape}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LSTM
# ══════════════════════════════════════════════════════════════════════════════
print("\n[LSTM] training ...", flush=True)
lmean=latents[:N_TRAIN].mean(0,keepdims=True)
lstd =latents[:N_TRAIN].std(0,keepdims=True); lstd=np.where(lstd<1e-8,1.0,lstd)
ln=(latents-lmean)/lstd
np.save(os.path.join(DATA_DIR, f"latent_norm_{TAG}.npy"),
        {"mean":lmean,"std":lstd}, allow_pickle=True)

def windows(a,s,e):
    X,Y=[],[]
    for i in range(s,e-TAU):
        X.append(a[i:i+TAU]); Y.append(a[i+TAU])
    return np.array(X,dtype=np.float32), np.array(Y,dtype=np.float32)
Xtr,Ytr=windows(ln,0,N_TRAIN)
Xva,Yva=windows(ln,N_TRAIN,N_TRAIN+N_VAL)
trl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Ytr)),
               batch_size=64, shuffle=True)
val=DataLoader(TensorDataset(torch.from_numpy(Xva),torch.from_numpy(Yva)),
               batch_size=64)

class LSTMP(nn.Module):
    def __init__(s,ld,hd,nl):
        super().__init__()
        s.lstm=nn.LSTM(ld,hd,nl,batch_first=True,dropout=0.1 if nl>1 else 0.0)
        s.fc=nn.Sequential(nn.Linear(hd,hd//2),nn.Tanh(),nn.Linear(hd//2,ld))
    def forward(s,x):
        o,_=s.lstm(x); return s.fc(o[:,-1,:])

lstm=LSTMP(LATENT_DIM,64,2).to(device)
opt=torch.optim.Adam(lstm.parameters(),lr=1e-3)
sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=30,factor=0.5,min_lr=1e-6)
best=float('inf'); lstm_path=os.path.join(DATA_DIR,f"lstm_{TAG}.pt")
for ep in range(args.lstm_epochs):
    lstm.train()
    for xb,yb in trl:
        xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
        l=crit(lstm(xb),yb); l.backward()
        torch.nn.utils.clip_grad_norm_(lstm.parameters(),1.0); opt.step()
    lstm.eval(); vl=0
    with torch.no_grad():
        for xb,yb in val:
            xb,yb=xb.to(device),yb.to(device); vl+=crit(lstm(xb),yb).item()*len(xb)
    vl/=len(Xva); sch.step(vl)
    if vl<best: best=vl; torch.save(lstm.state_dict(),lstm_path)
    if (ep+1)%50==0 or ep==0:
        print(f"  LSTM ep{ep+1}/{args.lstm_epochs} val={vl:.6f}", flush=True)
print(f"  LSTM best val: {best:.6f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROLLOUT + PHASE-LOCKED CORRECTION (v6)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[ROLLOUT + correction] ...", flush=True)
lstm.load_state_dict(torch.load(lstm_path,map_location=device)); lstm.eval()
TEST_START=N_TRAIN+N_VAL
ROLL=len(ln)-TEST_START
true_all=ln[TEST_START:TEST_START+ROLL]; n_truth=len(true_all)

seed=ln[TEST_START-TAU:TEST_START].copy(); win=seed.copy()
raw_pred=np.zeros((ROLL,LATENT_DIM),dtype=np.float32)
with torch.no_grad():
    for s in range(ROLL):
        inp=torch.tensor(win[np.newaxis],dtype=torch.float32).to(device)
        p=lstm(inp).cpu().numpy().squeeze(); raw_pred[s]=p
        win=np.vstack([win[1:],p])

dim_err=raw_pred[:n_truth]-true_all
St_expected = 0.20  # nominal

# Hilbert instantaneous phase
inst_phase=np.zeros((ROLL,LATENT_DIM)); freqs_meas=[]
for d in range(LATENT_DIM):
    sig=raw_pred[:,d]-raw_pred[:,d].mean()
    ph=np.unwrap(np.angle(hilbert(sig))); inst_phase[:,d]=ph
    freqs_meas.append(np.median(np.diff(ph)/(2*np.pi*DT)))
f_lstm=float(np.mean(freqs_meas))
St_lstm=f_lstm*D/U
print(f"  f_LSTM={f_lstm:.4f} Hz  →  St_LSTM={St_lstm:.4f}", flush=True)

# fit phase-conditioned correction on calibration window
t_full=np.arange(ROLL)*DT; t_cal=t_full[:CALIB_LEN]
params={}; caps={}
for d in range(LATENT_DIM):
    e=dim_err[:CALIB_LEN,d]; th=inst_phase[:CALIB_LEN,d]
    def mf(idx,A,g,psi):
        idx=idx.astype(int)
        return A*np.exp(g*t_cal[idx])*np.sin(th[idx]+psi)
    try:
        popt,_=curve_fit(mf,np.arange(CALIB_LEN,dtype=float),e,
            p0=[e.std(),0.1,0.0],
            bounds=([-1,-0.2,-2*np.pi],[1,1.0,2*np.pi]),maxfev=100000)
    except Exception:
        popt=[e.std(),0.1,0.0]
    params[d]=popt; caps[d]=np.abs(e).max()*1.5

corr=np.zeros((ROLL,LATENT_DIM),dtype=np.float32)
for d in range(LATENT_DIM):
    A,g,psi=params[d]
    corr[:,d]=np.clip(A*np.exp(g*t_full)*np.sin(inst_phase[:,d]+psi),
                      -caps[d],caps[d])
corrected=raw_pred-corr

# metrics
def errs(P): return np.array([np.linalg.norm(P[s]-true_all[s]) for s in range(n_truth)])
eu=errs(raw_pred[:n_truth]); ec=errs(corrected[:n_truth])
ext=slice(CALIB_LEN,n_truth)
imp=(1-ec[ext].mean()/eu[ext].mean())*100

# relative error (global energy norm) over extrapolation
def relerr(P):
    num=np.sqrt(np.sum((P[ext]-true_all[ext])**2))
    den=np.sqrt(np.sum(true_all[ext]**2))
    return num/den*100
rel_u=relerr(raw_pred); rel_c=relerr(corrected)

print(f"  extrap improvement : {imp:.1f}%", flush=True)
print(f"  rel err (unc/cor)  : {rel_u:.2f}% / {rel_c:.2f}%", flush=True)

# ── Compute rho = LSTM field error / CAE floor (field RMSE) ───────────────────
# Decode true latents, uncorrected preds, corrected preds → field space
def _decode(latn):
    lp = latn*lstd + lmean
    outs=[]
    with torch.no_grad():
        for i in range(0,len(lp),64):
            b=torch.tensor(lp[i:i+64],dtype=torch.float32).to(device)
            outs.append(cae.decode(b).cpu().numpy())
    return np.concatenate(outs,0)*ch_std + ch_mean

_snap = raw[TEST_START:TEST_START+n_truth]          # OpenFOAM truth fields
_f_true = _decode(true_all)                          # decode of TRUE latents = CAE floor
_f_unc  = _decode(raw_pred[:n_truth])
_f_cor  = _decode(corrected[:n_truth])

def _frmse(P,R): return np.sqrt(np.mean((P-R)**2))
cae_floor_rmse = _frmse(_f_true, _snap)              # best achievable (CAE limit)
lstm_unc_rmse  = _frmse(_f_unc,  _snap)
lstm_cor_rmse  = _frmse(_f_cor,  _snap)

# rho = how much the LSTM error exceeds the CAE floor.
# Use the PORTION of field error attributable to LSTM = total - floor (in quadrature-ish).
# Simple, interpretable definition:
lstm_only_err = max(lstm_unc_rmse**2 - cae_floor_rmse**2, 0.0) ** 0.5
rho = lstm_only_err / cae_floor_rmse if cae_floor_rmse > 0 else float('inf')

print(f"  CAE floor RMSE   : {cae_floor_rmse:.6f}", flush=True)
print(f"  LSTM unc  RMSE   : {lstm_unc_rmse:.6f}", flush=True)
print(f"  LSTM-only err    : {lstm_only_err:.6f}", flush=True)
print(f"  rho = LSTM/CAE   : {rho:.3f}   "
      f"({'LSTM-limited (correct!)' if rho>1 else 'CAE-limited (add latents)'})",
      flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE SUMMARY (the row for the cross-Re table)
# ══════════════════════════════════════════════════════════════════════════════
np.save(os.path.join(DATA_DIR, f"rollout_{TAG}.npy"),
    {'raw_pred':raw_pred,'corrected':corrected,'true':true_all,
     'err_unc':eu,'err_cor':ec,'inst_phase':inst_phase}, allow_pickle=True)

summary={
    're':RE, 'latent_dim':LATENT_DIM, 'cae_val':best,
    'f_lstm':f_lstm, 'St_lstm':St_lstm,
    'n_truth':n_truth, 'calib_len':CALIB_LEN,
    'improvement_pct':imp, 'relerr_unc':rel_u, 'relerr_cor':rel_c,
    'freqs_per_dim':freqs_meas, 'dt':DT, 'D':D, 'U':U,
    'cae_floor_rmse':cae_floor_rmse, 'lstm_unc_rmse':lstm_unc_rmse,
    'lstm_cor_rmse':lstm_cor_rmse, 'lstm_only_err':lstm_only_err,
    'rho':rho, 'cae_val':best,
}
np.save(os.path.join(DATA_DIR, f"pipeline_summary_{TAG}.npy"),
        summary, allow_pickle=True)

print("\n" + "=" * 64, flush=True)
print(f"DONE  Re={RE}", flush=True)
print(f"  f_LSTM={f_lstm:.4f}  St_LSTM={St_lstm:.4f}  "
      f"improvement={imp:.1f}%  relerr_cor={rel_c:.2f}%", flush=True)
print(f"  rho={rho:.3f}  summary → pipeline_summary_{TAG}.npy", flush=True)
print("=" * 64, flush=True)
