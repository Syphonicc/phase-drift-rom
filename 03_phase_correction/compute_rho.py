#!/usr/bin/env python3
"""
compute_rho.py
==============
Standalone rho computation for cases whose pipeline_summary lacks it
(e.g. Re=400, Re=600 processed by the older pipeline).

rho = (LSTM-only field error) / (CAE floor field RMSE)
      where LSTM-only err = sqrt( max(unc_rmse^2 - floor_rmse^2, 0) )

Same definition as run_pipeline.py. No retraining — loads saved CAE
weights, snapshots, latent-norm stats, and the rollout, decodes, and
computes field-space RMSEs.

Usage:
    python3 compute_rho.py --re 400 --latent 4
    python3 compute_rho.py --re 600 --latent 4

If your CAE / norm files are tagged (e.g. cae_Re400_L4.pt) it finds them;
if untagged (cae_Re400.pt) it finds those too.
"""

import os, argparse, glob, numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--re", type=int, required=True)
ap.add_argument("--latent", type=int, default=4)
args = ap.parse_args()

RE = args.re
LATENT_DIM = args.latent
DATA_DIR = os.path.expanduser("~/cylinder_lstm_data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── locate files, tolerating tagged / untagged names ─────────────────────────
def find_file(patterns):
    for p in patterns:
        full = os.path.join(DATA_DIR, p)
        if os.path.exists(full):
            return full
    return None

cae_path = find_file([f"cae_Re{RE}_L{LATENT_DIM}.pt", f"cae_Re{RE}.pt"])
norm_path = find_file([f"norm_Re{RE}_L{LATENT_DIM}.npy", f"norm_Re{RE}.npy"])
latnorm_path = find_file([f"latent_norm_Re{RE}_L{LATENT_DIM}.npy",
                          f"latent_norm_Re{RE}.npy"])
snap_path = find_file([f"snapshots_Re{RE}.npy"])
roll_path = find_file([f"rollout_Re{RE}_L{LATENT_DIM}.npy", f"rollout_Re{RE}.npy"])

for name, path in [("CAE", cae_path), ("norm", norm_path),
                   ("latent_norm", latnorm_path), ("snapshots", snap_path),
                   ("rollout", roll_path)]:
    if path is None:
        print(f"ERROR: could not find {name} file for Re={RE}. "
              f"Cannot compute rho.", flush=True)
        raise SystemExit(1)
    print(f"  {name:12s}: {os.path.basename(path)}", flush=True)

# ── load ──────────────────────────────────────────────────────────────────────
cae_norm = np.load(norm_path, allow_pickle=True).item()
ch_mean, ch_std = cae_norm['mean'], cae_norm['std']
lat_norm = np.load(latnorm_path, allow_pickle=True).item()
lmean, lstd = lat_norm['mean'], lat_norm['std']
snapshots = np.load(snap_path)
roll = np.load(roll_path, allow_pickle=True).item()

# handle old/new rollout key formats
raw_pred = roll['raw_pred'] if 'raw_pred' in roll else roll['predicted']
true_all = roll['true']
n_truth = min(len(raw_pred), len(true_all))
raw_pred = raw_pred[:n_truth]; true_all = true_all[:n_truth]

# infer latent dim from rollout if mismatch
actual_latent = raw_pred.shape[1]
if actual_latent != LATENT_DIM:
    print(f"  NOTE: rollout latent dim is {actual_latent}, using that.", flush=True)
    LATENT_DIM = actual_latent

# ── CAE decoder (must match training architecture) ───────────────────────────
FLAT = 64*8*16
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
        super().__init__(); s.f = nn.Linear(ld, FLAT)
        s.d = nn.Sequential(
            nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.Conv2d(16,3,3,padding=1))
    def forward(s,z): return s.d(s.f(z).view(-1,64,8,16))
class CAE(nn.Module):
    def __init__(s, ld):
        super().__init__(); s.encoder=Encoder(ld); s.decoder=Decoder(ld)
    def decode(s,z): return s.decoder(z)

cae = CAE(LATENT_DIM).to(device)
cae.load_state_dict(torch.load(cae_path, map_location=device))
cae.eval()

# ── figure out TEST_START the same way the pipeline did ──────────────────────
N = len(snapshots)
N_TRAIN = int(N*0.52); N_VAL = int(N*0.08)
TEST_START = N_TRAIN + N_VAL
true_snaps = snapshots[TEST_START:TEST_START+n_truth]

# ── decode helper ─────────────────────────────────────────────────────────────
def decode(latn):
    lp = latn*lstd + lmean
    outs=[]
    with torch.no_grad():
        for i in range(0,len(lp),64):
            b = torch.tensor(lp[i:i+64], dtype=torch.float32).to(device)
            outs.append(cae.decode(b).cpu().numpy())
    return np.concatenate(outs,0)*ch_std + ch_mean

f_true = decode(true_all)          # decode of TRUE latents → CAE floor
f_unc  = decode(raw_pred)          # decode of LSTM prediction

def frmse(P,R): return np.sqrt(np.mean((P-R)**2))
cae_floor = frmse(f_true, true_snaps)
unc_rmse  = frmse(f_unc,  true_snaps)
lstm_only = max(unc_rmse**2 - cae_floor**2, 0.0)**0.5
rho = lstm_only/cae_floor if cae_floor>0 else float('inf')

print("\n" + "="*50, flush=True)
print(f"Re={RE}  (latent={LATENT_DIM})", flush=True)
print(f"  CAE floor RMSE : {cae_floor:.6f}", flush=True)
print(f"  LSTM unc RMSE  : {unc_rmse:.6f}", flush=True)
print(f"  LSTM-only err  : {lstm_only:.6f}", flush=True)
print(f"  rho            : {rho:.3f}   "
      f"({'LSTM-limited (correction helps)' if rho>1 else 'CAE-limited (correction wont help)'})",
      flush=True)
print("="*50, flush=True)

# ── patch rho into the existing summary if present (so interp picks it up) ──
summ_path = find_file([f"pipeline_summary_Re{RE}_L{LATENT_DIM}.npy",
                       f"pipeline_summary_Re{RE}.npy"])
if summ_path:
    try:
        d = np.load(summ_path, allow_pickle=True).item()
        d['rho'] = rho
        d['cae_floor_rmse'] = cae_floor
        d['lstm_unc_rmse'] = unc_rmse
        np.save(summ_path, d, allow_pickle=True)
        print(f"  rho written into {os.path.basename(summ_path)}", flush=True)
    except Exception as e:
        print(f"  (could not patch summary: {e})", flush=True)
else:
    # create a minimal summary so the interpolation loader finds rho
    outp = os.path.join(DATA_DIR, f"pipeline_summary_Re{RE}.npy")
    np.save(outp, {'re':RE, 'rho':rho, 'cae_floor_rmse':cae_floor,
                   'lstm_unc_rmse':unc_rmse, 'latent_dim':LATENT_DIM},
            allow_pickle=True)
    print(f"  created {os.path.basename(outp)} with rho", flush=True)
