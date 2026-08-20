#!/usr/bin/env python3
"""
pooled_cae_sweep.py
===================
Step 1 of the held-out-Reynolds generalization test.

A single shared autoencoder must span all TRAINING Reynolds numbers
(300, 400, 600, 700, 800) in one latent space, so that a temporal model can be
trained across them and queried at the held-out Re=500. The latent dimension for
that shared CAE is not free: too small and it cannot represent five regimes (high
reconstruction floor, mechanism destroyed); too large and it stops being
compression-limited (the regime in which the phase-drift structure exists at all,
per the n_z=16 result).

This script chooses the dimension PRINCIPALLY, by measurement: it trains the
pooled CAE at several latent dimensions and reports, for each, the per-Re
reconstruction floor. We then pick the smallest latent dimension whose per-Re
floors are comparable to the single-case CAEs of the paper --- keeping the shared
model in the same compression-limited regime, rather than tuning the dimension to
flatter the result.

No LSTM, no rollout here --- CAE reconstruction only. Fast relative to the full test.

Usage
-----
    python3 pooled_cae_sweep.py --dry-run
    python3 pooled_cae_sweep.py --latents 8 12 16
    python3 pooled_cae_sweep.py --latents 8 12 16 --epochs 300

Writes pooled_cae_sweep.json with per-(latent, Re) reconstruction floors.
"""

import os, sys, argparse, json, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--train-re", type=int, nargs="+", default=[300, 400, 600, 700, 800],
                help="Reynolds numbers to pool for the shared CAE")
ap.add_argument("--holdout", type=int, default=500, help="held-out Re (checked too, for reference)")
ap.add_argument("--latents", type=int, nargs="+", default=[8, 12, 16])
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

if args.dry_run:
    args.epochs = 3
    args.latents = [8]

DATA_DIR = os.path.expanduser(args.data_dir)

def snap_path(re):
    return os.path.join(DATA_DIR, "snapshots_SQ100.npy" if re == 101 else f"snapshots_Re{re}.npy")

def load(re):
    p = snap_path(re)
    if not os.path.exists(p):
        sys.exit(f"ERROR: {p} not found")
    return np.load(p).astype(np.float32)

print("="*74, flush=True)
print(f"POOLED CAE SWEEP  train={args.train_re}  holdout={args.holdout}  "
      f"latents={args.latents}", flush=True)
print("="*74, flush=True)

# ---- load and pool training snapshots ---------------------------------------
if args.dry_run:
    rng = np.random.default_rng(0)
    per = {re: rng.standard_normal((200, 3, 64, 128)).astype(np.float32) for re in args.train_re}
    hold = rng.standard_normal((200, 3, 64, 128)).astype(np.float32)
else:
    per = {re: load(re) for re in args.train_re}
    hold = load(args.holdout)
for re, a in per.items():
    print(f"  Re={re}: {a.shape}", flush=True)
print(f"  holdout Re={args.holdout}: {hold.shape}", flush=True)

# ---- pooled normalization (channel-wise over ALL training data) -------------
pool = np.concatenate(list(per.values()), axis=0)
cm = pool.mean(axis=(0, 2, 3), keepdims=True)
cs = pool.std(axis=(0, 2, 3), keepdims=True); cs = np.where(cs < 1e-8, 1.0, cs)
print(f"  pooled {pool.shape[0]} snapshots for shared normalization", flush=True)

def normed(a):
    return (a - cm) / cs

# ---- torch CAE (same architecture as run_pipeline, parameterised by n_z) -----
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device: {dev}", flush=True)
FLAT = 64 * 8 * 16

class CAE(nn.Module):
    def __init__(s, ld):
        super().__init__()
        s.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU())
        s.efc = nn.Linear(FLAT, ld)
        s.dfc = nn.Linear(ld, FLAT)
        s.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1))
    def encode(s, x): return s.efc(s.enc(x).flatten(1))
    def decode(s, z): return s.dec(s.dfc(z).view(-1, 64, 8, 16))
    def forward(s, x): return s.decode(s.encode(x))

def floor_for(re_arr_normed, model):
    """per-Re reconstruction RMSE in normalized units"""
    model.eval(); errs = []
    with torch.no_grad():
        for i in range(0, len(re_arr_normed), 64):
            b = torch.tensor(re_arr_normed[i:i+64]).to(dev)
            r = model(b).cpu().numpy()
            errs.append(((r - re_arr_normed[i:i+64])**2).reshape(len(r), -1).mean(1))
    return float(np.sqrt(np.concatenate(errs).mean()))

Xpool = torch.tensor(normed(pool))
results = {}
for ld in args.latents:
    print(f"\n--- latent dim {ld} ---", flush=True)
    torch.manual_seed(0)
    cae = CAE(ld).to(dev)
    opt = torch.optim.Adam(cae.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5, min_lr=1e-5)
    nval = int(len(Xpool)*0.15); ntr = len(Xpool)-nval
    tr, va = random_split(TensorDataset(Xpool), [ntr, nval],
                          generator=torch.Generator().manual_seed(0))
    trl = DataLoader(tr, batch_size=32, shuffle=True)
    val = DataLoader(va, batch_size=32)
    crit = nn.MSELoss(); best = 1e9; bs = None
    for ep in range(args.epochs):
        cae.train()
        for (b,) in trl:
            b = b.to(dev); opt.zero_grad(); crit(cae(b), b).backward(); opt.step()
        cae.eval(); vl = 0
        with torch.no_grad():
            for (b,) in val: b = b.to(dev); vl += crit(cae(b), b).item()*len(b)
        vl /= nval; sch.step(vl)
        if vl < best: best = vl; bs = {k: v.cpu().clone() for k, v in cae.state_dict().items()}
        if (ep+1) % 50 == 0 or ep == 0:
            print(f"    ep{ep+1}/{args.epochs} val={vl:.6f}", flush=True)
    cae.load_state_dict(bs)
    floors = {re: floor_for(normed(per[re]), cae) for re in args.train_re}
    floors_holdout = floor_for(normed(hold), cae)
    results[ld] = dict(train_floors=floors, holdout_floor=floors_holdout, val=best)
    print(f"  per-Re reconstruction floor (normalized RMSE):", flush=True)
    for re, f in floors.items():
        print(f"     Re={re}: {f:.4f}", flush=True)
    print(f"     Re={args.holdout} (held out): {floors_holdout:.4f}", flush=True)

# ---- guidance ---------------------------------------------------------------
print("\n" + "="*74, flush=True)
print("CHOOSING THE LATENT DIMENSION", flush=True)
print("="*74, flush=True)
print("Single-case CAE floors in the paper are ~0.004-0.005 (normalized RMSE).", flush=True)
print("Pick the SMALLEST latent dim whose per-Re training floors are in that", flush=True)
print("neighbourhood -- that keeps the shared CAE compression-limited (where the", flush=True)
print("phase-drift mechanism lives) without over-capacity.\n", flush=True)
for ld in args.latents:
    f = results[ld]["train_floors"]
    mx = max(f.values())
    verdict = "compression-limited (good)" if mx < 0.010 else "floor too high -- underfit"
    print(f"  n_z={ld}: worst train floor {mx:.4f}  -> {verdict}", flush=True)

out = os.path.join(DATA_DIR, "pooled_cae_sweep.json")
json.dump(dict(train_re=args.train_re, holdout=args.holdout, results=results),
          open(out, "w"), indent=2, default=float)
print(f"\nSaved -> {out}", flush=True)
print("\nNext: choose n_z from the above, then run heldout_re_pipeline.py --latent <n_z>", flush=True)
