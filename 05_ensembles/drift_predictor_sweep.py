#!/usr/bin/env python3
"""
drift_predictor_sweep.py
========================
Investigates WHETHER a measurable property of a trained LSTM predicts its
rollout drift — i.e. tests the "Paper B" mechanism hypothesis.

Design (controlled):
  - Train the CAE ONCE (fixed seed) and freeze it, so the latent space is
    identical across all runs and predictors are comparable.
  - Train the LSTM across many seeds on that same latent space.
  - For each LSTM seed, log the DRIFT RATIO plus four candidate predictors:
        1. detuning       = |f_LSTM - f_true|      (frequency mismatch)
        2. amplitude_ratio= RMS(pred cycle)/RMS(true cycle)  (cycle inflation)
        3. onestep_err    = teacher-forced 1-step RMSE       (local accuracy)
        4. pert_growth    = closed-loop divergence rate       (instability)
  - Correlate each predictor with drift (Pearson + Spearman) and report
    which, if any, predicts drift.

Interpretation:
  - A predictor with |r| high AND significant  → candidate mechanism; worth
    a larger seed sweep + hold-out validation to confirm.
  - All predictors flat                         → drift is not predictable
    from these; report drift as a seed distribution (Paper A) and list this
    as future work.

Usage:
    python3 drift_predictor_sweep.py --re 500 --nseeds 10
    python3 drift_predictor_sweep.py --re 400 --nseeds 15 --reuse-cae

--reuse-cae : if cae_Re{RE}.pt exists, load it instead of training a new one.
"""

import os, argparse, numpy as np
from scipy.signal import hilbert
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

ap = argparse.ArgumentParser()
ap.add_argument("--re", type=int, required=True)
ap.add_argument("--nseeds", type=int, default=10)
ap.add_argument("--latent", type=int, default=4)
ap.add_argument("--cae-epochs", type=int, default=300)
ap.add_argument("--lstm-epochs", type=int, default=500)
ap.add_argument("--tau", type=int, default=20)
ap.add_argument("--dt", type=float, default=0.03)
ap.add_argument("--cae-seed", type=int, default=0, help="fixed seed for the CAE")
ap.add_argument("--reuse-cae", action="store_true",
                help="load existing cae_Re{RE}.pt instead of training")
args = ap.parse_args()

RE = args.re; LATENT_DIM = args.latent; DT = args.dt; TAU = args.tau
DATA_DIR = os.path.expanduser("~/cylinder_lstm_data")
OUT_DIR = os.path.join(DATA_DIR, f"drift_predictor_Re{RE}")
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*66, flush=True)
print(f"DRIFT PREDICTOR SWEEP  Re={RE}  nseeds={args.nseeds}  device={device}", flush=True)
print(f"(CAE fixed, LSTM varied — isolates LSTM drift variance)", flush=True)
print("="*66, flush=True)

raw = np.load(os.path.join(DATA_DIR, f"snapshots_Re{RE}.npy"))
N = len(raw); N_TRAIN = int(N*0.52); N_VAL = int(N*0.08)
TEST_START = N_TRAIN + N_VAL

# ── architectures ─────────────────────────────────────────────────────────────
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
    def forward(s,x): return s.decoder(s.encoder(x))
    def encode(s,x): return s.encoder(x)
class LSTMP(nn.Module):
    def __init__(s,ld,hd,nl):
        super().__init__()
        s.lstm=nn.LSTM(ld,hd,nl,batch_first=True,dropout=0.1 if nl>1 else 0.0)
        s.fc=nn.Sequential(nn.Linear(hd,hd//2),nn.Tanh(),nn.Linear(hd//2,ld))
    def forward(s,x):
        o,_=s.lstm(x); return s.fc(o[:,-1,:])

crit = nn.MSELoss()

# ── train / load CAE (once) ──────────────────────────────────────────────────
ch_mean = raw.mean(axis=(0,2,3), keepdims=True)
ch_std  = raw.std(axis=(0,2,3), keepdims=True); ch_std=np.where(ch_std<1e-8,1.0,ch_std)
norm = ((raw-ch_mean)/ch_std).astype(np.float32)
X = torch.tensor(norm)

cae = CAE(LATENT_DIM).to(device)
cae_path = os.path.join(DATA_DIR, f"cae_Re{RE}.pt")
if args.reuse_cae and os.path.exists(cae_path):
    cae.load_state_dict(torch.load(cae_path, map_location=device))
    cae.eval()
    print(f"Loaded existing CAE: {cae_path}", flush=True)
else:
    print("Training CAE (once, fixed seed) ...", flush=True)
    torch.manual_seed(args.cae_seed); np.random.seed(args.cae_seed)
    opt=torch.optim.Adam(cae.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=20,factor=0.5,min_lr=1e-5)
    nval=int(len(X)*0.15); ntr=len(X)-nval
    tr,va=random_split(TensorDataset(X),[ntr,nval],
                       generator=torch.Generator().manual_seed(args.cae_seed))
    trl=DataLoader(tr,batch_size=32,shuffle=True); val=DataLoader(va,batch_size=32)
    best=1e9; best_state=None
    for ep in range(args.cae_epochs):
        cae.train()
        for (b,) in trl:
            b=b.to(device); opt.zero_grad(); l=crit(cae(b),b); l.backward(); opt.step()
        cae.eval(); vl=0
        with torch.no_grad():
            for (b,) in val:
                b=b.to(device); vl+=crit(cae(b),b).item()*len(b)
        vl/=nval; sch.step(vl)
        if vl<best: best=vl; best_state={k:v.cpu().clone() for k,v in cae.state_dict().items()}
        if (ep+1)%100==0: print(f"  CAE ep{ep+1} val={vl:.6f}", flush=True)
    cae.load_state_dict(best_state); cae.eval()
    print(f"  CAE best val={best:.6f}", flush=True)

# encode once → fixed latent space
lat=[]
with torch.no_grad():
    for i in range(0,len(X),64):
        lat.append(cae.encode(X[i:i+64].to(device)).cpu().numpy())
latents=np.concatenate(lat,0)
lmean=latents[:N_TRAIN].mean(0,keepdims=True)
lstd=latents[:N_TRAIN].std(0,keepdims=True); lstd=np.where(lstd<1e-8,1.0,lstd)
ln=(latents-lmean)/lstd

ROLL=len(ln)-TEST_START
true_all=ln[TEST_START:TEST_START+ROLL]; n_truth=len(true_all)

# ── true shedding frequency (seed-independent) ───────────────────────────────
def median_freq(traj):
    fs=[]
    for d in range(traj.shape[1]):
        sig=traj[:,d]-traj[:,d].mean()
        ph=np.unwrap(np.angle(hilbert(sig)))
        fs.append(np.median(np.diff(ph)/(2*np.pi*DT)))
    return float(np.mean(fs))
f_true = median_freq(true_all)
true_amp = np.mean([true_all[:,d].std() for d in range(LATENT_DIM)])
print(f"\nTrue trajectory: f_true={f_true:.4f}  amp={true_amp:.4f}", flush=True)

def windows(a,s,e):
    Xw,Yw=[],[]
    for i in range(s,e-TAU):
        Xw.append(a[i:i+TAU]); Yw.append(a[i+TAU])
    return np.array(Xw,dtype=np.float32), np.array(Yw,dtype=np.float32)
Xtr,Ytr=windows(ln,0,N_TRAIN); Xva,Yva=windows(ln,N_TRAIN,N_TRAIN+N_VAL)

# ── train one LSTM, measure drift + predictors ───────────────────────────────
def run_lstm_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    trl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Ytr)),
                   batch_size=64,shuffle=True)
    val=DataLoader(TensorDataset(torch.from_numpy(Xva),torch.from_numpy(Yva)),batch_size=64)
    lstm=LSTMP(LATENT_DIM,64,2).to(device)
    opt=torch.optim.Adam(lstm.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=30,factor=0.5,min_lr=1e-6)
    best=1e9; best_state=None
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
        if vl<best: best=vl; best_state={k:v.cpu().clone() for k,v in lstm.state_dict().items()}
    lstm.load_state_dict(best_state); lstm.eval()

    # free rollout
    win=ln[TEST_START-TAU:TEST_START].copy()
    pred=np.zeros((ROLL,LATENT_DIM),dtype=np.float32)
    with torch.no_grad():
        for s in range(ROLL):
            inp=torch.tensor(win[np.newaxis],dtype=torch.float32).to(device)
            p=lstm(inp).cpu().numpy().squeeze(); pred[s]=p
            win=np.vstack([win[1:],p])
    err=np.array([np.linalg.norm(pred[i]-true_all[i]) for i in range(n_truth)])
    drift_ratio=err[-20:].mean()/err[:20].mean()

    # PREDICTOR 1: detuning
    f_lstm=median_freq(pred)
    detuning=abs(f_lstm-f_true)

    # PREDICTOR 2: amplitude ratio (cycle inflation) — use first third of rollout
    seg=pred[:n_truth//3]
    pred_amp=np.mean([seg[:,d].std() for d in range(LATENT_DIM)])
    amp_ratio=pred_amp/true_amp

    # PREDICTOR 3: teacher-forced one-step error over test region
    Xte,Yte=windows(ln,TEST_START,TEST_START+ROLL)
    with torch.no_grad():
        yp=lstm(torch.tensor(Xte,dtype=torch.float32).to(device)).cpu().numpy()
    onestep_err=np.sqrt(np.mean((yp-Yte)**2))

    # PREDICTOR 4: closed-loop perturbation growth
    #   perturb seed window, roll both K steps, measure divergence growth
    K=30; eps=1e-3; ndir=5; growths=[]
    base_win=ln[TEST_START-TAU:TEST_START].copy()
    with torch.no_grad():
        # clean short rollout
        w=base_win.copy(); clean=[]
        for s in range(K):
            inp=torch.tensor(w[np.newaxis],dtype=torch.float32).to(device)
            p=lstm(inp).cpu().numpy().squeeze(); clean.append(p); w=np.vstack([w[1:],p])
        clean=np.array(clean)
        for _ in range(ndir):
            wp=base_win.copy()
            wp[-1]=wp[-1]+eps*np.random.randn(LATENT_DIM)
            pert=[]
            for s in range(K):
                inp=torch.tensor(wp[np.newaxis],dtype=torch.float32).to(device)
                p=lstm(inp).cpu().numpy().squeeze(); pert.append(p); wp=np.vstack([wp[1:],p])
            pert=np.array(pert)
            sep=np.linalg.norm(pert-clean,axis=1)
            sep=np.where(sep<1e-12,1e-12,sep)
            # finite-time growth rate: log(sep_end/sep_start)/K
            growths.append(np.log(sep[-1]/max(sep[0],1e-12))/K)
    pert_growth=float(np.mean(growths))

    return {'seed':seed,'lstm_val':best,'drift_ratio':drift_ratio,
            'detuning':detuning,'amp_ratio':amp_ratio,
            'onestep_err':onestep_err,'pert_growth':pert_growth,
            'f_lstm':f_lstm}

# ── run all seeds ─────────────────────────────────────────────────────────────
results=[]
for seed in range(args.nseeds):
    r=run_lstm_seed(seed)
    results.append(r)
    print(f"  seed {seed:2d}: drift={r['drift_ratio']:6.2f}x  "
          f"detune={r['detuning']:.4f}  amp_ratio={r['amp_ratio']:.3f}  "
          f"1step={r['onestep_err']:.5f}  pert_growth={r['pert_growth']:+.4f}", flush=True)

# ── correlation analysis ─────────────────────────────────────────────────────
drift=np.array([r['drift_ratio'] for r in results])
predictors={
    'detuning'   : np.array([r['detuning']    for r in results]),
    'amp_ratio'  : np.array([r['amp_ratio']   for r in results]),
    'onestep_err': np.array([r['onestep_err'] for r in results]),
    'pert_growth': np.array([r['pert_growth'] for r in results]),
}

print("\n"+"="*66, flush=True)
print(f"CORRELATION WITH DRIFT — Re={RE}  ({args.nseeds} seeds)", flush=True)
print("="*66, flush=True)
print(f"  drift ratio: mean={drift.mean():.2f}x  std={drift.std():.2f}  "
      f"CV={drift.std()/drift.mean():.2f}", flush=True)
print(f"\n  {'predictor':>12}  {'Pearson r':>10}  {'p-value':>9}  "
      f"{'Spearman':>9}  verdict", flush=True)
print("  "+"-"*58, flush=True)
best_pred=None; best_abs=0
for name,vals in predictors.items():
    if np.std(vals)<1e-12:
        print(f"  {name:>12}  {'(constant)':>10}", flush=True); continue
    r_p,p_p=pearsonr(vals,drift)
    r_s,p_s=spearmanr(vals,drift)
    flag=""
    if abs(r_p)>0.6 and p_p<0.05: flag="** STRONG"
    elif abs(r_p)>0.4: flag="* moderate"
    else: flag="weak/none"
    print(f"  {name:>12}  {r_p:>+10.3f}  {p_p:>9.4f}  {r_s:>+9.3f}  {flag}", flush=True)
    if abs(r_p)>best_abs: best_abs=abs(r_p); best_pred=(name,r_p,p_p)

print("  "+"-"*58, flush=True)
if best_pred and abs(best_pred[1])>0.6 and best_pred[2]<0.05:
    print(f"\n  MOST PROMISING: '{best_pred[0]}' (r={best_pred[1]:+.3f}, "
          f"p={best_pred[2]:.4f})", flush=True)
    print(f"  → A drift predictor may exist. Worth a larger seed sweep +", flush=True)
    print(f"    hold-out validation to confirm (Paper B mechanism section).", flush=True)
else:
    print(f"\n  No predictor clears the bar (|r|>0.6, p<0.05) at {args.nseeds} seeds.", flush=True)
    print(f"  → Either genuinely no simple predictor (report drift as a", flush=True)
    print(f"    seed distribution, Paper A), or need more seeds to resolve.", flush=True)
print("="*66, flush=True)

np.save(os.path.join(OUT_DIR, f"drift_predictor_Re{RE}.npy"),
        {'re':RE,'results':results,'drift':drift,'predictors':predictors,
         'f_true':f_true}, allow_pickle=True)
print(f"\nSaved → {OUT_DIR}/drift_predictor_Re{RE}.npy", flush=True)
