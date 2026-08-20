#!/usr/bin/env python3
"""
seed_sweep_v2.py
================
Retrains the CAE+LSTM for a single Reynolds number across MANY random seeds
and asks a statistically honest question:

    Is the rollout drift predictable from any measurable training property,
    or is it training-run noise?

WHY THIS VERSION EXISTS (reviewer item 4)
-----------------------------------------
The n=10 study cannot distinguish "no predictor exists" from "a real moderate
predictor exists but we are underpowered": at n=10 the smallest correlation
resolvable at p<0.05 is |r|~=0.63, and the largest candidate was |r|=0.43.
A null at n=10 is therefore *absence of evidence*, not *evidence of absence*.

This version fixes that by (a) running enough seeds (default 40) that the
critical |r| drops to ~0.31, and (b) reporting, for every candidate predictor,
the Pearson r WITH ITS 95% CONFIDENCE INTERVAL (Fisher), the Spearman rank
correlation, p-values with a Bonferroni correction for the number of predictors,
and the minimum effect size the study is powered to detect. That lets the paper
say either "predictor X drives the drift (r=..., p=...)" or the much stronger
"we can rule out any correlation above |r|=X" -- neither of which n=10 supports.

Also: resume-safe (each seed written as it finishes; rerun skips finished seeds),
flush=True logging, and a --dry-run that exercises the whole pipeline on
synthetic data in seconds before you commit GPU hours.

Usage
-----
    python3 seed_sweep_v2.py --re 500 --n-seeds 40
    python3 seed_sweep_v2.py --re 500 --seeds 0 1 2 3 4      # explicit list
    python3 seed_sweep_v2.py --re 500 --n-seeds 40 --analyze-only   # just stats
    python3 seed_sweep_v2.py --dry-run                       # synthetic smoke test

Re-running the same command after a crash resumes where it stopped.
"""

import os, sys, argparse, json, numpy as np
from scipy.signal import hilbert
from scipy import stats

# ----------------------------------------------------------------------------- args
ap = argparse.ArgumentParser()
ap.add_argument("--re", type=int, default=500)
g = ap.add_mutually_exclusive_group()
g.add_argument("--n-seeds", type=int, help="run seeds 0..N-1 (recommend >=30)")
g.add_argument("--seeds", type=int, nargs='+', help="explicit seed list")
ap.add_argument("--latent", type=int, default=4)
ap.add_argument("--cae-epochs", type=int, default=300)
ap.add_argument("--lstm-epochs", type=int, default=500)
ap.add_argument("--tau", type=int, default=20)
ap.add_argument("--dt", type=float, default=0.03)
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--out-dir", default=None)
ap.add_argument("--alpha", type=float, default=0.05, help="significance level")
ap.add_argument("--analyze-only", action="store_true",
                help="skip training; just recompute stats from saved per-seed files")
ap.add_argument("--dry-run", action="store_true",
                help="synthetic data + tiny epochs, to validate the pipeline")
args = ap.parse_args()

if args.dry_run:
    args.re = args.re or 999
    args.n_seeds = args.n_seeds or 6
    args.cae_epochs = 3; args.lstm_epochs = 3
if args.seeds is not None:
    SEEDS = list(args.seeds)
else:
    SEEDS = list(range(args.n_seeds if args.n_seeds else 40))

RE, LATENT_DIM, DT, TAU = args.re, args.latent, args.dt, args.tau
DATA_DIR = os.path.expanduser(args.data_dir)
OUT_DIR = args.out_dir or os.path.join(DATA_DIR, f"seed_sweep_Re{RE}"
                                       + ("_dry" if args.dry_run else ""))
os.makedirs(OUT_DIR, exist_ok=True)

# Candidate predictors of drift. Keep this list stable and SMALL: every extra
# entry raises the Bonferroni bar. "avail" flags whether it's knowable BEFORE
# the rollout (a genuine predictor) or is itself a rollout co-outcome.
PREDICTORS = [
    ("cae_val",   "CAE reconstruction MSE (compression floor)", True),
    ("lstm_val",  "LSTM one-step validation MSE (fit quality)",  True),
    ("wnorm",     "LSTM weight L2 norm",                         True),
    ("amp_ratio", "predicted/true limit-cycle amplitude",        False),
    ("f_lstm",    "predicted shedding frequency",                False),
]
OUTCOME = "drift_ratio"   # err(last 20)/err(first 20); the paper's drift magnitude

# ----------------------------------------------------------------------------- torch
def build_torch():
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
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
    return torch, nn, DataLoader, TensorDataset, random_split, CAE, LSTMP

# ----------------------------------------------------------------------------- data
def load_snapshots():
    if args.dry_run:
        # synthetic travelling-wave field, shape (N, 3, 64, 128), so the whole
        # encode->latent->LSTM->rollout path runs without the real dataset.
        rng = np.random.default_rng(0)
        N, H, W = 240, 64, 128
        t = np.linspace(0, 12*np.pi, N)
        yy, xx = np.meshgrid(np.linspace(0,1,H), np.linspace(0,1,W), indexing='ij')
        base = np.sin(2*np.pi*xx[None] - t[:,None,None]) * np.exp(-3*(yy[None]-0.5)**2)
        raw = np.stack([base, 0.5*base, 0.2*base], axis=1).astype(np.float32)
        raw += 0.02*rng.standard_normal(raw.shape).astype(np.float32)
        return raw
    path = os.path.join(DATA_DIR, f"snapshots_Re{RE}.npy")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Set --data-dir or check --re.")
    return np.load(path)

# ----------------------------------------------------------------------------- one seed
def run_one_seed(seed, raw):
    torch, nn, DataLoader, TensorDataset, random_split, CAE, LSTMP = TORCH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = nn.MSELoss()
    torch.manual_seed(seed); np.random.seed(seed)

    N = len(raw)
    N_TRAIN = int(N*0.52); N_VAL = int(N*0.08); TEST_START = N_TRAIN + N_VAL

    ch_mean = raw.mean(axis=(0,2,3), keepdims=True)
    ch_std  = raw.std(axis=(0,2,3), keepdims=True); ch_std = np.where(ch_std<1e-8,1.0,ch_std)
    X = torch.tensor(((raw-ch_mean)/ch_std).astype(np.float32))

    # ---- CAE
    cae = CAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(cae.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5, min_lr=1e-5)
    nval=int(len(X)*0.15); ntr=len(X)-nval
    tr,va = random_split(TensorDataset(X),[ntr,nval],
                         generator=torch.Generator().manual_seed(seed))
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
    cae.load_state_dict(best_state); cae.eval(); cae_val=best

    lat=[]
    with torch.no_grad():
        for i in range(0,len(X),64):
            lat.append(cae.encode(X[i:i+64].to(device)).cpu().numpy())
    latents=np.concatenate(lat,0)
    lmean=latents[:N_TRAIN].mean(0,keepdims=True)
    lstd =latents[:N_TRAIN].std(0,keepdims=True); lstd=np.where(lstd<1e-8,1.0,lstd)
    ln=(latents-lmean)/lstd

    # ---- LSTM
    def windows(a,s,e):
        Xs,Ys=[],[]
        for i in range(s,e-TAU): Xs.append(a[i:i+TAU]); Ys.append(a[i+TAU])
        return np.array(Xs,dtype=np.float32), np.array(Ys,dtype=np.float32)
    Xtr,Ytr=windows(ln,0,N_TRAIN); Xva,Yva=windows(ln,N_TRAIN,N_TRAIN+N_VAL)
    trl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Ytr)),batch_size=64,shuffle=True)
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
    lstm.load_state_dict(best_state); lstm.eval(); lstm_val=best
    wnorm=float(np.sqrt(sum((p.detach().cpu().numpy()**2).sum() for p in lstm.parameters())))

    # ---- rollout
    ROLL=len(ln)-TEST_START
    true_all=ln[TEST_START:TEST_START+ROLL]
    win=ln[TEST_START-TAU:TEST_START].copy()
    pred=np.zeros((ROLL,LATENT_DIM),dtype=np.float32)
    with torch.no_grad():
        for s in range(ROLL):
            inp=torch.tensor(win[np.newaxis],dtype=torch.float32).to(device)
            p=lstm(inp).cpu().numpy().squeeze(); pred[s]=p
            win=np.vstack([win[1:],p])

    err=np.linalg.norm(pred-true_all,axis=1)
    drift_ratio=float(err[-20:].mean()/max(err[:20].mean(),1e-12))

    # amplitude ratio, predicted frequency, and phase-drift slope via analytic signal
    amp_r=[]; freqs=[]; slopes=[]
    tsteps=np.arange(ROLL)
    for d in range(LATENT_DIM):
        zp=hilbert(pred[:,d]-pred[:,d].mean()); zt=hilbert(true_all[:,d]-true_all[:,d].mean())
        amp_r.append(np.median(np.abs(zp))/max(np.median(np.abs(zt)),1e-12))
        php=np.unwrap(np.angle(zp)); freqs.append(np.median(np.diff(php))/(2*np.pi*DT))
        dphi=php-np.unwrap(np.angle(zt))
        slopes.append(np.polyfit(tsteps,dphi,1)[0])
    amp_ratio=float(np.mean(amp_r)); f_lstm=float(np.mean(freqs))
    drift_slope=float(np.mean(np.abs(slopes)))

    return dict(seed=int(seed), cae_val=float(cae_val), lstm_val=float(lstm_val),
                wnorm=wnorm, amp_ratio=amp_ratio, f_lstm=f_lstm,
                St=float(f_lstm*0.12/1.0), drift_ratio=drift_ratio,
                drift_slope=drift_slope,
                err_start=float(err[:20].mean()), err_end=float(err[-20:].mean()))

# ----------------------------------------------------------------------------- stats
def crit_r(n, alpha):
    if n < 4: return np.nan
    df=n-2; tc=stats.t.ppf(1-alpha/2, df); return tc/np.sqrt(df+tc*tc)

def fisher_ci(r, n, alpha=0.05):
    if n < 4 or abs(r) >= 1: return (np.nan, np.nan)
    z=np.arctanh(r); se=1/np.sqrt(n-3); zc=stats.norm.ppf(1-alpha/2)
    return (float(np.tanh(z-zc*se)), float(np.tanh(z+zc*se)))

def analyze(results, outcome=OUTCOME):
    n=len(results)
    y=np.array([r[outcome] for r in results], float)
    k=len(PREDICTORS); alpha=args.alpha; alpha_bonf=alpha/k
    print("\n"+"="*72, flush=True)
    print(f"CORRELATION ANALYSIS  Re={RE}   outcome = {outcome}", flush=True)
    print(f"  n = {n} seeds", flush=True)
    print(f"  drift: mean={y.mean():.3f}  std={y.std():.3f}  "
          f"min={y.min():.3f}  max={y.max():.3f}  CV={y.std()/max(abs(y.mean()),1e-9):.3f}", flush=True)
    print(f"  smallest |r| resolvable (p<{alpha}, n={n})           : "
          f"{crit_r(n,alpha):.3f}", flush=True)
    print(f"  smallest |r| resolvable (Bonferroni p<{alpha_bonf:.4f}) : "
          f"{crit_r(n,alpha_bonf):.3f}", flush=True)
    print("-"*72, flush=True)
    print(f"  {'predictor':<11} {'pearson r':>10} {'95% CI':>17} {'p':>8} "
          f"{'spearman':>9} {'sig?':>5}", flush=True)
    rows=[]; any_sig=False
    for key,desc,avail in PREDICTORS:
        x=np.array([r[key] for r in results], float)
        if np.allclose(x.std(),0):
            print(f"  {key:<11} (constant across seeds — skipped)", flush=True); continue
        r,p=stats.pearsonr(x,y); rho,_=stats.spearmanr(x,y); lo,hi=fisher_ci(r,n)
        sig = p<alpha_bonf; any_sig|=sig
        print(f"  {key:<11} {r:>10.3f} [{lo:>6.2f},{hi:>6.2f}] {p:>8.3f} "
              f"{rho:>9.3f} {'YES' if sig else '.':>5}", flush=True)
        rows.append(dict(predictor=key,desc=desc,avail=avail,r=float(r),p=float(p),
                         ci=(lo,hi),spearman=float(rho),significant=bool(sig)))
    print("-"*72, flush=True)
    if any_sig:
        print("  VERDICT: at least one predictor survives Bonferroni correction.", flush=True)
        print("           -> drift is (partly) predictable; report that predictor.", flush=True)
    else:
        rc=crit_r(n,alpha_bonf)
        print(f"  VERDICT: no predictor reaches significance. With n={n} the study is", flush=True)
        print(f"           powered to detect |r|>={rc:.2f}; the confidence intervals above", flush=True)
        print(f"           bound how large an undetected effect could be. This supports", flush=True)
        print(f"           'no predictor of drift stronger than |r|~={rc:.2f}', NOT merely", flush=True)
        print(f"           'p>0.05'.", flush=True)
    print("="*72, flush=True)
    return dict(re=RE,n=n,outcome=outcome,alpha=alpha,bonferroni=alpha_bonf,
                crit_r=float(crit_r(n,alpha)),crit_r_bonf=float(crit_r(n,alpha_bonf)),
                drift_mean=float(y.mean()),drift_std=float(y.std()),
                drift_cv=float(y.std()/max(abs(y.mean()),1e-9)),
                rows=rows,any_significant=bool(any_sig))

# ----------------------------------------------------------------------------- driver
def seed_path(seed): return os.path.join(OUT_DIR, f"seed_{seed:03d}.npy")

def main():
    print("="*72, flush=True)
    print(f"SEED SWEEP v2  Re={RE}  latent={LATENT_DIM}  seeds={SEEDS[0]}..{SEEDS[-1]} "
          f"({len(SEEDS)} total)  dry_run={args.dry_run}", flush=True)
    print("="*72, flush=True)

    results=[]
    if not args.analyze_only:
        global TORCH; TORCH=build_torch()
        raw=load_snapshots()
        print(f"snapshots: {raw.shape}", flush=True)
        for seed in SEEDS:
            sp=seed_path(seed)
            if os.path.exists(sp):                       # ---- resume
                print(f"[seed {seed}] already done -> skip", flush=True)
                results.append(np.load(sp,allow_pickle=True).item()); continue
            print(f"\n--- seed {seed} ---", flush=True)
            r=run_one_seed(seed, raw)
            np.save(sp, r, allow_pickle=True)            # ---- checkpoint immediately
            results.append(r)
            print(f"  cae_val={r['cae_val']:.6f} lstm_val={r['lstm_val']:.6f} "
                  f"wnorm={r['wnorm']:.2f}", flush=True)
            print(f"  DRIFT RATIO={r['drift_ratio']:.2f}x  amp_ratio={r['amp_ratio']:.3f} "
                  f"St={r['St']:.4f}", flush=True)
    else:
        for seed in SEEDS:
            sp=seed_path(seed)
            if os.path.exists(sp): results.append(np.load(sp,allow_pickle=True).item())
        print(f"loaded {len(results)} finished seeds from {OUT_DIR}", flush=True)

    if len(results) < 4:
        sys.exit(f"Only {len(results)} seeds available; need >=4 for correlation.")

    summ_ratio = analyze(results, "drift_ratio")
    summ_slope = analyze(results, "drift_slope")   # cross-check on the phase-rate outcome

    out=os.path.join(OUT_DIR, f"seed_sweep_Re{RE}_analysis.npy")
    np.save(out, dict(results=results, drift_ratio=summ_ratio, drift_slope=summ_slope),
            allow_pickle=True)
    with open(os.path.join(OUT_DIR, f"seed_sweep_Re{RE}_analysis.json"),"w") as f:
        json.dump({"drift_ratio":summ_ratio,"drift_slope":summ_slope}, f, indent=2)
    print(f"\nSaved analysis -> {out}", flush=True)
    print("Send the .json (or .npy) to finalise the item-4 wording.", flush=True)

if __name__ == "__main__":
    main()
