#!/usr/bin/env python3
"""
noise_injection_test.py
=======================
Reviewer M5: does training-side noise injection simply remove the phase drift,
making the post-hoc correction unnecessary?

Trains matched seed ensembles at several noise levels and reports, for each:
    omega      the fitted phase-drift rate
    R^2        linearity of the phase fit on the calibration window
    I_pha      improvement from the phase correction
    drift      end/start error ratio of the uncorrected rollout
    lstm_val   one-step validation error (noise should not wreck this)

Interpretation:
  - if |omega| and drift fall towards zero as noise rises, and I_pha collapses
    because there is nothing left to correct, then noise injection is an
    alternative remedy and the paper must say so;
  - if omega and drift are unchanged while lstm_val degrades, the drift is not
    a distribution-shift artefact and the correction remains necessary. Either
    result is worth reporting.

Everything except the noise is identical to run_pipeline.py / phase_correction.py.

Usage
-----
    python3 noise_injection_test.py --dry-run
    python3 noise_injection_test.py --re 400 --latent 4 --dt 0.03 --n-seeds 10
    python3 noise_injection_test.py --re 400 --noise 0.0 0.01 0.03 0.10 --n-seeds 10
    python3 noise_injection_test.py --re 400 --analyze-only

Resume-safe:each (noise, seed) pair is checkpointed; rerunning skips finished work.
Writes noise_injection_Re{RE}_L{LD}_summary.json  -> send to Claude.
"""

import os, sys, argparse, json, numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress

ap = argparse.ArgumentParser()
ap.add_argument("--re", type=int, default=400)
ap.add_argument("--latent", type=int, default=4)
ap.add_argument("--dt", type=float, default=0.03)
ap.add_argument("--calib", type=int, default=150)
ap.add_argument("--tau", type=int, default=20)
ap.add_argument("--noise", type=float, nargs='+', default=[0.0, 0.01, 0.03, 0.10],
                help="std of Gaussian noise added to LSTM inputs during training")
ap.add_argument("--n-seeds", type=int, default=10)
ap.add_argument("--cae-epochs", type=int, default=300)
ap.add_argument("--lstm-epochs", type=int, default=500)
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--analyze-only", action="store_true")
args = ap.parse_args()

if args.dry_run:
    args.n_seeds = min(args.n_seeds, 3)
    args.cae_epochs = 3; args.lstm_epochs = 3
    args.noise = [0.0, 0.05]

RE, LD, DT, TAU, NCAL = args.re, args.latent, args.dt, args.tau, args.calib
SEEDS = list(range(args.n_seeds))
DATA_DIR = os.path.expanduser(args.data_dir)
OUT_DIR = os.path.join(DATA_DIR, f"noise_Re{RE}_L{LD}" + ("_dry" if args.dry_run else ""))
os.makedirs(OUT_DIR, exist_ok=True)
SNAP = "snapshots_SQ100.npy" if RE == 101 else f"snapshots_Re{RE}.npy"


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


def load_snapshots():
    if args.dry_run:
        rng=np.random.default_rng(0); N,H,W=600,64,128
        tt=np.linspace(0,12*np.pi,N)
        yy,xx=np.meshgrid(np.linspace(0,1,H),np.linspace(0,1,W),indexing='ij')
        base=np.sin(2*np.pi*xx[None]-tt[:,None,None])*np.exp(-3*(yy[None]-0.5)**2)
        raw=np.stack([base,0.5*base,0.2*base],axis=1).astype(np.float32)
        return raw+0.02*rng.standard_normal(raw.shape).astype(np.float32)
    p=os.path.join(DATA_DIR, SNAP)
    if not os.path.exists(p): sys.exit(f"ERROR: {p} not found (check --re / --data-dir).")
    return np.load(p)


def analytic(sig):
    m=sig.mean(); z=hilbert(sig-m); return m, np.abs(z), np.unwrap(np.angle(z))


def phase_correct(pred, true):
    """verbatim phase_correction.py: fit omega on the calibration window, realign"""
    n=min(len(pred),len(true)); pred=pred[:n].astype(float); true=true[:n].astype(float)
    LDn=pred.shape[1]; t=np.arange(n)*DT
    m_p=np.zeros(LDn); A_p=np.zeros((n,LDn)); th_p=np.zeros((n,LDn)); th_t=np.zeros((n,LDn))
    om=np.zeros(LDn); r2=np.zeros(LDn)
    for d in range(LDn):
        _,_,th_t[:,d]=analytic(true[:,d])
        m_p[d],A_p[:,d],th_p[:,d]=analytic(pred[:,d])
        phi=(th_p[:,d]-th_t[:,d]); phi-=phi[0]
        lr=linregress(t[:NCAL],phi[:NCAL]); om[d]=lr.slope; r2[d]=lr.rvalue**2
    corr=np.zeros_like(pred)
    for d in range(LDn): corr[:,d]=A_p[:,d]*np.cos(th_p[:,d]-om[d]*t)+m_p[d]
    e_unc=np.linalg.norm(pred-true,axis=1); e_cor=np.linalg.norm(corr-true,axis=1)
    ext=slice(NCAL,n)
    return dict(omega_mean=float(np.mean(om)), omega_absmean=float(np.mean(np.abs(om))),
                r2_mean=float(np.mean(r2)),
                I_pha=float((1-e_cor[ext].mean()/e_unc[ext].mean())*100),
                drift_ratio=float(e_unc[-20:].mean()/max(e_unc[:20].mean(),1e-12)),
                err_unc=float(e_unc[ext].mean()))


def run_one(seed, noise, raw):
    torch, nn, DataLoader, TensorDataset, random_split, CAE, LSTMP = TORCH
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); crit=nn.MSELoss()
    torch.manual_seed(seed); np.random.seed(seed)
    N=len(raw); N_TRAIN=int(N*0.52); N_VAL=int(N*0.08); TEST_START=N_TRAIN+N_VAL
    cm=raw.mean(axis=(0,2,3),keepdims=True); cs=raw.std(axis=(0,2,3),keepdims=True)
    cs=np.where(cs<1e-8,1.0,cs)
    Xt=torch.tensor(((raw-cm)/cs).astype(np.float32))

    cae=CAE(LD).to(dev); opt=torch.optim.Adam(cae.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=20,factor=0.5,min_lr=1e-5)
    nval=int(len(Xt)*0.15); ntr=len(Xt)-nval
    tr,va=random_split(TensorDataset(Xt),[ntr,nval],
                       generator=torch.Generator().manual_seed(seed))
    trl=DataLoader(tr,batch_size=32,shuffle=True); val=DataLoader(va,batch_size=32)
    best=1e9; bs=None
    for ep in range(args.cae_epochs):
        cae.train()
        for (b,) in trl: b=b.to(dev); opt.zero_grad(); crit(cae(b),b).backward(); opt.step()
        cae.eval(); vl=0
        with torch.no_grad():
            for (b,) in val: b=b.to(dev); vl+=crit(cae(b),b).item()*len(b)
        vl/=nval; sch.step(vl)
        if vl<best: best=vl; bs={k:v.cpu().clone() for k,v in cae.state_dict().items()}
    cae.load_state_dict(bs); cae.eval(); cae_val=best

    lat=[]
    with torch.no_grad():
        for i in range(0,len(Xt),64): lat.append(cae.encode(Xt[i:i+64].to(dev)).cpu().numpy())
    L=np.concatenate(lat,0)
    lm=L[:N_TRAIN].mean(0,keepdims=True); ls=L[:N_TRAIN].std(0,keepdims=True)
    ls=np.where(ls<1e-8,1.0,ls); ln=(L-lm)/ls

    def windows(a,s,e):
        Xs,Ys=[],[]
        for i in range(s,e-TAU): Xs.append(a[i:i+TAU]); Ys.append(a[i+TAU])
        return np.array(Xs,dtype=np.float32),np.array(Ys,dtype=np.float32)
    Xtr,Ytr=windows(ln,0,N_TRAIN); Xva,Yva=windows(ln,N_TRAIN,N_TRAIN+N_VAL)
    if len(Xtr)==0 or len(Xva)==0:
        raise ValueError(f"empty windows (train={len(Xtr)}, val={len(Xva)})")
    trl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Ytr)),
                   batch_size=64,shuffle=True)
    val=DataLoader(TensorDataset(torch.from_numpy(Xva),torch.from_numpy(Yva)),batch_size=64)
    lstm=LSTMP(LD,64,2).to(dev); opt=torch.optim.Adam(lstm.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=30,factor=0.5,min_lr=1e-6)
    best=1e9; bs=None
    for ep in range(args.lstm_epochs):
        lstm.train()
        for xb,yb in trl:
            xb,yb=xb.to(dev),yb.to(dev)
            if noise>0: xb = xb + noise*torch.randn_like(xb)     # <-- the intervention
            opt.zero_grad(); l=crit(lstm(xb),yb); l.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(),1.0); opt.step()
        lstm.eval(); vl=0
        with torch.no_grad():                                    # validation is CLEAN
            for xb,yb in val:
                xb,yb=xb.to(dev),yb.to(dev); vl+=crit(lstm(xb),yb).item()*len(xb)
        vl/=len(Xva); sch.step(vl)
        if vl<best: best=vl; bs={k:v.cpu().clone() for k,v in lstm.state_dict().items()}
    lstm.load_state_dict(bs); lstm.eval(); lstm_val=best

    ROLL=len(ln)-TEST_START; true=ln[TEST_START:TEST_START+ROLL]
    win=ln[TEST_START-TAU:TEST_START].copy(); pred=np.zeros((ROLL,LD),dtype=np.float32)
    with torch.no_grad():
        for s in range(ROLL):
            inp=torch.tensor(win[np.newaxis],dtype=torch.float32).to(dev)
            p=lstm(inp).cpu().numpy().squeeze(); pred[s]=p; win=np.vstack([win[1:],p])

    r=phase_correct(pred, true)
    r.update(seed=int(seed), noise=float(noise),
             cae_val=float(cae_val), lstm_val=float(lstm_val))
    return r


def cpath(noise, seed):
    return os.path.join(OUT_DIR, f"n{noise:g}_seed{seed:03d}.npy")

def stats(v):
    a=np.array(v,float); a=a[np.isfinite(a)]
    if a.size==0: return dict(mean=float('nan'),std=float('nan'),n=0)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=1) if a.size>1 else 0.0),
                min=float(a.min()), max=float(a.max()), n=int(a.size))

KEYS=["omega_absmean","r2_mean","I_pha","drift_ratio","lstm_val","cae_val","err_unc"]

def main():
    print("="*78, flush=True)
    print(f"NOISE-INJECTION TEST  Re={RE} L={LD}  noise={args.noise}  "
          f"{len(SEEDS)} seeds each  dry={args.dry_run}", flush=True)
    print("="*78, flush=True)
    results={}
    if not args.analyze_only:
        global TORCH; TORCH=build_torch(); raw=load_snapshots()
        print(f"snapshots: {raw.shape}", flush=True)
        for nz in args.noise:
            results[nz]=[]
            for sd in SEEDS:
                p=cpath(nz,sd)
                if os.path.exists(p):
                    print(f"[noise={nz:g} seed={sd}] done -> skip", flush=True)
                    results[nz].append(np.load(p,allow_pickle=True).item()); continue
                print(f"\n--- noise={nz:g}  seed={sd} ---", flush=True)
                r=run_one(sd,nz,raw); np.save(p,r,allow_pickle=True); results[nz].append(r)
                print(f"    omega={r['omega_absmean']:.3e}  R2={r['r2_mean']:.3f}  "
                      f"I_pha={r['I_pha']:+.1f}%  drift={r['drift_ratio']:.2f}  "
                      f"lstm_val={r['lstm_val']:.2e}", flush=True)
    else:
        for nz in args.noise:
            results[nz]=[np.load(cpath(nz,sd),allow_pickle=True).item()
                         for sd in SEEDS if os.path.exists(cpath(nz,sd))]

    print("\n"+"="*78, flush=True)
    print(f"SUMMARY  Re={RE} L={LD}", flush=True)
    print("="*78, flush=True)
    print(f"  {'noise':>7}{'n':>4}{'|omega|':>13}{'R^2':>14}{'I_pha %':>15}"
          f"{'drift':>13}{'lstm_val':>12}", flush=True)
    summ={}
    for nz in args.noise:
        R=results.get(nz,[])
        if not R: continue
        s={k:stats([r[k] for r in R]) for k in KEYS}; summ[str(nz)]=s
        print(f"  {nz:>7g}{len(R):>4}"
              f"{s['omega_absmean']['mean']:>9.2e}±{s['omega_absmean']['std']:<.0e}"
              f"{s['r2_mean']['mean']:>9.3f}±{s['r2_mean']['std']:<4.2f}"
              f"{s['I_pha']['mean']:>9.1f}±{s['I_pha']['std']:<4.1f}"
              f"{s['drift_ratio']['mean']:>8.2f}±{s['drift_ratio']['std']:<4.2f}"
              f"{s['lstm_val']['mean']:>12.2e}", flush=True)

    if len(summ)>1:
        ks=sorted(summ, key=float)
        o0=summ[ks[0]]['omega_absmean']['mean']; oN=summ[ks[-1]]['omega_absmean']['mean']
        d0=summ[ks[0]]['drift_ratio']['mean'];   dN=summ[ks[-1]]['drift_ratio']['mean']
        v0=summ[ks[0]]['lstm_val']['mean'];      vN=summ[ks[-1]]['lstm_val']['mean']
        print("\n  VERDICT", flush=True)
        print(f"    |omega|  {o0:.2e} -> {oN:.2e}   ({100*(oN/o0-1):+.0f}%)", flush=True)
        print(f"    drift    {d0:.2f} -> {dN:.2f}       ({100*(dN/d0-1):+.0f}%)", flush=True)
        print(f"    val err  {v0:.2e} -> {vN:.2e}   ({100*(vN/v0-1):+.0f}%)", flush=True)
        if dN < 0.5*d0:
            print("    -> noise injection SUBSTANTIALLY REDUCES the drift;", flush=True)
            print("       the paper must report it as an alternative remedy.", flush=True)
        else:
            print("    -> drift SURVIVES noise injection; the correction remains", flush=True)
            print("       necessary and this is a positive result for the paper.", flush=True)
    print("="*78, flush=True)

    out=os.path.join(OUT_DIR, f"noise_injection_Re{RE}_L{LD}_summary.json")
    json.dump(dict(re=RE, latent=LD, dt=DT, calib=NCAL, seeds=len(SEEDS),
                   noise_levels=args.noise, summary=summ,
                   per_run={str(nz):[{k:r[k] for k in ["seed"]+KEYS}
                                     for r in results.get(nz,[])] for nz in args.noise}),
              open(out,"w"), indent=2, default=float)
    print(f"\nSaved -> {out}", flush=True)

if __name__ == "__main__":
    main()
