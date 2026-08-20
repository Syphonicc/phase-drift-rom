#!/usr/bin/env python3
"""
seed_correction_sweep.py — error bars on the correction-efficacy tables (reviewer #1).
Retrains CAE+LSTM across seeds for ONE case, applies the EXACT phase correction and
sinusoid baseline from phase_correction.py, reports I_pha / I_sin as mean +/- SD.

  python3 seed_correction_sweep.py --re 400 --latent 4 --dt 0.03   --n-seeds 10
  python3 seed_correction_sweep.py --re 300 --latent 4 --dt 0.03   --n-seeds 10
  python3 seed_correction_sweep.py --re 800 --latent 8 --dt 0.03   --n-seeds 10
  python3 seed_correction_sweep.py --re 101 --latent 4 --dt 0.0438 --n-seeds 10
  python3 seed_correction_sweep.py --dry-run
Resume-safe; outputs seed_correction_Re{RE}_L{LD}_summary.json
"""
import os, sys, argparse, json, numpy as np
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.stats import linregress

ap = argparse.ArgumentParser()
ap.add_argument("--re", type=int, default=400)
ap.add_argument("--latent", type=int, default=4)
ap.add_argument("--dt", type=float, default=0.03)
ap.add_argument("--calib", type=int, default=150)
g = ap.add_mutually_exclusive_group()
g.add_argument("--n-seeds", type=int)
g.add_argument("--seeds", type=int, nargs='+')
ap.add_argument("--cae-epochs", type=int, default=300)
ap.add_argument("--lstm-epochs", type=int, default=500)
ap.add_argument("--tau", type=int, default=20)
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--analyze-only", action="store_true")
args = ap.parse_args()

if args.dry_run:
    args.n_seeds = args.n_seeds or 4
    args.cae_epochs = 3; args.lstm_epochs = 3
SEEDS = list(args.seeds) if args.seeds is not None else list(range(args.n_seeds or 10))
RE, LD, DT, TAU, NCAL = args.re, args.latent, args.dt, args.tau, args.calib
DATA_DIR = os.path.expanduser(args.data_dir)
OUT_DIR = os.path.join(DATA_DIR, f"seed_corr_Re{RE}_L{LD}" + ("_dry" if args.dry_run else ""))
os.makedirs(OUT_DIR, exist_ok=True)
HORIZ_FRACS = [0.25, 0.5, 0.75, 1.0]

def build_torch():
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    FLAT = 64*8*16
    class Encoder(nn.Module):
        def __init__(s, ld):
            super().__init__()
            s.c = nn.Sequential(nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
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
    p=os.path.join(DATA_DIR,f"snapshots_Re{RE}.npy")
    if not os.path.exists(p): sys.exit(f"ERROR: {p} not found (check --re / --data-dir).")
    return np.load(p)

def analytic(sig):
    m=sig.mean(); z=hilbert(sig-m); return m, np.abs(z), np.unwrap(np.angle(z))

def apply_corrections(pred, true):
    n=min(len(pred),len(true)); pred=pred[:n].astype(float); true=true[:n].astype(float)
    LDn=pred.shape[1]; t=np.arange(n)*DT
    m_p=np.zeros(LDn); A_p=np.zeros((n,LDn)); th_p=np.zeros((n,LDn))
    th_t=np.zeros((n,LDn)); omega=np.zeros(LDn); r2=np.zeros(LDn)
    for d in range(LDn):
        _,_,th_t[:,d]=analytic(true[:,d])
        m_p[d],A_p[:,d],th_p[:,d]=analytic(pred[:,d])
        phi=(th_p[:,d]-th_t[:,d]); phi-=phi[0]
        lr=linregress(t[:NCAL],phi[:NCAL]); omega[d]=lr.slope; r2[d]=lr.rvalue**2
    corr_phase=np.zeros_like(pred)
    for d in range(LDn):
        corr_phase[:,d]=A_p[:,d]*np.cos(th_p[:,d]-omega[d]*t)+m_p[d]
    dim_err=pred-true; t_cal=t[:NCAL]; corr_sin_sig=np.zeros_like(pred)
    for d in range(LDn):
        e=dim_err[:NCAL,d]; th=th_p[:NCAL,d]
        def mf(idx,A,gg,psi):
            idx=idx.astype(int); return A*np.exp(gg*t_cal[idx])*np.sin(th[idx]+psi)
        try:
            popt,_=curve_fit(mf,np.arange(NCAL,dtype=float),e,p0=[e.std(),0.1,0.0],
                             bounds=([-1,0.0,-2*np.pi],[1,1.0,2*np.pi]),maxfev=100000)
        except Exception:
            popt=[e.std(),0.1,0.0]
        A,gg,psi=popt; cap=np.abs(e).max()*1.5
        corr_sin_sig[:,d]=np.clip(A*np.exp(gg*t)*np.sin(th_p[:,d]+psi),-cap,cap)
    corr_sin=pred-corr_sin_sig
    def emag(P): return np.linalg.norm(P-true,axis=1)
    e_unc=emag(pred); e_sin=emag(corr_sin); e_pha=emag(corr_phase)
    ext=slice(NCAL,n)
    I_sin=float((1-e_sin[ext].mean()/e_unc[ext].mean())*100)
    I_pha=float((1-e_pha[ext].mean()/e_unc[ext].mean())*100)
    drift_ratio=float(e_unc[-20:].mean()/max(e_unc[:20].mean(),1e-12))
    horizon=[]
    for f in HORIZ_FRACS:
        hi=NCAL+int((n-NCAL)*f); sl=slice(NCAL,hi)
        horizon.append(float((1-e_pha[sl].mean()/e_unc[sl].mean())*100))
    return dict(I_sin=I_sin, I_pha=I_pha, r2_mean=float(np.mean(r2)),
                drift_ratio=drift_ratio, horizon=horizon)

def run_one_seed(seed, raw):
    torch, nn, DataLoader, TensorDataset, random_split, CAE, LSTMP = TORCH
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); crit=nn.MSELoss()
    torch.manual_seed(seed); np.random.seed(seed)
    N=len(raw); N_TRAIN=int(N*0.52); N_VAL=int(N*0.08); TEST_START=N_TRAIN+N_VAL
    ch_mean=raw.mean(axis=(0,2,3),keepdims=True); ch_std=raw.std(axis=(0,2,3),keepdims=True)
    ch_std=np.where(ch_std<1e-8,1.0,ch_std)
    Xt=torch.tensor(((raw-ch_mean)/ch_std).astype(np.float32))
    cae=CAE(LD).to(device); opt=torch.optim.Adam(cae.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=20,factor=0.5,min_lr=1e-5)
    nval=int(len(Xt)*0.15); ntr=len(Xt)-nval
    tr,va=random_split(TensorDataset(Xt),[ntr,nval],generator=torch.Generator().manual_seed(seed))
    trl=DataLoader(tr,batch_size=32,shuffle=True); val=DataLoader(va,batch_size=32)
    best=1e9; bs=None
    for ep in range(args.cae_epochs):
        cae.train()
        for (b,) in trl: b=b.to(device); opt.zero_grad(); crit(cae(b),b).backward(); opt.step()
        cae.eval(); vl=0
        with torch.no_grad():
            for (b,) in val: b=b.to(device); vl+=crit(cae(b),b).item()*len(b)
        vl/=nval; sch.step(vl)
        if vl<best: best=vl; bs={k:v.cpu().clone() for k,v in cae.state_dict().items()}
    cae.load_state_dict(bs); cae.eval(); cae_val=best
    lat=[]
    with torch.no_grad():
        for i in range(0,len(Xt),64): lat.append(cae.encode(Xt[i:i+64].to(device)).cpu().numpy())
    latents=np.concatenate(lat,0)
    lmean=latents[:N_TRAIN].mean(0,keepdims=True); lstd=latents[:N_TRAIN].std(0,keepdims=True)
    lstd=np.where(lstd<1e-8,1.0,lstd); ln=(latents-lmean)/lstd
    def windows(a,s,e):
        Xs,Ys=[],[]
        for i in range(s,e-TAU): Xs.append(a[i:i+TAU]); Ys.append(a[i+TAU])
        return np.array(Xs,dtype=np.float32),np.array(Ys,dtype=np.float32)
    Xtr,Ytr=windows(ln,0,N_TRAIN); Xva,Yva=windows(ln,N_TRAIN,N_TRAIN+N_VAL)
    if len(Xtr)==0 or len(Xva)==0:
        raise ValueError(f"empty windows (train={len(Xtr)}, val={len(Xva)}); N={N}, TAU={TAU}")
    trl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Ytr)),batch_size=64,shuffle=True)
    val=DataLoader(TensorDataset(torch.from_numpy(Xva),torch.from_numpy(Yva)),batch_size=64)
    lstm=LSTMP(LD,64,2).to(device); opt=torch.optim.Adam(lstm.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=30,factor=0.5,min_lr=1e-6)
    best=1e9; bs=None
    for ep in range(args.lstm_epochs):
        lstm.train()
        for xb,yb in trl:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); l=crit(lstm(xb),yb); l.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(),1.0); opt.step()
        lstm.eval(); vl=0
        with torch.no_grad():
            for xb,yb in val: xb,yb=xb.to(device),yb.to(device); vl+=crit(lstm(xb),yb).item()*len(xb)
        vl/=len(Xva); sch.step(vl)
        if vl<best: best=vl; bs={k:v.cpu().clone() for k,v in lstm.state_dict().items()}
    lstm.load_state_dict(bs); lstm.eval(); lstm_val=best
    ROLL=len(ln)-TEST_START; true_all=ln[TEST_START:TEST_START+ROLL]
    win=ln[TEST_START-TAU:TEST_START].copy(); pred=np.zeros((ROLL,LD),dtype=np.float32)
    with torch.no_grad():
        for s in range(ROLL):
            inp=torch.tensor(win[np.newaxis],dtype=torch.float32).to(device)
            p=lstm(inp).cpu().numpy().squeeze(); pred[s]=p; win=np.vstack([win[1:],p])
    c=apply_corrections(pred, true_all)
    return dict(seed=int(seed), cae_val=float(cae_val), lstm_val=float(lstm_val), **c)

def seed_path(s): return os.path.join(OUT_DIR,f"seed_{s:03d}.npy")

def summarise(results):
    def ms(key):
        a=np.array([r[key] for r in results],float)
        return dict(mean=float(a.mean()),std=float(a.std(ddof=1) if len(a)>1 else 0.0),
                    min=float(a.min()),max=float(a.max()),n=len(a))
    hor=np.array([r["horizon"] for r in results],float)
    horizon=dict(fracs=HORIZ_FRACS, mean=hor.mean(0).tolist(),
                 std=(hor.std(0,ddof=1) if len(hor)>1 else np.zeros(hor.shape[1])).tolist())
    return dict(re=RE, latent=LD, dt=DT, calib=NCAL, n_seeds=len(results),
                I_pha=ms("I_pha"), I_sin=ms("I_sin"), r2_mean=ms("r2_mean"),
                drift_ratio=ms("drift_ratio"), horizon=horizon,
                per_seed=[{k:r[k] for k in ("seed","I_pha","I_sin","r2_mean","drift_ratio")}
                          for r in results])

def main():
    print("="*72, flush=True)
    print(f"SEED CORRECTION SWEEP  Re={RE} L={LD} dt={DT} calib={NCAL}  "
          f"{len(SEEDS)} seeds  dry={args.dry_run}", flush=True)
    print("="*72, flush=True)
    results=[]
    if not args.analyze_only:
        global TORCH; TORCH=build_torch(); raw=load_snapshots()
        print(f"snapshots: {raw.shape}", flush=True)
        for seed in SEEDS:
            sp=seed_path(seed)
            if os.path.exists(sp):
                print(f"[seed {seed}] done -> skip", flush=True)
                results.append(np.load(sp,allow_pickle=True).item()); continue
            print(f"\n--- seed {seed} ---", flush=True)
            r=run_one_seed(seed, raw); np.save(sp,r,allow_pickle=True); results.append(r)
            print(f"  I_pha={r['I_pha']:+.1f}%  I_sin={r['I_sin']:+.1f}%  "
                  f"R2={r['r2_mean']:.3f}  drift={r['drift_ratio']:.2f}x", flush=True)
    else:
        for seed in SEEDS:
            sp=seed_path(seed)
            if os.path.exists(sp): results.append(np.load(sp,allow_pickle=True).item())
    if len(results)<2: sys.exit("need >=2 finished seeds to summarise.")
    S=summarise(results)
    print("\n"+"="*72, flush=True)
    print(f"SUMMARY  Re={RE} L={LD}   ({S['n_seeds']} seeds)", flush=True)
    print("="*72, flush=True)
    print(f"  phase correction  I_pha = {S['I_pha']['mean']:+.1f} +/- {S['I_pha']['std']:.1f}%"
          f"   (min {S['I_pha']['min']:+.1f}, max {S['I_pha']['max']:+.1f})", flush=True)
    print(f"  sinusoid baseline I_sin = {S['I_sin']['mean']:+.1f} +/- {S['I_sin']['std']:.1f}%", flush=True)
    print(f"  calib R^2               = {S['r2_mean']['mean']:.3f} +/- {S['r2_mean']['std']:.3f}", flush=True)
    print(f"  drift ratio             = {S['drift_ratio']['mean']:.2f} +/- {S['drift_ratio']['std']:.2f}x", flush=True)
    print(f"  I_pha vs horizon (mean +/- SD):", flush=True)
    for f,m,s in zip(S['horizon']['fracs'],S['horizon']['mean'],S['horizon']['std']):
        print(f"      {int(f*100):>3}% of extrap: {m:+.1f} +/- {s:.1f}%", flush=True)
    print("="*72, flush=True)
    out=os.path.join(OUT_DIR,f"seed_correction_Re{RE}_L{LD}_summary.json")
    with open(out,"w") as f: json.dump(S,f,indent=2)
    np.save(out.replace(".json",".npy"), S, allow_pickle=True)
    print(f"\nSaved -> {out}", flush=True)

if __name__ == "__main__":
    main()
