#!/usr/bin/env python3
"""
seed_correction_field.py
========================
Seed-ensemble version of field_validation.py  (reviewer item 9).

seed_correction_sweep.py gave error bars on the LATENT improvement. Table V
(field RMSE, G, rho) is still single-seed, and G = 92-99% is the most quotable
number in the paper -- so it is the one most exposed to the CoV 0.70 seed
variance. This script retrains CAE+LSTM per seed and, for each seed, decodes all
four latent trajectories through that seed's own autoencoder to give
field-space metrics as a distribution.

Per seed it reports, over the extrapolation region:
    eps_floor  CAE floor RMSE      (decode of the TRUE latents)
    eps_unc    uncorrected RMSE
    eps_sin    sinusoid-corrected RMSE
    eps_pha    phase-corrected RMSE
    i_sin_f, i_pha_f   field improvement (%) vs uncorrected
    G          fraction of the CORRECTABLE field error removed
    rho_before, rho_after
    r2         calibration-window R^2 (for the R^2>0.3 conditional column)

Correction code is copied verbatim from phase_correction.py; decode/normalisation
logic verbatim from field_validation.py, so the numbers are directly comparable
with the single-seed table.

Usage
-----
    python3 seed_correction_field.py --dry-run
    python3 seed_correction_field.py --re 400 --latent 4 --dt 0.03    --n-seeds 10
    python3 seed_correction_field.py --re 300 --latent 4 --dt 0.03    --n-seeds 10
    python3 seed_correction_field.py --re 800 --latent 8 --dt 0.03    --n-seeds 10
    python3 seed_correction_field.py --re 101 --latent 4 --dt 0.0438  --n-seeds 10

Resume-safe (per-seed checkpoints; rerun skips finished seeds).
Writes seed_field_Re{RE}_L{LD}_summary.json  -> send to Claude for Table V.

NOTE on --trim: the analytic signal has end artefacts, and omega shifts by
4-14% if the first samples are included in the regression. Default trims 10
samples, matching the treatment adopted in the paper.
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
ap.add_argument("--trim", type=int, default=10,
                help="samples discarded at the start of the phase regression")
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
    args.n_seeds = args.n_seeds or 3
    args.cae_epochs = 3; args.lstm_epochs = 3
SEEDS = list(args.seeds) if args.seeds is not None else list(range(args.n_seeds or 10))

RE, LD, DT, TAU, NCAL, TRIM = (args.re, args.latent, args.dt,
                               args.tau, args.calib, args.trim)
DATA_DIR = os.path.expanduser(args.data_dir)
OUT_DIR = os.path.join(DATA_DIR, f"seed_field_Re{RE}_L{LD}"
                       + ("_dry" if args.dry_run else ""))
os.makedirs(OUT_DIR, exist_ok=True)

# square-cylinder snapshots are tagged SQ100, not Re101 (as in field_validation.py)
SNAP_NAME = "snapshots_SQ100.npy" if RE == 101 else f"snapshots_Re{RE}.npy"


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
        def decode(s,z): return s.decoder(z)
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
    p=os.path.join(DATA_DIR, SNAP_NAME)
    if not os.path.exists(p): sys.exit(f"ERROR: {p} not found (check --re / --data-dir).")
    return np.load(p)


# ---------------------------------------------------------------- corrections
def analytic(sig):
    m=sig.mean(); z=hilbert(sig-m); return m, np.abs(z), np.unwrap(np.angle(z))

def build_corrections(pred, true):
    """returns corr_phase, corr_sin, omega, r2  -- verbatim phase_correction.py
    apart from the TRIM applied to the phase regression."""
    n=min(len(pred),len(true)); pred=pred[:n].astype(float); true=true[:n].astype(float)
    LDn=pred.shape[1]; t=np.arange(n)*DT
    m_p=np.zeros(LDn); A_p=np.zeros((n,LDn)); th_p=np.zeros((n,LDn)); th_t=np.zeros((n,LDn))
    omega=np.zeros(LDn); r2=np.zeros(LDn)
    for d in range(LDn):
        _,_,th_t[:,d]=analytic(true[:,d])
        m_p[d],A_p[:,d],th_p[:,d]=analytic(pred[:,d])
        phi=(th_p[:,d]-th_t[:,d]); phi-=phi[0]
        sl=slice(TRIM, NCAL)
        lr=linregress(t[sl],phi[sl]); omega[d]=lr.slope; r2[d]=lr.rvalue**2
    corr_phase=np.zeros_like(pred)
    for d in range(LDn):
        corr_phase[:,d]=A_p[:,d]*np.cos(th_p[:,d]-omega[d]*t)+m_p[d]
    dim_err=pred-true; t_cal=t[:NCAL]; sig=np.zeros_like(pred)
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
        sig[:,d]=np.clip(A*np.exp(gg*t)*np.sin(th_p[:,d]+psi),-cap,cap)
    return corr_phase, pred-sig, omega, r2


# ---------------------------------------------------------------- one seed
def run_one_seed(seed, raw):
    torch, nn, DataLoader, TensorDataset, random_split, CAE, LSTMP = TORCH
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); crit=nn.MSELoss()
    torch.manual_seed(seed); np.random.seed(seed)

    N=len(raw); N_TRAIN=int(N*0.52); N_VAL=int(N*0.08); TEST_START=N_TRAIN+N_VAL
    ch_mean=raw.mean(axis=(0,2,3),keepdims=True); ch_std=raw.std(axis=(0,2,3),keepdims=True)
    ch_std=np.where(ch_std<1e-8,1.0,ch_std)
    Xt=torch.tensor(((raw-ch_mean)/ch_std).astype(np.float32))

    # ---- CAE (kept in memory: we need its decoder below)
    cae=CAE(LD).to(device); opt=torch.optim.Adam(cae.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=20,factor=0.5,min_lr=1e-5)
    nval=int(len(Xt)*0.15); ntr=len(Xt)-nval
    tr,va=random_split(TensorDataset(Xt),[ntr,nval],
                       generator=torch.Generator().manual_seed(seed))
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

    # ---- LSTM
    def windows(a,s,e):
        Xs,Ys=[],[]
        for i in range(s,e-TAU): Xs.append(a[i:i+TAU]); Ys.append(a[i+TAU])
        return np.array(Xs,dtype=np.float32),np.array(Ys,dtype=np.float32)
    Xtr,Ytr=windows(ln,0,N_TRAIN); Xva,Yva=windows(ln,N_TRAIN,N_TRAIN+N_VAL)
    if len(Xtr)==0 or len(Xva)==0:
        raise ValueError(f"empty windows (train={len(Xtr)}, val={len(Xva)}); N={N}, TAU={TAU}")
    trl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Ytr)),
                   batch_size=64,shuffle=True)
    val=DataLoader(TensorDataset(torch.from_numpy(Xva),torch.from_numpy(Yva)),batch_size=64)
    lstm=LSTMP(LD,64,2).to(device); opt=torch.optim.Adam(lstm.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=30,factor=0.5,min_lr=1e-6)
    best=1e9; bs=None
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
        if vl<best: best=vl; bs={k:v.cpu().clone() for k,v in lstm.state_dict().items()}
    lstm.load_state_dict(bs); lstm.eval(); lstm_val=best

    # ---- rollout
    ROLL=len(ln)-TEST_START; true_all=ln[TEST_START:TEST_START+ROLL]
    win=ln[TEST_START-TAU:TEST_START].copy(); pred=np.zeros((ROLL,LD),dtype=np.float32)
    with torch.no_grad():
        for s in range(ROLL):
            inp=torch.tensor(win[np.newaxis],dtype=torch.float32).to(device)
            p=lstm(inp).cpu().numpy().squeeze(); pred[s]=p; win=np.vstack([win[1:],p])

    corr_pha, corr_sin, omega, r2 = build_corrections(pred, true_all)

    # ---- latent scores (so this file also reproduces the Table III columns)
    def emag(P): return np.linalg.norm(P-true_all,axis=1)
    e_unc=emag(pred); e_sin=emag(corr_sin); e_pha=emag(corr_pha)
    ext=slice(NCAL,ROLL)
    I_sin=float((1-e_sin[ext].mean()/e_unc[ext].mean())*100)
    I_pha=float((1-e_pha[ext].mean()/e_unc[ext].mean())*100)
    drift_ratio=float(e_unc[-20:].mean()/max(e_unc[:20].mean(),1e-12))

    # ---- FIELD SPACE: decode through THIS seed's own autoencoder
    def decode(latn):
        lp=latn*lstd+lmean; outs=[]
        with torch.no_grad():
            for i in range(0,len(lp),64):
                b=torch.tensor(lp[i:i+64],dtype=torch.float32).to(device)
                outs.append(cae.decode(b).cpu().numpy())
        return np.concatenate(outs,0)*ch_std+ch_mean

    ref = raw[TEST_START:TEST_START+ROLL]           # solver snapshots
    f_floor=decode(true_all); f_unc=decode(pred)
    f_sin=decode(corr_sin);   f_pha=decode(corr_pha)
    def frmse(P): return float(np.sqrt(np.mean((P[ext]-ref[ext])**2)))
    eps_floor=frmse(f_floor); eps_unc=frmse(f_unc)
    eps_sin=frmse(f_sin);     eps_pha=frmse(f_pha)
    field_rms=float(np.sqrt(np.mean(ref[ext]**2)))

    i_sin_f=float((1-eps_sin/eps_unc)*100)
    i_pha_f=float((1-eps_pha/eps_unc)*100)
    gap_unc=eps_unc-eps_floor; gap_pha=eps_pha-eps_floor
    G=float((1-gap_pha/gap_unc)*100) if gap_unc>0 else float('nan')
    def rho_of(e):
        v=e**2-eps_floor**2
        return float(np.sqrt(max(v,0.0))/eps_floor) if eps_floor>0 else float('inf')

    return dict(seed=int(seed), cae_val=float(cae_val), lstm_val=float(lstm_val),
                r2_mean=float(np.mean(r2)), omega_mean=float(np.mean(omega)),
                drift_ratio=drift_ratio, I_sin=I_sin, I_pha=I_pha,
                eps_floor=eps_floor, eps_unc=eps_unc, eps_sin=eps_sin, eps_pha=eps_pha,
                field_rms=field_rms, eps_floor_norm=eps_floor/field_rms,
                eps_unc_norm=eps_unc/field_rms, eps_pha_norm=eps_pha/field_rms,
                i_sin_f=i_sin_f, i_pha_f=i_pha_f, G=G,
                rho_before=rho_of(eps_unc), rho_after=rho_of(eps_pha),
                radicand_positive=bool(eps_unc**2-eps_floor**2 > 0))


# ---------------------------------------------------------------- driver
def seed_path(s): return os.path.join(OUT_DIR, f"seed_{s:03d}.npy")

def stats(vals):
    a=np.array(vals,float); a=a[np.isfinite(a)]
    if a.size==0: return dict(mean=float('nan'),std=float('nan'),
                              min=float('nan'),max=float('nan'),n=0)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=1) if a.size>1 else 0.0),
                min=float(a.min()), max=float(a.max()), n=int(a.size))

KEYS=["I_pha","I_sin","i_pha_f","i_sin_f","G","rho_before","rho_after",
      "eps_floor_norm","eps_unc_norm","eps_pha_norm","r2_mean","drift_ratio"]

def summarise(res):
    acc=[r for r in res if r["r2_mean"]>0.3]
    out=dict(re=RE, latent=LD, dt=DT, calib=NCAL, trim=TRIM, n_seeds=len(res),
             n_accepted=len(acc), snapshot_file=SNAP_NAME,
             all_seeds={k:stats([r[k] for r in res]) for k in KEYS},
             accepted_r2_gt_0p3={k:stats([r[k] for r in acc]) for k in KEYS} if acc else {},
             all_radicands_positive=bool(all(r["radicand_positive"] for r in res)),
             per_seed=[{k:r[k] for k in ["seed"]+KEYS} for r in res])
    return out

def main():
    print("="*74, flush=True)
    print(f"SEED FIELD SWEEP  Re={RE} L={LD} dt={DT} calib={NCAL} trim={TRIM}  "
          f"{len(SEEDS)} seeds  dry={args.dry_run}", flush=True)
    print("="*74, flush=True)
    res=[]
    if not args.analyze_only:
        global TORCH; TORCH=build_torch(); raw=load_snapshots()
        print(f"snapshots: {raw.shape}  ({SNAP_NAME})", flush=True)
        for seed in SEEDS:
            sp=seed_path(seed)
            if os.path.exists(sp):
                print(f"[seed {seed}] done -> skip", flush=True)
                res.append(np.load(sp,allow_pickle=True).item()); continue
            print(f"\n--- seed {seed} ---", flush=True)
            r=run_one_seed(seed, raw); np.save(sp,r,allow_pickle=True); res.append(r)
            print(f"  latent : I_pha={r['I_pha']:+.1f}%  R2={r['r2_mean']:.3f}", flush=True)
            print(f"  field  : i_pha_f={r['i_pha_f']:+.1f}%  G={r['G']:.1f}%  "
                  f"rho {r['rho_before']:.2f}->{r['rho_after']:.2f}", flush=True)
    else:
        for seed in SEEDS:
            sp=seed_path(seed)
            if os.path.exists(sp): res.append(np.load(sp,allow_pickle=True).item())
    if len(res)<2: sys.exit("need >=2 finished seeds.")

    S=summarise(res)
    A=S["all_seeds"]; B=S.get("accepted_r2_gt_0p3",{})
    print("\n"+"="*74, flush=True)
    print(f"SUMMARY  Re={RE} L={LD}   {S['n_seeds']} seeds "
          f"({S['n_accepted']} with R^2>0.3)", flush=True)
    print("="*74, flush=True)
    def line(lbl,k,unit="%"):
        a=A[k]; s=f"  {lbl:<30}{a['mean']:>8.2f} +/- {a['std']:<7.2f}"
        if B: s+=f" | R2>0.3: {B[k]['mean']:>7.2f} +/- {B[k]['std']:.2f}"
        print(s, flush=True)
    line("latent I_pha (%)","I_pha"); line("latent I_sin (%)","I_sin")
    line("FIELD i_pha_f (%)","i_pha_f"); line("FIELD i_sin_f (%)","i_sin_f")
    line("G, correctable removed (%)","G")
    line("rho before","rho_before"); line("rho after","rho_after")
    line("eps_floor / field RMS","eps_floor_norm")
    line("eps_unc   / field RMS","eps_unc_norm")
    line("eps_pha   / field RMS","eps_pha_norm")
    print(f"\n  radicand eps_unc^2-eps_CAE^2 positive in every seed: "
          f"{S['all_radicands_positive']}   (rho well defined)", flush=True)
    print("="*74, flush=True)

    out=os.path.join(OUT_DIR, f"seed_field_Re{RE}_L{LD}_summary.json")
    with open(out,"w") as f: json.dump(S,f,indent=2)
    np.save(out.replace(".json",".npy"), S, allow_pickle=True)
    print(f"\nSaved -> {out}", flush=True)
    print("Send this JSON to Claude for the Table V error bars.", flush=True)

if __name__ == "__main__":
    main()
