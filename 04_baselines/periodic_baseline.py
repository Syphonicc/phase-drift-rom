#!/usr/bin/env python3
"""
periodic_baseline.py
====================
The baseline reviewer M4 asks for, and the one the community will reach for.

On a saturated limit cycle, the cheapest possible predictor is to TILE THE LAST
OBSERVED PERIOD FORWARD. It needs no model, no training and no latent space --
only the assumption that the flow is periodic. If it beats the corrected
CAE-LSTM, the pipeline needs a different justification (parametric or 3D
generalisation) and the paper must say so.

Four variants are evaluated, all using ONLY data available at prediction time
(the calibration window, i.e. exactly what the phase correction is given):

  (a) latent tiling          -- repeat the last measured period of the latent
                                trajectory, period estimated from the calibration
                                window by autocorrelation with parabolic refinement
  (b) latent tiling, phase-locked -- as (a) but with the tiling offset chosen to
                                minimise mismatch at the splice
  (c) field tiling           -- repeat the last measured period of the SOLVER
                                snapshots directly (no autoencoder at all): the
                                true zero-model baseline in physical space
  (d) mean field             -- predict the time-mean field (a sanity floor)

Scored against the same truth, over the same extrapolation region, with the same
metric as Eq. (16), so the numbers drop straight into the paper's tables.

Usage
-----
    python3 periodic_baseline.py --re 400 --latent 4 --dt 0.03
    python3 periodic_baseline.py --re 300 --latent 4 --dt 0.03
    python3 periodic_baseline.py --re 800 --latent 8 --dt 0.03
    python3 periodic_baseline.py --re 101 --latent 4 --dt 0.0438
    python3 periodic_baseline.py --re 400 --latent 4 --no-field   # skip decoding

Outputs periodic_baseline_Re{RE}_L{LD}.json -> send to Claude.
"""

import os, sys, argparse, json, numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress
from scipy.optimize import curve_fit

ap = argparse.ArgumentParser()
ap.add_argument("--re", type=int, required=True)
ap.add_argument("--latent", type=int, default=4)
ap.add_argument("--dt", type=float, default=0.03)
ap.add_argument("--calib", type=int, default=150)
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--no-field", action="store_true")
args = ap.parse_args()

RE, LD, DT, NCAL = args.re, args.latent, args.dt, args.calib
DATA_DIR = os.path.expanduser(args.data_dir)
TAG = f"Re{RE}_L{LD}"
SNAP = "snapshots_SQ100.npy" if RE == 101 else f"snapshots_Re{RE}.npy"

def find(c, required=True):
    for x in c:
        p = os.path.join(DATA_DIR, x)
        if os.path.exists(p): return p
    if required: sys.exit(f"ERROR: none of {c} in {DATA_DIR}")
    return None

print("="*76, flush=True)
print(f"PERIODIC-EXTENSION BASELINE   Re={RE}  latent={LD}  calib={NCAL}", flush=True)
print("="*76, flush=True)

roll = np.load(find([f"rollout_{TAG}.npy", f"rollout_Re{RE}.npy"]), allow_pickle=True).item()
true = roll['true'].astype(float)
pred = (roll['raw_pred'] if 'raw_pred' in roll else roll['predicted']).astype(float)
n = min(len(true), len(pred)); true, pred = true[:n], pred[:n]
LDn = true.shape[1]; t = np.arange(n)*DT
print(f"  rollout {n} steps, {LDn} latent dims; extrapolation = steps {NCAL}..{n}", flush=True)

# ---------------------------------------------------------------- tiling helper
def estimate_period(sig, lo=4):
    """samples per period, from autocorrelation of the CALIBRATION window only"""
    x = sig - sig.mean()
    ac = np.correlate(x, x, mode='full')[len(x)-1:]
    ac /= ac[0]
    d = np.diff(ac)
    start = np.argmax(d > 0)                     # first rise after the central lobe
    k = start + np.argmax(ac[start:])            # first true peak
    if k < lo or k+1 >= len(ac): return None
    a0, a1, a2 = ac[k-1], ac[k], ac[k+1]         # parabolic refinement
    denom = (a0 - 2*a1 + a2)
    return k + (0.5*(a0-a2)/denom if abs(denom) > 1e-12 else 0.0)

def tile(series, period, upto, total, shift=0.0):
    """Periodic extension with a FRACTIONAL period.

    Rounding the period to an integer would make the tiling accumulate its own
    phase error (0.17 samples per period here, i.e. a fifth of a cycle over the
    horizon) -- precisely the drift this paper is about, and an unfair handicap
    on the baseline. We therefore map each future time back into the last
    observed period at fractional resolution and interpolate.
    """
    out = series.copy()
    m = len(series)
    for k in range(upto, total):
        # fold t_k back into the window [upto-period, upto)
        src = upto - period + ((k - upto + shift) % period)
        i0 = int(np.floor(src)); w = src - i0
        i0 = min(max(i0, 0), m-2)
        out[k] = (1.0-w)*series[i0] + w*series[i0+1]
    return out

def period_fft(sig, dt, pad=32):
    """period from a zero-padded periodogram with parabolic peak refinement --
    substantially more precise than autocorrelation on a short window, and the
    precision of P is what limits periodic extension over a long horizon."""
    x = sig - sig.mean()
    N = len(x)*pad
    F = np.abs(np.fft.rfft(x*np.hanning(len(x)), n=N))
    f = np.fft.rfftfreq(N, dt)
    k = np.argmax(F[1:])+1
    a0,a1,a2 = F[k-1],F[k],F[k+1]
    den = a0-2*a1+a2
    kk = k + (0.5*(a0-a2)/den if abs(den)>1e-12 else 0.0)
    f0 = kk*(f[1]-f[0])
    return 1.0/(f0*dt) if f0>0 else None

cands = {}
pa = [estimate_period(true[:NCAL,d]) for d in range(LDn)]
pa = [p for p in pa if p is not None]
if pa: cands['autocorrelation'] = float(np.median(pa))
pf = [period_fft(true[:NCAL,d], DT) for d in range(LDn)]
pf = [p for p in pf if p is not None]
if pf: cands['periodogram'] = float(np.median(pf))
if not cands: sys.exit("ERROR: could not estimate a period from the calibration window.")
# pick whichever reproduces the calibration window best when tiled within it
def _score(Pc):
    half = NCAL//2
    try:
        c = tile(true.copy(), Pc, half, NCAL)
        return float(np.linalg.norm(c[half:NCAL]-true[half:NCAL],axis=1).mean())
    except Exception: return np.inf
for nm,v in cands.items(): print(f"  period candidate ({nm}): {v:.4f} samples", flush=True)
P = min(cands.values(), key=_score)
print(f"  period from calibration window: {P:.3f} samples "
      f"({P*DT:.4f} time units, St = {0.12/(P*DT):.4f})", flush=True)

# ---------------------------------------------------------------- (a),(b) latent tiling

tiled_latent = tile(true.copy(), P, NCAL, n)     # truth known only up to NCAL

best_off, best_err = 0.0, np.inf
for off in np.arange(-1.0, 1.0, 0.02):        # sub-sample splice alignment
    cand = tile(true.copy(), P, NCAL, n, shift=off)
    e = np.linalg.norm(cand[NCAL:NCAL+20]-true[NCAL:NCAL+20], axis=1).mean()
    if e < best_err: best_err, best_off = e, float(off)
tiled_locked = tile(true.copy(), P, NCAL, n, shift=best_off)
print(f"  phase-locked tiling shift: {best_off:+.3f} samples", flush=True)

# ---------------------------------------------------------------- phase correction
def analytic(s):
    m=s.mean(); z=hilbert(s-m); return m, np.abs(z), np.unwrap(np.angle(z))
A_p=np.zeros((n,LDn)); th_p=np.zeros((n,LDn)); m_p=np.zeros(LDn); th_t=np.zeros((n,LDn))
om=np.zeros(LDn); r2=np.zeros(LDn)
for d in range(LDn):
    _,_,th_t[:,d]=analytic(true[:,d]); m_p[d],A_p[:,d],th_p[:,d]=analytic(pred[:,d])
    phi=th_p[:,d]-th_t[:,d]; phi-=phi[0]
    lr=linregress(t[:NCAL],phi[:NCAL]); om[d]=lr.slope; r2[d]=lr.rvalue**2
corr_pha=np.zeros_like(pred)
for d in range(LDn): corr_pha[:,d]=A_p[:,d]*np.cos(th_p[:,d]-om[d]*t)+m_p[d]

de=pred-true; tc=t[:NCAL]; sg=np.zeros_like(pred)
for d in range(LDn):
    e=de[:NCAL,d]; th=th_p[:NCAL,d]
    def mf(i,A,g,ps):
        i=i.astype(int); return A*np.exp(g*tc[i])*np.sin(th[i]+ps)
    try:
        po,_=curve_fit(mf,np.arange(NCAL,dtype=float),e,p0=[e.std(),.1,0.],
                       bounds=([-1,0.,-2*np.pi],[1,1.,2*np.pi]),maxfev=100000)
    except Exception: po=[e.std(),.1,0.]
    A,g,ps=po; cap=np.abs(e).max()*1.5
    sg[:,d]=np.clip(A*np.exp(g*t)*np.sin(th_p[:,d]+ps),-cap,cap)
corr_sin=pred-sg

# ---------------------------------------------------------------- latent scores
ext=slice(NCAL,n)
def emag(P_): return np.linalg.norm(P_-true,axis=1)
base=emag(pred)[ext].mean()
lat={}
for nm,P_ in [('lstm_uncorrected',pred),('lstm_sinusoid',corr_sin),
              ('lstm_phase',corr_pha),('periodic_tiling',tiled_latent),
              ('periodic_tiling_locked',tiled_locked)]:
    m=float(emag(P_)[ext].mean())
    lat[nm]=dict(mean_err=m, ratio_to_uncorrected=float(m/base),
                 improvement_pct=float((1-m/base)*100))
print("\n  LATENT SPACE (extrapolation region)", flush=True)
print(f"    {'method':<26}{'mean err':>12}{'ratio':>9}{'improvement':>13}", flush=True)
for k,v in lat.items():
    print(f"    {k:<26}{v['mean_err']:>12.5f}{v['ratio_to_uncorrected']:>9.3f}"
          f"{v['improvement_pct']:>+12.1f}%", flush=True)

# ---------------------------------------------------------------- field space
field={}
if not args.no_field:
    try:
        import torch, torch.nn as nn
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        FLAT=64*8*16
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
        cae=CAE(LDn).to(dev)
        sd=torch.load(find([f"cae_{TAG}.pt", f"cae_Re{RE}.pt"]), map_location=dev)
        if any(k.startswith("encoder.conv") for k in sd):     # legacy layer names
            sd={k.replace("encoder.conv.","encoder.c.").replace("encoder.fc.","encoder.f.")
                 .replace("decoder.fc.","decoder.f.").replace("decoder.deconv.","decoder.d."):v
                for k,v in sd.items()}
        cae.load_state_dict(sd); cae.eval()
        cn=np.load(find([f"norm_{TAG}.npy", f"norm_Re{RE}.npy"]),allow_pickle=True).item()
        lnm=np.load(find([f"latent_norm_{TAG}.npy", f"latent_norm_Re{RE}.npy"]),allow_pickle=True).item()
        snaps=np.load(find([SNAP]))
        N=len(snaps); TEST_START=int(N*0.52)+int(N*0.08)
        ref=snaps[TEST_START:TEST_START+n]
        def decode(latn):
            lp=latn*lnm['std']+lnm['mean']; outs=[]
            with torch.no_grad():
                for i in range(0,len(lp),64):
                    b=torch.tensor(lp[i:i+64],dtype=torch.float32).to(dev)
                    outs.append(cae.decode(b).cpu().numpy())
            return np.concatenate(outs,0)*cn['std']+cn['mean']
        def frmse(Pf): return float(np.sqrt(np.mean((Pf[ext]-ref[ext])**2)))

        # (c) FIELD tiling: repeat the last observed period of the SOLVER fields.
        #     No autoencoder, no model -- the true zero-model baseline.
        field_tiled=ref.copy().astype(float)
        for k in range(NCAL, n):
            src=NCAL - P + ((k-NCAL+best_off) % P)
            i0=int(np.floor(src)); w=src-i0; i0=min(max(i0,0),len(ref)-2)
            field_tiled[k]=(1.0-w)*ref[i0]+w*ref[i0+1]
        # (d) mean field
        field_mean=np.repeat(ref[:NCAL].mean(0,keepdims=True), n, axis=0)

        f_floor=decode(true); f_unc=decode(pred)
        f_sin=decode(corr_sin); f_pha=decode(corr_pha)
        f_til=decode(tiled_locked)
        base_f=frmse(f_unc); rms=float(np.sqrt(np.mean(ref[ext]**2)))
        for nm,Pf in [('cae_floor',f_floor),('lstm_uncorrected',f_unc),
                      ('lstm_sinusoid',f_sin),('lstm_phase',f_pha),
                      ('periodic_tiling_latent',f_til),
                      ('periodic_tiling_field',field_tiled),
                      ('mean_field',field_mean)]:
            r=frmse(Pf)
            field[nm]=dict(rmse=r, normalised=r/rms,
                           improvement_pct=float((1-r/base_f)*100))
        print(f"\n  FIELD SPACE (field RMS = {rms:.4f})", flush=True)
        print(f"    {'method':<26}{'RMSE':>12}{'/RMS':>10}{'vs unc':>10}", flush=True)
        for k,v in field.items():
            print(f"    {k:<26}{v['rmse']:>12.6f}{v['normalised']:>10.4f}"
                  f"{v['improvement_pct']:>+9.1f}%", flush=True)
    except Exception as ex:
        print(f"\n  [field space skipped: {ex}]", flush=True)

# ---------------------------------------------------------------- verdict
print("\n" + "="*76, flush=True)
pt = lat['periodic_tiling_locked']['ratio_to_uncorrected']
ph = lat['lstm_phase']['ratio_to_uncorrected']
if pt < ph:
    print(f"  PERIODIC TILING WINS in latent space ({pt:.3f} vs {ph:.3f}).", flush=True)
    print("  The pipeline needs a justification beyond accuracy on this flow.", flush=True)
else:
    print(f"  Phase-corrected CAE-LSTM wins ({ph:.3f} vs {pt:.3f}); "
          f"tiling is {pt/ph:.1f}x worse.", flush=True)
if field:
    ptf=field.get('periodic_tiling_field',{}).get('rmse',np.inf)
    phf=field['lstm_phase']['rmse']
    print(f"  Field space: model-free tiling {ptf:.6f} vs phase-corrected {phf:.6f} "
          f"-> {'TILING WINS' if ptf<phf else 'model wins'}", flush=True)
print("="*76, flush=True)

out=os.path.join(DATA_DIR, f"periodic_baseline_{TAG}.json")
json.dump(dict(re=RE, latent=LDn, dt=DT, calib=NCAL, n=n,
               period_samples=P, period_time=P*DT, tiling_offset=best_off,
               r2_mean=float(np.mean(r2)), latent_scores=lat, field=field),
          open(out,"w"), indent=2, default=float)
print(f"\nSaved -> {out}", flush=True)
