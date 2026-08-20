#!/usr/bin/env python3
"""
dmd_baseline.py
===============
Linear-propagator baselines for the CAE latent dynamics (reviewer item 1).

Fits, on the SAME latent trajectory the LSTM was trained on:
  (a) DMD  -- one linear operator  x_{k+1} = A x_k     (raw least squares)
  (b) DMD with eigenvalues projected onto the unit circle (no growth/decay)
  (c) DMD refit including the calibration window (fairest to the baseline)
  (d) a harmonic / Fourier model at the dominant frequency (+ 2nd harmonic)

and scores each against the CAE-LSTM in BOTH latent and field space.

Also measures each baseline's OWN phase drift, because a linear propagator
does NOT eliminate phase drift -- its frequency is estimated from data, so a
frequency error delta_w produces a phase error delta_w * t, linear in time,
exactly the paper's mechanism. This gives a second (non-recurrent) propagator
exhibiting the same failure mode.

Usage:
    python3 dmd_baseline.py --re 400 --latent 4 --dt 0.03
    python3 dmd_baseline.py --re 800 --latent 8 --dt 0.03
    python3 dmd_baseline.py --re 101 --latent 4 --dt 0.0438
    python3 dmd_baseline.py --re 400 --latent 4 --no-field   # skip decoding

Outputs (per case):
    dmd_baseline_Re{RE}_L{LD}.json   <- summary numbers, send to Claude
    dmd_baseline_Re{RE}_L{LD}.npy    <- full trajectories, for the plot script
"""

import os, sys, argparse, json, numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress

ap = argparse.ArgumentParser()
ap.add_argument("--re", type=int, required=True)
ap.add_argument("--latent", type=int, default=4)
ap.add_argument("--dt", type=float, default=0.03)
ap.add_argument("--calib", type=int, default=150)
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--no-field", action="store_true",
                help="skip field-space scoring (no torch / no CAE checkpoint needed)")
args = ap.parse_args()

RE, LD, DT, NCAL = args.re, args.latent, args.dt, args.calib
DATA_DIR = os.path.expanduser(args.data_dir)
TAG = f"Re{RE}_L{LD}"

def find(cands, required=True):
    for c in cands:
        p = os.path.join(DATA_DIR, c)
        if os.path.exists(p): return p
    if required: sys.exit(f"ERROR: none of {cands} found in {DATA_DIR}")
    return None

print("="*74, flush=True)
print(f"DMD / LINEAR BASELINE   Re={RE}  latent={LD}  dt={DT}  calib={NCAL}", flush=True)
print("="*74, flush=True)

# ---------------------------------------------------------------- load
lat  = np.load(find([f"latents_{TAG}.npy", f"latents_Re{RE}.npy"]))
roll = np.load(find([f"rollout_{TAG}.npy", f"rollout_Re{RE}.npy"]), allow_pickle=True).item()
N = len(lat)
N_TRAIN = int(N*0.52); N_VAL = int(N*0.08); TEST_START = N_TRAIN + N_VAL

lmean = lat[:N_TRAIN].mean(0, keepdims=True)
lstd  = lat[:N_TRAIN].std(0, keepdims=True); lstd = np.where(lstd < 1e-8, 1.0, lstd)
ln = (lat - lmean) / lstd

true_all = roll['true'].astype(float)
lstm_raw = (roll['raw_pred'] if 'raw_pred' in roll else roll['predicted']).astype(float)
lstm_cor = roll['corrected'].astype(float) if 'corrected' in roll else None
ROLL = len(true_all)
print(f"  latents {lat.shape}  TEST_START={TEST_START}  ROLL={ROLL}", flush=True)
chk = np.abs(ln[TEST_START:TEST_START+ROLL] - true_all).max()
print(f"  consistency check (stored truth vs recomputed): {chk:.2e}", flush=True)

# ---------------------------------------------------------------- DMD
def fit_A(end):
    """least-squares one-step linear operator on ln[0:end]"""
    X, Xp = ln[0:end-1].T, ln[1:end].T
    return Xp @ np.linalg.pinv(X)

def unit_circle(A):
    """project eigenvalues onto |lambda|=1 (pure oscillation, no growth/decay)"""
    w, V = np.linalg.eig(A)
    w = w / np.abs(w)
    return np.real(V @ np.diag(w) @ np.linalg.inv(V))

def propagate(A, x0, n):
    out = np.zeros((n, len(x0))); x = x0.copy()
    for k in range(n): x = A @ x; out[k] = x
    return out

def spectrum(A):
    w = np.linalg.eig(A)[0]
    om = np.log(w.astype(complex)) / DT
    return w, np.real(om), np.imag(om)/(2*np.pi)     # lambda, growth, freq[Hz]

A_tr = fit_A(N_TRAIN)
lam, growth, freq = spectrum(A_tr)
print("\n  DMD spectrum (fit on training window):", flush=True)
for j in range(len(lam)):
    print(f"    |lambda|={abs(lam[j]):.4f}  growth={growth[j]:+.4f}/t  f={freq[j]:+.4f} Hz  "
          f"St={abs(freq[j])*0.12:.4f}", flush=True)

preds = {}
preds['dmd_train']      = propagate(A_tr, ln[TEST_START-1], ROLL)
preds['dmd_unitcircle'] = propagate(unit_circle(A_tr), ln[TEST_START-1], ROLL)
# fairest: refit including the same calibration window the phase correction uses
A_cal = fit_A(TEST_START + NCAL)
full = np.zeros((ROLL, LD)); full[:NCAL] = np.nan
full[NCAL:] = propagate(A_cal, ln[TEST_START+NCAL-1], ROLL-NCAL)
preds['dmd_calib'] = full

# ---------------------------------------------------------------- harmonic
def harmonic(end, nharm=2):
    n = end; t = np.arange(n)*DT; out = np.zeros((ROLL, LD)); f0s = []
    for d in range(LD):
        s = ln[:n, d] - ln[:n, d].mean()
        F = np.abs(np.fft.rfft(s*np.hanning(n))); f = np.fft.rfftfreq(n, DT)
        b = np.where((f > 0.2) & (f < 10))[0]; i = b[np.argmax(F[b])]
        a0, a1, a2 = F[i-1], F[i], F[i+1]           # parabolic peak refinement
        f0 = f[i] + 0.5*(a0-a2)/(a0-2*a1+a2)*(f[1]-f[0]); f0s.append(f0)
        cols = [np.ones(n)]
        for h in range(1, nharm+1):
            cols += [np.cos(2*np.pi*h*f0*t), np.sin(2*np.pi*h*f0*t)]
        M = np.column_stack(cols)
        c, *_ = np.linalg.lstsq(M, ln[:n, d], rcond=None)
        resid = float(np.sqrt(np.mean((M@c - ln[:n, d])**2)))
        tf = (np.arange(ROLL) + n)*DT
        colsf = [np.ones(ROLL)]
        for h in range(1, nharm+1):
            colsf += [np.cos(2*np.pi*h*f0*tf), np.sin(2*np.pi*h*f0*tf)]
        out[:, d] = np.column_stack(colsf) @ c
        if d == 0:
            print(f"\n  harmonic fit dim0: f0={f0:.4f} Hz, train RMS residual={resid:.5f}",
                  flush=True)
    return out, f0s
preds['harmonic'], f0s = harmonic(TEST_START)

# ---------------------------------------------------------------- latent scoring
ext = slice(NCAL, ROLL)
def emag(P): return np.linalg.norm(P[:ROLL] - true_all, axis=1)
e_lstm = emag(lstm_raw); base = e_lstm[ext].mean()
scores = {}
def add(name, P):
    e = emag(P); m = float(np.nanmean(e[ext]))
    scores[name] = dict(mean_err=m, vs_lstm_pct=float((1 - m/base)*100),
                        ratio_to_lstm=float(m/base))
    return m
add('lstm_uncorrected', lstm_raw)
if lstm_cor is not None: add('lstm_phase_corrected', lstm_cor)
for k, P in preds.items(): add(k, P)

print("\n  LATENT-SPACE, extrapolation region (steps %d-%d):" % (NCAL, ROLL), flush=True)
print(f"    {'method':<26}{'mean err':>12}{'vs LSTM':>11}{'ratio':>9}", flush=True)
for k, v in scores.items():
    print(f"    {k:<26}{v['mean_err']:>12.5f}{v['vs_lstm_pct']:>10.1f}%{v['ratio_to_lstm']:>8.1f}x",
          flush=True)

# ---------------------------------------------------------------- phase drift OF the baselines
def drift_rate(P):
    """linear phase-drift rate of prediction P vs truth, per latent coord"""
    t = np.arange(ROLL)*DT; om = []; r2 = []
    for d in range(LD):
        p = P[:, d]; q = true_all[:, d]
        if np.any(np.isnan(p)):
            om.append(np.nan); r2.append(np.nan); continue
        php = np.unwrap(np.angle(hilbert(p - np.nanmean(p))))
        pht = np.unwrap(np.angle(hilbert(q - q.mean())))
        phi = php - pht; phi -= phi[0]
        lr = linregress(t[:NCAL], phi[:NCAL])
        om.append(float(lr.slope)); r2.append(float(lr.rvalue**2))
    return float(np.nanmean(om)), float(np.nanmean(r2))

print("\n  PHASE DRIFT of each propagator (the paper's mechanism):", flush=True)
print(f"    {'method':<26}{'omega [rad/t]':>15}{'R^2':>9}", flush=True)
drift = {}
for name, P in [('lstm_uncorrected', lstm_raw)] + list(preds.items()):
    om, r2 = drift_rate(P); drift[name] = dict(omega=om, r2=r2)
    print(f"    {name:<26}{om:>15.3e}{r2:>9.3f}", flush=True)

# ---------------------------------------------------------------- field space
field = {}
if not args.no_field:
    try:
        import torch, torch.nn as nn
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        FLAT = 64*8*16
        class Decoder(nn.Module):
            def __init__(s, ld):
                super().__init__(); s.f = nn.Linear(ld, FLAT)
                s.d = nn.Sequential(
                    nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
                    nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
                    nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
                    nn.Conv2d(16,3,3,padding=1))
            def forward(s,z): return s.d(s.f(z).view(-1,64,8,16))
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
        class CAE(nn.Module):
            def __init__(s, ld):
                super().__init__(); s.encoder=Encoder(ld); s.decoder=Decoder(ld)
            def decode(s,z): return s.decoder(z)
        cae = CAE(LD).to(dev)
        cae.load_state_dict(torch.load(find([f"cae_{TAG}.pt", f"cae_Re{RE}.pt"]),
                                       map_location=dev))
        cae.eval()
        nrm = np.load(find([f"norm_{TAG}.npy", f"norm_Re{RE}.npy"]), allow_pickle=True).item()
        ch_mean, ch_std = nrm['mean'], nrm['std']
        snaps = np.load(find([f"snapshots_Re{RE}.npy"]))[TEST_START:TEST_START+ROLL]

        def decode(latn):
            lp = latn*lstd + lmean; outs=[]
            with torch.no_grad():
                for i in range(0, len(lp), 64):
                    b = torch.tensor(np.nan_to_num(lp[i:i+64]), dtype=torch.float32).to(dev)
                    outs.append(cae.decode(b).cpu().numpy())
            return np.concatenate(outs,0)*ch_std + ch_mean
        def frmse(P, R, sl): return float(np.sqrt(np.mean((P[sl]-R[sl])**2)))

        f_true = decode(true_all); floor = frmse(f_true, snaps, ext)
        rms_field = float(np.sqrt(np.mean(snaps[ext]**2)))
        field['cae_floor_rmse'] = floor
        field['field_rms'] = rms_field
        print(f"\n  FIELD-SPACE (CAE floor RMSE = {floor:.6f}, field RMS = {rms_field:.4f}):",
              flush=True)
        print(f"    {'method':<26}{'RMSE':>12}{'/field RMS':>12}{'vs LSTM':>10}", flush=True)
        f_lstm = decode(lstm_raw); base_f = frmse(f_lstm, snaps, ext)
        items = [('lstm_uncorrected', lstm_raw)]
        if lstm_cor is not None: items.append(('lstm_phase_corrected', lstm_cor))
        items += list(preds.items())
        for name, P in items:
            r = frmse(decode(P), snaps, ext)
            field[name] = dict(rmse=r, normalised=r/rms_field,
                               vs_lstm_pct=float((1-r/base_f)*100))
            print(f"    {name:<26}{r:>12.6f}{r/rms_field:>12.4f}{(1-r/base_f)*100:>9.1f}%",
                  flush=True)
    except Exception as ex:
        print(f"\n  [field-space skipped: {ex}]", flush=True)

# ---------------------------------------------------------------- save
summary = dict(re=RE, latent=LD, dt=DT, calib=NCAL, roll=ROLL, test_start=TEST_START,
               dmd_eigs=dict(abs=np.abs(lam).tolist(), growth=growth.tolist(),
                             freq_hz=freq.tolist(), St=(np.abs(freq)*0.12).tolist()),
               harmonic_f0=f0s, latent_scores=scores, phase_drift=drift, field=field)
jp = os.path.join(DATA_DIR, f"dmd_baseline_{TAG}.json")
with open(jp, "w") as fh: json.dump(summary, fh, indent=2, default=float)
np.save(os.path.join(DATA_DIR, f"dmd_baseline_{TAG}.npy"),
        dict(summary=summary, true=true_all, lstm_raw=lstm_raw, lstm_cor=lstm_cor,
             preds=preds, e_lstm=e_lstm, ln=ln, TEST_START=TEST_START), allow_pickle=True)
print(f"\n  Saved -> {jp}", flush=True)
print(f"  Saved -> dmd_baseline_{TAG}.npy  (trajectories, for the plot script)", flush=True)
print("="*74, flush=True)
