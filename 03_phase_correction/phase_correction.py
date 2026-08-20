#!/usr/bin/env python3
"""
phase_correction.py
===================
Corrects rollout drift by removing the PHASE ERROR ANALYTICALLY.

WHY THIS SHOULD WORK
--------------------
decompose_error.py established that the rollout error is 96-101% PHASE error:
the LSTM reproduces the amplitude of the limit cycle essentially perfectly
(ratio 1.000-1.001) but its phase slips linearly in time.

The earlier time-warp attempt failed for a NUMERICAL reason, not a physical
one: the required time shift is ~0.001 time units against a sample spacing of
0.03, so np.interp was being asked to do sub-sample resampling of an
oscillating signal and introduced far more error than it removed.

THE FIX
-------
Do not resample. Reconstruct the corrected signal directly from its analytic
representation. Write the prediction as

    z_hat(t) = A_hat(t) * cos( theta_hat(t) )     [Hilbert]

and simply SUBTRACT the modelled phase error from the phase:

    z_corrected(t) = A_hat(t) * cos( theta_hat(t) - phi_model(t) )

where phi_model(t) = omega * t is fitted on the calibration window ONLY.
No interpolation. Exact at every sample.

ONE parameter per latent dimension (omega). Compare against:
  - uncorrected
  - the existing phase-locked SINUSOID correction (3 params/dim + cap)

Usage:
    python3 phase_correction.py --re 400 --latent 4
    python3 phase_correction.py --re 101 --latent 4 --dt 0.0438
"""

import os, argparse, numpy as np
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--re",     type=int,   required=True)
ap.add_argument("--latent", type=int,   default=4)
ap.add_argument("--dt",     type=float, default=0.03)
ap.add_argument("--calib",  type=int,   default=150)
args = ap.parse_args()

RE, LD, DT, NCAL = args.re, args.latent, args.dt, args.calib
DATA_DIR = os.path.expanduser("~/cylinder_lstm_data")

def find(c):
    for x in c:
        p = os.path.join(DATA_DIR, x)
        if os.path.exists(p): return p
    raise FileNotFoundError(c)

roll = np.load(find([f"rollout_Re{RE}_L{LD}.npy", f"rollout_Re{RE}.npy"]),
               allow_pickle=True).item()
pred = roll['raw_pred'] if 'raw_pred' in roll else roll['predicted']
true = roll['true']
n = min(len(pred), len(true))
pred, true = pred[:n].astype(float), true[:n].astype(float)
LD = pred.shape[1]
t = np.arange(n)*DT

print("="*76, flush=True)
print(f"ANALYTIC PHASE CORRECTION — Re={RE}  latent={LD}  calib={NCAL}", flush=True)
print(f"  rollout {n} steps; scoring on steps {NCAL}..{n} (extrapolation only)",
      flush=True)
print("="*76, flush=True)

def analytic(sig):
    m = sig.mean()
    z = hilbert(sig - m)
    return m, np.abs(z), np.unwrap(np.angle(z))

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — fit the phase-drift rate on the CALIBRATION window only
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Fitting phase-drift rate omega (calibration window only)\n", flush=True)

m_t = np.zeros(LD); A_t = np.zeros((n,LD)); th_t = np.zeros((n,LD))
m_p = np.zeros(LD); A_p = np.zeros((n,LD)); th_p = np.zeros((n,LD))
omega = np.zeros(LD); r2 = np.zeros(LD)

for d in range(LD):
    m_t[d], A_t[:,d], th_t[:,d] = analytic(true[:,d])
    m_p[d], A_p[:,d], th_p[:,d] = analytic(pred[:,d])
    phi = (th_p[:,d] - th_t[:,d]); phi -= phi[0]
    lr = linregress(t[:NCAL], phi[:NCAL])
    omega[d] = lr.slope
    r2[d] = lr.rvalue**2
    print(f"    z{d}: omega = {omega[d]:+.6f} rad/t   R^2(calib) = {r2[d]:.4f}",
          flush=True)

# how well does calibration-fitted omega extrapolate?
print("\n    Extrapolation quality of the linear phase model:", flush=True)
for d in range(LD):
    phi = (th_p[:,d] - th_t[:,d]); phi -= phi[0]
    resid = np.abs(omega[d]*t - phi)[NCAL:]
    denom = np.abs(phi[NCAL:]).mean()
    print(f"    z{d}: mean |residual| = {resid.mean():.5f} rad "
          f"({resid.mean()/max(denom,1e-12)*100:.1f}% of |phi|)", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — ANALYTIC PHASE CORRECTION (no interpolation)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Applying the analytic phase correction", flush=True)
print("    z_corr(t) = A_hat(t) * cos( theta_hat(t) - omega*t )  +  mean",
      flush=True)

corr_phase = np.zeros_like(pred)
for d in range(LD):
    phi_model = omega[d] * t
    corr_phase[:,d] = A_p[:,d] * np.cos(th_p[:,d] - phi_model) + m_p[d]

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — baseline: the existing phase-locked sinusoid correction
# ══════════════════════════════════════════════════════════════════════════════
print("[3] Applying the existing sinusoid correction (baseline)", flush=True)
dim_err = pred - true
t_cal = t[:NCAL]
corr_sin_sig = np.zeros_like(pred)
for d in range(LD):
    e  = dim_err[:NCAL, d]
    th = th_p[:NCAL, d]
    def mf(idx, A, g, psi):
        idx = idx.astype(int)
        return A*np.exp(g*t_cal[idx])*np.sin(th[idx] + psi)
    try:
        popt,_ = curve_fit(mf, np.arange(NCAL,dtype=float), e,
                           p0=[e.std(),0.1,0.0],
                           bounds=([-1,0.0,-2*np.pi],[1,1.0,2*np.pi]),
                           maxfev=100000)
    except Exception:
        popt=[e.std(),0.1,0.0]
    A,g,psi = popt
    cap = np.abs(e).max()*1.5
    corr_sin_sig[:,d] = np.clip(A*np.exp(g*t)*np.sin(th_p[:,d]+psi), -cap, cap)
corr_sin = pred - corr_sin_sig

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — score
# ══════════════════════════════════════════════════════════════════════════════
def emag(P): return np.linalg.norm(P-true, axis=1)
e_unc = emag(pred); e_sin = emag(corr_sin); e_pha = emag(corr_phase)
ext = slice(NCAL, n)
def imp(e): return (1 - e[ext].mean()/e_unc[ext].mean())*100
I_sin, I_pha = imp(e_sin), imp(e_pha)

print("\n" + "="*76, flush=True)
print("RESULTS — extrapolation region", flush=True)
print("="*76, flush=True)
print(f"  {'method':<38} {'mean err':>10} {'improvement':>13} {'params':>7}",
      flush=True)
print("  " + "-"*72, flush=True)
print(f"  {'uncorrected':<38} {e_unc[ext].mean():>10.5f} {'--':>13} {'--':>7}",
      flush=True)
print(f"  {'phase-locked sinusoid (baseline)':<38} {e_sin[ext].mean():>10.5f} "
      f"{I_sin:>+12.1f}% {'3/dim':>7}", flush=True)
print(f"  {'ANALYTIC PHASE CORRECTION (new)':<38} {e_pha[ext].mean():>10.5f} "
      f"{I_pha:>+12.1f}% {'1/dim':>7}", flush=True)
print("  " + "-"*72, flush=True)

if I_pha > I_sin + 2:
    print(f"\n  ==> PHASE CORRECTION WINS by {I_pha-I_sin:.1f} points, "
          f"with 1/3 the parameters.", flush=True)
    print(f"      Correcting the cause beats fitting the symptom.", flush=True)
elif I_pha > I_sin - 2:
    print(f"\n  ==> COMPARABLE. The sinusoid was implicitly doing the same thing,",
          flush=True)
    print(f"      but the phase correction achieves it with 1 parameter and no cap.",
          flush=True)
else:
    print(f"\n  ==> Sinusoid still better by {I_sin-I_pha:.1f} points.", flush=True)

# horizon dependence — the key test: does phase correction hold up longer?
print("\n  Improvement vs horizon (does it degrade?):", flush=True)
print(f"  {'steps past calib':>18} {'sinusoid':>10} {'phase':>9}", flush=True)
for frac in [0.25, 0.5, 0.75, 1.0]:
    hi = NCAL + int((n-NCAL)*frac)
    sl = slice(NCAL, hi)
    i_s = (1 - e_sin[sl].mean()/e_unc[sl].mean())*100
    i_p = (1 - e_pha[sl].mean()/e_unc[sl].mean())*100
    print(f"  {hi-NCAL:>18} {i_s:>+9.1f}% {i_p:>+8.1f}%", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2,2, figsize=(14,9))
fig.suptitle(f"Analytic phase correction vs sinusoid — Re={RE}, latent={LD}",
             fontsize=13, fontweight='bold')

ax = axes[0,0]
ax.plot(t, e_unc, color='tomato',    lw=1.2, label='uncorrected', alpha=.85)
ax.plot(t, e_sin, color='orange',    lw=1.2, label=f'sinusoid ({I_sin:+.0f}%)', alpha=.85)
ax.plot(t, e_pha, color='seagreen',  lw=1.3, label=f'phase corr ({I_pha:+.0f}%)', alpha=.9)
ax.axvline(NCAL*DT, color='grey', ls='--', label='calib end')
ax.set_xlabel('t'); ax.set_ylabel('latent L2 error')
ax.set_title('Error trajectory'); ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axes[0,1]
ax.semilogy(t, e_unc+1e-9, color='tomato',   lw=1.2, label='uncorrected', alpha=.85)
ax.semilogy(t, e_sin+1e-9, color='orange',   lw=1.2, label='sinusoid', alpha=.85)
ax.semilogy(t, e_pha+1e-9, color='seagreen', lw=1.3, label='phase corr', alpha=.9)
ax.axvline(NCAL*DT, color='grey', ls='--')
ax.set_xlabel('t'); ax.set_ylabel('latent L2 error (log)')
ax.set_title('Log scale'); ax.legend(fontsize=8); ax.grid(alpha=.3, which='both')

ax = axes[1,0]
for d in range(LD):
    phi = th_p[:,d]-th_t[:,d]; phi -= phi[0]
    ax.plot(t, phi, lw=1.0, alpha=.6, label=f'z{d} measured')
    ax.plot(t, omega[d]*t, ls='--', lw=1.2, alpha=.9, label=f'z{d} model')
ax.axvline(NCAL*DT, color='grey', ls='--')
ax.set_xlabel('t'); ax.set_ylabel(r'$\phi$ [rad]')
ax.set_title(r'Phase error: measured vs linear model $\omega t$ (fit on calib only)')
ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)

ax = axes[1,1]
d = 0
sl = slice(int(n*0.88), n)
ax.plot(t[sl], true[sl,d],       color='k',        lw=2.0, label='truth', alpha=.85)
ax.plot(t[sl], pred[sl,d],       color='tomato',   lw=1.3, ls='--', label='prediction')
ax.plot(t[sl], corr_phase[sl,d], color='seagreen', lw=1.3, label='phase-corrected')
ax.set_xlabel('t'); ax.set_ylabel(f'$z_{d}$')
ax.set_title('Latent trajectory, late horizon (zoom)\nthe phase slip should be visibly removed')
ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout()
fp = os.path.join(DATA_DIR, f"phasecorr_Re{RE}_L{LD}.png")
plt.savefig(fp, dpi=150, bbox_inches='tight')
print(f"\n  Figure -> {fp}", flush=True)

np.save(os.path.join(DATA_DIR, f"phasecorr_Re{RE}_L{LD}_ncal{NCAL}.npy"),
        {'omega':omega, 'I_sin':I_sin, 'I_pha':I_pha, 'r2':r2,
         'e_unc':e_unc, 'e_sin':e_sin, 'e_pha':e_pha,
         'corrected':corr_phase, 'ncal':NCAL, 'dt':DT}, allow_pickle=True)
print("="*76, flush=True)
