#!/usr/bin/env python3
"""
timewarp_correction.py
======================
Corrects rollout drift AT ITS SOURCE by re-timing the prediction, rather than
fitting and subtracting the error signal it produces.

THE IDEA
--------
The verification (verify_mechanism.py) showed that the accumulated phase error
between prediction and truth,

    phi(t) = theta_pred(t) - theta_true(t),

grows LINEARLY in time:   phi(t) ~= omega * t.

That means the LSTM is reproducing the right waveform on a clock that runs at
slightly the wrong rate. The prediction at time t is showing the state the true
flow was in at a shifted time:

    z_hat(t)  ~=  z_true( t - phi(t) / (2*pi*f_s) )

So instead of modelling the ERROR, we undo the CLOCK ERROR: resample the
predicted trajectory onto a corrected time base.

    tau(t) = t + phi(t) / (2*pi*f_s)        (corrected time)
    z_corrected(t) = z_hat evaluated at tau(t)

ONE parameter per latent dim (omega). No amplitude, no growth rate, no cap.
Because phi is linear, extrapolating omega beyond the calibration window is
EXACT, not an approximation.

WHAT IS COMPARED
----------------
  (a) uncorrected rollout
  (b) the existing phase-locked SINUSOID correction  (3 params/dim, capped)
  (c) the new TIME-WARP correction                   (1 param/dim)

All three use the SAME calibration budget (first N_CAL steps) and are scored on
the SAME extrapolation region (everything after).

Usage:
    python3 timewarp_correction.py --re 400 --latent 4
    python3 timewarp_correction.py --re 101 --latent 4 --dt 0.0438
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
ap.add_argument("--D",      type=float, default=0.12)
ap.add_argument("--U",      type=float, default=1.0)
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
t = np.arange(n) * DT

print("="*72, flush=True)
print(f"TIME-WARP CORRECTION — Re={RE}  latent={LD}  dt={DT}  calib={NCAL}", flush=True)
print(f"rollout {n} steps ({t[-1]:.2f} time units); "
      f"extrapolation region = steps {NCAL}..{n}", flush=True)
print("="*72, flush=True)

def iphase(sig):
    return np.unwrap(np.angle(hilbert(sig - sig.mean())))

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — measure the phase-drift rate omega from the CALIBRATION window only
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Measuring phase-drift rate on the calibration window", flush=True)
print("    (ground truth is available here, exactly as for the sinusoid fit)\n",
      flush=True)

omega   = np.zeros(LD)     # rad per time unit
f_shed  = np.zeros(LD)     # true shedding frequency per dim
phi_cal_r2 = np.zeros(LD)

th_pred_full = np.zeros((n, LD))
th_true_full = np.zeros((n, LD))
for d in range(LD):
    th_pred_full[:, d] = iphase(pred[:, d])
    th_true_full[:, d] = iphase(true[:, d])

for d in range(LD):
    phi = th_pred_full[:, d] - th_true_full[:, d]
    phi = phi - phi[0]
    lr = linregress(t[:NCAL], phi[:NCAL])       # fit ONLY on calibration
    omega[d] = lr.slope
    phi_cal_r2[d] = lr.rvalue**2
    f_shed[d] = np.median(np.diff(th_true_full[:, d])) / (2*np.pi*DT)
    print(f"    z{d}: omega = {omega[d]:+.6f} rad/t   "
          f"(R^2 on calib = {phi_cal_r2[d]:.4f})   "
          f"f_shed = {f_shed[d]:.4f} Hz", flush=True)

# how well does the calibration-fitted omega predict phi over the FULL rollout?
print("\n    Extrapolation check — does omega from calib predict phi later?",
      flush=True)
for d in range(LD):
    phi = th_pred_full[:, d] - th_true_full[:, d]; phi -= phi[0]
    phi_pred_lin = omega[d]*t
    ext = slice(NCAL, n)
    err = np.abs(phi_pred_lin[ext] - phi[ext])
    rel = err.mean()/max(np.abs(phi[ext]).mean(), 1e-12)
    print(f"    z{d}: mean |phi_predicted - phi_actual| in extrap region "
          f"= {err.mean():.5f} rad  ({rel*100:.1f}% of |phi|)", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — TIME-WARP correction
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Applying the time-warp correction", flush=True)

corrected_tw = np.zeros_like(pred)
for d in range(LD):
    fs = f_shed[d]
    # time shift implied by the accumulated phase error
    #   phi(t) = omega*t  ->  time lag  = phi / (2*pi*f_s)
    dt_shift = (omega[d]*t) / (2*np.pi*fs)       # seconds of clock error
    tau = t + dt_shift                            # corrected sample times
    # resample the PREDICTION at the corrected times
    corrected_tw[:, d] = np.interp(tau, t, pred[:, d])

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — the EXISTING sinusoid correction, for comparison
# ══════════════════════════════════════════════════════════════════════════════
print("[3] Applying the existing phase-locked sinusoid correction (baseline)",
      flush=True)

dim_err = pred - true
t_cal = t[:NCAL]
corr_sin = np.zeros_like(pred)
for d in range(LD):
    e  = dim_err[:NCAL, d]
    th = th_pred_full[:NCAL, d]
    def mf(idx, A, g, psi):
        idx = idx.astype(int)
        return A*np.exp(g*t_cal[idx])*np.sin(th[idx] + psi)
    try:
        popt, _ = curve_fit(mf, np.arange(NCAL, dtype=float), e,
                            p0=[e.std(), 0.1, 0.0],
                            bounds=([-1, 0.0, -2*np.pi], [1, 1.0, 2*np.pi]),
                            maxfev=100000)
    except Exception:
        popt = [e.std(), 0.1, 0.0]
    A, g, psi = popt
    cap = np.abs(e).max()*1.5
    corr_sin[:, d] = np.clip(A*np.exp(g*t)*np.sin(th_pred_full[:, d] + psi),
                             -cap, cap)
corrected_sin = pred - corr_sin

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — score all three
# ══════════════════════════════════════════════════════════════════════════════
def errmag(P):
    return np.linalg.norm(P - true, axis=1)

e_unc = errmag(pred)
e_sin = errmag(corrected_sin)
e_twp = errmag(corrected_tw)

ext = slice(NCAL, n)
def imp(e):
    return (1 - e[ext].mean()/e_unc[ext].mean())*100

I_sin = imp(e_sin)
I_twp = imp(e_twp)

print("\n" + "="*72, flush=True)
print("RESULTS — extrapolation region only", flush=True)
print("="*72, flush=True)
print(f"  {'method':<34} {'mean err':>10} {'improvement':>13} {'params/dim':>11}",
      flush=True)
print("  " + "-"*68, flush=True)
print(f"  {'uncorrected':<34} {e_unc[ext].mean():>10.5f} {'--':>13} {'--':>11}",
      flush=True)
print(f"  {'phase-locked sinusoid (baseline)':<34} {e_sin[ext].mean():>10.5f} "
      f"{I_sin:>+12.1f}% {3:>11}", flush=True)
print(f"  {'TIME-WARP (new)':<34} {e_twp[ext].mean():>10.5f} "
      f"{I_twp:>+12.1f}% {1:>11}", flush=True)
print("  " + "-"*68, flush=True)

if I_twp > I_sin + 2:
    verdict = "TIME-WARP WINS — correcting the clock beats modelling the error."
elif I_twp > I_sin - 2:
    verdict = "COMPARABLE — the sinusoid was implicitly doing the same thing."
else:
    verdict = ("SINUSOID WINS — the residual (non-phase) error matters more "
               "than the phase drift.")
print(f"\n  {verdict}", flush=True)

# how does the advantage evolve with horizon? (the key claim: time-warp should
# hold up better at LONG horizons, since it does not extrapolate an envelope)
print("\n  Improvement as a function of horizon:", flush=True)
print(f"  {'steps past calib':>18} {'sinusoid':>10} {'time-warp':>11}", flush=True)
for frac in [0.25, 0.5, 0.75, 1.0]:
    hi = NCAL + int((n-NCAL)*frac)
    sl = slice(NCAL, hi)
    i_s = (1 - e_sin[sl].mean()/e_unc[sl].mean())*100
    i_t = (1 - e_twp[sl].mean()/e_unc[sl].mean())*100
    print(f"  {hi-NCAL:>18} {i_s:>+9.1f}% {i_t:>+10.1f}%", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f"Time-warp vs sinusoid correction — Re={RE}, latent={LD}",
             fontsize=13, fontweight='bold')

ax = axes[0,0]
ax.plot(t, e_unc, color='tomato',    lw=1.2, label='uncorrected', alpha=.85)
ax.plot(t, e_sin, color='orange',    lw=1.2, label=f'sinusoid ({I_sin:+.0f}%)', alpha=.85)
ax.plot(t, e_twp, color='steelblue', lw=1.2, label=f'time-warp ({I_twp:+.0f}%)', alpha=.9)
ax.axvline(NCAL*DT, color='grey', ls='--', label='calib end')
ax.set_xlabel('t'); ax.set_ylabel('latent L2 error')
ax.set_title('Error trajectory'); ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axes[0,1]
ax.semilogy(t, e_unc+1e-9, color='tomato',    lw=1.2, label='uncorrected', alpha=.85)
ax.semilogy(t, e_sin+1e-9, color='orange',    lw=1.2, label='sinusoid', alpha=.85)
ax.semilogy(t, e_twp+1e-9, color='steelblue', lw=1.2, label='time-warp', alpha=.9)
ax.axvline(NCAL*DT, color='grey', ls='--')
ax.set_xlabel('t'); ax.set_ylabel('latent L2 error (log)')
ax.set_title('Log scale'); ax.legend(fontsize=8); ax.grid(alpha=.3, which='both')

ax = axes[1,0]
for d in range(LD):
    phi = th_pred_full[:,d] - th_true_full[:,d]; phi -= phi[0]
    ax.plot(t, phi, lw=1.0, alpha=.7, label=f'z{d} measured')
    ax.plot(t, omega[d]*t, ls='--', lw=1.0, alpha=.9,
            label=f'z{d} calib-fit $\\omega t$')
ax.axvline(NCAL*DT, color='grey', ls='--')
ax.set_xlabel('t'); ax.set_ylabel(r'$\phi$ [rad]')
ax.set_title('Phase drift: measured vs linear model fitted on calibration only')
ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)

ax = axes[1,1]
d = 0
ax.plot(t, true[:,d], color='k',         lw=1.2, label='truth', alpha=.7)
ax.plot(t, pred[:,d], color='tomato',    lw=1.0, label='prediction', alpha=.7)
ax.plot(t, corrected_tw[:,d], color='steelblue', lw=1.0,
        label='time-warped', alpha=.85)
ax.axvline(NCAL*DT, color='grey', ls='--')
ax.set_xlim(t[-1]*0.75, t[-1])     # zoom the LATE horizon, where drift is worst
ax.set_xlabel('t'); ax.set_ylabel(f'$z_{d}$')
ax.set_title('Latent trajectory, late horizon (zoom)')
ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout()
fp = os.path.join(DATA_DIR, f"timewarp_Re{RE}_L{LD}.png")
plt.savefig(fp, dpi=150, bbox_inches='tight')
print(f"\n  Figure -> {fp}", flush=True)

np.save(os.path.join(DATA_DIR, f"timewarp_Re{RE}_L{LD}.npy"),
        {'omega': omega, 'f_shed': f_shed, 'I_sin': I_sin, 'I_twp': I_twp,
         'e_unc': e_unc, 'e_sin': e_sin, 'e_twp': e_twp,
         'corrected_tw': corrected_tw, 'ncal': NCAL, 'dt': DT},
        allow_pickle=True)
print("="*72, flush=True)
