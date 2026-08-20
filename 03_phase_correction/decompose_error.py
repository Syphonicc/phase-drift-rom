#!/usr/bin/env python3
"""
decompose_error.py
==================
Determines WHAT the rollout error actually is, by decomposing it into a phase
component and an amplitude component.

BACKGROUND
----------
The pure phase-drift model predicts

    e(t) = 2A sin(phi/2) cos(2 pi f_s t + phi/2)

The time-warp experiment refuted this: the measured phase drift phi is tiny
(~0.02 rad) and correcting it does not reduce the error. So the error must be
dominated by something else.

THE DECOMPOSITION
-----------------
Write both the true and predicted latent trajectories in analytic form:

    z_true(t) = A(t)      * cos( theta(t) )
    z_pred(t) = A_hat(t)  * cos( theta_hat(t) )

Both A and theta are obtained from the Hilbert transform. Then the error

    e(t) = z_pred - z_true

can be split EXACTLY into two contributions, by constructing two hybrid signals:

    e_amp(t)   = A_hat(t)*cos(theta(t))     - A(t)*cos(theta(t))
                 -> the error you would get from the AMPLITUDE being wrong,
                    with the phase perfectly correct

    e_phase(t) = A(t)*cos(theta_hat(t))     - A(t)*cos(theta(t))
                 -> the error you would get from the PHASE being wrong,
                    with the amplitude perfectly correct

    e_cross    = e - e_amp - e_phase        -> the remainder (nonlinear cross term)

We then report the variance of each contribution as a fraction of the total.
Whichever dominates IS the mechanism.

Usage:
    python3 decompose_error.py --re 400 --latent 4
    python3 decompose_error.py --re 101 --latent 4 --dt 0.0438
"""

import os, argparse, numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--re",     type=int,   required=True)
ap.add_argument("--latent", type=int,   default=4)
ap.add_argument("--dt",     type=float, default=0.03)
args = ap.parse_args()

RE, LD, DT = args.re, args.latent, args.dt
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

print("="*74, flush=True)
print(f"ERROR DECOMPOSITION — Re={RE}  latent={LD}", flush=True)
print("  Is the rollout error a PHASE error or an AMPLITUDE error?", flush=True)
print("="*74, flush=True)

def analytic(sig):
    """Return (mean, amplitude(t), phase(t)) of a real signal."""
    m = sig.mean()
    z = hilbert(sig - m)
    return m, np.abs(z), np.unwrap(np.angle(z))

rows = []
fig, axes = plt.subplots(3, LD, figsize=(4.3*LD, 11))
if LD == 1: axes = axes.reshape(3,1)

for d in range(LD):
    zt, zp = true[:, d], pred[:, d]

    m_t, A_t, th_t = analytic(zt)     # truth: amplitude and phase
    m_p, A_p, th_p = analytic(zp)     # prediction: amplitude and phase

    e = zp - zt                        # the actual error

    # --- the two hypothetical errors -----------------------------------------
    # amplitude-only error: right phase, wrong amplitude
    e_amp   = (A_p - A_t) * np.cos(th_t)
    # phase-only error: right amplitude, wrong phase
    e_phase = A_t * (np.cos(th_p) - np.cos(th_t))
    # mean-offset error (the DC term, usually negligible)
    e_mean  = (m_p - m_t) * np.ones_like(t)
    # whatever is left over
    e_cross = e - e_amp - e_phase - e_mean

    # --- variance shares ------------------------------------------------------
    var_tot   = np.var(e)
    share_amp   = np.var(e_amp)   / var_tot
    share_phase = np.var(e_phase) / var_tot
    share_mean  = np.var(e_mean)  / var_tot
    share_cross = np.var(e_cross) / var_tot

    # --- how much do amplitude and phase each drift? --------------------------
    amp_ratio_end  = A_p[-50:].mean() / A_t[-50:].mean()
    phase_drift_end = (th_p - th_t)[-1] - (th_p - th_t)[0]

    rows.append(dict(d=d, share_amp=share_amp, share_phase=share_phase,
                     share_mean=share_mean, share_cross=share_cross,
                     amp_ratio=amp_ratio_end, phi_end=phase_drift_end,
                     rms_e=np.sqrt(np.mean(e**2)),
                     rms_amp=np.sqrt(np.mean(e_amp**2)),
                     rms_phase=np.sqrt(np.mean(e_phase**2))))

    # ---- plots ----
    ax = axes[0, d]
    ax.plot(t, A_t, color='k',         lw=1.2, label='true amplitude $A$')
    ax.plot(t, A_p, color='tomato',    lw=1.2, label=r'predicted $\hat{A}$')
    ax.set_title(f'z{d}: amplitude\n(final ratio {amp_ratio_end:.3f})')
    ax.set_xlabel('t'); ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = axes[1, d]
    ax.plot(t, th_p - th_t - (th_p-th_t)[0], color='steelblue', lw=1.2)
    ax.set_title(f'z{d}: phase error $\\phi(t)$\n'
                 f'(total {phase_drift_end:+.4f} rad '
                 f'= {phase_drift_end/(2*np.pi):+.5f} cycles)')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\phi$ [rad]'); ax.grid(alpha=.3)

    ax = axes[2, d]
    ax.plot(t, e,       color='k',         lw=1.0, alpha=.8, label='actual error')
    ax.plot(t, e_amp,   color='tomato',    lw=1.0, alpha=.8,
            label=f'amplitude part ({share_amp*100:.0f}%)')
    ax.plot(t, e_phase, color='steelblue', lw=1.0, alpha=.8,
            label=f'phase part ({share_phase*100:.0f}%)')
    ax.set_title(f'z{d}: error decomposition')
    ax.set_xlabel('t'); ax.legend(fontsize=7); ax.grid(alpha=.3)

plt.tight_layout()
fp = os.path.join(DATA_DIR, f"decompose_Re{RE}_L{LD}.png")
plt.savefig(fp, dpi=150, bbox_inches='tight')

# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\nVARIANCE SHARE OF EACH ERROR COMPONENT", flush=True)
print("  (what fraction of the error variance does each mechanism explain?)\n",
      flush=True)
print(f"  {'dim':>3} {'AMPLITUDE':>11} {'PHASE':>9} {'mean':>8} {'cross':>8}",
      flush=True)
print("  " + "-"*45, flush=True)
for r in rows:
    print(f"  {r['d']:>3} {r['share_amp']*100:>10.1f}% {r['share_phase']*100:>8.1f}% "
          f"{r['share_mean']*100:>7.1f}% {r['share_cross']*100:>7.1f}%", flush=True)

mA = np.mean([r['share_amp']   for r in rows])*100
mP = np.mean([r['share_phase'] for r in rows])*100
mM = np.mean([r['share_mean']  for r in rows])*100
mC = np.mean([r['share_cross'] for r in rows])*100
print("  " + "-"*45, flush=True)
print(f"  {'mean':>3} {mA:>10.1f}% {mP:>8.1f}% {mM:>7.1f}% {mC:>7.1f}%", flush=True)

print("\n\nHOW MUCH DOES EACH QUANTITY ACTUALLY DRIFT?\n", flush=True)
print(f"  {'dim':>3} {'amp ratio (end)':>16} {'phase drift (cycles)':>22}", flush=True)
print("  " + "-"*44, flush=True)
for r in rows:
    print(f"  {r['d']:>3} {r['amp_ratio']:>16.4f} "
          f"{r['phi_end']/(2*np.pi):>22.5f}", flush=True)

print("\n" + "="*74, flush=True)
print("VERDICT", flush=True)
print("="*74, flush=True)
if mA > 60:
    print(f"  The error is dominated by AMPLITUDE error ({mA:.0f}% of variance).",
          flush=True)
    print(f"  The prediction's oscillation has the right timing but the wrong size.",
          flush=True)
    print(f"  Phase drift accounts for only {mP:.0f}%.", flush=True)
    print(f"\n  => The phase-drift derivation is NOT the mechanism.", flush=True)
    print(f"  => The error is an amplitude modulation of the shedding mode,", flush=True)
    print(f"     which is why it still peaks at f_s in the spectrum.", flush=True)
elif mP > 60:
    print(f"  The error is dominated by PHASE error ({mP:.0f}% of variance).", flush=True)
    print(f"  The phase-drift derivation IS the mechanism.", flush=True)
else:
    print(f"  Neither dominates: amplitude {mA:.0f}%, phase {mP:.0f}%.", flush=True)
    print(f"  The error is a genuine mixture and needs a two-component model.",
          flush=True)
print("="*74, flush=True)
print(f"\n  Figure -> {fp}", flush=True)

np.save(os.path.join(DATA_DIR, f"decompose_Re{RE}_L{LD}.npy"),
        {'rows': rows, 're': RE, 'latent': LD,
         'share_amp': mA, 'share_phase': mP}, allow_pickle=True)
