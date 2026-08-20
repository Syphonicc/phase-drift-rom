#!/usr/bin/env python3
"""
verify_mechanism.py
===================
Tests the ANALYTICAL MECHANISM claimed in the Discussion (Eq. 12):

    e(t) = 2A sin(phi(t)/2) cos(2*pi*f_s*t + phi(t)/2)

i.e. the rollout error is a cosine at the SHEDDING frequency, modulated by an
envelope 2A|sin(phi/2)| where phi(t) is the accumulated phase error between
prediction and truth.

Three separate tests, in increasing order of strictness:

  TEST 1  Is phi(t) LINEAR in t?
          Theory: phi(t) = 2*pi*delta_f*t, with delta_f the detuning.
          Fit a line; report R^2 and compare the fitted slope to the
          independently measured detuning.

  TEST 2  Does the ERROR ENVELOPE follow 2A|sin(phi/2)|?
          Extract the envelope of the error (Hilbert), compare against the
          prediction from the measured phi(t). This is the load-bearing test:
          it checks the FUNCTIONAL FORM, not just a correlation.

  TEST 3  Is the error's instantaneous frequency equal to f_s (not f_LSTM)?
          Eq. 12 says the carrier is at the TRUE shedding frequency, with a
          slowly-varying phase offset phi/2.

Reads existing rollouts. No retraining.

Usage:
    python3 verify_mechanism.py --re 400 --latent 4
    python3 verify_mechanism.py --re 400 --latent 4 --dt 0.03
    python3 verify_mechanism.py --re 101 --latent 4 --dt 0.0438   # square cyl
"""

import os, argparse, numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--re",     type=int,   required=True)
ap.add_argument("--latent", type=int,   default=4)
ap.add_argument("--dt",     type=float, default=0.03)
ap.add_argument("--D",      type=float, default=0.12)
ap.add_argument("--U",      type=float, default=1.0)
args = ap.parse_args()

RE, LD, DT = args.re, args.latent, args.dt
DATA_DIR = os.path.expanduser("~/cylinder_lstm_data")

def find(cands):
    for c in cands:
        p = os.path.join(DATA_DIR, c)
        if os.path.exists(p): return p
    raise FileNotFoundError(f"none of {cands}")

roll = np.load(find([f"rollout_Re{RE}_L{LD}.npy", f"rollout_Re{RE}.npy"]),
               allow_pickle=True).item()
pred = roll['raw_pred'] if 'raw_pred' in roll else roll['predicted']
true = roll['true']
n = min(len(pred), len(true))
pred, true = pred[:n], true[:n]
LD = pred.shape[1]
t = np.arange(n) * DT

print("="*70, flush=True)
print(f"MECHANISM VERIFICATION — Re={RE}, latent={LD}, dt={DT}", flush=True)
print(f"rollout length {n} steps = {t[-1]:.2f} time units", flush=True)
print("="*70, flush=True)

# ── helper: unwrapped instantaneous phase of a real signal ───────────────────
def inst_phase(sig):
    s = sig - sig.mean()
    return np.unwrap(np.angle(hilbert(s)))

def inst_freq(ph):
    return np.median(np.diff(ph) / (2*np.pi*DT))

# ══════════════════════════════════════════════════════════════════════════════
#  Per-dimension analysis
# ══════════════════════════════════════════════════════════════════════════════
results = []
fig, axes = plt.subplots(3, LD, figsize=(4.2*LD, 11))
if LD == 1: axes = axes.reshape(3,1)

for d in range(LD):
    zt = true[:, d]          # true latent trajectory
    zp = pred[:, d]          # predicted latent trajectory
    e  = zp - zt             # the rollout error

    th_true = inst_phase(zt)
    th_pred = inst_phase(zp)

    f_true = inst_freq(th_true)          # shedding frequency (from truth)
    f_pred = inst_freq(th_pred)          # LSTM effective frequency
    delta_f = f_pred - f_true            # THE DETUNING

    # -------- TEST 1: is phi(t) linear? --------
    phi = th_pred - th_true              # accumulated phase error
    phi = phi - phi[0]                   # reference to zero at rollout start
    lr = linregress(t, phi)
    slope_measured = lr.slope            # rad per time unit
    slope_theory   = 2*np.pi*delta_f     # theory: dphi/dt = 2*pi*delta_f
    r2 = lr.rvalue**2

    # -------- TEST 2: does the envelope follow 2A|sin(phi/2)|? --------
    # amplitude of the true oscillation (A in the derivation)
    A = np.abs(hilbert(zt - zt.mean())).mean()
    env_theory = 2*A*np.abs(np.sin(phi/2))          # predicted envelope
    env_actual = np.abs(hilbert(e - e.mean()))      # actual error envelope
    # correlate (the theory has no free parameters here!)
    mask = env_theory > 1e-12
    if mask.sum() > 20:
        r_env, p_env = pearsonr(env_theory[mask], env_actual[mask])
    else:
        r_env, p_env = np.nan, np.nan
    # also a scale check: ratio of actual to theoretical envelope
    scale = np.median(env_actual[mask] / env_theory[mask]) if mask.sum()>20 else np.nan

    # -------- TEST 3: carrier frequency of the error --------
    f_err = inst_freq(inst_phase(e))

    results.append(dict(d=d, f_true=f_true, f_pred=f_pred, delta_f=delta_f,
                        slope_measured=slope_measured, slope_theory=slope_theory,
                        r2=r2, r_env=r_env, p_env=p_env, scale=scale,
                        f_err=f_err, A=A, phi_end=phi[-1]))

    # ---- plots ----
    ax = axes[0, d]
    ax.plot(t, phi, color='steelblue', lw=1.2, label=r'measured $\phi(t)$')
    ax.plot(t, lr.intercept + lr.slope*t, 'r--', lw=1.2,
            label=f'linear fit ($R^2$={r2:.3f})')
    ax.plot(t, slope_theory*t, 'g:', lw=1.5,
            label=r'theory $2\pi\,\delta f\,t$')
    ax.set_title(f'z{d}: accumulated phase error')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\phi$ [rad]')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, d]
    ax.plot(t, env_actual, color='tomato', lw=1.0, label='actual error envelope')
    ax.plot(t, env_theory, color='green', lw=1.4, ls='--',
            label=r'theory $2A|\sin(\phi/2)|$')
    ax.set_title(f'z{d}: error envelope  (r={r_env:+.3f})')
    ax.set_xlabel('t'); ax.set_ylabel('envelope')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2, d]
    ax.scatter(env_theory[mask], env_actual[mask], s=4, alpha=0.4, color='purple')
    lim = max(env_theory[mask].max(), env_actual[mask].max())
    ax.plot([0,lim],[0,lim],'k--',lw=1, label='1:1')
    ax.set_title(f'z{d}: envelope, theory vs actual\n(scale factor {scale:.2f})')
    ax.set_xlabel('theory'); ax.set_ylabel('actual')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
fp = os.path.join(DATA_DIR, f"mechanism_Re{RE}_L{LD}.png")
plt.savefig(fp, dpi=150, bbox_inches='tight')

# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\nTEST 1 — Is the accumulated phase error phi(t) linear in t?", flush=True)
print("  theory: phi(t) = 2*pi*delta_f*t\n", flush=True)
print(f"  {'dim':>3} {'f_true':>8} {'f_LSTM':>8} {'delta_f':>9} "
      f"{'slope_meas':>11} {'slope_thry':>11} {'R^2':>7}", flush=True)
for r in results:
    print(f"  {r['d']:>3} {r['f_true']:>8.4f} {r['f_pred']:>8.4f} "
          f"{r['delta_f']:>+9.5f} {r['slope_measured']:>+11.5f} "
          f"{r['slope_theory']:>+11.5f} {r['r2']:>7.4f}", flush=True)

mean_r2 = np.mean([r['r2'] for r in results])
slope_agree = np.mean([abs(r['slope_measured']-r['slope_theory']) /
                       max(abs(r['slope_theory']),1e-9) for r in results])
print(f"\n  mean R^2 of linear fit          : {mean_r2:.4f}", flush=True)
print(f"  mean |slope error| / |theory|   : {slope_agree*100:.1f}%", flush=True)
if mean_r2 > 0.9:
    print("  ==> phi(t) IS essentially linear. Theory supported.", flush=True)
elif mean_r2 > 0.6:
    print("  ==> phi(t) is roughly linear but with structure. Partially supported.",
          flush=True)
else:
    print("  ==> phi(t) is NOT linear. The simple detuning model is incomplete.",
          flush=True)

print("\n" + "-"*70, flush=True)
print("TEST 2 — Does the error envelope follow 2A|sin(phi/2)|?", flush=True)
print("  (this is the load-bearing test: NO free parameters)\n", flush=True)
print(f"  {'dim':>3} {'r(theory,actual)':>17} {'p':>9} {'scale':>8}", flush=True)
for r in results:
    print(f"  {r['d']:>3} {r['r_env']:>+17.4f} {r['p_env']:>9.2e} "
          f"{r['scale']:>8.3f}", flush=True)
mean_renv = np.nanmean([r['r_env'] for r in results])
mean_scale = np.nanmean([r['scale'] for r in results])
print(f"\n  mean correlation : {mean_renv:+.4f}", flush=True)
print(f"  mean scale factor: {mean_scale:.3f}   (1.0 = perfect amplitude match)",
      flush=True)
if mean_renv > 0.8:
    print("  ==> Envelope follows the theory closely. MECHANISM VERIFIED.", flush=True)
elif mean_renv > 0.5:
    print("  ==> Envelope partially follows theory. Mechanism is a good "
          "first-order description.", flush=True)
else:
    print("  ==> Envelope does NOT follow theory. Mechanism claim needs revision.",
          flush=True)

print("\n" + "-"*70, flush=True)
print("TEST 3 — Is the error's carrier frequency f_s (the TRUE shedding freq)?",
      flush=True)
print(f"\n  {'dim':>3} {'f_true':>9} {'f_error':>9} {'diff %':>8}", flush=True)
for r in results:
    diff = (r['f_err']-r['f_true'])/r['f_true']*100
    print(f"  {r['d']:>3} {r['f_true']:>9.4f} {r['f_err']:>9.4f} {diff:>+8.2f}",
          flush=True)

print("\n" + "="*70, flush=True)
print("SUMMARY", flush=True)
print("="*70, flush=True)
St_true = np.mean([r['f_true'] for r in results])*args.D/args.U
St_err  = np.mean([r['f_err']  for r in results])*args.D/args.U
print(f"  St (flow)          : {St_true:.4f}", flush=True)
print(f"  St (error carrier) : {St_err:.4f}", flush=True)
print(f"  mean detuning      : {np.mean([r['delta_f'] for r in results]):+.5f} Hz",
      flush=True)
print(f"  total phase drift  : {np.mean([r['phi_end'] for r in results]):+.3f} rad "
      f"({np.mean([r['phi_end'] for r in results])/(2*np.pi):+.3f} cycles)", flush=True)
print(f"\n  Figure -> {fp}", flush=True)

np.save(os.path.join(DATA_DIR, f"mechanism_Re{RE}_L{LD}.npy"),
        {'results': results, 're': RE, 'latent': LD, 'dt': DT}, allow_pickle=True)
