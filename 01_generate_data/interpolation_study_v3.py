#!/usr/bin/env python3
"""
interpolation_study.py
======================
Tests whether the correction parameters vary smoothly and *predictably*
with Reynolds number, via leave-one-out validation.

Core claim being tested:
  For an unseen Re, we can SYNTHESISE the correction by interpolating
  parameters from neighbouring Re — without refitting on that case.

Method:
  1. For each periodic-regime Re, recompute correction parameters from
     its saved rollout (raw prediction + truth). Parameters:
        - f_LSTM  (detuned frequency, via Hilbert instantaneous freq)
        - per-dim amplitude A_d, growth γ_d, phase ψ_d
     We summarise per-case with the MEAN amplitude/growth across dims
     and the frequency, since those are what a Re-map would predict.
  2. Fit smooth maps Re -> {f_LSTM, A_mean, gamma_mean}.
  3. LEAVE-ONE-OUT: for each interior Re, refit maps on the OTHER five,
     predict the held-out Re's parameters, synthesise its correction,
     apply to that case's actual rollout, and compare the resulting
     improvement to the directly-fitted correction.

Only periodic-regime cases: 300,400,500,600,700,800  (NOT 1000).

Usage:  python3 interpolation_study.py
"""

import os, numpy as np
from scipy.optimize import curve_fit
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR   = os.path.expanduser("~/cylinder_lstm_data")
DT         = 0.03
D, U       = 0.12, 1.0
CALIB_LEN  = 150
RES        = [300, 400, 500, 600, 700, 800]   # periodic regime only
INTERIOR   = [400, 500, 600, 700]             # valid held-out (not endpoints)

# ── Load a rollout, handling both old and new key formats ────────────────────
def load_rollout(re):
    d = np.load(os.path.join(DATA_DIR, f"rollout_Re{re}.npy"),
                allow_pickle=True).item()
    if 'raw_pred' in d:
        pred = d['raw_pred']
    elif 'predicted' in d:
        pred = d['predicted']
    else:
        raise KeyError(f"Re={re}: no prediction key found")
    true = d['true']
    n = min(len(pred), len(true))
    return pred[:n], true[:n]


def load_rho(re):
    """Try to read rho from a pipeline summary; return None if unavailable."""
    import glob
    # try tagged then untagged summary names
    candidates = [
        os.path.join(DATA_DIR, f"pipeline_summary_Re{re}_L4.npy"),
        os.path.join(DATA_DIR, f"pipeline_summary_Re{re}_L8.npy"),
        os.path.join(DATA_DIR, f"pipeline_summary_Re{re}.npy"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                d = np.load(c, allow_pickle=True).item()
                if 'rho' in d:
                    return d['rho']
            except Exception:
                pass
    return None

# ── Recompute correction parameters from a rollout ───────────────────────────
def growing_sinusoid(t, A, gamma, phi, freq):
    return A * np.exp(gamma * t) * np.sin(2*np.pi*freq*t + phi)

def fit_parameters(pred, true):
    """Return f_LSTM (mean), and per-dim (A, gamma, psi) fitted on calib window."""
    n_truth = len(true)
    latent_dim = pred.shape[1]
    dim_err = pred - true

    # Hilbert instantaneous frequency per dim → f_LSTM
    freqs_meas = []
    inst_phase = np.zeros((n_truth, latent_dim))
    for d in range(latent_dim):
        sig = pred[:, d] - pred[:, d].mean()
        ph = np.unwrap(np.angle(hilbert(sig)))
        inst_phase[:, d] = ph
        freqs_meas.append(np.median(np.diff(ph) / (2*np.pi*DT)))
    f_lstm = float(np.mean(freqs_meas))

    # Fit growing sinusoid per dim on calibration window (frequency free)
    calib = min(CALIB_LEN, n_truth)
    t_cal = np.arange(calib) * DT
    A_list, g_list, psi_list = [], [], []
    for d in range(latent_dim):
        e = dim_err[:calib, d]
        fft = np.fft.rfft(e - e.mean())
        fr  = np.fft.rfftfreq(calib, d=DT)
        idx = np.argmax(np.abs(fft[1:])**2) + 1
        f0  = fr[idx] if fr[idx] > 0 else f_lstm
        try:
            popt, _ = curve_fit(
                growing_sinusoid, t_cal, e,
                p0=[e.std()*0.5, 0.1, np.angle(fft[idx]), f0],
                bounds=([-1,0.0,-2*np.pi,f0-0.3],[1,1.0,2*np.pi,f0+0.3]),
                maxfev=100000)
            A_list.append(abs(popt[0])); g_list.append(popt[1]); psi_list.append(popt[2])
        except Exception:
            A_list.append(e.std()); g_list.append(0.1); psi_list.append(0.0)

    # Model-free effective amplitude: RMS of the error signal in calib window
    # (per dim, then averaged). No curve_fit, no degeneracy — robust & comparable.
    A_eff = float(np.mean([np.sqrt(np.mean(dim_err[:calib, d]**2))
                           for d in range(latent_dim)]))
    return {
        'f_lstm'   : f_lstm,
        'St'       : f_lstm * D / U,
        'A_mean'   : float(np.mean(A_list)),
        'A_eff'    : A_eff,
        'gamma_mean': float(np.mean(g_list)),
        'latent_dim': latent_dim,
        'n_truth'  : n_truth,
    }

# ── Apply a correction with GIVEN parameters, measure improvement ────────────
def apply_and_score(pred, true, f_use, A_use, gamma_use):
    """
    Build a correction using supplied (f, A, gamma) — phase/psi still fit
    per-dim from the case's own calibration window (phase is not interpolable,
    it's case-specific), but frequency/amplitude/growth come from `_use`.
    Returns extrapolation-region improvement %.
    """
    n_truth = len(true); latent_dim = pred.shape[1]
    dim_err = pred - true
    inst_phase = np.zeros((n_truth, latent_dim))
    for d in range(latent_dim):
        sig = pred[:, d] - pred[:, d].mean()
        inst_phase[:, d] = np.unwrap(np.angle(hilbert(sig)))

    calib = min(CALIB_LEN, n_truth)
    t_cal = np.arange(calib)*DT
    t_full = np.arange(n_truth)*DT

    corr = np.zeros((n_truth, latent_dim))
    for d in range(latent_dim):
        e = dim_err[:calib, d]
        # fit ONLY psi, with f/A/gamma fixed to supplied values
        def fn(idx, psi):
            idx = idx.astype(int)
            return A_use*np.exp(gamma_use*t_cal[idx])*np.sin(inst_phase[:calib,d][idx]+psi)
        try:
            popt,_ = curve_fit(fn, np.arange(calib,dtype=float), e,
                               p0=[0.0], bounds=([-2*np.pi],[2*np.pi]), maxfev=20000)
            psi = popt[0]
        except Exception:
            psi = 0.0
        cap = np.abs(e).max()*1.5
        raw = A_use*np.exp(gamma_use*t_full)*np.sin(inst_phase[:,d]+psi)
        corr[:,d] = np.clip(raw, -cap, cap)

    corrected = pred - corr
    ext = slice(calib, n_truth)
    eu = np.array([np.linalg.norm(pred[s]-true[s])      for s in range(n_truth)])
    ec = np.array([np.linalg.norm(corrected[s]-true[s]) for s in range(n_truth)])
    return (1 - ec[ext].mean()/eu[ext].mean())*100

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — recompute parameters for all six cases
# ══════════════════════════════════════════════════════════════════════════════
print("="*64, flush=True)
print("STEP 1 — Recomputing correction parameters from rollouts", flush=True)
print("="*64, flush=True)

params = {}
rollouts = {}
rho_map = {}
for re in RES:
    pred, true = load_rollout(re)
    rollouts[re] = (pred, true)
    p = fit_parameters(pred, true)
    params[re] = p
    rho_map[re] = load_rho(re)
    print(f"  Re={re}: St={p['St']:.4f}  f_LSTM={p['f_lstm']:.4f}  "
          f"A_eff={p['A_eff']:.5f}  A_fit={p['A_mean']:.5f}  "
          f"gamma={p['gamma_mean']:.4f}", flush=True)

re_arr = np.array(RES, dtype=float)
St_arr = np.array([params[r]['St'] for r in RES])
f_arr  = np.array([params[r]['f_lstm'] for r in RES])
A_arr  = np.array([params[r]['A_mean'] for r in RES])
Aeff_arr = np.array([params[r]['A_eff'] for r in RES])
g_arr  = np.array([params[r]['gamma_mean'] for r in RES])

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — leave-one-out on interior points
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*64, flush=True)
print("STEP 2 — Leave-one-out interpolation on interior points", flush=True)
print("="*64, flush=True)

def fit_map(x, y, degree=2):
    """Low-order polynomial fit; degree auto-capped by #points."""
    deg = min(degree, len(x)-1)
    return np.polyfit(x, y, deg)

loo_results = {}
for held in INTERIOR:
    others = [r for r in RES if r != held]
    xo = np.array(others, dtype=float)
    f_fit = fit_map(xo, np.array([params[r]['f_lstm'] for r in others]))
    A_fit = fit_map(xo, np.array([params[r]['A_mean'] for r in others]))
    Aeff_fit = fit_map(xo, np.array([params[r]['A_eff'] for r in others]))
    g_fit = fit_map(xo, np.array([params[r]['gamma_mean'] for r in others]))

    f_pred = np.polyval(f_fit, held)
    A_pred = np.polyval(A_fit, held)
    Aeff_pred = np.polyval(Aeff_fit, held)
    g_pred = np.polyval(g_fit, held)

    f_true = params[held]['f_lstm']
    A_true = params[held]['A_mean']
    Aeff_true = params[held]['A_eff']
    g_true = params[held]['gamma_mean']

    # Apply predicted params vs true-fitted params to the held-out case
    pred_roll, true_roll = rollouts[held]
    imp_synth  = apply_and_score(pred_roll, true_roll, f_pred, A_pred, g_pred)
    imp_direct = apply_and_score(pred_roll, true_roll, f_true, A_true, g_true)

    loo_results[held] = {
        'f_pred': f_pred, 'f_true': f_true,
        'A_pred': A_pred, 'A_true': A_true,
        'Aeff_pred': Aeff_pred, 'Aeff_true': Aeff_true,
        'g_pred': g_pred, 'g_true': g_true,
        'imp_synth': imp_synth, 'imp_direct': imp_direct,
    }
    print(f"\n  Held-out Re={held}:", flush=True)
    print(f"    f_LSTM  predicted={f_pred:.4f}  true={f_true:.4f}  "
          f"err={abs(f_pred-f_true)/f_true*100:.1f}%", flush=True)
    print(f"    A_eff   predicted={Aeff_pred:.5f}  true={Aeff_true:.5f}  "
          f"err={abs(Aeff_pred-Aeff_true)/Aeff_true*100:.1f}%  <- model-free", flush=True)
    print(f"    A_fit   predicted={A_pred:.5f}  true={A_true:.5f}  "
          f"err={abs(A_pred-A_true)/A_true*100:.1f}%", flush=True)
    print(f"    gamma   predicted={g_pred:.4f}  true={g_true:.4f}", flush=True)
    print(f"    IMPROVEMENT: synthesised={imp_synth:.1f}%  "
          f"direct-fit={imp_direct:.1f}%  "
          f"(ratio={imp_synth/imp_direct*100:.0f}% of direct)", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — figures
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Reynolds-Number Interpolation of Correction Parameters\n"
             "Leave-one-out validation (periodic regime, Re 300-800)",
             fontsize=12, fontweight='bold')

# Panel 1: St vs Re with fit
ax = axes[0,0]
re_fine = np.linspace(300, 800, 100)
ax.plot(re_arr, St_arr, 'o', ms=9, color='steelblue', label='Measured St', zorder=5)
St_full_fit = fit_map(re_arr, St_arr)
ax.plot(re_fine, np.polyval(St_full_fit, re_fine), '-', color='tomato',
        label='Smooth fit', alpha=0.7)
for held in INTERIOR:
    ax.plot(held, loo_results[held]['f_pred']*D/U, 'x', ms=12,
            color='green', mew=3, zorder=6)
ax.plot([],[],'x',color='green',mew=3,label='LOO prediction')
ax.set_xlabel("Reynolds number"); ax.set_ylabel("Strouhal number St")
ax.set_title("St(Re): measured vs leave-one-out predicted")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 2: Amplitude vs Re
ax = axes[0,1]
ax.plot(re_arr, A_arr, 'o', ms=9, color='steelblue', label='Measured A', zorder=5)
A_full_fit = fit_map(re_arr, A_arr)
ax.plot(re_fine, np.polyval(A_full_fit, re_fine), '-', color='tomato',
        label='Smooth fit', alpha=0.7)
for held in INTERIOR:
    ax.plot(held, loo_results[held]['A_pred'], 'x', ms=12, color='green', mew=3, zorder=6)
ax.plot([],[],'x',color='green',mew=3,label='LOO prediction')
ax.set_xlabel("Reynolds number"); ax.set_ylabel("Mean correction amplitude")
ax.set_title("A(Re): measured vs predicted")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 3: improvement synth vs direct
ax = axes[1,0]
hs = list(INTERIOR)
x = np.arange(len(hs)); w = 0.35
ax.bar(x-w/2, [loo_results[h]['imp_direct'] for h in hs], w,
       label='Direct-fit correction', color='steelblue')
ax.bar(x+w/2, [loo_results[h]['imp_synth'] for h in hs], w,
       label='Synthesised (interpolated)', color='seagreen')
ax.set_xticks(x); ax.set_xticklabels([f'Re={h}' for h in hs])
ax.set_ylabel("Extrapolation improvement (%)")
ax.set_title("Correction quality: interpolated vs directly-fitted\n"
             "(close bars = interpolation works)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

# Panel 4: prediction error of each parameter
ax = axes[1,1]
metrics = ['f_LSTM', 'A_mean']
for held in INTERIOR:
    fe = abs(loo_results[held]['f_pred']-loo_results[held]['f_true'])/loo_results[held]['f_true']*100
    ae = abs(loo_results[held]['A_pred']-loo_results[held]['A_true'])/loo_results[held]['A_true']*100
    ax.plot(['f_LSTM','A_mean'], [fe, ae], 'o-', label=f'Re={held}', alpha=0.7)
ax.set_ylabel("Leave-one-out prediction error (%)")
ax.set_title("How accurately each parameter interpolates")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.axhline(5, color='grey', ls='--', alpha=0.5)

plt.tight_layout()
fig_path = os.path.join(DATA_DIR, "interpolation_study.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n  Figure → {fig_path}", flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
np.save(os.path.join(DATA_DIR, "interpolation_results.npy"),
    {'params': params, 'loo_results': loo_results, 'rho_map': rho_map,
     'St_fit': St_full_fit, 'RES': RES, 'INTERIOR': INTERIOR},
    allow_pickle=True)

print("\n"+"="*72, flush=True)
print("UNIFIED SUMMARY — rho (correction worthwhile?) vs interpolation quality", flush=True)
print("="*72, flush=True)
print(f"  {'Re':>5}  {'rho':>7}  {'regime':>13}  {'St err':>7}  "
      f"{'synth':>8}  {'direct':>8}", flush=True)
print("  " + "-"*66, flush=True)
for held in INTERIOR:
    r = loo_results[held]
    fe = abs(r['f_pred']-r['f_true'])/r['f_true']*100
    rho = rho_map.get(held)
    if rho is None:
        rho_s, regime = "  n/a", "unknown"
    else:
        rho_s = f"{rho:6.2f}"
        regime = "LSTM-limited" if rho > 1 else "CAE-limited"
    print(f"  {held:>5}  {rho_s}  {regime:>13}  {fe:>6.1f}%  "
          f"{r['imp_synth']:>7.1f}%  {r['imp_direct']:>7.1f}%", flush=True)
print("  " + "-"*66, flush=True)
print("  Reading: interpolation is meaningful only where rho>1 (correction helps).", flush=True)
print("  Where rho<1, neither synthesised nor direct correction is expected to help.", flush=True)
print("="*72, flush=True)
