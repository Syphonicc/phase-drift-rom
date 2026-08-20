#!/usr/bin/env python3
"""
collect_phase_results.py
========================
Gathers the outputs of run_all_phase.sh into the summary tables the paper needs.

Produces:
  1. Master comparison table: uncorrected / sinusoid / phase correction, all cases
  2. Error-decomposition table: phase vs amplitude share, all cases
  3. omega(Re) — the phase-drift rate as a function of Reynolds number
  4. A master figure: improvement vs case, both methods side by side

Run AFTER run_all_phase.sh.

Usage:  python3 collect_phase_results.py
"""

import os, glob, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.expanduser("~/cylinder_lstm_data")

CASES = [
    (300, 4, 0.03,   'circular'),
    (400, 4, 0.03,   'circular'),
    (500, 4, 0.03,   'circular'),
    (600, 4, 0.03,   'circular'),
    (700, 4, 0.03,   'circular'),
    (800, 4, 0.03,   'circular'),
    (800, 8, 0.03,   'circular'),
    (1000,4, 0.03,   'circular'),
    (1000,8, 0.03,   'circular'),
    (101, 4, 0.0438, 'square'),
    (101, 8, 0.0438, 'square'),
]

def load(fn):
    p = os.path.join(DATA_DIR, fn)
    if not os.path.exists(p): return None
    try:
        return np.load(p, allow_pickle=True).item()
    except Exception:
        return None

rows = []
for RE, LD, DT, GEO in CASES:
    pc = load(f"phasecorr_Re{RE}_L{LD}.npy")
    dc = load(f"decompose_Re{RE}_L{LD}.npy")
    if pc is None:
        rows.append(dict(re=RE, ld=LD, geo=GEO, missing=True))
        continue
    rows.append(dict(
        re=RE, ld=LD, geo=GEO, missing=False,
        I_sin=pc['I_sin'], I_pha=pc['I_pha'],
        omega=np.mean(np.abs(pc['omega'])),
        omega_signed=np.mean(pc['omega']),
        r2=np.mean(pc['r2']),
        share_phase=(dc['share_phase'] if dc else np.nan),
        share_amp=(dc['share_amp'] if dc else np.nan),
    ))

# ══════════════════════════════════════════════════════════════════════════════
#  TABLE 1 — master comparison
# ══════════════════════════════════════════════════════════════════════════════
print("="*84, flush=True)
print("TABLE 1 — Correction performance: sinusoid vs analytic phase correction",
      flush=True)
print("="*84, flush=True)
print(f"  {'geometry':<10} {'Re':>5} {'n_z':>4} | {'sinusoid':>10} {'phase':>9} "
      f"{'gain':>8} | {'|omega|':>9} {'R^2':>6}", flush=True)
print("  " + "-"*78, flush=True)
for r in rows:
    if r['missing']:
        print(f"  {r['geo']:<10} {r['re']:>5} {r['ld']:>4} | "
              f"{'-- missing --':>30}", flush=True)
        continue
    gain = r['I_pha'] - r['I_sin']
    print(f"  {r['geo']:<10} {r['re']:>5} {r['ld']:>4} | "
          f"{r['I_sin']:>+9.1f}% {r['I_pha']:>+8.1f}% {gain:>+7.1f} | "
          f"{r['omega']:>9.5f} {r['r2']:>6.3f}", flush=True)
print("  " + "-"*78, flush=True)

ok = [r for r in rows if not r['missing']]
if ok:
    print(f"  {'MEAN':<10} {'':>5} {'':>4} | "
          f"{np.mean([r['I_sin'] for r in ok]):>+9.1f}% "
          f"{np.mean([r['I_pha'] for r in ok]):>+8.1f}% "
          f"{np.mean([r['I_pha']-r['I_sin'] for r in ok]):>+7.1f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABLE 2 — error decomposition
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*84, flush=True)
print("TABLE 2 — What IS the error? (variance share)", flush=True)
print("="*84, flush=True)
print(f"  {'geometry':<10} {'Re':>5} {'n_z':>4} | {'PHASE':>8} {'AMPLITUDE':>11}",
      flush=True)
print("  " + "-"*45, flush=True)
for r in ok:
    if np.isnan(r['share_phase']): continue
    print(f"  {r['geo']:<10} {r['re']:>5} {r['ld']:>4} | "
          f"{r['share_phase']:>7.1f}% {r['share_amp']:>10.1f}%", flush=True)
valid = [r for r in ok if not np.isnan(r['share_phase'])]
if valid:
    print("  " + "-"*45, flush=True)
    print(f"  {'MEAN':<10} {'':>5} {'':>4} | "
          f"{np.mean([r['share_phase'] for r in valid]):>7.1f}% "
          f"{np.mean([r['share_amp'] for r in valid]):>10.1f}%", flush=True)
    print(f"\n  => The rollout error is "
          f"{np.mean([r['share_phase'] for r in valid]):.0f}% PHASE error.",
          flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABLE 3 — omega(Re), for the interpolation story
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*84, flush=True)
print("TABLE 3 — Phase-drift rate omega vs Reynolds number (circular, n_z=4)",
      flush=True)
print("="*84, flush=True)
circ = [r for r in ok if r['geo']=='circular' and r['ld']==4]
circ.sort(key=lambda r: r['re'])
print(f"  {'Re':>6} {'omega (signed)':>16} {'|omega|':>10} {'R^2':>7}", flush=True)
print("  " + "-"*44, flush=True)
for r in circ:
    print(f"  {r['re']:>6} {r['omega_signed']:>+16.6f} {r['omega']:>10.5f} "
          f"{r['r2']:>7.3f}", flush=True)
print("\n  (If omega varies smoothly with Re, it can be interpolated —", flush=True)
print("   and it is a SINGLE physical parameter, unlike the 3 sinusoid params.)",
      flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE — master comparison
# ══════════════════════════════════════════════════════════════════════════════
if ok:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    labels = [f"{r['geo'][:4]}\nRe{r['re']}\nL{r['ld']}" for r in ok]
    x = np.arange(len(ok)); w = 0.38
    ax.bar(x-w/2, [r['I_sin'] for r in ok], w, label='sinusoid (3 params)',
           color='orange', alpha=.85)
    ax.bar(x+w/2, [r['I_pha'] for r in ok], w, label='phase correction (1 param)',
           color='seagreen', alpha=.85)
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('extrapolation improvement (%)')
    ax.set_title('Correction performance across all cases')
    ax.legend(); ax.grid(alpha=.3, axis='y')

    ax = axes[1]
    if len(circ) > 1:
        res = [r['re'] for r in circ]
        oms = [r['omega_signed'] for r in circ]
        ax.plot(res, oms, 'o-', color='steelblue', ms=8, lw=1.5)
        ax.set_xlabel('Reynolds number')
        ax.set_ylabel(r'phase-drift rate $\omega$ [rad/t]')
        ax.set_title(r'$\omega$(Re) — a single physical parameter'
                     '\n(smooth => interpolable across Re)')
        ax.grid(alpha=.3)
        ax.axhline(0, color='k', lw=.8, ls=':')

    plt.tight_layout()
    fp = os.path.join(DATA_DIR, "phase_master_summary.png")
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    print(f"\n  Master figure -> {fp}", flush=True)

np.save(os.path.join(DATA_DIR, "phase_collected.npy"),
        {'rows': rows}, allow_pickle=True)
print("\n" + "="*84, flush=True)
