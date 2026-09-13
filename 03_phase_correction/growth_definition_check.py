#!/usr/bin/env python3
"""
p0s4b_growth_definition_check.py

The reproduction run gave tangential growth of 2.5-2.7x against the paper's
11-17x. That is very likely a definitional mismatch, not a disagreement:
p0s4 measured first-fifth vs last-fifth of the SCORED window [150:], while the
paper probably measures from the start of the rollout.

This tries every plausible definition and reports which one lands in 11-17x.
Whichever it is, that is the definition the library should document and the
paper should state explicitly, because right now the number is unreproducible
by anyone who doesn't already know how it was computed.

Also fits the growth law: linear in t (phase drift accumulating at a constant
rate) vs exponential (instability). The paper's mechanism predicts linear.

Run from the repo root:
    python p0s4b_growth_definition_check.py
"""

import os
import sys

import numpy as np

try:
    import rolldrift as rd
except ImportError:
    sys.exit("rolldrift not importable. Run `pip install -e .` first.")

DATA = os.path.expanduser("~/cylinder_lstm_data")

CASES = [
    ("circular Re=300", 300, 4, 0.03),
    ("circular Re=400", 400, 4, 0.03),
    ("circular Re=800", 800, 8, 0.03),
    ("square   Re=101", 101, 4, 0.0438),
]


def load(re_, nz):
    for name in (f"rollout_Re{re_}_L{nz}.npy", f"rollout_Re{re_}.npy"):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True).item()
            return (np.asarray(d["raw_pred"], float),
                    np.asarray(d["true"], float), name)
    raise FileNotFoundError(f"Re={re_}, nz={nz}")


def series(pred, true, dt):
    """Per-step tangential and transverse magnitudes over the FULL rollout."""
    r = rd.diagnose(pred, true, dt)
    return r.tangential_t, r.transverse_t


def ratio(a, i0, i1, w):
    """Mean of a over a window of width w at i0, vs at i1."""
    lo = a[i0:i0 + w]
    hi = a[i1 - w:i1]
    lo, hi = lo[np.isfinite(lo)], hi[np.isfinite(hi)]
    if lo.size == 0 or hi.size == 0 or lo.mean() <= 0:
        return np.nan
    return hi.mean() / lo.mean()


print("=" * 84, flush=True)
print("Tangential growth under different definitions   [paper reports 11-17x]", flush=True)
print("=" * 84, flush=True)

for label, re_, nz, dt in CASES:
    try:
        pred, true, fname = load(re_, nz)
    except FileNotFoundError as e:
        print(f"\n{label}: SKIPPED ({e})", flush=True)
        continue

    T = len(true)
    par, perp = series(pred, true, dt)

    print(f"\n{label}   ({fname}, T={T})", flush=True)
    print(f"  {'definition':<52}{'tangential':>12}{'transverse':>12}", flush=True)
    print("  " + "-" * 76, flush=True)

    w = max(3, T // 20)
    defs = [
        ("full rollout, first 5% vs last 5%",            0,   T,   w),
        ("full rollout, first 20% vs last 20%",          0,   T,   T // 5),
        ("from step 150, first 20% vs last 20%",         150, T,   (T - 150) // 5),
        ("full rollout, steps 1-10 vs last 10",          1,   T,   10),
    ]
    for name, i0, i1, ww in defs:
        gp = ratio(par, i0, i1, ww)
        gt = ratio(perp, i0, i1, ww)
        mark = "  <== matches paper" if 11.0 <= gp <= 17.0 else ""
        print(f"  {name:<52}{gp:>11.2f}x{gt:>11.2f}x{mark}", flush=True)

    # single-step endpoints, the most literal reading
    gp = par[1] and par[-1] / par[1]
    print(f"  {'single step: last / first':<52}{gp:>11.2f}x"
          f"{perp[-1] / perp[1]:>11.2f}x", flush=True)

    # growth law: linear vs exponential over the full rollout
    t = np.arange(T) * dt
    m = np.isfinite(par) & (par > 0)
    m[:5] = False
    if m.sum() > 20:
        lin = np.polyfit(t[m], par[m], 1)
        r_lin = 1 - np.var(par[m] - np.polyval(lin, t[m])) / np.var(par[m])
        exp = np.polyfit(t[m], np.log(par[m]), 1)
        pred_exp = np.exp(np.polyval(exp, t[m]))
        r_exp = 1 - np.var(par[m] - pred_exp) / np.var(par[m])
        print(f"  growth law fit over full rollout: "
              f"linear R2={r_lin:.4f}   exponential R2={r_exp:.4f}   "
              f"-> {'LINEAR' if r_lin > r_exp else 'EXPONENTIAL'}", flush=True)
        print(f"    (linear mechanism = constant-rate phase drift, "
              f"as the paper argues)", flush=True)

print("\n" + "=" * 84, flush=True)
print("Whichever definition matches 11-17x is the one the paper used.", flush=True)
print("State it explicitly in the manuscript -- as written the number cannot be", flush=True)
print("reproduced by a reader who doesn't already know the convention.", flush=True)
