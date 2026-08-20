#!/usr/bin/env python3
"""
phasecorr_on_dmd.py
===================
Applies the paper's phase correction to a Hankel-DMD rollout instead of to the
CAE-LSTM rollout, on the modulated-inflow cases.

WHY
---
The correction is currently demonstrated only on the LSTM. If it also repairs a
linear propagator that drifts, the claim generalises from "our network's error is
correctable" to "phase realignment repairs any autoregressive propagator whose
temporal representation is slightly wrong". That is architecture-independent.

THE DESIGN
----------
Two delay counts per case, deliberately:

  q = 60   the operator is INACCURATE and drifts (omega_d ~ -0.1, R^2 ~ 0.85).
           The correction should help, and the R^2 gate should ACCEPT.

  q = 300  the operator is ACCURATE and does not drift (|omega_d| <= 6e-4,
           R^2 <= 0.29). There is nothing to correct, and the gate should
           REJECT.

Both outcomes are results. The second is arguably the better one: a diagnostic
that declines to act when there is no drift is doing exactly what it claims.

The correction itself is copied from phase_correction.py, deliberately unchanged:
    z_corr(t) = A_hat(t) * cos( theta_hat(t) - omega*t ) + mean_hat
with omega fitted by least squares on the first NCAL steps of the rollout only.

USAGE
    python3 phasecorr_on_dmd.py
    python3 phasecorr_on_dmd.py --cases 998:4 --delays 60 300 --ncal 150
"""
import argparse, importlib.util, os, sys
import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress

DATA = os.path.expanduser("~/cylinder_lstm_data")

ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="+", default=["998:4", "999:4"])
ap.add_argument("--delays", type=int, nargs="+", default=[60, 300])
ap.add_argument("--rank", type=int, default=128)
ap.add_argument("--max-rank", type=int, default=200)
ap.add_argument("--train-frac", type=float, default=0.6)
ap.add_argument("--ncal", type=int, default=150)
ap.add_argument("--dt", type=float, default=0.03)
ap.add_argument("--r2-threshold", type=float, default=0.3)
ap.add_argument("--out", default="phasecorr_on_dmd.json")
args = ap.parse_args()


# ---------------------------------------------------------------- load hankel
def load_hankel_module():
    """Import hankel_dmd.py without letting its argparse eat our argv."""
    path = os.path.join(DATA, "hankel_dmd.py")
    saved = sys.argv
    sys.argv = ["hankel_dmd.py"]          # force its defaults
    try:
        spec = importlib.util.spec_from_file_location("hk", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    for fn in ("hankel", "fit_dmd", "rollout_dmd"):
        if not hasattr(mod, fn):
            sys.exit(f"hankel_dmd.py has no {fn}(); inspect it and adjust")
    return mod


hk = load_hankel_module()
print("imported hankel_dmd.py helpers", flush=True)


# ------------------------------------------------------------ the correction
def analytic(x):
    """Mean, envelope, unwrapped phase of a real signal. As phase_correction.py."""
    m = x.mean()
    z = hilbert(x - m)
    return m, np.abs(z), np.unwrap(np.angle(z))


def phase_correct(pred, true, t, ncal):
    """Fit one drift rate per coordinate on t[:ncal], then realign the phase.

    Returns corrected trajectory, per-coordinate omega, per-coordinate R^2.
    """
    n, nz = pred.shape
    corr = np.empty_like(pred)
    omega = np.zeros(nz)
    r2 = np.zeros(nz)
    for d in range(nz):
        m_t, A_t, th_t = analytic(true[:, d])
        m_p, A_p, th_p = analytic(pred[:, d])
        phi = th_p - th_t
        phi = phi - phi[0]
        lr = linregress(t[:ncal], phi[:ncal])
        omega[d] = lr.slope
        r2[d] = lr.rvalue ** 2
        corr[:, d] = A_p * np.cos(th_p - omega[d] * t) + m_p
    return corr, omega, r2


def mean_l2(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean())


# --------------------------------------------------------------------- main
results = {}
for case in args.cases:
    re_, nz = (int(v) for v in case.split(":"))
    tag = f"Re{re_}_L{nz}"
    f = os.path.join(DATA, f"latents_{tag}.npy")
    if not os.path.exists(f):
        print(f"  MISSING {f} -- skipping", flush=True)
        continue

    Z = np.load(f)
    # Normalise exactly as hankel_dmd.py does, so errors are comparable with
    # Table XII. If a stored normalisation exists, prefer it (footgun 4).
    nf = os.path.join(DATA, f"latent_norm_{tag}.npy")
    if os.path.exists(nf):
        nd = np.load(nf, allow_pickle=True).item()
        m, sd = np.asarray(nd["mean"], float), np.asarray(nd["std"], float)
        src = "stored latent_norm"
    else:
        m, sd = Z.mean(0), Z.std(0)
        src = "self-normalised"
    Z = (Z - m) / sd

    n = Z.shape[0]
    split = int(args.train_frac * n)
    t_roll = np.arange(n - split) * args.dt

    print(f"\n{'='*74}\n{tag}   n={n}  split={split}  extrapolate={n-split}"
          f"  ({src})\n{'='*74}", flush=True)
    print(f"{'q':>5} {'DMD err':>10} {'corrected':>10} {'change':>9} "
          f"{'omega_d':>11} {'R^2':>7}  verdict", flush=True)

    results[tag] = []
    for q in args.delays:
        try:
            pred, r = hk.rollout_dmd(Z, q, split, args.rank, args.max_rank)
        except Exception as e:
            print(f"{q:>5}  rollout_dmd failed: {e}", flush=True)
            continue
        if pred is None:
            print(f"{q:>5}  rollout_dmd returned None", flush=True)
            continue

        pred = np.asarray(pred, float)
        true = Z[split:split + pred.shape[0]]
        assert pred.shape == true.shape, (pred.shape, true.shape)
        t = t_roll[:pred.shape[0]]

        corr, omega, r2 = phase_correct(pred, true, t, args.ncal)

        # score on the extrapolation region only -- the calibration window is
        # not evidence, it is what the parameter was fitted on
        e_unc = mean_l2(pred[args.ncal:], true[args.ncal:])
        e_cor = mean_l2(corr[args.ncal:], true[args.ncal:])
        I = 100.0 * (1.0 - e_cor / e_unc)

        r2_mean = float(r2.mean())
        accept = r2_mean >= args.r2_threshold
        verdict = "ACCEPT" if accept else "reject"

        print(f"{q:>5} {e_unc:>10.5f} {e_cor:>10.5f} {I:>+8.1f}% "
              f"{omega.mean():>+11.5f} {r2_mean:>7.3f}  {verdict}"
              f"{'' if accept else '  (correction would not be applied)'}",
              flush=True)

        results[tag].append(dict(q=q, rank_used=int(r), err_uncorrected=e_unc,
                                 err_corrected=e_cor, improvement_pct=I,
                                 omega_per_coord=omega.tolist(),
                                 r2_per_coord=r2.tolist(),
                                 r2_mean=r2_mean, accepted=bool(accept)))

import json
out = os.path.join(DATA, args.out)
json.dump(results, open(out, "w"), indent=2, default=float)
print(f"\nSaved -> {out}", flush=True)

print("""
Reading the output
------------------
  q = 60  is the regime where the operator is inaccurate and drifts. If the
          correction improves it and the gate ACCEPTS, the correction is not
          specific to the LSTM -- it repairs any propagator whose temporal
          representation is slightly wrong.

  q = 300 is the regime where the operator is accurate and does not drift. The
          gate should REJECT. A diagnostic that declines to act when there is
          nothing to correct is behaving exactly as advertised, and that is a
          result worth reporting in its own right.

  If the gate ACCEPTS at q = 300 and the correction then makes things worse,
  that is a false accept -- the first in the paper -- and it must be reported.
""", flush=True)
