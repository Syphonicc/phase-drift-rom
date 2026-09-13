#!/usr/bin/env python3
"""
p0s4_reproduce_paper_rollouts.py

Step 4: point rolldrift at the actual CAE-LSTM rollout data behind the
phase-drift paper.

Three questions, in order of importance:

  Q1 REPRODUCTION
     Does kappa / f_par agree with the paper's 97-98% phase-error figure?
     If yes, the library is validated against published work. If no, we find
     out privately, now.

  Q2 DILATION CONFOUND
     A pure AMPLITUDE error on a non-circular orbit carries a real tangential
     component (verified: circle 0.000, Van der Pol mu=1 0.180, mu=5 0.749,
     mu=10 0.884). How much tangential content does a pure dilation of THESE
     latent orbits produce? That number is the floor your claim has to clear.

  Q3 ORBIT GEOMETRY
     How circular are the latent orbits? The singular-value spectrum of the
     mean-centred trajectory answers this directly and tells you how seriously
     to take Q2. A dominant equal pair => near-circular => confound is small.

Also reported: the isotropic null check (a sanity check on the data), the
per-window kappa profile (because aggregate kappa is horizon-diluted), and
tangential vs transverse growth.

SAFETY NOTES BAKED IN
  - uses only raw_pred and true, which are in the SAME normalised space
  - never touches `corrected` (sinusoid baseline, misleading name)
  - never touches latents_*.npy (raw unnormalised -> the 11x footgun)
  - scores on [150:] only; the first 150 steps are the calibration window
  - prints the resolved filename for every case, since the suffixed /
    unsuffixed naming is how the wrong file gets loaded silently

Run:
    python p0s4_reproduce_paper_rollouts.py
"""

import os
import sys

import numpy as np

try:
    import rolldrift as rd
except ImportError:
    sys.exit("rolldrift not importable. Run `pip install -e .` in the repo first.")

DATA = os.path.expanduser("~/cylinder_lstm_data")
CALIB_STEPS = 150          # first 150 steps are the calibration window
DILATION_EPS = 0.02        # 2% pure amplitude perturbation for the confound test

# (label, Re, n_z, dt, shedding period or None)
CASES = [
    ("circular Re=300",  300,  4, 0.03,   None),
    ("circular Re=400",  400,  4, 0.03,   0.548),
    ("circular Re=800",  800,  8, 0.03,   None),
    ("square   Re=101",  101,  4, 0.0438, 0.876),
]


# ----------------------------------------------------------------- loading

def load(re_, nz, dt):
    """Suffixed first, then unsuffixed. Returns (pred, true, t, resolved_name)."""
    tried = []
    for name in (f"rollout_Re{re_}_L{nz}.npy", f"rollout_Re{re_}.npy"):
        p = os.path.join(DATA, name)
        tried.append(name)
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True).item()
            missing = [k for k in ("raw_pred", "true") if k not in d]
            if missing:
                raise KeyError(f"{name}: missing {missing}; keys are {sorted(d)}")
            pred = np.asarray(d["raw_pred"], dtype=float)
            true = np.asarray(d["true"], dtype=float)
            if pred.shape != true.shape:
                raise ValueError(f"{name}: shape mismatch {pred.shape} vs {true.shape}")
            return pred, true, np.arange(len(true)) * dt, name
    raise FileNotFoundError(f"Re={re_}, nz={nz}: tried {tried} in {DATA}")


# ----------------------------------------------------------------- helpers

def orbit_geometry(traj):
    """Singular-value spectrum of the mean-centred trajectory.

    A planar circular orbit gives two equal leading values and nothing else.
    Departure from that is what creates the dilation confound.
    """
    X = traj - traj.mean(axis=0)
    s = np.linalg.svd(X, compute_uv=False)
    energy = s ** 2 / np.sum(s ** 2)
    pair_ratio = s[1] / s[0] if len(s) > 1 and s[0] > 0 else np.nan
    planarity = float(energy[0] + energy[1]) if len(energy) > 1 else float(energy[0])
    return s, energy, pair_ratio, planarity


def windowed_kappa(pred, true, dt, n_win=5):
    """kappa in equal windows across the scored horizon, to expose dilution."""
    T = len(true)
    edges = np.linspace(0, T, n_win + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 5:
            out.append(np.nan)
            continue
        out.append(rd.diagnose(pred[a:b], true[a:b], dt).kappa)
    return out


def growth(report, frac=0.2):
    """Tangential and transverse magnitude, first window vs last."""
    n = len(report.err_norm_t)
    k = max(3, int(n * frac))
    par_0, par_1 = report.tangential_t[1:k].mean(), report.tangential_t[-k:].mean()
    perp_0, perp_1 = report.transverse_t[1:k].mean(), report.transverse_t[-k:].mean()
    return (par_1 / par_0 if par_0 > 0 else np.nan,
            perp_1 / perp_0 if perp_0 > 0 else np.nan)


# ----------------------------------------------------------------- main

def main():
    if not os.path.isdir(DATA):
        sys.exit(f"data directory not found: {DATA}")

    print("=" * 78, flush=True)
    print("STEP 4 -- rolldrift against the paper's CAE-LSTM rollouts", flush=True)
    print(f"data: {DATA}", flush=True)
    print(f"scoring on steps [{CALIB_STEPS}:] (calibration window excluded)", flush=True)
    print("=" * 78, flush=True)

    summary = []

    for label, re_, nz, dt, T_shed in CASES:
        print(f"\n{'-' * 78}\n{label}   (n_z={nz}, dt={dt})\n{'-' * 78}", flush=True)
        try:
            pred_full, true_full, t_full, fname = load(re_, nz, dt)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  SKIPPED: {e}", flush=True)
            continue

        print(f"  resolved file : {fname}", flush=True)
        print(f"  full shape    : {true_full.shape}", flush=True)

        pred = pred_full[CALIB_STEPS:]
        true = true_full[CALIB_STEPS:]
        n_steps = len(true)
        span = n_steps * dt
        print(f"  scored steps  : {n_steps}  ({span:.2f} time units"
              + (f", {span / T_shed:.1f} shedding cycles)" if T_shed else ")"),
              flush=True)

        # ---------- Q3: orbit geometry (do this first; it frames Q2) --------
        s, energy, pair_ratio, planarity = orbit_geometry(true)
        print(f"\n  [Q3] orbit geometry of the reference latent trajectory", flush=True)
        print(f"       singular values      : "
              + "  ".join(f"{v:.4g}" for v in s), flush=True)
        print(f"       energy fractions     : "
              + "  ".join(f"{v:.4f}" for v in energy), flush=True)
        print(f"       s2/s1 (1.0 = circular): {pair_ratio:.4f}", flush=True)
        print(f"       planarity (s1+s2)    : {planarity:.4f}", flush=True)

        # ---------- Q1: reproduction ---------------------------------------
        r = rd.diagnose(pred, true, dt)
        print(f"\n  [Q1] reproduction", flush=True)
        print(f"       kappa (aggregate)    : {r.kappa:+.4f}", flush=True)
        print(f"       kappa (first fifth)  : {r.kappa_early:+.4f}", flush=True)
        print(f"       f_par (raw fraction) : {r.f_par:.4f}   "
              f"[paper reports 0.97-0.98]", flush=True)
        print(f"       isotropic null (1/n) : {r.f_par_null:.4f}", flush=True)
        print(f"       enrichment           : {r.enrichment:.3g}x", flush=True)
        print(f"       transverse fraction  : {r.transverse_fraction:.4f}", flush=True)
        print(f"       -> {r.interpretation()}", flush=True)

        g_par, g_perp = growth(r)
        print(f"       tangential growth    : {g_par:.2f}x   "
              f"[paper reports 11-17x]", flush=True)
        print(f"       transverse growth    : {g_perp:.2f}x   "
              f"[paper: does not grow]", flush=True)

        wk = windowed_kappa(pred, true, dt)
        print(f"       kappa by window      : "
              + "  ".join(f"{v:+.3f}" for v in wk), flush=True)

        if r.warnings:
            print(f"       warnings:", flush=True)
            for w in r.warnings:
                print(f"         - {w}", flush=True)

        # ---------- Q2: dilation confound ----------------------------------
        # Pure amplitude error about the orbit centre. On a circular orbit this
        # is purely transverse; on an asymmetric one it is not.
        mu = true.mean(axis=0)
        dilated = mu + (true - mu) * (1.0 + DILATION_EPS)
        r_dil = rd.diagnose(dilated, true, dt)

        # Isotropic control: random error must score ~1/n. Sanity check.
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(true.shape)
        noise *= (np.linalg.norm(pred - true) / np.linalg.norm(noise))
        r_iso = rd.diagnose(true + noise, true, dt)

        print(f"\n  [Q2] confound floors", flush=True)
        print(f"       pure {DILATION_EPS:.0%} dilation  -> f_par {r_dil.f_par:.4f}, "
              f"kappa {r_dil.kappa:+.4f}", flush=True)
        print(f"       isotropic noise      -> f_par {r_iso.f_par:.4f}, "
              f"kappa {r_iso.kappa:+.4f}   (null {r.f_par_null:.4f})", flush=True)

        margin = r.f_par - r_dil.f_par
        if margin > 0.3:
            verdict = "CLEAR -- well above what amplitude error alone can produce"
        elif margin > 0.1:
            verdict = "OK but narrower than ideal -- report the floor in the paper"
        else:
            verdict = ("PROBLEM -- the claim is not separated from the dilation "
                       "confound; investigate before resubmission")
        print(f"       margin over confound : {margin:+.4f}   [{verdict}]", flush=True)

        summary.append(dict(label=label, file=fname, n=r.n_state, kappa=r.kappa,
                            kappa_early=r.kappa_early, f_par=r.f_par,
                            null=r.f_par_null, dil=r_dil.f_par, iso=r_iso.f_par,
                            margin=margin, pair_ratio=pair_ratio,
                            g_par=g_par, g_perp=g_perp))

    # ------------------------------------------------------------- summary
    if not summary:
        sys.exit("\nNo cases loaded.")

    print("\n" + "=" * 78, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 78, flush=True)
    hdr = (f"{'case':<18}{'n':>3}{'f_par':>8}{'kappa':>8}{'k_early':>9}"
           f"{'dilat':>8}{'iso':>7}{'null':>7}{'s2/s1':>7}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for s_ in summary:
        print(f"{s_['label']:<18}{s_['n']:>3}{s_['f_par']:>8.4f}{s_['kappa']:>+8.4f}"
              f"{s_['kappa_early']:>+9.4f}{s_['dil']:>8.4f}{s_['iso']:>7.4f}"
              f"{s_['null']:>7.4f}{s_['pair_ratio']:>7.4f}", flush=True)

    print("\nHow to read this:", flush=True)
    print("  f_par vs 0.97-0.98 : does the library reproduce the paper?", flush=True)
    print("  kappa vs k_early   : large gap = horizon dilution, not a mixed mode", flush=True)
    print("  dilat              : the floor. f_par must clear it by a wide margin", flush=True)
    print("  iso vs null        : sanity check; these should agree", flush=True)
    print("  s2/s1              : ~1.0 means circular, so the dilation floor is low", flush=True)


if __name__ == "__main__":
    main()
