#!/usr/bin/env python3
"""
tangential_transverse.py
========================
Direct geometric test of the phase-reduction argument.

The paper argues that phase dominance is a property of the attractor, not of the
network: a stable limit cycle has one neutrally stable direction (the phase, zero
Floquet exponent) while transverse directions contract exponentially, so a
perturbation applied repeatedly to a trajectory on that cycle must accumulate
ALONG the cycle and not away from it.

At each rollout step we form the error vector in the full latent space,

    e(t) = z_hat(t) - z_true(t),

and split it against the local tangent of the TRUE trajectory,

    t_hat(t) = (dz_true/dt) / |dz_true/dt|,
    e_par    = (e . t_hat) t_hat        motion along the cycle  -> timing error
    e_perp   = e - e_par                displacement off the cycle

Prediction: |e_par| grows linearly (constant phase drift rate) while |e_perp|
shows no systematic trend. If both grow, the mechanism claim is wrong.

Computed on the full latent vector; uses no Hilbert transform, so it is
independent of the per-coordinate amplitude/phase decomposition used elsewhere.


CHANGES FROM THE PREVIOUS VERSION
---------------------------------
1. DEFAULT --smooth IS NOW 1, WAS 5.
   The tangent was previously estimated by a central difference over +/-5 steps,
   spanning 2*k*dt = 0.54-0.93 of a shedding period. For a pure sinusoid that is
   harmless (a scalar rescale leaves the direction unchanged), but harmonic
   content is distorted: at k=5 the second harmonic acquires a factor
   sin(3.44)/3.44 = -0.084, attenuated AND sign-flipped. The tangent is then
   rotated, and tangent misalignment biases the tangential share DOWNWARD as
   cos^2(alpha). The old default therefore UNDERSTATED the result.

   Run with --stencil-sweep to reproduce the convergence evidence. The share
   rises monotonically as k falls and plateaus by k=1 (Re=400 moves 0.0249 from
   k=5 to k=1, but only 0.0021 from k=2 to k=1). The size of the penalty tracks
   the energy outside the leading POD pair, which is the predicted mechanism.

2. THE DECILE GROWTH RATIO IS DEMOTED, NOT REPORTED AS A HEADLINE.
   It is convention-dependent: the same data yields 2.5x to 428x depending on
   the windowing choice, and the published 11-17x is specific to the last-decile
   / first-decile convention over the full rollout. It is still computed for
   continuity but is flagged in the output.

   The linear slope and R^2 replace it. Those are convention-free and are the
   statistics that actually support a CONSTANT drift rate, which is what makes a
   one-parameter correction exact.

3. NEW: DRIFT RATE PER SHEDDING PERIOD, orbit geometry, and the dilation floor.
   The drift rate is the convention-free statement of the mechanism. The
   dilation floor is the tangential share produced by a PURE amplitude error on
   the same orbit: it bounds how much of the measured share could be an artefact
   of orbit asymmetry rather than genuine phase drift.

Inputs (already produced by run_pipeline.py, no retraining):
    <data-dir>/rollout_Re{RE}_L{NZ}.npy   with keys 'raw_pred', 'true'

Usage
-----
    python3 tangential_transverse.py --dry-run
    python3 tangential_transverse.py --cases 300:4 400:4 800:8 101:4
    python3 tangential_transverse.py --stencil-sweep
    python3 tangential_transverse.py --cases 400:4 --data-dir ~/cylinder_lstm_data
"""

import os, sys, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="+", default=["300:4", "400:4", "800:8", "101:4"],
                help="RE:NZ pairs, e.g. 400:4")
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--dt", type=float, default=0.03,
                help="timestep (0.0438 for the square cylinder)")
ap.add_argument("--dt-square", type=float, default=0.0438,
                help="timestep used for Re=101 (square cylinder)")
ap.add_argument("--smooth", type=int, default=1,
                help="central-difference half-width for the tangent, in steps. "
                     "Default 1. Values above ~2 span an appreciable fraction of "
                     "a shedding period and bias the tangential share downward; "
                     "see --stencil-sweep.")
ap.add_argument("--skip", type=int, default=5,
                help="skip this many steps at EACH end (tangent ill-conditioned there)")
ap.add_argument("--dilation-eps", type=float, default=0.02,
                help="amplitude perturbation used for the confound floor")
ap.add_argument("--stencil-sweep", action="store_true",
                help="sweep the tangent half-width and report convergence")
ap.add_argument("--out-prefix", default="tangential_transverse")
ap.add_argument("--no-plot", action="store_true")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

DATA = os.path.expanduser(args.data_dir)


def load_rollout(re_, nz):
    for name in (f"rollout_Re{re_}_L{nz}.npy", f"rollout_Re{re_}.npy"):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True).item()
            pred = d["raw_pred"] if "raw_pred" in d else d["predicted"]
            true = d["true"]
            n = min(len(pred), len(true))
            return np.asarray(pred[:n], float), np.asarray(true[:n], float), name
    return None, None, None


def synthetic(nz=4, n=1200, dt=0.03, drift=0.004, amp_err=0.0013, seed=0):
    """A limit cycle with a small clock error and a small radial error, i.e.
    exactly the situation the theory describes. Used for --dry-run."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    w = 2 * np.pi * 1.8
    true = np.zeros((n, nz)); pred = np.zeros((n, nz))
    for d in range(nz):
        ph = 2 * np.pi * d / nz
        true[:, d] = np.cos(w * t + ph)
        pred[:, d] = (1 + amp_err) * np.cos((w + drift) * t + ph)
    pred += rng.normal(0, 1e-4, pred.shape)
    return pred, true


# ─────────────────────────────────────────────────────────── geometry helpers

def tangent(true, dt, k):
    """Unit tangent by central difference over +/-k steps."""
    n = len(true)
    k = max(1, int(k))
    tang = np.zeros_like(true)
    tang[k:n - k] = (true[2 * k:] - true[:n - 2 * k]) / (2 * k * dt)
    tang[:k] = (true[1:k + 1] - true[:k]) / dt
    tang[n - k:] = (true[n - k:] - true[n - k - 1:n - 1]) / dt
    speed = np.linalg.norm(tang, axis=1)
    ok = speed > 1e-12
    that = np.zeros_like(tang)
    that[ok] = tang[ok] / speed[ok, None]
    return that, speed


def split(pred, true, that):
    """Magnitudes of the error along and orthogonal to the tangent."""
    e = pred - true
    par_signed = np.einsum("ij,ij->i", e, that)
    tot2 = np.einsum("ij,ij->i", e, e)
    perp = np.sqrt(np.maximum(tot2 - par_signed ** 2, 0.0))
    return np.abs(par_signed), perp, np.sqrt(tot2), par_signed


def share_of(par, perp, sl):
    p, q = par[sl], perp[sl]
    denom = np.sum(p ** 2 + q ** 2)
    return float(np.sum(p ** 2) / denom) if denom > 0 else float("nan")


def linfit(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum((y - A @ c) ** 2) / ss) if ss > 0 else float("nan")
    return float(c[0]), float(c[1]), r2


def orbit_geometry(true, dt):
    """Dominant period, orbit radius, and how circular the orbit is."""
    n = len(true)
    mu = true.mean(axis=0)
    X = true - mu
    s = np.linalg.svd(X, compute_uv=False)
    energy = s ** 2 / np.sum(s ** 2)
    sig = X[:, int(np.argmax(X.var(axis=0)))]
    spec = np.abs(np.fft.rfft(sig * np.hanning(n))) ** 2
    fr = np.fft.rfftfreq(n, d=dt)
    T_shed = 1.0 / fr[int(np.argmax(spec[1:])) + 1]
    return dict(T_shed=float(T_shed),
                radius=float(np.median(np.linalg.norm(X, axis=1))),
                s2_over_s1=float(s[1] / s[0]) if s[0] > 0 else float("nan"),
                s3_over_s1=float(s[2] / s[0]) if len(s) > 2 and s[0] > 0 else 0.0,
                planarity=float(energy[0] + energy[1]) if len(energy) > 1
                else float(energy[0]))


def decompose(pred, true, dt, smooth, skip, dilation_eps):
    n = len(true)
    k = max(1, int(smooth))
    that, speed = tangent(true, dt, k)
    par, perp, tot, par_signed = split(pred, true, that)

    sl = slice(skip, max(skip + 1, n - max(skip, k)))
    t = np.arange(n) * dt
    tt = t[sl]

    out = dict(t=t, e_par=par, e_perp=perp, e_tot=tot,
               e_par_signed=par_signed, speed=speed, valid=sl)

    out["share_par"] = share_of(par, perp, sl)
    out["share_perp"] = 1.0 - out["share_par"]

    sp, ip, r2p = linfit(tt, par[sl])
    sq, iq, r2q = linfit(tt, perp[sl])
    out.update(slope_par=sp, intercept_par=ip, r2_par=r2p,
               slope_perp=sq, intercept_perp=iq, r2_perp=r2q)

    # convention-dependent, kept for continuity with earlier versions
    m = len(par[sl]) // 10 or 1
    out["growth_par"] = float(par[sl][-m:].mean() / max(par[sl][:m].mean(), 1e-15))
    out["growth_perp"] = float(perp[sl][-m:].mean() / max(perp[sl][:m].mean(), 1e-15))

    out["mean_par"] = float(par[sl].mean())
    out["mean_perp"] = float(perp[sl].mean())

    geo = orbit_geometry(true, dt)
    out.update(geo)
    # tangential displacement accumulated per shedding period, as a percentage
    # of the orbit circumference -> a convention-free drift rate
    out["drift_pct_per_period"] = float(
        100.0 * sp * geo["T_shed"] / (2 * np.pi * geo["radius"]))

    # confound floor: tangential share produced by a PURE amplitude error
    mu = true.mean(axis=0)
    dil = mu + (true - mu) * (1.0 + dilation_eps)
    par_d, perp_d, _, _ = split(dil, true, that)
    out["dilation_floor"] = share_of(par_d, perp_d, sl)
    out["dilation_eps"] = float(dilation_eps)
    out["smooth"] = int(k)
    out["skip"] = int(skip)
    return out


SUMMARY_KEYS = ("share_par", "share_perp", "slope_par", "slope_perp",
                "r2_par", "r2_perp", "growth_par", "growth_perp",
                "mean_par", "mean_perp", "T_shed", "radius", "s2_over_s1",
                "s3_over_s1", "planarity", "drift_pct_per_period",
                "dilation_floor", "dilation_eps", "smooth", "skip")

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 78, flush=True)
print("TANGENTIAL / TRANSVERSE DECOMPOSITION OF THE ROLLOUT ERROR", flush=True)
print("=" * 78, flush=True)
print("Prediction from phase reduction: the component ALONG the cycle grows", flush=True)
print("linearly, the component OFF the cycle shows no systematic trend.", flush=True)
print(f"tangent half-width k = {args.smooth} "
      f"({'converged' if args.smooth <= 2 else 'WIDE -- see --stencil-sweep'})\n",
      flush=True)

results, panels = {}, []

for spec in args.cases:
    try:
        re_s, nz_s = spec.split(":"); re_, nz = int(re_s), int(nz_s)
    except Exception:
        print(f"  skipping malformed case '{spec}' (want RE:NZ)", flush=True); continue

    dt = args.dt_square if re_ == 101 else args.dt

    if args.dry_run:
        pred, true = synthetic(nz=nz); src = "synthetic"
    else:
        pred, true, src = load_rollout(re_, nz)
        if pred is None:
            print(f"  Re={re_} L={nz}: no rollout file found, skipping", flush=True)
            continue

    d = decompose(pred, true, dt, args.smooth, args.skip, args.dilation_eps)
    key = f"Re{re_}_L{nz}"
    results[key] = {k: v for k, v in d.items() if k in SUMMARY_KEYS}
    panels.append((f"Re={re_}, $n_z$={nz}" if re_ != 101 else f"square, $n_z$={nz}", d))

    print(f"  {src}  (dt={dt}, k={d['smooth']})", flush=True)
    print(f"    tangential share      : {100*d['share_par']:6.2f}%    "
          f"transverse {100*d['share_perp']:5.2f}%", flush=True)
    print(f"    linear fit, tangential: slope {d['slope_par']:+.4e}  "
          f"R^2 {d['r2_par']:.4f}   <- supports a constant drift rate", flush=True)
    print(f"    linear fit, transverse: slope {d['slope_perp']:+.4e}  "
          f"R^2 {d['r2_perp']:.4f}   <- no systematic trend", flush=True)
    print(f"    drift rate            : {d['drift_pct_per_period']:.4f}% of the "
          f"orbit per shedding period (T={d['T_shed']:.4f})", flush=True)
    print(f"    confound floor        : {d['dilation_floor']:.4f}  "
          f"(pure {100*d['dilation_eps']:.0f}% amplitude error; "
          f"s2/s1 {d['s2_over_s1']:.3f}, planarity {d['planarity']:.4f})", flush=True)
    print(f"    [decile growth ratio  : tangential x{d['growth_par']:.2f}  "
          f"transverse x{d['growth_perp']:.2f}  -- CONVENTION-DEPENDENT, "
          f"not reported]", flush=True)

    ok = (d["share_par"] > 0.5
          and d["r2_par"] > 0.8
          and d["r2_perp"] < 0.1
          and d["share_par"] > 10 * d["dilation_floor"])
    print(f"    -> {'consistent with the mechanism' if ok else 'NOT consistent -- check'}\n",
          flush=True)

if not results:
    sys.exit("No cases processed.")

# ─────────────────────────────────────────────────────────── stencil sweep
if args.stencil_sweep:
    print("=" * 78, flush=True)
    print("STENCIL CONVERGENCE  (evidence for the default k=1)", flush=True)
    print("=" * 78, flush=True)
    print("Tangent misalignment biases the share DOWNWARD as cos^2(alpha), so a", flush=True)
    print("share that RISES as k falls means a wide stencil was understating it.\n",
          flush=True)
    sweep = {}
    for spec in args.cases:
        re_, nz = (int(x) for x in spec.split(":"))
        dt = args.dt_square if re_ == 101 else args.dt
        if args.dry_run:
            pred, true = synthetic(nz=nz)
        else:
            pred, true, _ = load_rollout(re_, nz)
            if pred is None:
                continue
        geo = orbit_geometry(true, dt)
        key = f"Re{re_}_L{nz}"
        print(f"  {key}   T_shed={geo['T_shed']:.4f}   s3/s1={geo['s3_over_s1']:.4f}",
              flush=True)
        print(f"    {'k':>3}{'span/T':>9}{'share_par':>12}", flush=True)
        rec = {}
        for k in (1, 2, 3, 5, 8):
            that, _ = tangent(true, dt, k)
            par, perp, _, _ = split(pred, true, that)
            sl = slice(args.skip, max(args.skip + 1, len(true) - max(args.skip, k)))
            s_k = share_of(par, perp, sl)
            rec[str(k)] = s_k
            print(f"    {k:>3}{2*k*dt/geo['T_shed']:>9.3f}{s_k:>12.4f}", flush=True)
        rec["penalty_k5_vs_k1"] = rec["1"] - rec["5"]
        rec["s3_over_s1"] = geo["s3_over_s1"]
        sweep[key] = rec
        print(f"    understated by {rec['penalty_k5_vs_k1']:+.4f} at k=5\n", flush=True)
    fp = os.path.join("." if args.dry_run else DATA, "stencil_convergence.json")
    json.dump(sweep, open(fp, "w"), indent=2, default=float)
    print(f"Saved -> {fp}\n", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
out_json = os.path.join("." if args.dry_run else DATA, f"{args.out_prefix}.json")
json.dump(results, open(out_json, "w"), indent=2, default=float)
print(f"Saved -> {out_json}", flush=True)

if not args.no_plot:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ncol = 2 if len(panels) > 1 else 1
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow),
                                 squeeze=False)
        for ax, (title, d) in zip(axes.ravel(), panels):
            sl = d["valid"]; t = d["t"][sl]
            ax.plot(t, d["e_par"][sl], lw=1.0, color="C3",
                    label=r"tangential $|e_\parallel|$ (along the cycle)")
            ax.plot(t, d["e_perp"][sl], lw=1.0, color="C0",
                    label=r"transverse $|e_\perp|$ (off the cycle)")
            ax.plot(t, d["slope_par"] * t + d["intercept_par"], ls="--", lw=1.0,
                    color="k", alpha=.7,
                    label=rf"linear fit, $R^2$={d['r2_par']:.3f}")
            ax.axhline(d["mean_perp"], ls=":", lw=1.0, color="C0", alpha=.8)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("t"); ax.set_ylabel("error magnitude")
            ax.grid(alpha=.3)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=7, loc="upper left")
        fig.suptitle("Rollout error resolved along and across the limit cycle",
                     fontsize=11)
        fig.tight_layout()
        fp = os.path.join("." if args.dry_run else DATA, f"{args.out_prefix}.pdf")
        fig.savefig(fp, dpi=200, bbox_inches="tight")
        fig.savefig(fp.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        print(f"Figure -> {fp} (and .png)", flush=True)
    except Exception as e:
        print(f"[plot skipped: {e}]", flush=True)

print("""
Reading the output:
  * 'tangential share' is the headline: how much of the squared error lies along
    the cycle rather than off it. Phase reduction predicts most of it.
  * The TANGENTIAL R^2 is the statistic that supports a constant drift rate,
    which is what makes a one-parameter correction exact. The TRANSVERSE R^2
    being near zero is what supports boundedness. Report both; do not report the
    decile growth ratio, which depends on an unstated windowing convention.
  * 'drift rate' states the mechanism without reference to any convention.
  * 'confound floor' bounds how much of the share a pure amplitude error could
    explain on this orbit. The share should exceed it by a wide margin.
""", flush=True)
