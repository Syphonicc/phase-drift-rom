#!/usr/bin/env python3
"""
tangential_transverse_v2.py  (2026-11-24, presentation-only revision of tangential_transverse.py)
Changes: figure sized for one column (3.35 in) with 7-8 pt fonts; shared legend above the panels whose
line colours match the data (tangential red, transverse blue); per-panel R^2_mag of the tangential fit in
the panel title; upright Re; no suptitle; --dry-run removed. ALL outputs (json, pdf) go to --out.
Paper figure: --cases 300:4 400:4 800:8 101:4 --smooth 1 (tangent half-width k=1).
========================
Direct geometric test of the phase-reduction argument.

The paper now argues that phase dominance is a property of the attractor, not of
the network: a stable limit cycle has one neutrally stable direction (the phase,
zero Floquet exponent) while transverse directions contract exponentially, so a
perturbation applied repeatedly to a trajectory on that cycle must accumulate
ALONG the cycle and not away from it.

This script tests that claim directly, in the geometry the theory refers to,
rather than through the per-coordinate amplitude/phase decomposition used
elsewhere in the paper. At each rollout step we form the error vector in the full
latent space,

    e(t) = z_hat(t) - z_true(t),

and split it against the local tangent of the TRUE trajectory,

    t_hat(t) = (dz_true/dt) / |dz_true/dt|,
    e_par    = (e . t_hat) t_hat        motion along the cycle  -> timing error
    e_perp   = e - e_par                displacement off the cycle

Prediction: |e_par| grows (approximately linearly, as the phase drifts) while
|e_perp| stays bounded. If both grow, the mechanism claim is wrong.

This is an independent measurement, not a replot of the amplitude/phase table:
it is computed on the full latent vector and uses no Hilbert transform.

Inputs (already produced by run_pipeline.py, no retraining):
    <data-dir>/rollout_Re{RE}_L{NZ}.npy   with keys 'raw_pred', 'true'

Usage
-----
    python3 tangential_transverse.py --dry-run
    python3 tangential_transverse.py --cases 300:4 400:4 800:8 101:4
    python3 tangential_transverse.py --cases 400:4 --data-dir ~/cylinder_lstm_data
"""

import os, sys, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="+", default=["300:4", "400:4", "800:8", "101:4"],
                help="RE:NZ pairs, e.g. 400:4")
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--dt", type=float, default=0.03, help="timestep (0.0438 for the square cylinder)")
ap.add_argument("--dt-square", type=float, default=0.0438,
                help="timestep used for Re=101 (square cylinder)")
ap.add_argument("--smooth", type=int, default=5,
                help="central-difference half-width for the tangent, in steps")
ap.add_argument("--skip", type=int, default=5,
                help="skip this many steps at EACH end (tangent ill-conditioned there)")
ap.add_argument("--out-prefix", default="fig09_tangential_transverse_k1_v2")
ap.add_argument("--out", required=True, help="output directory for json and pdf")
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
        # prediction: same geometry, slightly wrong clock + tiny radial error
        pred[:, d] = (1 + amp_err) * np.cos((w + drift) * t + ph)
    pred += rng.normal(0, 1e-4, pred.shape)
    return pred, true


def decompose(pred, true, dt, smooth=5, skip=5):
    """Split the error into components along and orthogonal to the local tangent
    of the true trajectory."""
    n = len(true)
    e = pred - true

    # local tangent by central difference over +/- `smooth` steps (smoother and
    # better conditioned than a one-step difference)
    k = max(1, int(smooth))
    tang = np.zeros_like(true)
    tang[k:n-k] = (true[2*k:] - true[:n-2*k]) / (2 * k * dt)
    # endpoints: one-sided
    tang[:k] = (true[1:k+1] - true[:k]) / dt
    tang[n-k:] = (true[n-k:] - true[n-k-1:n-1]) / dt

    speed = np.linalg.norm(tang, axis=1)
    ok = speed > 1e-12
    that = np.zeros_like(tang)
    that[ok] = tang[ok] / speed[ok, None]

    e_par_signed = np.einsum("ij,ij->i", e, that)          # signed component along
    e_par_vec = e_par_signed[:, None] * that
    e_perp_vec = e - e_par_vec

    mag_par = np.abs(e_par_signed)
    mag_perp = np.linalg.norm(e_perp_vec, axis=1)
    mag_tot = np.linalg.norm(e, axis=1)

    # skip both ends: the one-sided tangent at the endpoints is ill-conditioned
    sl = slice(skip, max(skip + 1, n - max(skip, k)))
    out = dict(t=np.arange(n) * dt, e_par=mag_par, e_perp=mag_perp, e_tot=mag_tot,
               e_par_signed=e_par_signed, speed=speed, valid=sl)

    # ---- summary statistics over the valid window --------------------------
    p, q = mag_par[sl], mag_perp[sl]
    tt = out["t"][sl]
    # share of the squared error carried by each component
    out["share_par"] = float(np.sum(p**2) / np.sum(p**2 + q**2))
    out["share_perp"] = 1.0 - out["share_par"]
    # linear fit to the tangential magnitude (the theory says it grows)
    A = np.vstack([tt, np.ones_like(tt)]).T
    coef_p, *_ = np.linalg.lstsq(A, p, rcond=None)
    coef_q, *_ = np.linalg.lstsq(A, q, rcond=None)
    out["slope_par"] = float(coef_p[0]); out["intercept_par"] = float(coef_p[1])
    out["slope_perp"] = float(coef_q[0]); out["intercept_perp"] = float(coef_q[1])
    # R^2 of those fits
    def r2(y, c):
        yh = A @ c
        ss = np.sum((y - y.mean())**2)
        return float(1 - np.sum((y - yh)**2) / ss) if ss > 0 else float("nan")
    out["r2_par"] = r2(p, coef_p); out["r2_perp"] = r2(q, coef_q)
    # growth over the window: last decile mean / first decile mean
    m = len(p) // 10 or 1
    out["growth_par"] = float(p[-m:].mean() / max(p[:m].mean(), 1e-15))
    out["growth_perp"] = float(q[-m:].mean() / max(q[:m].mean(), 1e-15))
    out["mean_par"] = float(p.mean()); out["mean_perp"] = float(q.mean())
    return out


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 76, flush=True)
print("TANGENTIAL / TRANSVERSE DECOMPOSITION OF THE ROLLOUT ERROR", flush=True)
print("=" * 76, flush=True)
print("Prediction from phase reduction: the component ALONG the cycle grows,", flush=True)
print("the component OFF the cycle stays bounded.\n", flush=True)

results = {}
panels = []

for spec in args.cases:
    try:
        re_s, nz_s = spec.split(":"); re_, nz = int(re_s), int(nz_s)
    except Exception:
        print(f"  skipping malformed case '{spec}' (want RE:NZ)", flush=True); continue

    dt = args.dt_square if re_ == 101 else args.dt

    if True:
        pred, true, src = load_rollout(re_, nz)
        if pred is None:
            print(f"  Re={re_} L={nz}: no rollout file found, skipping", flush=True)
            continue

    d = decompose(pred, true, dt, args.smooth, args.skip)
    results[f"Re{re_}_L{nz}"] = {k: v for k, v in d.items()
                                 if k in ("share_par", "share_perp", "slope_par",
                                          "slope_perp", "r2_par", "r2_perp",
                                          "growth_par", "growth_perp",
                                          "mean_par", "mean_perp")}
    panels.append((rf"$\mathrm{{Re}}={re_}$, $n_z={nz}$" if re_ != 101 else rf"square, $n_z={nz}$", d))

    print(f"  {src}  (dt={dt})", flush=True)
    print(f"    share of squared error: tangential {100*d['share_par']:5.1f}%   "
          f"transverse {100*d['share_perp']:5.1f}%", flush=True)
    print(f"    growth over rollout   : tangential x{d['growth_par']:6.2f}   "
          f"transverse x{d['growth_perp']:6.2f}", flush=True)
    print(f"    linear fit slope      : tangential {d['slope_par']:+.3e} (R^2 {d['r2_par']:.3f})   "
          f"transverse {d['slope_perp']:+.3e} (R^2 {d['r2_perp']:.3f})", flush=True)
    verdict = ("consistent with the mechanism"
               if d["share_par"] > 0.5 and d["growth_par"] > d["growth_perp"]
               else "NOT consistent -- check")
    print(f"    -> {verdict}\n", flush=True)

if not results:
    sys.exit("No cases processed.")

# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(args.out, exist_ok=True)
out_json = os.path.join(args.out, f"{args.out_prefix}.json")
json.dump(results, open(out_json, "w"), indent=2, default=float)
print(f"Saved -> {out_json}", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6})
ncol = 2; nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.35, 1.6 * nrow + 0.35), squeeze=False)
letters = "abcdefgh"
for i, (ax, (title, d)) in enumerate(zip(axes.ravel(), panels)):
    sl = d["valid"]; t = d["t"][sl]
    ax.plot(t, d["e_par"][sl], lw=0.6, color="C3", label=r"tangential $|e_\parallel|$")
    ax.plot(t, d["e_perp"][sl], lw=0.6, color="C0", label=r"transverse $|e_\perp|$")
    ax.plot(t, d["slope_par"] * t + d["intercept_par"], ls="--", lw=0.8,
            color="k", alpha=.7, label="linear fit (tangential)")
    ax.axhline(d["mean_perp"], ls=":", lw=0.8, color="C0", alpha=.8, label=r"mean $|e_\perp|$")
    ax.set_title(f"({letters[i]}) {title}\n" + r"$R^2_{\mathrm{mag}}$" + f"$={d['r2_par']:.3f}$", pad=2, linespacing=1.1)
    ax.set_xlabel(r"$t$", labelpad=1)
    if i % ncol == 0: ax.set_ylabel("error magnitude", labelpad=2)
    ax.tick_params(length=2, width=0.5, pad=1.5)
    ax.grid(alpha=.3, lw=0.4)
for ax in axes.ravel()[len(panels):]:
    ax.axis("off")
h, l = axes.ravel()[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.0, handlelength=1.8, handletextpad=0.4, borderaxespad=0.1)
fig.tight_layout(pad=0.2, h_pad=0.4, w_pad=0.6, rect=(0, 0, 1, 1 - 0.32 / (1.6 * nrow + 0.35)))
fp = os.path.join(args.out, f"{args.out_prefix}.pdf")
fig.savefig(fp, bbox_inches="tight", pad_inches=0.02)
print(f"Figure -> {fp}", flush=True)
