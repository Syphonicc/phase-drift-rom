#!/usr/bin/env python3
"""
hankel_dmd.py
=============
Delay-embedded (Hankel) DMD as a competent linear propagator, and a test of the
paper's central prediction.

The single-snapshot DMD reported elsewhere in the paper is a weak comparator for
oscillatory dynamics: one linear map on the instantaneous state cannot represent
a trajectory whose latent coordinates are not themselves a Koopman-invariant set.
Delay embedding removes that objection --- with enough delays the Hankel-DMD
operator spans the dominant Koopman modes of a periodic flow and becomes a strong
predictor.

That makes it the right instrument for a different question. The paper argues that
phase dominance follows from the geometry of the attractor rather than from the
architecture of the propagator: the direction along a limit cycle is neutrally
stable, so any repeatedly-applied propagator accumulates error there. If that is
right, a linear operator fitted in closed form -- no training, no nonlinearity, no
recurrence -- should drift in phase exactly as the network does.

This script therefore does two things at once:

  (a) BASELINE. Sweeps the number of delays and reports rollout accuracy against
      the LSTM, so the linear comparator is a fair one rather than a straw man.

  (b) TEST. Runs the same phase analysis on the DMD rollout as on the LSTM's:
      accumulated phase phi(t), its drift rate and linearity, and the geometric
      tangential/transverse split. If the drift is present and linear here too,
      the mechanism is a property of the flow, not of the network.

Everything operates on the LATENT trajectory already produced by run_pipeline.py,
so the propagator is the only thing that changes between the two.

Usage
-----
    python3 hankel_dmd.py --dry-run
    python3 hankel_dmd.py --cases 300:4 400:4 800:8 101:4
    python3 hankel_dmd.py --cases 400:4 --delays 1 5 10 20 40 60
"""

import os, sys, json, argparse
import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress

ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="+", default=["300:4", "400:4", "800:8", "101:4"])
ap.add_argument("--delays", type=int, nargs="+", default=[1, 5, 10, 20, 40, 60])
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--dt", type=float, default=0.03)
ap.add_argument("--dt-square", type=float, default=0.0438)
ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64],
                help="explicit SVD ranks to try (energy criteria fail here)")
ap.add_argument("--max-rank", type=int, default=200)
ap.add_argument("--train-frac", type=float, default=0.60,
                help="fraction of the latent series used to FIT the operator; the "
                     "remainder is the extrapolation region, matching the LSTM split")
ap.add_argument("--out-prefix", default="hankel_dmd")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

DATA = os.path.expanduser(args.data_dir)


# ── data ────────────────────────────────────────────────────────────────────
def load_latents(re_, nz):
    """Load the latent series and NORMALISE it exactly as run_pipeline.py does
    before training the LSTM.

    This matters: latents_*.npy is stored raw (RMS ~11 for Re=400), whereas the
    rollout arrays -- and therefore the LSTM error we compare against -- are in
    normalised units (RMS 1). Comparing a DMD error computed on raw latents with
    an LSTM error computed on normalised ones is an order-of-magnitude mistake.
    """
    lat = None
    for nm in (f"latents_Re{re_}_L{nz}.npy", f"latents_Re{re_}.npy"):
        p = os.path.join(DATA, nm)
        if os.path.exists(p):
            lat, src = np.load(p).astype(float), nm
            break
    if lat is None:
        for name in (f"rollout_Re{re_}_L{nz}.npy", f"rollout_Re{re_}.npy"):
            q = os.path.join(DATA, name)
            if os.path.exists(q):
                d = np.load(q, allow_pickle=True).item()
                # already normalised
                return np.asarray(d["true"], float), name + " (test segment, normalised)"
        return None, None

    for nm in (f"latent_norm_Re{re_}_L{nz}.npy", f"latent_norm_Re{re_}.npy"):
        p = os.path.join(DATA, nm)
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True).item()
            m = np.asarray(d["mean"]).ravel(); sd = np.asarray(d["std"]).ravel()
            sd = np.where(sd < 1e-12, 1.0, sd)
            return (lat - m) / sd, src + " (normalised)"
    # no norm file: standardise on the training portion so units are comparable
    k = int(len(lat) * 0.60)
    m, sd = lat[:k].mean(0), lat[:k].std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (lat - m) / sd, src + " (self-normalised)"


def load_lstm_error(re_, nz):
    """Mean latent L2 error of the LSTM over its extrapolation region, for
    reference. Returns None if unavailable."""
    for name in (f"rollout_Re{re_}_L{nz}.npy", f"rollout_Re{re_}.npy"):
        q = os.path.join(DATA, name)
        if os.path.exists(q):
            d = np.load(q, allow_pickle=True).item()
            pred = d["raw_pred"] if "raw_pred" in d else d["predicted"]
            true = d["true"]
            n = min(len(pred), len(true))
            return float(np.linalg.norm(np.asarray(pred[:n], float)
                                        - np.asarray(true[:n], float), axis=1).mean())
    return None


def synthetic_latents(nz=4, n=1400, dt=0.03, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    w = 2 * np.pi * 1.8
    z = np.zeros((n, nz))
    for d in range(nz):
        z[:, d] = np.cos(w * t + 2 * np.pi * d / nz) + 0.15 * np.cos(2 * w * t)
    return z + rng.normal(0, 1e-3, z.shape)


# ── Hankel DMD ──────────────────────────────────────────────────────────────
def hankel(Z, q):
    """Stack q delays. Z is (n, nz) -> H is (q*nz, n-q+1)."""
    n, nz = Z.shape
    m = n - q + 1
    H = np.empty((q * nz, m))
    for i in range(q):
        H[i * nz:(i + 1) * nz, :] = Z[i:i + m].T
    return H


def fit_dmd(H, rank, max_rank):
    """Exact DMD on the Hankel pair with an EXPLICIT rank.

    An energy criterion is the wrong instrument here. For a periodic signal the
    leading three modes carry >99.9% of the energy, yet the harmonics beyond them
    -- individually tiny in energy -- are what make the operator predictive. A
    0.9999-energy cut keeps ~5 modes and loses two orders of magnitude of
    accuracy; the rank must therefore be set directly.
    """
    X, Y = H[:, :-1], H[:, 1:]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = max(2, min(int(rank), max_rank, len(S)))
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T
    Atil = Ur.T @ Y @ Vr @ np.diag(1.0 / Sr)
    return Ur, Atil, r


def rollout_dmd(Z, q, split, rank, max_rank):
    """Fit on Z[:split], then advance the delay state autoregressively over the
    remainder. Returns the predicted latent series for the extrapolation region."""
    nz = Z.shape[1]
    Htr = hankel(Z[:split], q)
    if Htr.shape[1] < 3:
        return None, None
    Ur, Atil, r = fit_dmd(Htr, rank, max_rank)

    # advance in the reduced coordinates; state is the last delay column
    h = Htr[:, -1].copy()
    a = Ur.T @ h
    n_pred = len(Z) - split
    out = np.zeros((n_pred, nz))
    for k in range(n_pred):
        a = Atil @ a
        h = Ur @ a
        # column j of the Hankel matrix is [Z[j], Z[j+1], ..., Z[j+q-1]], so the
        # NEWEST snapshot is the last block, not the first.
        out[k] = h[-nz:]
    return out, r


# ── phase analysis (identical treatment to the LSTM) ────────────────────────
def iphase(x):
    return np.unwrap(np.angle(hilbert(x - x.mean())))


def phase_drift(pred, true, dt, ncal=150):
    """Accumulated phase difference per coordinate; drift rate and linearity
    measured on a calibration window, exactly as for the network."""
    n, nz = pred.shape
    t = np.arange(n) * dt
    om, r2 = [], []
    for d in range(nz):
        phi = iphase(pred[:, d]) - iphase(true[:, d])
        phi -= phi[0]
        m = min(ncal, n - 1)
        lr = linregress(t[:m], phi[:m])
        om.append(lr.slope); r2.append(lr.rvalue ** 2)
    return np.array(om), np.array(r2)


def tangential_split(pred, true, dt, k=5, skip=5):
    n = len(true)
    e = pred - true
    tang = np.zeros_like(true)
    tang[k:n-k] = (true[2*k:] - true[:n-2*k]) / (2 * k * dt)
    tang[:k] = (true[1:k+1] - true[:k]) / dt
    tang[n-k:] = (true[n-k:] - true[n-k-1:n-1]) / dt
    sp = np.linalg.norm(tang, axis=1); ok = sp > 1e-12
    that = np.zeros_like(tang); that[ok] = tang[ok] / sp[ok, None]
    epar = np.einsum("ij,ij->i", e, that)
    eperp = np.linalg.norm(e - epar[:, None] * that, axis=1)
    sl = slice(skip, max(skip + 1, n - max(skip, k)))
    p, q_ = np.abs(epar[sl]), eperp[sl]
    share = float(np.sum(p**2) / np.sum(p**2 + q_**2))
    tt = (np.arange(n) * dt)[sl]
    A = np.vstack([tt, np.ones_like(tt)]).T
    cp, *_ = np.linalg.lstsq(A, p, rcond=None)
    ss = np.sum((p - p.mean())**2)
    r2p = float(1 - np.sum((p - A @ cp)**2) / ss) if ss > 0 else float("nan")
    return share, float(cp[0]), r2p


# ── run ─────────────────────────────────────────────────────────────────────
print("=" * 78, flush=True)
print("DELAY-EMBEDDED (HANKEL) DMD — competent linear propagator", flush=True)
print("=" * 78, flush=True)
print("Question: does a linear operator, fitted in closed form with no training,", flush=True)
print("show the same linear phase drift as the network?\n", flush=True)

results = {}
for spec in args.cases:
    try:
        re_s, nz_s = spec.split(":"); re_, nz = int(re_s), int(nz_s)
    except Exception:
        print(f"  malformed case '{spec}', skipping", flush=True); continue
    dt = args.dt_square if re_ == 101 else args.dt

    if args.dry_run:
        Z, src = synthetic_latents(nz=nz), "synthetic"
        lstm_err = None
    else:
        Z, src = load_latents(re_, nz)
        if Z is None:
            print(f"  Re={re_} L={nz}: no latent series found, skipping", flush=True)
            continue
        lstm_err = load_lstm_error(re_, nz)

    split = int(len(Z) * args.train_frac)
    true_ex = Z[split:]
    print(f"  Re={re_} n_z={nz}   source: {src}", flush=True)
    print(f"    series {Z.shape}, fit on {split}, extrapolate {len(true_ex)}", flush=True)
    if lstm_err is not None:
        print(f"    LSTM mean latent error over its rollout: {lstm_err:.5f}", flush=True)
    print(f"    {'delays':>7} {'rank':>5} {'mean err':>10} {'omega_d':>11} "
          f"{'R^2(phi)':>9} {'tang.share':>11} {'R^2(tang)':>10}", flush=True)

    per_case = []
    for q in args.delays:
        if split - q < 10:
            continue
        for rank in args.ranks:
            if rank > q * nz:            # cannot exceed the Hankel row count
                continue
            try:
                pred, r = rollout_dmd(Z, q, split, rank, args.max_rank)
            except np.linalg.LinAlgError:
                print(f"    {q:>7} {'--':>5}  SVD failed", flush=True); continue
            if pred is None:
                continue
            n = min(len(pred), len(true_ex))
            pred, tru = pred[:n], true_ex[:n]
            if not np.all(np.isfinite(pred)):
                print(f"    {q:>7} {r:>5}   diverged (non-finite)", flush=True)
                per_case.append(dict(delays=q, rank=int(r), diverged=True)); continue
            err = float(np.linalg.norm(pred - tru, axis=1).mean())
            om, r2 = phase_drift(pred, tru, dt)
            share, slope_p, r2_tan = tangential_split(pred, tru, dt)
            flag = ""
            if lstm_err is not None and err < lstm_err:
                flag = "  <- beats LSTM"
            print(f"    {q:>7} {r:>5} {err:>10.5f} {np.mean(om):>+11.5f} "
                  f"{np.mean(r2):>9.3f} {100*share:>10.1f}% {r2_tan:>10.3f}{flag}",
                  flush=True)
            per_case.append(dict(delays=q, rank=int(r), mean_err=err,
                                 omega_mean=float(np.mean(om)),
                                 omega_per_dim=[float(x) for x in om],
                                 r2_phi=float(np.mean(r2)),
                                 tang_share=float(share), tang_slope=float(slope_p),
                                 r2_tangential=float(r2_tan), diverged=False))
    ok = [p for p in per_case if not p.get("diverged")]
    if ok and lstm_err:
        best = min(ok, key=lambda p: p["mean_err"])
        rel = best["mean_err"] / lstm_err
        print(f"    best: {best['delays']} delays, error {best['mean_err']:.5f} "
              f"= {rel:.2f}x the LSTM", flush=True)
    results[f"Re{re_}_L{nz}"] = dict(lstm_mean_err=lstm_err, sweep=per_case)
    print("", flush=True)

if not results:
    sys.exit("No cases processed.")

out = os.path.join("." if args.dry_run else DATA, f"{args.out_prefix}.json")
json.dump(results, open(out, "w"), indent=2, default=float)
print(f"Saved -> {out}", flush=True)
print("""
Reading the output:
  * 'mean err' against the LSTM figure says whether the linear comparator is
    competent. With enough delays it may well be; that is the point of running it.
  * 'omega_d' and 'R^2(phi)' are the test. A non-zero drift rate with a high R^2
    means the linear operator accumulates phase error linearly, exactly as the
    network does -- so the mechanism belongs to the flow, not the architecture.
  * 'tang.share' repeats the geometric decomposition for the DMD rollout. If it
    is as high here as for the network, the same conclusion follows independently
    of the Hilbert transform.
  * Compare omega_d across delays: a more competent propagator (more delays,
    lower error) that STILL drifts is the strongest form of the result.
""", flush=True)
