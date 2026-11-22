#!/usr/bin/env python3
"""R3.9/3.10, v2.  v1 clipped the linear-interpolation index after computing
the weight, which extrapolated in the last sample interval; v1 numbers are void.
Two fold conventions:
  pub    : exactly as periodic_baseline.py -- fractional source position in
           [NCAL-P, NCAL); the last interval interpolates towards sample NCAL
           (one sample beyond the calibration window).
  noleak : fold into [NCAL-1-P, NCAL-1), so only samples 0..NCAL-1 are used.
Interpolants: linear (published), cubic spline, band-limited trigonometric
least-squares fit (8 harmonics, last 3 periods at their fractional phases).
Also: 'incal' = tile the first half of the calibration window into its second
half, where the truth is known -- isolates interpolation error from any
genuine change of the flow."""
import os, json, numpy as np
from scipy.interpolate import make_interp_spline
from paths import DATA, OUT_REV
NCAL = 150; H = 8

def evaluate(base, src, method, upto, P):
    """values of the periodic extension at fractional positions src; base usable up to index upto (inclusive)"""
    b = base[:upto + 1].astype(float)
    if method == "linear":
        i0 = np.clip(np.floor(src).astype(int), 0, upto - 1); w = src - i0
        w = w.reshape((-1,) + (1,) * (b.ndim - 1)); return (1 - w) * b[i0] + w * b[i0 + 1]
    if method == "cubic":
        sp = make_interp_spline(np.arange(upto + 1), b.reshape(upto + 1, -1), k=3)
        return sp(src).reshape((len(src),) + b.shape[1:])
    if method == "fourier":
        j = np.arange(upto + 1 - int(np.ceil(3 * P)), upto + 1); ph = 2 * np.pi * j / P
        X = lambda q: np.column_stack([np.ones_like(q)] + [f(h * q) for h in range(1, H + 1) for f in (np.cos, np.sin)])
        c, *_ = np.linalg.lstsq(X(ph), b[j].reshape(len(j), -1), rcond=None)
        return (X(2 * np.pi * src / P) @ c).reshape((len(src),) + b.shape[1:])

def tile(series, P, n, method, conv, ncal=NCAL):
    ks = np.arange(ncal, n)
    if conv == "pub":  src, upto = ncal - P + ((ks - ncal) % P), ncal
    else:              src, upto = ncal - 1 - P + ((ks - ncal + 1) % P), ncal - 1
    out = series[:n].astype(float).copy(); out[ks] = evaluate(series, src, method, upto, P); return out

res = {}
for spec in ["400:4", "300:4", "800:8", "101:4"]:
    re_, ld = map(int, spec.split(":")); tag = f"Re{re_}_L{ld}"
    pb = json.load(open(f"{DATA}/periodic_baseline_{tag}.json")); P = pb["period_samples"]
    rp = f"{DATA}/rollout_{tag}.npy"; rp = rp if os.path.exists(rp) else f"{DATA}/rollout_Re{re_}.npy"
    roll = np.load(rp, allow_pickle=True).item()
    true = roll["true"].astype(float); pred = (roll["raw_pred"] if "raw_pred" in roll else roll["predicted"]).astype(float)
    n = min(len(true), len(pred)); true, pred = true[:n], pred[:n]; ext = slice(NCAL, n)
    em = lambda Z, sl=ext: float(np.linalg.norm(Z[sl] - true[sl], axis=1).mean())
    r = dict(P=P, lstm_uncorrected=em(pred), lstm_phase_published=pb["latent_scores"]["lstm_phase"]["mean_err"],
             tiling_published=pb["latent_scores"]["periodic_tiling"]["mean_err"])
    for conv in ("pub", "noleak"):
        for m in ("linear", "cubic", "fourier"):
            r[f"latent_{conv}_{m}"] = em(tile(true, P, n, m, conv))
    half = NCAL // 2
    for m in ("linear", "cubic", "fourier"):
        z = tile(true, P, NCAL, m, "noleak", ncal=half)
        r[f"incal_{m}"] = em(z, slice(half, NCAL))
    snaps = np.load(f"{DATA}/" + ("snapshots_SQ100.npy" if re_ == 101 else f"snapshots_Re{re_}.npy"), mmap_mode="r")
    N = len(snaps); ts = int(N * 0.52) + int(N * 0.08); ref = np.asarray(snaps[ts:ts + n], float)
    rms = float(np.sqrt(np.mean(ref[ext] ** 2)))
    r["field_published"] = pb["field"]["periodic_tiling_field"]["normalised"]
    r["field_lstm_phase_published"] = pb["field"]["lstm_phase"]["normalised"]
    r["field_cae_floor_published"] = pb["field"]["cae_floor"]["normalised"]
    for m in ("linear", "cubic", "fourier"):
        ft = tile(ref, P, n, m, "pub")
        r[f"field_pub_{m}"] = float(np.sqrt(np.mean((ft[ext] - ref[ext]) ** 2)) / rms)
    res[tag] = r
    print(tag, {k: round(v, 5) for k, v in r.items()}, flush=True)
json.dump(res, open(os.path.join(OUT_REV, "periodic_tiling_interp_v2.json"), "w"), indent=1)
