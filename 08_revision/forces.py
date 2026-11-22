#!/usr/bin/env python3
"""Force coefficients of an OpenFOAM case (forceCoeffs1, all restart segments joined,
later segments overriding overlaps).  Window statistics: mean Cd, rms Cl', St from
the mean period of upward zero crossings of Cl' (linear interpolation; force data are
written every time step), cycle-to-cycle period/amplitude scatter, and the change of
mean period and amplitude between the two halves of the window.
Also resamples Cl onto a uniform grid (for the lift-only model, R2.4).
Usage: forces.py CASE T0 T1 [--D 0.12 --U 1] [--resample DT OUT.npy]"""
import sys, os, glob, json, argparse, numpy as np

def load(case):
    segs = []
    for d in glob.glob(os.path.join(case, "postProcessing/forceCoeffs1/*")):
        f = os.path.join(d, "coefficient.dat")
        if not os.path.exists(f): continue
        a = np.loadtxt(f, comments="#", usecols=(0, 1, 4))
        segs.append((float(os.path.basename(d)), a))
    segs.sort(key=lambda x: x[0]); out = None
    for t0, a in segs:
        out = a if out is None else np.vstack([out[out[:, 0] < a[0, 0]], a])
    return out[:, 0], out[:, 1], out[:, 2]

def stats(t, cd, cl, t0, t1, D=0.12, U=1.0):
    k = (t >= t0) & (t <= t1); t, cd, cl = t[k], cd[k], cl[k]
    s = cl - cl.mean(); i = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    tc = t[i] - s[i] * (t[i + 1] - t[i]) / (s[i + 1] - s[i]); T = np.diff(tc)
    A = np.array([0.5 * (s[(t >= a) & (t < b)].max() - s[(t >= a) & (t < b)].min()) for a, b in zip(tc[:-1], tc[1:])])
    h = len(T) // 2
    return dict(t0=t0, t1=t1, n_cycles=int(len(T)), Cd_mean=float(cd.mean()), Cl_rms=float(s.std()),
                Cl_amp=float(A.mean()), St=float(D / (U * T.mean())), T_cv=float(T.std() / T.mean()),
                A_cv=float(A.std() / A.mean()), dT_halves=float(T[h:].mean() / T[:h].mean() - 1),
                dA_halves=float(A[h:].mean() / A[:h].mean() - 1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("case"); ap.add_argument("t0", type=float); ap.add_argument("t1", type=float)
    ap.add_argument("--D", type=float, default=0.12); ap.add_argument("--U", type=float, default=1.0)
    ap.add_argument("--resample", nargs=2, default=None); ap.add_argument("--json", default=None)
    a = ap.parse_args(); t, cd, cl = load(a.case); st = stats(t, cd, cl, a.t0, a.t1, a.D, a.U)
    print(json.dumps(st))
    if a.json: json.dump(st, open(a.json, "w"), indent=1)
    if a.resample:
        dt = float(a.resample[0]); tg = np.arange(a.t0, a.t1 + 1e-9, dt)
        np.save(a.resample[1], np.interp(tg, t, cl).astype(np.float32)); print("resampled", len(tg), "->", a.resample[1])
