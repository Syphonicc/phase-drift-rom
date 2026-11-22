#!/usr/bin/env python3
"""
v2: perturbations are applied to the WHOLE input window (v1 kicked only the newest of
20 inputs, which the other 19 dilute). Kicks: amplitude (window scaled about the
training mean by 1+eps), out-of-plane offset (eps*n, n orthogonal to each sample's
tangent), and time shift (window advanced along its own derivative by eps). Metrics:
amplitude recovery from the Hilbert envelope of perturbed vs reference rollout
(|A_p/A_r-1|/eps averaged over a cycle), and persistence of the phase offset
(mean phase difference over cycles 8-9 relative to cycles 1-2).
(fix: perturbed and reference rollouts are now aligned step for step; checked by an
 unperturbed rerun reproducing the reference.)
R3.8 and R4.1 -- properties of the LEARNED map, from trained LSTMs only.

R3.8  free-run clock.  True shedding frequency f_true from the TRAINING latents only
      (linear fit of unwrapped Hilbert phase, one cycle trimmed at each end, averaged
      over coordinates).  Model frequency f_free from a long autonomous run of the LSTM
      started from the last training window (NFREE steps, first NBURN discarded).
      Predicted drift rate  w_pred = 2*pi*(f_free - f_true)  [rad per unit time], to be
      compared with the drift measured in the rollout (w_roll, slope of phi over the
      whole rollout) and with the calibration-window estimate (w_cal).  A causal
      correction uses w_pred in place of w_cal (Eq. 13); efficacy reported over the
      same extrapolation region (steps >= NCAL).
R4.1  perturbation.  At rollout step S0 the most recent input state is displaced by
      eps*n, with n (i) a random unit vector orthogonal to the local tangent of the
      unperturbed rollout ("transverse") or (ii) the unit tangent ("tangential").
      The difference D(t) between perturbed and unperturbed rollouts is resolved into
      components along/across the unperturbed trajectory.  Transverse: decay factor per
      shedding period from a log-linear fit of |D_perp| over cycles 1-8 after the kick.
      Tangential: the persistent phase offset after 10 cycles relative to the imposed
      one (1 = neutral, as on a limit cycle).
Inputs: a directory of per-seed models in seed_sweep_v3 or v4 format
        (seed_NNN_lstm.pt + seed_NNN_art.npz with latents/lmean/lstd/test_start/ntrain).
"""
import os, sys, glob, json, argparse, numpy as np
from scipy.signal import hilbert
ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--nfree", type=int, default=4000)
ap.add_argument("--nburn", type=int, default=200)
ap.add_argument("--ncal", type=int, default=150)
ap.add_argument("--s0", type=int, default=60)
ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.1])
ap.add_argument("--ndir", type=int, default=6)
ap.add_argument("--out", required=True)
args = ap.parse_args()
import torch, torch.nn as nn
torch.set_num_threads(2)
class LSTMP(nn.Module):
    def __init__(s, ld, hd=64, nl=2):
        super().__init__()
        s.lstm = nn.LSTM(ld, hd, nl, batch_first=True, dropout=0.1 if nl > 1 else 0.0)
        s.fc = nn.Sequential(nn.Linear(hd, hd // 2), nn.Tanh(), nn.Linear(hd // 2, ld))
    def forward(s, x): o, _ = s.lstm(x); return s.fc(o[:, -1, :])

def rollout(m, win, n):
    win = win.astype(np.float32).copy(); out = np.zeros((n, win.shape[1]), np.float32)
    with torch.no_grad():
        for s in range(n):
            p = m(torch.tensor(win[None])).numpy()[0]; out[s] = p; win = np.vstack([win[1:], p])
    return out

def phase(x): return np.unwrap(np.angle(hilbert(x - x.mean())))
def freq(z, dt, trim):
    t = np.arange(len(z)) * dt; k = slice(trim, len(z) - trim)
    return np.mean([np.polyfit(t[k], phase(z[:, d])[k], 1)[0] for d in range(z.shape[1])]) / (2 * np.pi)

res = []
for D in args.dirs:
    for lp in sorted(glob.glob(os.path.join(os.path.expanduser(D), "seed_*_lstm.pt"))):
        pre = lp[:-len("_lstm.pt")]; a = np.load(pre + "_art.npz")
        sd = torch.load(lp, map_location="cpu"); ld = sd["fc.2.weight"].shape[0]
        m = LSTMP(ld); m.load_state_dict(sd); m.eval()
        tau = int(a["tau"]) if "tau" in a else 20; dt = float(a["dt"]) if "dt" in a else 0.03
        lat = a["latents"]; ln = ((lat - a["lmean"]) / a["lstd"]).astype(np.float32)
        ts = int(a["test_start"]); ntr = int(a["ntrain"])
        pred = a["pred"].astype(float); true = ln[ts:ts + len(pred)].astype(float); n = len(pred)
        f_true_tr = None
        per = 1.0
        # --- R3.8 free-run clock
        f0 = freq(ln[:ntr], dt, 1); per = int(round(1 / (f0 * dt)))
        f_true_tr = freq(ln[:ntr], dt, per)
        free = rollout(m, ln[ntr - tau:ntr].astype(np.float32), args.nfree)[args.nburn:]
        f_free = freq(free.astype(float), dt, per)
        amp_free = float(np.mean(np.abs(hilbert(free - free.mean(0), axis=0))[per:-per]) /
                         np.mean(np.abs(hilbert(ln[:ntr] - ln[:ntr].mean(0), axis=0))[per:-per]))
        w_pred = 2 * np.pi * (f_free - f_true_tr)
        t = np.arange(n) * dt
        th_p = np.stack([phase(pred[:, d]) for d in range(ld)], 1); th_t = np.stack([phase(true[:, d]) for d in range(ld)], 1)
        phi = th_p - th_t; phi -= phi[0]
        w_roll = float(np.mean([np.polyfit(t, phi[:, d], 1)[0] for d in range(ld)]))
        w_cal = [np.polyfit(t[:args.ncal], phi[:args.ncal, d], 1)[0] for d in range(ld)]
        Ap = np.abs(hilbert(pred - pred.mean(0), axis=0)); mp = pred.mean(0)
        def corrected(w): return np.stack([Ap[:, d] * np.cos(th_p[:, d] - w[d] * t) + mp[d] for d in range(ld)], 1)
        ext = slice(args.ncal, n); eu = np.linalg.norm(pred - true, axis=1)[ext].mean()
        I_cal = 100 * (1 - np.linalg.norm(corrected(w_cal) - true, axis=1)[ext].mean() / eu)
        I_free = 100 * (1 - np.linalg.norm(corrected([w_pred] * ld) - true, axis=1)[ext].mean() / eu)
        # --- R4.1 perturbation (whole window)
        full = np.concatenate([ln[ts - tau:ts], pred.astype(np.float32)]); s0 = args.s0
        win0 = full[s0:s0 + tau].copy(); ref = pred[s0:].astype(np.float32); nn_ = len(ref)
        chk = rollout(m, win0, 50); assert np.abs(chk - ref[:50]).max() < 1e-4, "window/reference misaligned"
        mu = ln[:ntr].mean(0); dwin = np.gradient(full, axis=0)[s0:s0 + tau]
        tw = dwin / np.linalg.norm(dwin, axis=1, keepdims=True); rng = np.random.default_rng(0)
        Ar = np.abs(hilbert(ref - ref.mean(0), axis=0)); thr = np.stack([phase(ref[:, d]) for d in range(ld)], 1)
        cyc = np.arange(nn_) / per; out_p = {}
        for eps in args.eps:
            v = rng.standard_normal(ld); off = np.array([v - (v @ tw[i]) * tw[i] for i in range(tau)])
            off /= np.linalg.norm(off, axis=1, keepdims=True)
            kicks = {"amplitude": mu + (1 + eps) * (win0 - mu), "offset": win0 + eps * off,
                     "timeshift": win0 + (eps / np.linalg.norm(dwin[-1])) * dwin}
            rec = {}
            for nm, w in kicks.items():
                pp = rollout(m, w.astype(np.float32), nn_)
                Ap = np.abs(hilbert(pp - pp.mean(0), axis=0)); thp = np.stack([phase(pp[:, d]) for d in range(ld)], 1)
                amp_dev = np.abs(Ap / Ar - 1).mean(1) / eps; dph = (thp - thr).mean(1)
                cm = lambda x, c0, c1: float(np.mean(x[(cyc >= c0) & (cyc < c1)]))
                rec[nm] = dict(amp_dev_step1=float(amp_dev[1]), amp_dev_cyc0_5_1=cm(amp_dev, 0.5, 1), amp_dev_cyc1_2=cm(amp_dev, 1, 2),
                               amp_dev_cyc5_6=cm(amp_dev, 5, 6), phase_off_cyc1_2=cm(dph, 1, 2), phase_off_cyc8_9=cm(dph, 8, 9),
                               phase_persist=cm(dph, 8, 9) / cm(dph, 1, 2) if abs(cm(dph, 1, 2)) > 1e-9 else float("nan"))
            out_p[str(eps)] = rec
        r = dict(dir=D, seed=os.path.basename(pre), ld=ld, f_true_train=float(f_true_tr), f_free=float(f_free),
                 amp_free_ratio=amp_free, w_pred=float(w_pred), w_roll=w_roll, w_cal_mean=float(np.mean(w_cal)),
                 I_cal=float(I_cal), I_free=float(I_free), perturb=out_p)
        res.append(r)
        print(f"{r['seed']}: w_pred={w_pred:+.2e} w_roll={w_roll:+.2e} w_cal={np.mean(w_cal):+.2e} "
              f"I_cal={I_cal:+.1f}% I_free={I_free:+.1f}% amp_free={amp_free:.3f} "
              f"amp-kick dev 0.5-1/1-2 cyc={[(round(out_p[e]['amplitude']['amp_dev_cyc0_5_1'],4),round(out_p[e]['amplitude']['amp_dev_cyc1_2'],4)) for e in out_p]} "
              f"phase persist(timeshift)={[round(out_p[e]['timeshift']['phase_persist'],3) for e in out_p]}", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
w = np.array([[r["w_pred"], r["w_roll"]] for r in res])
if len(w) > 2:
    c = np.corrcoef(w.T)[0, 1]
    print(f"\nN={len(w)}  corr(w_pred, w_roll)={c:.3f}  median |w_pred-w_roll|/|w_roll|="
          f"{np.median(np.abs(w[:,0]-w[:,1])/np.abs(w[:,1])):.2f}  sign agreement={np.mean(np.sign(w[:,0])==np.sign(w[:,1])):.2f}")
    print(f"I_cal median {np.median([r['I_cal'] for r in res]):.1f}%   I_free median {np.median([r['I_free'] for r in res]):.1f}%")
