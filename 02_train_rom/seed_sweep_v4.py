#!/usr/bin/env python3
"""
seed_sweep_v4.py -- PoF revision (POF26-AR-14315), group-B experiments.

Training/rollout identical to seed_sweep_v3.py by default (same CAE, LSTM 2x64,
tau=20, Adam 1e-3, 300/500 epochs, 52/8/40 split, one-step teacher forcing).
Switches (each isolates one factor):
  --cae-mode all        CAE fitted on a random 85/15 split of ALL snapshots (as published)
             causal     CAE fitted only on the first 60% (train+val) -- leakage check
             fixed:P    load CAE + normalisations from artifact prefix P (seed_..._art.npz/_cae.pt);
                        only the LSTM is trained -> isolates LSTM initialisation (R4.6, R3.16)
             pod        POD of the first 60% (rank n_z) instead of a CAE (R3.16)
  --scalar F            1-D series (e.g. C_L) from .npy instead of fields; no encoder (R2.4)
  --lstm-seed S         LSTM init + batch order seeded with S, independent of --seeds (R4.6)
  --tau T               input window length (R4.4)
  --loss multistep --k-steps K   unrolled K-step loss under the model's own feedback (R4.5)
  --split-n N0          split fractions computed on the first N0 samples; rollout runs to
                        the end of the data (long-horizon test, R4.2)
  --snap F              snapshot file (default <data-dir>/snapshots_Re<RE>.npy)
Every seed writes seed_NNN.npy (scalar metrics, the paper's diagnostics) and
seed_NNN_art.npz (pred, true, latent normalisation) [+ _lstm.pt, _cae.pt].
Resume-safe: finished seeds are skipped.
"""
import os, sys, json, argparse, time, numpy as np
from scipy.signal import hilbert

ap = argparse.ArgumentParser()
ap.add_argument("--re", type=int, default=400)
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--latent", type=int, default=4)
ap.add_argument("--cae-epochs", type=int, default=300)
ap.add_argument("--lstm-epochs", type=int, default=500)
ap.add_argument("--tau", type=int, default=20)
ap.add_argument("--dt", type=float, default=0.03)
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--snap", default=None)
ap.add_argument("--scalar", default=None)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--cae-mode", default="all")
ap.add_argument("--lstm-seed", type=int, default=None)
ap.add_argument("--loss", default="onestep", choices=["onestep", "multistep"])
ap.add_argument("--k-steps", type=int, default=10)
ap.add_argument("--split-n", type=int, default=None)
ap.add_argument("--ncal", type=int, default=150)
ap.add_argument("--no-save-models", action="store_true")
args = ap.parse_args()

import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = os.path.expanduser(args.data_dir); OUT = os.path.expanduser(args.out_dir)
os.makedirs(OUT, exist_ok=True)
json.dump(vars(args), open(os.path.join(OUT, "config.json"), "w"), indent=1)
LD, TAU, DT, NCAL = args.latent, args.tau, args.dt, args.ncal
FLAT = 64 * 8 * 16

class Encoder(nn.Module):
    def __init__(s, ld):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU())
        s.f = nn.Linear(FLAT, ld)
    def forward(s, x): return s.f(s.c(x).flatten(1))
class Decoder(nn.Module):
    def __init__(s, ld):
        super().__init__(); s.f = nn.Linear(ld, FLAT)
        s.d = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1))
    def forward(s, z): return s.d(s.f(z).view(-1, 64, 8, 16))
class CAE(nn.Module):
    def __init__(s, ld):
        super().__init__(); s.encoder = Encoder(ld); s.decoder = Decoder(ld)
    def forward(s, x): return s.decoder(s.encoder(x))
    def encode(s, x): return s.encoder(x)
class LSTMP(nn.Module):
    def __init__(s, ld, hd, nl):
        super().__init__()
        s.lstm = nn.LSTM(ld, hd, nl, batch_first=True, dropout=0.1 if nl > 1 else 0.0)
        s.fc = nn.Sequential(nn.Linear(hd, hd // 2), nn.Tanh(), nn.Linear(hd // 2, ld))
    def forward(s, x):
        o, _ = s.lstm(x); return s.fc(o[:, -1, :])

# ------------------------------------------------------------------ diagnostics (paper definitions)
def analytic(x):
    m = x.mean(); z = hilbert(x - m); return m, np.abs(z), np.unwrap(np.angle(z))

def diagnostics(pred, true, dt=DT, ncal=NCAL):
    n, ld = true.shape; t = np.arange(n) * dt; ext = slice(ncal, n)
    out = {}
    # amplitude/phase variance shares (decompose_error.py)
    sa, sp, om, r2, amp, A_p, th_p, m_p = [], [], [], [], [], [], [], []
    for d in range(ld):
        mt, At, tht = analytic(true[:, d]); mp, Ap, thp = analytic(pred[:, d])
        e = pred[:, d] - true[:, d]; v = max(np.var(e), 1e-30)
        sa.append(np.var((Ap - At) * np.cos(tht)) / v); sp.append(np.var(At * (np.cos(thp) - np.cos(tht))) / v)
        phi = thp - tht; phi -= phi[0]
        c = np.polyfit(t[:ncal], phi[:ncal], 1); res = phi[:ncal] - np.polyval(c, t[:ncal])
        om.append(c[0]); r2.append(1 - res.var() / max(phi[:ncal].var(), 1e-30))
        amp.append(np.median(Ap) / max(np.median(At), 1e-12)); A_p.append(Ap); th_p.append(thp); m_p.append(mp)
    out.update(share_phase=float(np.mean(sp)), share_amp=float(np.mean(sa)), omega_cal=[float(x) for x in om],
               omega_mean=float(np.mean(om)), r2_cal=float(np.mean(r2)), amp_ratio=float(np.mean(amp)))
    # phase correction (phase_correction.py)
    corr = np.stack([A_p[d] * np.cos(th_p[d] - om[d] * t) + m_p[d] for d in range(ld)], 1)
    eu = np.linalg.norm(pred - true, axis=1)[ext].mean(); ec = np.linalg.norm(corr - true, axis=1)[ext].mean()
    out.update(err_ext=float(eu), err_ext_corrected=float(ec), I_phase=float(100 * (1 - ec / eu)))
    # tangential/transverse (tangential_transverse.py, k=1, skip=5)
    k = 1; tang = np.zeros_like(true); tang[k:n - k] = (true[2 * k:] - true[:n - 2 * k]) / (2 * k * dt)
    tang[:k] = (true[1:k + 1] - true[:k]) / dt; tang[n - k:] = (true[n - k:] - true[n - k - 1:n - 1]) / dt
    th = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    e = pred - true; ep = np.einsum("ij,ij->i", e, th); eq = np.linalg.norm(e - ep[:, None] * th, axis=1)
    sl = slice(5, n - 5); p_, q_ = np.abs(ep[sl]), eq[sl]; m10 = max(1, len(p_) // 10)
    out.update(share_tan=float(np.sum(p_ ** 2) / np.sum(p_ ** 2 + q_ ** 2)),
               growth_tan=float(p_[-m10:].mean() / max(p_[:m10].mean(), 1e-12)),
               growth_perp=float(q_[-m10:].mean() / max(q_[:m10].mean(), 1e-12)))
    # clock: frequency of prediction and truth from a linear fit of the unwrapped phase
    fp = np.mean([np.polyfit(t, th_p[d], 1)[0] for d in range(ld)]) / (2 * np.pi)
    ft = np.mean([np.polyfit(t, analytic(true[:, d])[2], 1)[0] for d in range(ld)]) / (2 * np.pi)
    out.update(f_pred=float(fp), f_true=float(ft), clock_bias=float(fp / ft - 1))
    return out

# ------------------------------------------------------------------ data
if args.scalar:
    series = np.load(os.path.expanduser(args.scalar)).astype(np.float32).reshape(-1, 1); raw = None
    N = len(series); LD = 1
else:
    snap = os.path.expanduser(args.snap) if args.snap else os.path.join(DATA, f"snapshots_Re{args.re}.npy")
    raw = np.load(snap); N = len(raw)
N0 = args.split_n or N
N_TRAIN = int(N0 * 0.52); N_VAL = int(N0 * 0.08); TEST_START = N_TRAIN + N_VAL

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed); crit = nn.MSELoss(); info = {}
    mode = args.cae_mode
    if args.scalar:
        lat = series.copy(); lm = lat[:N_TRAIN].mean(0, keepdims=True); ls = lat[:N_TRAIN].std(0, keepdims=True)
        cae_val = np.nan
    elif mode.startswith("fixed:"):
        P = os.path.expanduser(mode.split(":", 1)[1]); a = np.load(P + "_art.npz")
        cm, cs = a["ch_mean"].reshape(1, 3, 1, 1), a["ch_std"].reshape(1, 3, 1, 1)
        cae = CAE(LD).to(dev); cae.load_state_dict(torch.load(P + "_cae.pt", map_location=dev)); cae.eval()
        X = torch.tensor(((raw - cm) / cs).astype(np.float32))
        with torch.no_grad():
            lat = np.concatenate([cae.encode(X[i:i + 64].to(dev)).cpu().numpy() for i in range(0, len(X), 64)])
        lm, ls = a["lmean"], a["lstd"]; cae_val = np.nan; info["fixed_cae"] = P
    elif mode == "pod":
        cm = raw[:TEST_START].mean(axis=(0, 2, 3), keepdims=True); cs = raw[:TEST_START].std(axis=(0, 2, 3), keepdims=True)
        Xf = ((raw - cm) / cs).reshape(N, -1).astype(np.float64); mu = Xf[:TEST_START].mean(0)
        U, S, Vt = np.linalg.svd(Xf[:TEST_START] - mu, full_matrices=False)
        lat = ((Xf - mu) @ Vt[:LD].T).astype(np.float32)
        info["pod_energy"] = float((S[:LD] ** 2).sum() / (S ** 2).sum())
        rec = (lat[:TEST_START] @ Vt[:LD]) + mu; cae_val = float(np.mean((rec - Xf[:TEST_START]) ** 2))
        lm = lat[:N_TRAIN].mean(0, keepdims=True); ls = lat[:N_TRAIN].std(0, keepdims=True)
    else:
        cm = raw.mean(axis=(0, 2, 3), keepdims=True); cs = raw.std(axis=(0, 2, 3), keepdims=True)
        cs = np.where(cs < 1e-8, 1.0, cs); X = torch.tensor(((raw - cm) / cs).astype(np.float32))
        Xfit = X if mode == "all" else X[:TEST_START]
        cae = CAE(LD).to(dev); opt = torch.optim.Adam(cae.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5, min_lr=1e-5)
        nval = int(len(Xfit) * 0.15); tr, va = random_split(TensorDataset(Xfit), [len(Xfit) - nval, nval],
                                                            generator=torch.Generator().manual_seed(seed))
        trl = DataLoader(tr, batch_size=32, shuffle=True); vll = DataLoader(va, batch_size=32)
        best, bs = 1e9, None
        for ep in range(args.cae_epochs):
            cae.train()
            for (b,) in trl:
                b = b.to(dev); opt.zero_grad(); l = crit(cae(b), b); l.backward(); opt.step()
            cae.eval(); vl = 0
            with torch.no_grad():
                for (b,) in vll: b = b.to(dev); vl += crit(cae(b), b).item() * len(b)
            vl /= nval; sch.step(vl)
            if vl < best: best, bs = vl, {k: v.cpu().clone() for k, v in cae.state_dict().items()}
        cae.load_state_dict(bs); cae.eval(); cae_val = best
        with torch.no_grad():
            lat = np.concatenate([cae.encode(X[i:i + 64].to(dev)).cpu().numpy() for i in range(0, len(X), 64)])
        lm = lat[:N_TRAIN].mean(0, keepdims=True); ls = lat[:N_TRAIN].std(0, keepdims=True)
        if not args.no_save_models:
            torch.save(cae.state_dict(), os.path.join(OUT, f"seed_{seed:03d}_cae.pt"))
        info["ch_mean"], info["ch_std"] = cm.squeeze(), cs.squeeze()
    ls = np.where(ls < 1e-8, 1.0, ls); ln = ((lat - lm) / ls).astype(np.float32)

    # ---------------- LSTM
    if args.lstm_seed is not None: torch.manual_seed(args.lstm_seed); np.random.seed(args.lstm_seed)
    elif not mode == "all" or args.scalar: torch.manual_seed(seed + 1000)
    K = args.k_steps if args.loss == "multistep" else 1
    def windows(s, e):
        Xs, Ys = [], []
        for i in range(s, e - TAU - K + 1): Xs.append(ln[i:i + TAU]); Ys.append(ln[i + TAU:i + TAU + K])
        return torch.tensor(np.array(Xs)), torch.tensor(np.array(Ys))
    Xtr, Ytr = windows(0, N_TRAIN); Xva, Yva = windows(N_TRAIN, N_TRAIN + N_VAL)
    trl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True)
    lstm = LSTMP(LD, 64, 2).to(dev); opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.5, min_lr=1e-6)
    def unroll(xb, k):
        w = xb; outs = []
        for _ in range(k):
            p = lstm(w); outs.append(p); w = torch.cat([w[:, 1:], p[:, None]], 1)
        return torch.stack(outs, 1)
    best, bs = 1e9, None
    for ep in range(args.lstm_epochs):
        lstm.train()
        for xb, yb in trl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad()
            l = crit(unroll(xb, K), yb); l.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0); opt.step()
        lstm.eval()
        with torch.no_grad():   # validation always on the ONE-STEP error, comparable across losses
            vl = crit(lstm(Xva.to(dev)), Yva[:, 0].to(dev)).item()
        sch.step(vl)
        if vl < best: best, bs = vl, {k: v.cpu().clone() for k, v in lstm.state_dict().items()}
    lstm.load_state_dict(bs); lstm.eval(); lstm_val = best

    # ---------------- rollout to the end of the data
    ROLL = N - TEST_START; true = ln[TEST_START:]; win = ln[TEST_START - TAU:TEST_START].copy()
    pred = np.zeros((ROLL, LD), np.float32)
    with torch.no_grad():
        for s in range(ROLL):
            p = lstm(torch.tensor(win[None]).to(dev)).cpu().numpy()[0]; pred[s] = p; win = np.vstack([win[1:], p])
    d = diagnostics(pred.astype(float), true.astype(float))
    d.update(seed=int(seed), cae_val=float(cae_val), lstm_val=float(lstm_val), n_roll=int(ROLL),
             n_cycles=float(ROLL * DT * d["f_true"]), **{k: v for k, v in info.items() if np.size(v) == 1 or isinstance(v, str)})
    if not args.no_save_models: torch.save(lstm.state_dict(), os.path.join(OUT, f"seed_{seed:03d}_lstm.pt"))
    np.savez_compressed(os.path.join(OUT, f"seed_{seed:03d}_art.npz"), pred=pred, true=true, lmean=lm, lstd=ls,
                        latents=lat.astype(np.float32), test_start=TEST_START, ntrain=N_TRAIN, tau=TAU, dt=DT,
                        **({"ch_mean": info["ch_mean"], "ch_std": info["ch_std"]} if "ch_mean" in info else {}))
    return d

print(f"v4 Re={args.re} LD={LD} tau={TAU} cae={args.cae_mode} loss={args.loss}(K={args.k_steps}) "
      f"N={N} split_n={N0} out={OUT}", flush=True)
for s in args.seeds:
    fn = os.path.join(OUT, f"seed_{s:03d}.npy")
    if os.path.exists(fn): print(f"seed {s}: done, skipping", flush=True); continue
    t0 = time.time(); d = run(s); np.save(fn, d, allow_pickle=True)
    print(f"seed {s}: {time.time()-t0:.0f}s  phase={d['share_phase']:.3f} tan={d['share_tan']:.3f} "
          f"omega={d['omega_mean']:+.2e} r2cal={d['r2_cal']:.2f} I={d['I_phase']:+.1f}% "
          f"bias={d['clock_bias']:+.2e} err={d['err_ext']:.4f}", flush=True)
