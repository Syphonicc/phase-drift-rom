"""Diagnostics copied verbatim from seed_sweep_v4.py (keep in sync)."""
import numpy as np
from scipy.signal import hilbert
DT=0.03; NCAL=150
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

