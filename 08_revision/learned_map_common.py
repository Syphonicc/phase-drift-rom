"""Definitions shared by the learned-map scripts (2026-11-24).
Copied verbatim from clock_freerun_perturb_v2.py (lines 46-65), which make_fig_learned_map_v5.py and
long_rollout_analysis_v2.py previously took by exec() of the part of that file before "res = []"."""
import numpy as np
from scipy.signal import hilbert
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
