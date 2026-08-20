# 05 — Ensembles

Multi-seed studies. This is the expensive part: ninety networks in total.

| script | does |
|---|---|
| `seed_correction_sweep.py` | retrains across seeds, records latent-space efficacy per seed |
| `seed_correction_field.py` | the same in field space, recording G and rho |
| `seed_sweep_v2.py` | the broader parameter sweep |
| `seed_noise_sweep.py` | noise injection across seeds |
| `heldout_threshold.py` | out-of-sample validation of the R2 threshold: stratified split, leave-one-configuration-out, and a threshold sweep |
| `pooled_cae_sweep.py` | shared-autoencoder variant |
| `drift_predictor_sweep.py` | tests whether any network property predicts the drift magnitude — a null result, reported with an explicit power bound |

`heldout_threshold.py` is the one that matters most: it establishes that the
threshold transfers to configurations it was not calibrated on, including across
geometry, with no false accepts.

**The per-seed records store summary scalars only** — efficacies, mean R2, drift
ratio — and no trajectory. Any analysis needing a refit cannot be run on these
ensembles without retraining. That is a real constraint, not an oversight to work
around.

Output directories ending `_trim10` were produced with a different phase-series
trim and are **not** the values reported in the paper; those ending `_dry` are
dry-run output. See [`../docs/DATA.md`](../docs/DATA.md), trap 3.
