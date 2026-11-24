# 05 — Ensembles

Multi-seed studies. This is the expensive part: ninety networks in total.

| script | does |
|---|---|
| `seed_correction_sweep.py` | retrains across seeds, records latent-space efficacy per seed |
| `seed_correction_field.py` | the same in field space, recording G and rho |
| `seed_sweep_v2.py` | the broader parameter sweep |
| `seed_sweep_v3.py` | forty-network Re = 500 ensemble with stored networks (used for the free-running clock and perturbation tests) and the Re = 400 mesh ensembles |
| `seed_noise_sweep.py` | noise injection across seeds |
| `heldout_threshold.py` | out-of-sample validation of the R2 threshold: stratified split, leave-one-configuration-out, and a threshold sweep |
| `pooled_cae_sweep.py` | shared-autoencoder variant |
| `drift_predictor_sweep.py` | tests whether any network property predicts the drift magnitude — a null result, reported with an explicit power bound |

`heldout_threshold.py` is the one that matters most. The independent test is the
fixed threshold R²_cal > 0.3 applied once to the 90 ensemble networks, none of which was
used to choose it: it accepts 48, none of which is made worse by the correction. The
stratified splits and the threshold sweep describe the sensitivity to the threshold
value, not generalisation. The criterion is conservative but not sufficient: it also
rejects networks that the correction would improve, and it accepts the two baseline-mesh
Re = 1000 runs, for which the correction recovers only 14–23% (see the paper, Sec. III G).

**The per-seed records store summary scalars only** — efficacies, mean R2, drift
ratio — and no trajectory. Any analysis needing a refit cannot be run on these
ensembles without retraining. That is a real constraint, not an oversight to work
around.

Output directories ending `_trim10` were produced with a different phase-series
trim and are **not** the values reported in the paper; those ending `_dry` are
dry-run output. See [`../docs/DATA.md`](../docs/DATA.md), trap 3.
