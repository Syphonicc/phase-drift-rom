# 02 — Training the reduced-order model

| script | does |
|---|---|
| `seed_sweep_v4.py` | trains and rolls out ensembles of the same model with switches for the revision variants (`--cae-mode all/causal/fixed:<prefix>/pod`, `--lstm-seed`, `--tau`, `--loss multistep --k-steps K`, `--scalar`, `--snap`, `--split-n`) and writes per-network diagnostics; exact commands in [`../08_revision/README.md`](../08_revision/README.md) |
| `run_pipeline.py` | trains the convolutional autoencoder and the LSTM, runs the autoregressive rollout, writes `latents_*`, `rollout_*`, `pipeline_summary_*` and the model checkpoints |

```bash
python3 run_pipeline.py --re 400 --latent 4
```

The split is contiguous in time: roughly 52% training, 8% validation, 40% rollout.
The validation window is therefore a later portion of the same saturated limit
cycle, so the one-step validation error measures interpolation on a memorised
orbit rather than generalisation. The paper is explicit about this; it matters
because that error is very small (order 1e-6) and it would be easy to read as
evidence of a good long-horizon model, which is precisely the illusion the paper
is about.

`pipeline_summary_*.npy['improvement_pct']` is the **sinusoid** correction, not the
phase correction. See [`../docs/DATA.md`](../docs/DATA.md), trap 1.
