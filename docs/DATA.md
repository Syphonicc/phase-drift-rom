# Data layout and known traps

This file describes the arrays and JSON records the scripts read and write, and
documents seven ways in which those files are easy to misread. **Read the traps
section before deriving any number from these outputs.** Each one caused a real
error during preparation of the paper; they are not hypothetical.

The audit tooling in [`../07_audit/`](../07_audit) exists to make these traps
detectable rather than silent. `audit_numbers.py` walks every output and records
each reported quantity with its provenance — case, latent dimension, calibration
length, timestep, trim setting, source file — and flags any quantity whose sources
disagree. If you are re-deriving results, run it first and work from the CSV.

---

## Traps

### 1. `corrected` and `improvement_pct` are the *sinusoid* baseline

`rollout_Re*.npy['corrected']` and `pipeline_summary_Re*.npy['improvement_pct']`
hold the **error-fitted sinusoid** correction, not the phase correction. The name
is historical and misleading.

The same applies to `dmd_baseline_Re*_L*.json`, whose key
`field.lstm_phase_corrected` is also the sinusoid despite its name. The genuine
phase result is in `dmd_baseline_Re*_L*_fixed.json` under
`latent_scores.lstm_phase_true`.

**For field space, neither JSON is correct.** The field-space phase result lives in
`fieldval_Re*_L*.npy` as `i_pha_f`, a percentage improvement; the bar-chart ratio
is `1 - i_pha_f/100`. This trap produced a published figure that plotted the
sinusoid baseline while labelling it the phase correction.

### 2. Unsuffixed `phasecorr` files are `ncal=200`

`phasecorr_Re400_L4.npy` (no suffix) was written with a 200-step calibration
window. The paper reports 150. Always use the explicit
`phasecorr_Re400_L4_ncal150.npy`. The difference is 84.17% against 82.79% — small
enough to look like a rounding discrepancy and large enough to be wrong.

Every calibration length in the sweep is stored:
`_ncal25`, `_ncal50`, `_ncal75`, `_ncal100`, `_ncal150`, `_ncal200`.

### 3. `seed_field_*` exists in two variants under the same filenames

`seed_field_Re300_L4/` and `seed_field_Re300_L4_trim10/` contain files with
identical names but different contents — the second was produced with 10 samples
trimmed from each end of the phase series. **Only `trim=0` is correct** for the
values reported in the paper. `heldout_threshold.py` rejects `trim != 0` records
explicitly; anything else reading these directories must do the same.

Directories ending `_dry` are dry-run output and contain no valid results.

### 4. Latents are stored unnormalised; rollouts are normalised

`latents_Re*_L*.npy` holds the raw encoder output. `rollout_Re*.npy['raw_pred']`
and `['true']` are in normalised coordinates. Comparing across the two gives
errors inflated by roughly a factor of eleven. Where a stored normalisation exists
(`latent_norm_Re*_L*.npy`, keys `mean` and `std`) it should be preferred over
recomputing one.

### 5. Hankel-DMD needs an explicit rank

An energy criterion fails here. The three leading modes of the Hankel matrices
carry more than 99.9% of the energy, so any conventional retained-energy threshold
keeps only those three and discards the low-energy harmonics that carry the
operator's predictive accuracy — costing about two orders of magnitude. `--ranks`
is therefore explicit, not derived.

Related: results at different delay counts are only comparable if swept over the
**same rank range**. The stationary cases in the paper use ranks 8–64; the
modulated cases use 8–128 at every delay count, because the comparison there is
about memory length and rank must be held constant across it.

### 6. `hankel_dmd.json` is overwritten on import

`hankel_dmd.py` runs its sweep at module level, so importing it — which
`phasecorr_on_dmd.py` does, to reuse `hankel()`, `fit_dmd()` and `rollout_dmd()` —
re-runs the sweep and rewrites `hankel_dmd.json` with default arguments. If you
have a sweep you want to keep, write it to a distinct `--out-prefix` first.

### 7. Some reported numbers are printed but not saved

The single-run columns of the horizon table are printed to console by
`phase_correction.py` and never written to disk, so no file can confirm them.
`audit_numbers.py` reports this rather than passing over it. The ensemble means
for the same quantity *are* saved and are audited.

---

## File reference

### Snapshots and metadata

| file | contents |
|---|---|
| `snapshots_Re{RE}.npy` | `(n_t, 3, 64, 128)` float32 — `[u_x, u_y, p]` on a uniform interpolation grid |
| `metadata_Re{RE}.npy` | dict: `x_grid`, `y_grid`, `nx`, `ny`, `t_start`, `re` |

Circular cylinder: 1334 snapshots at `dt = 0.03`. Square: 1370 at `dt = 0.0438`.
Modulated cases (`Re998` = ε 0.05, `Re999` = ε 0.15): 3000 snapshots at
`dt = 0.03`.

Note the grid is an interpolation of the solver field, not the solver mesh, and
its spacing is anisotropic. The circular domain spans 14.35D x 9.96D at 128 x 64,
giving 8.9 points per diameter streamwise and 6.4 laterally -- a pixel aspect
ratio of 0.72. Anything treating these arrays as images with square pixels (a
convolutional autoencoder among them) inherits that anisotropy. Claims about
reconstruction fidelity in this paper are claims about the interpolated field, not
the solver field.

### Latent space and rollouts

| file | contents |
|---|---|
| `latents_Re{RE}_L{NZ}.npy` | `(n_t, n_z)` — encoder output, **unnormalised** |
| `latent_norm_Re{RE}_L{NZ}.npy` | dict: `mean`, `std` |
| `rollout_Re{RE}[_L{NZ}].npy` | dict: `raw_pred`, `true`, `corrected` (sinusoid — see trap 1), `err_unc`, `err_cor`, `inst_phase` |
| `pipeline_summary_Re{RE}[_L{NZ}].npy` | dict: `re`, `latent_dim`, `cae_val`, `f_lstm`, `St_lstm`, `n_truth`, `calib_len`, `dt`, `D`, `U` |

Some cases use the unsuffixed form (`rollout_Re400.npy`) and some the suffixed
(`rollout_Re300_L4.npy`); several exist in both. Scripts search suffixed first.

### Correction and validation

| file | contents |
|---|---|
| `phasecorr_Re{RE}_L{NZ}_ncal{N}.npy` | `omega` `(n_z,)`, `r2` `(n_z,)` (calibration window), `I_sin`, `I_pha`, `e_unc`/`e_sin`/`e_pha` `(n_t,)`, `corrected` `(n_t, n_z)`, `ncal`, `dt` |
| `fieldval_Re{RE}_L{NZ}.npy` | `r_unc`, `r_sin`, `r_pha`, `r_floor` `(n_t,)` field RMSE series; `i_sin_f`, `i_pha_f` scalars; `closed` (the G statistic); `omega`, `r2`, `ncal` |
| `decompose_Re{RE}_L{NZ}.npy` | `share_amp`, `share_phase` scalars; `rows` — per coordinate `share_amp`, `share_phase`, `share_mean`, `share_cross`, `amp_ratio`, `phi_end`, `rms_e`, `rms_amp`, `rms_phase` |
| `mechanism_Re{RE}_L{NZ}.npy` | `results` — per coordinate `r2` (**rollout-wide** φ fit, not the calibration-window value), `slope_measured`, `slope_theory`, `f_pred`, `f_true`, `delta_f`, `r_env` |
| `tangential_transverse.json` | tangential/transverse shares, growth factors, linear-fit R² |

The `r2` in `phasecorr_*` is the **calibration-window** fit; the `r2` in
`mechanism_*` is the **rollout-wide** fit. At Re=400 these are 0.530 and 0.949.
The paper quotes both and they must not be interchanged. A third quantity, the R²
of the fit to tangential error *magnitude*, appears in the tangential table and is
different again.

### Baselines and ensembles

| file | contents |
|---|---|
| `dmd_baseline_Re*_L*_fixed.json` | `latent_scores` (use `lstm_phase_true`), `field` (sinusoid only — see trap 1) |
| `hankel_dmd.json` | per case: `lstm_mean_err`, `sweep` — each entry `delays`, `rank`, `mean_err`, `omega_mean`, `r2_phi`, `tang_share`, `diverged` |
| `hankel_modulated_longq.json` | same schema, extended sweep on the modulated cases |
| `slope_tstat.json` | per case: `per_coord` (`omega`, `se`, `t`, `r2`), `r2_mean`, `t_mean`, `t_min` |
| `seed_corr_Re*_L*/seed_*.npy` | per seed: `I_pha`, `I_sin`, `r2_mean`, `drift_ratio`, `horizon`, `cae_val`, `lstm_val` — **scalars only, no trajectory** |
| `seed_field_Re*_L*/` | as above for field space; see trap 3 |
| `spectral_robustness.json` | per estimator: peak location, `log10_ratio` (peak-to-floor decades), in-band fraction |

The per-seed records hold summary scalars and no stored trajectory. Any analysis
requiring a refit — a different regression, a different statistic — cannot be run
on the ensembles without retraining the networks.
