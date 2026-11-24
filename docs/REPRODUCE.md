# Reproducing the paper

Which script produces which table and figure. Read
[`DATA.md`](DATA.md) first — several output files are easy to misread, and the
traps documented there caused real errors during preparation.

**Data directory.** Not every script takes `--data-dir`. The scripts of steps 01–06 were
written for a fixed working directory, `~/cylinder_lstm_data`, and the following read
*and write* there regardless of any option (some even when called with `--help`):
`01_generate_data/{extract_snapshots_v2,extract_square,interpolation_study_v3}.py`,
`02_train_rom/run_pipeline.py`,
`03_phase_correction/{phase_correction,decompose_error,field_validation,compute_rho,verify_mechanism,timewarp_correction,slope_tstat,collect_phase_results}.py`,
`04_baselines/phasecorr_on_dmd.py`, `05_ensembles/drift_predictor_sweep.py`, and
`06_figures/{make_fig01,make_paper_figures,sync_figures}.py`. Change the `DATA`/`HOME`
constant at the top of these scripts, or run them in a copy, before using them on data you
want to keep. The other scripts of steps 01–06 take `--data-dir` (default
`~/cylinder_lstm_data`); the scripts of `08_revision/` use the environment variables of
[`08_revision/paths.py`](../08_revision/paths.py), and the revised figure scripts
(`*_v2.py`, `field_timing_check_v3.py`) write only to the directory given by `--out`.

---

## Order of operations

```
01_generate_data   OpenFOAM cases -> snapshots_Re*.npy
02_train_rom       snapshots -> autoencoder, LSTM, rollout
03_phase_correction rollout -> correction, decomposition, field validation
04_baselines       rollout -> DMD, Hankel-DMD, tiling, harmonic, noise injection
05_ensembles       repeat 02-03 across seeds -> error bars, threshold validation
06_figures         all of the above -> figures
07_audit           all of the above -> provenance CSV, manuscript cross-check
```

Steps 4 and 5 are independent of each other. Step 5 is by far the most expensive:
it trains 90 networks. Step 8 (revision analyses) uses a further 225 networks
(165 from `02_train_rom/seed_sweep_v4.py`, 20 for the Re = 400 mesh comparison and 40
at Re = 500 from `05_ensembles/seed_sweep_v3.py`) and the additional simulations; see
`08_revision/README.md`.

---

## Figures

| figure (revised paper) | script | notes |
|---|---|---|
| 1 — flow configurations | `06_figures/make_fig01_v2.py --out DIR` | reads `snapshots_*` and `metadata_*` directly; `make_fig01.py` (and `make_paper_figures.py`) wrote the earlier version of `fig01_configurations.pdf` |
| 2 — force coefficients vs Re | `08_revision/make_fig_forces_v2.py` | reads `postProcessing/forceCoeffs*` of the solved cases via `08_revision/forces.py` |
| 3 — pipeline schematic | `06_figures/fig02_pipeline.tex` | TikZ, no script |
| 4, 5 — error spectra | `06_figures/make_paper_figures.py` | unwindowed periodogram; y-limits are clipped deliberately, see below |
| 6 — amplitude/phase decomposition | `06_figures/make_paper_figures_v2.py --out DIR` | panel (b) is the **rollout-wide** φ fit |
| 7 — tangential/transverse | `03_phase_correction/tangential_transverse_v2.py --smooth 1` | tangent stencil k = 1 |
| 8 — signed error and phase | `08_revision/make_fig_signed_phase_v2.py` | |
| 9 — 395-cycle rollout | `08_revision/long_rollout_analysis_v3.py` | ensemble `S8_longroll`, reference `snapshots_Re400ext.npy` |
| 10 — latent correction | `06_figures/make_paper_figures_v2.py --out DIR` | reads `phasecorr_*_ncal150.npy` |
| 11 — field-space timing error | `08_revision/field_timing_check_v3.py --out DIR` | needs torch |
| 12 — field correction | `06_figures/make_paper_figures_v2.py --out DIR` | |
| 13 — linear-propagator baselines | `06_figures/plot_dmd_baseline_v2.py --out DIR` | field panel reads `fieldval_*.npy`, **not** the DMD JSON |
| 14 — learned map (free-running clock, perturbations) | `08_revision/make_fig_learned_map_v6.py` | reads `$PDROM_ENSEMBLE_DIR/clock_perturb_v2_Re500_40seeds.json` written by `08_revision/clock_freerun_perturb_v2.py` (see `08_revision/README.md`) |

The spectrum figures set an explicit `ylim` of `(1e-6, 2e0)`. The `+1e-20` guard
that keeps the DC null finite on a log axis would otherwise drag the axis down
some fifteen empty decades and hide the peak-to-floor separation the figure
exists to show.

`06_figures/sync_figures.py` regenerates all figures and copies them into a LaTeX
bundle, verifying that every `\includegraphics` resolves. It locates the bundle by
searching for `main.tex` rather than by a fixed path, because repeated downloads
leave numbered duplicate directories and it is easy to regenerate into one while
compiling another.

**Note on figure numbering.** Filenames and printed figure numbers do not
correspond one-to-one: LaTeX numbers by declaration order, and the tangential
figure is declared before the correction figures. `\ref` handles this correctly;
manual references to "Figure N" may not.

---

## Tables

| table (revised paper) | script | source file |
|---|---|---|
| I — limit-cycle saturation | `08_revision/saturation_check_v3.py` | `saturation_check_v3.json` |
| II — grid and timestep convergence | solver output; Re = 800/1000 rows from `08_revision/make_fig_forces.py` | `forces_vs_re.json` |
| III — summary of cases | — | |
| IV — amplitude/phase decomposition | `03_phase_correction/decompose_error.py` | `decompose_Re*_L*.npy` |
| V — tangential/transverse | `03_phase_correction/tangential_transverse.py` | `tangential_transverse.json` |
| VI — correction efficacy | `03_phase_correction/phase_correction.py` | `phasecorr_*_ncal150.npy` |
| VII — horizon | `03_phase_correction/phase_correction.py` | **printed only, not saved** — see `DATA.md` trap 7 |
| VIII — field-space G and ρ | `03_phase_correction/field_validation.py`, `compute_rho.py` | `fieldval_*.npy` |
| IX — efficacy by R²_cal band | `05_ensembles/seed_correction_sweep.py` | `seed_corr_*/` |
| X — calibration-length sweep | `03_phase_correction/phase_correction.py --calib ...` | one file per length |
| XI — threshold response | `05_ensembles/heldout_threshold.py` | `seed_corr_*/`, `seed_field_*/` |
| XII — periodic continuation | `04_baselines/periodic_baseline.py`; cubic-spline columns from `08_revision/periodic_tiling_interp_v2.py` | `periodic_tiling_interp_v2.json` |
| XIII — DMD and harmonic baselines | `04_baselines/dmd_baseline.py` | `dmd_baseline_*_fixed.json` |
| XIV — Hankel-DMD | `04_baselines/hankel_dmd.py` | `hankel_dmd.json`, `hankel_modulated_longq.json` |
| XV — modulated case | `03_phase_correction/phase_correction.py --re 998`; tiling from `08_revision/periodic_tiling_interp_v3.py 998:4 999:4` | |
| XVI — drift predictors | `05_ensembles/drift_predictor_sweep.py` | |
| XVII — sensitivity | `08_revision/sensitivity_table_v2.py` | `sensitivity_table_v2.json`; ensembles listed in `08_revision/README.md` |
| S1 (supplementary) — estimator bias | `03_phase_correction/spectral_robustness.py` | synthetic signals |

The held-out evidence in the revised paper is the fixed threshold applied once to all
90 ensemble networks (none of which was used to choose it); the stratified-split and
leave-one-configuration-out outputs of `heldout_threshold.py` are not used as
evidence of generalisation.

The Hankel table reports the best result over a rank sweep at each delay count.
The stationary rows sweep ranks 8–64; the modulated rows sweep 8–128 at **every**
delay count, so that delay length is the only quantity varying between them.
Mixing rank ranges across rows of the same comparison makes the delay effect
unreadable — this is trap 5 in `DATA.md` and it produced a wrong table during
preparation.

---

## The audit

```bash
python3 07_audit/audit_numbers.py --data-dir .
python3 07_audit/check_manuscript.py --csv audit_numbers.csv --tex <bundle-dir>
python3 07_audit/audit_numbers_v3.py --data-dir . --quiet   # revised manuscript: adds the revision outputs and a claims check
python3 07_audit/audit_numbers_v4.py --data-dir . --quiet   # v3 plus the claims added on 2026-11-24 (77 in total)
```

`audit_numbers.py` writes a CSV row for every reported quantity, tagged with case,
latent dimension, calibration length, timestep, trim setting and source file, and
flags any quantity whose sources disagree by more than a tolerance. Some flagged
disagreements are expected — the calibration-length sweep stores the same quantity
at six different `ncal` values, and these group together — so read the `ncal`
column before concluding anything.

`check_manuscript.py` then cross-checks the LaTeX source against that CSV in both
directions: whether every audited quantity is accounted for, and whether every
printed number has a source. Expect a long list of unmatched entries; range
endpoints, derived percentages and values from sources the audit does not read all
appear there legitimately. **A number that should be traceable and is not is the
finding.**

This tooling is included because it repeatedly caught errors that manual checking
missed, including a figure plotting one quantity while labelled another, a table
built from two incompatible parameter sweeps, and a drift rate that survived a
manual search-and-replace because the search terms did not include it.
