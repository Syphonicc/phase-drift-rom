# Reproducing the paper

Which script produces which table and figure. Read
[`DATA.md`](DATA.md) first — several output files are easy to misread, and the
traps documented there caused real errors during preparation.

Every script takes `--data-dir`; the examples below assume outputs are written to
a single working directory.

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
it trains 90 networks.

---

## Figures

| figure | script | notes |
|---|---|---|
| 1 — flow configurations | `06_figures/make_fig01.py` | reads `snapshots_*` and `metadata_*` directly |
| 2 — pipeline schematic | `fig02_pipeline.tex` | TikZ, no script |
| 3, 4 — error spectra | `06_figures/make_paper_figures.py` | unwindowed periodogram; y-limits are clipped deliberately, see below |
| 5 — amplitude/phase decomposition | `06_figures/make_paper_figures.py` | panel (b) is the **rollout-wide** φ fit |
| 6 — tangential/transverse | `03_phase_correction/tangential_transverse.py --out-prefix fig06_tangential_transverse` | writes its own figure |
| 7 — latent correction | `06_figures/make_paper_figures.py` | reads `phasecorr_*_ncal150.npy` |
| 8 — field correction | `06_figures/make_paper_figures.py` | |
| 9 — linear-propagator baselines | `06_figures/plot_dmd_baseline.py` | field panel reads `fieldval_*.npy`, **not** the DMD JSON |

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

| table | script | source file |
|---|---|---|
| Grid convergence | — | reported from solver output |
| Amplitude/phase decomposition | `03_phase_correction/decompose_error.py` | `decompose_Re*_L*.npy` |
| Estimator bias | `03_phase_correction/spectral_robustness.py` | synthetic signals |
| Tangential/transverse | `03_phase_correction/tangential_transverse.py` | `tangential_transverse.json` |
| Correction efficacy | `03_phase_correction/phase_correction.py` | `phasecorr_*_ncal150.npy` |
| Horizon | `03_phase_correction/phase_correction.py` | **printed only, not saved** — see `DATA.md` trap 7 |
| Field-space G and ρ | `03_phase_correction/field_validation.py`, `compute_rho.py` | `fieldval_*.npy` |
| Calibration-length sweep | `03_phase_correction/phase_correction.py --ncal ...` | one file per length |
| Held-out threshold | `05_ensembles/heldout_threshold.py` | `seed_corr_*/`, `seed_field_*/` |
| DMD and harmonic baselines | `04_baselines/dmd_baseline.py` | `dmd_baseline_*_fixed.json` |
| Hankel-DMD | `04_baselines/hankel_dmd.py` | `hankel_dmd.json`, `hankel_modulated_longq.json` |
| Periodic tiling | `04_baselines/periodic_baseline.py` | |
| Modulated case | `03_phase_correction/phase_correction.py --re 998` | |
| Seed study | `05_ensembles/seed_correction_sweep.py` | `seed_corr_*/` |
| Drift predictors | `05_ensembles/drift_predictor_sweep.py` | |

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
