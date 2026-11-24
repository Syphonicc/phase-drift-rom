# Revision analyses

Scripts added for the revised manuscript: the additional ensembles, the sensitivity
study, the free-running clock and perturbation tests of the learned map, the
395-cycle rollout, the field-space timing analysis, the force coefficients, the
limit-cycle saturation and near-wall checks, and the corrected periodic-continuation
baseline. Every script here was rerun from this folder and reproduces the results
used in the paper exactly.

## Data locations

The scripts read and write the directories defined in [`paths.py`](paths.py), set
by environment variables:

| variable | contents | default |
|---|---|---|
| `PDROM_DATA_DIR` | outputs of steps 01–05 (`snapshots_Re*.npy`, `rollout_*`, `cae_*`, `seed_sweep_v3_Re500/`) | `~/cylinder_lstm_data` |
| `PDROM_REVISION_DIR` | revision snapshots and results | `$PDROM_DATA_DIR/revision` |
| `PDROM_ENSEMBLE_DIR` | revision ensembles (below) | `$PDROM_REVISION_DIR/B` |
| `PDROM_CASES_DIR` | solved cases of [`01_generate_data/circular_cylinder/revision`](../01_generate_data/circular_cylinder/revision) | `~/OpenFOAM/run/Cylinder2D_study/revision` |
| `PDROM_ARCHIVE_DIR` | solved production cases (for the near-wall check) | `~/cfd_archive` |
| `PDROM_FINE_DIR` | solved refined-mesh Re = 800 and 1000 cases | `$PDROM_ARCHIVE_DIR/CYL2D_fine` |
| `PDROM_OUT_DIR` | where results are written | next to the inputs |

Set `PDROM_OUT_DIR` to a scratch directory to regenerate results without
overwriting existing ones.

## 1. Snapshots of the additional simulations

[`01_generate_data/extract_snapshots_v5.py`](../01_generate_data/extract_snapshots_v5.py)
samples a case onto the production 128 × 64 grid (taken from
`metadata_Re400.npy`). With `--decomposed` it reads the processor directories
directly, using `cellProcAddressing`; this reproduces `reconstructPar` followed by
serial extraction exactly.

```bash
python 01_generate_data/extract_snapshots_v5.py --decomposed --case $PDROM_CASES_DIR/Re100           --out-tag Re100c     --tstart 210.01
python 01_generate_data/extract_snapshots_v5.py --decomposed --case $PDROM_CASES_DIR/Re400_continued --out-tag Re400cont --tstart 90.01
# likewise: Re400_outlet25D -> Re400L25D, Re400_walls10D -> Re400W10D,
#           Re800_finemesh -> Re800fine, Re1000_finemesh -> Re1000fine   (default window t = 50.01-90)
```

`snapshots_Re400ext.npy` is the production `snapshots_Re400.npy` (t = 50.01–90)
followed by `snapshots_Re400cont.npy` (t = 90.03–290), 8000 snapshots, continuous at
the join. The lift-coefficient series for `S9_lift` is
`python 08_revision/forces.py $PDROM_CASES_DIR/Re400_continued 90.03 290 --resample 0.03 cl_Re400_t90_290.npy`
(the lift coefficient of the continued Re = 400 run sampled every 0.03), of which the
first 1334 samples (`cl_Re400_1334.npy`) are used.

## 2. Ensembles

All revision ensembles were trained with
[`02_train_rom/seed_sweep_v4.py`](../02_train_rom/seed_sweep_v4.py) (same
architecture and hyper-parameters as the paper, with switches for the variants)
and are at Re = 400 with n_z = 4 unless stated:

| ensemble | n | what changes |
|---|---|---|
| `S7b_base_v4` | 10 | baseline (production data) for S6, S7, S9, S11 |
| `S6_causalCAE` | 10 | autoencoder fitted on the first 60 % of snapshots only |
| `S7_long25D` | 10 | outlet at 25D |
| `S11_wide10D` | 10 | lateral boundaries at ±10D (5 % blockage; inlet and outlet also moved, see `01_generate_data/circular_cylinder/revision/README.md`) |
| `S9_lift` | 10 | scalar LSTM on the lift coefficient alone |
| `S1_fixedCAE` | 10 | autoencoder fixed, LSTM initialisation varied; reference for τ, loss and POD blocks |
| `S2_fixedLSTMseed` | 10 | LSTM initialisation fixed, autoencoder varied |
| `S3_pod` | 10 | POD (rank 4) instead of the autoencoder |
| `S4_tau5` | 5 | input window τ = 5 |
| `S4_tau10` | 5 | input window τ = 10 |
| `S4_tau40` | 5 | input window τ = 40 |
| `S4_tau80` | 5 | input window τ = 80 |
| `S5_ms5` | 5 | unrolled loss over K = 5 own predictions |
| `S5_ms20` | 5 | unrolled loss over K = 20 own predictions |
| `S8_longroll` | 5 | 395-cycle rollout against the continued simulation (Fig. 9) |
| `S10_Re100` | 10 | circular cylinder at Re = 100 (Sec. III M) |
| `S12a_Re800_coarse_L8` | 10 | Re = 800, production mesh, n_z = 8 |
| `S12b_Re800_fine_L8` | 10 | Re = 800, refined mesh, n_z = 8 |
| `S13a_Re1000_coarse` | 10 | Re = 1000, production mesh |
| `S13b_Re1000_fine` | 10 | Re = 1000, refined mesh |
| `S0_v4_check` | 1 | check that v4 reproduces v3 settings (one network; not used in the paper) |

The two mesh ensembles of the sensitivity table at Re = 400 (`gridconv_Re400_coarse`,
`gridconv_Re400_fine`, in `$PDROM_REVISION_DIR`) and the forty networks at Re = 500
used for the free-running clock (`seed_sweep_v3_Re500`, the second Re = 500 ensemble of
Sec. III L; the first, `seed_sweep_Re500`, was trained with `05_ensembles/seed_sweep_v2.py`
and stores scalars only) were trained with
[`05_ensembles/seed_sweep_v3.py`](../05_ensembles/seed_sweep_v3.py):

```bash
for s in $(seq 0 9); do
  python 05_ensembles/seed_sweep_v3.py --re 400 --seeds $s --data-dir $DATA --out-dir $REV/gridconv_Re400_coarse
  python 05_ensembles/seed_sweep_v3.py --re 400 --seeds $s --data-dir $REV/fine400 --out-dir $REV/gridconv_Re400_fine
done      # $REV/fine400 holds snapshots_Re400.npy and metadata_Re400.npy linked to the refined-mesh files
python 05_ensembles/seed_sweep_v3.py --re 500 --n-seeds 40 --data-dir $DATA --out-dir $DATA/seed_sweep_v3_Re500
```

Exact commands for the `seed_sweep_v4` ensembles (each writes `config.json` with its
full settings; `DATA`, `REV` and `ENS` are the directories above):

```bash
# S7b_base_v4
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 9) --latent 4 --data-dir $DATA --out-dir $ENS/S7b_base_v4
# S6_causalCAE
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 9) --latent 4 --cae-mode causal --data-dir $DATA --out-dir $ENS/S6_causalCAE
# S7_long25D
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 9) --latent 4 --snap $REV/snapshots_Re400L25D.npy --data-dir $DATA --out-dir $ENS/S7_long25D
# S11_wide10D
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 9) --latent 4 --snap $REV/snapshots_Re400W10D.npy --data-dir $DATA --out-dir $ENS/S11_wide10D
# S9_lift
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 9) --latent 4 --scalar $REV/cl_Re400_1334.npy --data-dir $DATA --out-dir $ENS/S9_lift
# S1_fixedCAE
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 9) --latent 4 --cae-mode fixed:$REV/gridconv_Re400_coarse/seed_000 --data-dir $DATA --out-dir $ENS/S1_fixedCAE
# S2_fixedLSTMseed
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 9) --latent 4 --lstm-seed 0 --data-dir $DATA --out-dir $ENS/S2_fixedLSTMseed
# S3_pod
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 9) --latent 4 --cae-mode pod --data-dir $DATA --out-dir $ENS/S3_pod
# S4_tau5
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 4) --latent 4 --tau 5 --cae-mode fixed:$REV/gridconv_Re400_coarse/seed_000 --data-dir $DATA --out-dir $ENS/S4_tau5
# S4_tau10
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 4) --latent 4 --tau 10 --cae-mode fixed:$REV/gridconv_Re400_coarse/seed_000 --data-dir $DATA --out-dir $ENS/S4_tau10
# S4_tau40
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 4) --latent 4 --tau 40 --cae-mode fixed:$REV/gridconv_Re400_coarse/seed_000 --data-dir $DATA --out-dir $ENS/S4_tau40
# S4_tau80
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 4) --latent 4 --tau 80 --cae-mode fixed:$REV/gridconv_Re400_coarse/seed_000 --data-dir $DATA --out-dir $ENS/S4_tau80
# S5_ms5
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 4) --latent 4 --cae-mode fixed:$REV/gridconv_Re400_coarse/seed_000 --loss multistep --k-steps 5 --data-dir $DATA --out-dir $ENS/S5_ms5
# S5_ms20
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 4) --latent 4 --cae-mode fixed:$REV/gridconv_Re400_coarse/seed_000 --loss multistep --k-steps 20 --data-dir $DATA --out-dir $ENS/S5_ms20
# S8_longroll
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 4) --latent 4 --snap $REV/snapshots_Re400ext.npy --cae-mode fixed:$REV/gridconv_Re400_coarse/seed_000 --split-n 1334 --data-dir $DATA --out-dir $ENS/S8_longroll
# S10_Re100
python 02_train_rom/seed_sweep_v4.py --re 100 --seeds $(seq 0 9) --latent 4 --snap $REV/snapshots_Re100c.npy --data-dir $DATA --out-dir $ENS/S10_Re100
# S12a_Re800_coarse_L8
python 02_train_rom/seed_sweep_v4.py --re 800 --seeds $(seq 0 9) --latent 8 --snap $DATA/snapshots_Re800.npy --data-dir $DATA --out-dir $ENS/S12a_Re800_coarse_L8
# S12b_Re800_fine_L8
python 02_train_rom/seed_sweep_v4.py --re 800 --seeds $(seq 0 9) --latent 8 --snap $REV/snapshots_Re800fine.npy --data-dir $DATA --out-dir $ENS/S12b_Re800_fine_L8
# S13a_Re1000_coarse
python 02_train_rom/seed_sweep_v4.py --re 1000 --seeds $(seq 0 9) --latent 4 --snap $DATA/snapshots_Re1000.npy --data-dir $DATA --out-dir $ENS/S13a_Re1000_coarse
# S13b_Re1000_fine
python 02_train_rom/seed_sweep_v4.py --re 1000 --seeds $(seq 0 9) --latent 4 --snap $REV/snapshots_Re1000fine.npy --data-dir $DATA --out-dir $ENS/S13b_Re1000_fine
# S0_v4_check
python 02_train_rom/seed_sweep_v4.py --re 400 --seeds $(seq 0 0) --latent 4 --data-dir $DATA --out-dir $ENS/S0_v4_check
```

Training on the GPU is not bit-for-bit deterministic, so a network with a given seed
is not reproduced exactly between runs or code versions. Ensembles are therefore
compared only as distributions (Mann–Whitney test on |ω_d|), never network by
network.

## 3. Analyses → paper

| paper | script | output |
|---|---|---|
| Table I (saturation) | `saturation_check_v3.py` | `saturation_check_v3.json` |
| near-wall resolution (Sec. II B 1) | `nearwall_resolution_v2.py` | `nearwall_resolution_v2.json` |
| Fig. 2, Table II rows at Re = 800/1000 | `forces.py`, `make_fig_forces_v2.py` (case names of `01_generate_data/circular_cylinder/revision`, all under `$PDROM_CASES_DIR`) | `forces_vs_re.json`, `fig_forces_vs_re_v2.pdf` |
| Fig. 8 (signed error and phase) | `make_fig_signed_phase_v2.py` | `fig_signed_phase_v2.{json,pdf}` |
| Fig. 9 (395-cycle rollout) | `long_rollout_analysis_v3.py` | `long_rollout_v2.json`, `fig_long_rollout_v2.pdf` |
| Fig. 11 (field-space timing) | `field_timing_check_v2.py` | `field_timing_check_v2.json`, `fig_field_timing_v2.pdf` |
| Table XII (periodic continuation, cubic spline) | `periodic_tiling_interp_v2.py`; `periodic_tiling_interp_v3.py 998:4 999:4` (modulated) | `periodic_tiling_interp_v2.json`, `periodic_tiling_interp_v3_*.json` |
| Table XVII (sensitivity) | `sensitivity_table_v3.py` (uses `diag.py`) | `sensitivity_table_v3.json`, `sensitivity_table_rows_v3.tex` |
| per-ensemble summaries (Sec. III M, N) | `aggregate_B_v2.py <ensemble dirs>` | `aggregate_B_*.json` |
| Fig. 14, Sec. III O (free-running clock, perturbations) | `clock_freerun_perturb_v2.py $DATA/seed_sweep_v3_Re500 --out $ENS/clock_perturb_v2_Re500_40seeds.json`; `make_fig_learned_map_v6.py` (reads that file, or `--perturb <file>`) | `fig_learned_map_v6.pdf`, `learned_map_panel_c_v6.json` |
| table-of-contents image | `make_highlight_image_v2.py` | `highlight_image.{png,pdf}` |

The earlier versions (`make_fig_forces.py`, `long_rollout_analysis_v2.py`,
`sensitivity_table_v2.py`, `aggregate_B.py`, `make_fig_learned_map_v5.py`) are kept for the
record; they expected the case names used while the paper was prepared, skipped missing
inputs silently, or took definitions by `exec()`. The new versions stop with an error when an
input is missing, import the shared definitions from `learned_map_common.py`, and were
checked to reproduce the stored results exactly (2026-11-24).

`diag.py` holds the per-network diagnostics (phase and amplitude shares, tangential
share, growth factors, R²_cal, ω_d, correction efficacy, clock bias); it is the same
code as in `seed_sweep_v4.py`, so diagnostics are identical for every ensemble format.

Every number quoted in the paper is checked against these outputs by
[`07_audit/audit_numbers_v3.py`](../07_audit/audit_numbers_v3.py).

Two traps found while preparing the revision:

- `phasecorr_Re400_L4.npy` and `phasecorr_Re101_L4.npy` were overwritten by the
  calibration-length sweep and hold N_cal = 200 results; use the `*_ncal150.npy`
  files for the default N_cal = 150.
- A network can be retrained in place if two runs write to the same ensemble
  directory; regenerate the sensitivity table and rerun the audit after any
  retraining.
