# Phase-drift diagnosis and correction for latent-space reduced-order models

Code and OpenFOAM cases for the paper:

> **Phase drift dominates long-horizon error in autoencoder–LSTM models of
> periodic cylinder wakes**
> *(under review, Physics of Fluids; revised version)*

Autoregressive convolutional-autoencoder / LSTM reduced-order models of
bluff-body wakes accumulate rollout error that is almost entirely a **coherent
phase drift at the vortex-shedding frequency**, not broadband noise: 97–98% of
the error variance is phase, while the limit-cycle amplitude is reproduced to
within 0.14%. Because the drift is linear in time it can be removed by a
one-parameter-per-latent-coordinate phase realignment, fitted on a short
calibration window, with a built-in diagnostic (the R² of the phase fit) that
reports in advance whether the correction applies.

The revised paper traces the drift to the shedding clock: the trained network
reproduces the Strouhal number to better than 0.01%, this small error is fixed when
training ends and is predicted by the frequency of the free-running network
(correlation 0.99 over forty networks), and the learned dynamics restore amplitude
perturbations within one shedding period while leaving the phase neutral. In
physical space the error is a displacement of the shed vortices along the street.
The structure is unchanged by solver mesh, domain length, blockage, POD in place of
the autoencoder, input window, an unrolled training loss, and a model of the lift
coefficient alone.

This repository reproduces every figure and table in the paper.

**What this repository does not claim.** The correction is *post hoc* and
non-causal: it uses the analytic signal of the complete predicted trajectory, so
it cannot be run online. And on every flow tested here a delay-embedded linear
(Hankel-DMD) operator with an adequate memory is more accurate than the corrected
network and shows no drift; on the stationary wakes, so is periodic continuation
with a cubic-spline interpolant. The contribution is the diagnosis — that the
long-horizon failure of a learned autoregressive surrogate on an oscillatory flow
is a *timing* failure, measurable and reportable — not a claim that correcting it
beats fitting the flow directly. The baselines that outperform the pipeline are
included in `04_baselines/` so this can be checked.

---

## How to navigate this repository

The folders are numbered in the order you would run them.

| step | folder | what it does |
|------|--------|--------------|
| 1 | [`01_generate_data/`](01_generate_data) | OpenFOAM cases (circular + square cylinder) and snapshot extraction |
| 2 | [`02_train_rom/`](02_train_rom) | train the convolutional autoencoder and LSTM, produce a rollout |
| 3 | [`03_phase_correction/`](03_phase_correction) | the method: phase realignment, error decomposition, field-space validation |
| 4 | [`04_baselines/`](04_baselines) | linear (DMD and Hankel-DMD), harmonic, periodic-extension and noise-injection baselines |
| 5 | [`05_ensembles/`](05_ensembles) | multi-seed studies (efficacy spread, field-space G and ρ, drift-predictor null, held-out threshold validation) |
| 6 | [`06_figures/`](06_figures) | figure generation and bundle synchronisation |
| 7 | [`07_audit/`](07_audit) | provenance audit and manuscript cross-check |
| 8 | [`08_revision/`](08_revision) | revision analyses: sensitivity ensembles, free-running clock and perturbation tests, 395-cycle rollout, field-space timing, forces, saturation and near-wall checks |

**Core path** (to see the finding): 1 → 2 → 3 → 6.
**Full reproduction** (with error bars and baselines): 1 → 6, then 8.

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the table/figure → script map and
[`docs/DATA.md`](docs/DATA.md) for the data-file layout.

## Method in one paragraph

For each latent coordinate the analytic signal (Hilbert transform) gives an
instantaneous amplitude and phase. The predicted amplitude is essentially exact
(RMS deviation 0.12–0.14%); the phase drifts linearly, at a rate `omega_d`
estimated by least-squares regression of the phase difference over a calibration
window of 150 steps. The correction re-references the predicted phase by
`omega_d · t` and restores the coordinate mean, leaving the predicted amplitude
untouched. The R² of that regression is the diagnostic: high R² means a coherent
drift is present and correctable; low R² means it is not, and the correction is
not applied. Note that the diagnostic tests *timing*, not amplitude — it should
not be applied to a propagator whose predicted trajectory has left the attractor.

## Quick start

```bash
pip install -r requirements.txt
# obtain the snapshots: run step 1, or see docs/DATA.md

# core path on one case:
python 02_train_rom/run_pipeline.py                --re 400 --latent 4
python 03_phase_correction/phase_correction.py     --re 400 --latent 4 --dt 0.03
python 06_figures/make_paper_figures.py
```

## A note on the audit tooling

`07_audit/` is unusual for a paper repository and is included deliberately.
`audit_numbers.py` walks every `.npy` and `.json` output and writes a CSV of every
reported quantity together with its provenance — case, latent dimension,
calibration length, timestep, trim setting, and source file — flagging any
quantity whose sources disagree. `check_manuscript.py` then cross-checks every
number printed in the LaTeX source against that audit, in both directions:
coverage (is every audited quantity accounted for?) and unmatched claims (does
every printed number have a source?).

These were written because several data files are genuinely easy to confuse — for
example, a key named `lstm_phase_corrected` holds the *sinusoid* baseline rather
than the phase correction, and an unsuffixed `phasecorr_Re400_L4.npy` is a
different calibration length from the one the paper reports. Those traps are
documented in `docs/DATA.md`. Anyone re-deriving numbers from these outputs should
read that file first.

## Data availability

The OpenFOAM cases in `01_generate_data/` regenerate the raw simulations from
scratch, including mesh generation (`blockMesh` for the circular cylinder,
`blockMesh` plus `snappyHexMesh` for the square), and the additional simulations of
the revision are in `01_generate_data/circular_cylinder/revision/`. The processed snapshot arrays
used for training are approximately 5 GB; they are not held in this repository but
can be regenerated from the case files, and are available from the corresponding
author on reasonable request.

## Citation

If you use this code, please cite the paper — see [`CITATION.cff`](CITATION.cff).

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The OpenFOAM
case files are configuration data under the same license; OpenFOAM itself is GPL
and is not redistributed here.

## AI-assistance disclosure

Consistent with the disclosure in the paper: a large language model (Claude,
Anthropic — models Opus 4.8 and Opus 5) was used to assist with editing and with
generating analysis and figure-generation code in this repository. All code was
reviewed and verified against the simulation data by the authors, who take full
responsibility for it.
