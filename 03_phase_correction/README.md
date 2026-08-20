# 03 — The method

Diagnosis and correction of the phase drift, plus the checks that establish the
diagnosis is not an artefact of how it was measured.

| script | does |
|---|---|
| `phase_correction.py` | the correction itself: fits one drift rate per latent coordinate on a calibration window, realigns the phase, scores on the extrapolation region |
| `decompose_error.py` | amplitude/phase variance partition via the analytic signal |
| `tangential_transverse.py` | the independent geometric check: resolves error against the tangent of the true trajectory, no Hilbert transform |
| `verify_mechanism.py` | rollout-wide phase fit and the frequency-difference cross-check (`slope_measured` against `2*pi*(f_pred - f_true)`) |
| `field_validation.py` | decodes corrected latents and measures field-space error against the autoencoder floor |
| `compute_rho.py` | the ratio of temporal error to compression floor |
| `slope_tstat.py` | slope t-statistic beside the R2 of the same regression |
| `omega_sensitivity.py` | efficacy against deliberate error in the fitted drift rate |
| `spectral_robustness.py` | the error spectrum under different estimators (periodogram, Hann, Welch) |
| `timewarp_correction.py` | equivalent formulation as a time warp rather than a phase shift |
| `collect_phase_results.py` | gathers per-case results into the comparison table |

**Three different R-squared values circulate in this work and must not be
interchanged**: the calibration-window fit to phi(t) (the applicability
diagnostic, 0.530 at Re=400), the rollout-wide fit to phi(t) (0.949), and the fit
to tangential error *magnitude* (0.946). They live in `phasecorr_*`, `mechanism_*`
and `tangential_transverse.json` respectively. The near-coincidence of the last
two is a coincidence.

The diagnostic tests **timing, not amplitude**. It should not be applied to a
propagator whose predicted trajectory has left the attractor: the analytic
envelope of a diverged trajectory does not describe an oscillation, and realigning
its phase cannot help however linear the measured phi(t) appears.
