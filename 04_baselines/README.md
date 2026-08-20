# 04 — Baselines

The alternatives the reduced-order model is measured against. Several of them beat
it, which is reported in the paper and is the reason this directory matters.

| script | does |
|---|---|
| `dmd_baseline.py` | dynamic mode decomposition on the latent trajectory, plus a harmonic fit |
| `hankel_dmd.py` | delay-embedded DMD, swept over delay count and truncation rank |
| `periodic_baseline.py` | model-free: tile the last observed shedding period forward |
| `noise_injection_test.py` | training-time noise injection, the standard remedy for rollout drift |
| `phasecorr_on_dmd.py` | applies the phase correction to a DMD rollout instead of the network's |

Two results worth knowing before you run anything here. A delay-embedded operator
with an adequate memory is more accurate than the corrected network on every flow
tested, and shows no drift. And on the square cylinder, tiling the last period of
raw solver snapshots beats the entire pipeline in field space.

`phasecorr_on_dmd.py` returns a **negative** result: the correction degrades every
DMD rollout it is applied to, because those rollouts fail in amplitude as well as
timing. It is included because the experiment is an obvious one to try and the
answer is worth having on record.

Two traps live in this directory. Hankel-DMD requires an **explicit** rank — an
energy criterion keeps three modes and loses two orders of magnitude of accuracy.
And importing `hankel_dmd.py` as a module re-runs its sweep and overwrites
`hankel_dmd.json`. See [`../docs/DATA.md`](../docs/DATA.md), traps 5 and 6.
