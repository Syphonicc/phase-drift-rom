# Additional circular-cylinder simulations (revision)

These cases use the solver settings and boundary conditions of the production
case [`../Re400`](../Re400) (2D laminar `pimpleFoam`, OpenFOAM v2512, second-order
backward time integration, maximum Courant number 0.8, inlet velocity
`(1 0.01 0)` whose 1% transverse component triggers shedding, zero-gradient outlet
velocity with fixed outlet pressure, symmetry planes at the lateral boundaries,
no-slip cylinder). They differ only in viscosity, mesh, domain or time span. Only
the case definitions are stored (`system/`, `constant/` dictionaries, `0/`); meshes
and solutions are regenerated.

| case | Re (ν) | mesh | time | purpose (section of the paper) |
|---|---|---|---|---|
| `Re100` | 100 (1.2e-3) | production | 0–250 | wake with a strongly stable limit cycle (III M) |
| `Re400_continued` | 400 (3.0e-4) | production | 90–290 | reference for the 395-cycle rollout (III D) |
| `Re1000` | 1000 (1.2e-4) | production | 0–90 | baseline mesh at Re=1000; irregular (II B 1, III N) |
| `Re800_finemesh` | 800 (1.5e-4) | refined (69,296 cells) | 0–90 | mesh sensitivity at Re=800 (II B 1, III N) |
| `Re1000_finemesh` | 1000 (1.2e-4) | refined (69,296 cells) | 0–90 | mesh sensitivity at Re=1000; clean limit cycle (II B 1, III N) |
| `Re400_outlet25D` | 400 (3.0e-4) | outlet at 25D, 72,612 cells | 0–90 | domain length (II B, III N) |
| `Re400_walls10D` | 400 (3.0e-4) | lateral walls at ±10D (5% blockage), 120,204 cells | 0–90 | blockage (II B, III N) |
| `forces_only/Re300…Re700` | 300–700 | production | 0–90 | force coefficients of Fig. 2 (forces written, no fields) |

Mesh generation:

- **production mesh** (`Re100`, `Re400_continued`, `Re1000`, `forces_only/*`): as in
  `../Re400` — `blockMesh -dict system/blockMeshDict.main && mirrorMesh -overwrite`
  (34,764 cells). The `forces_only` cases have no `blockMeshDict`; copy
  `../Re400/system/{blockMeshDict.main,mirrorMeshDict}` into them first.
- **refined mesh** (`Re800_finemesh`, `Re1000_finemesh`): the √2 refinement of
  `../Re400_finemesh` (`system/blockMeshDict.fine`). The `blockMeshDict.main` in these
  two cases is the production dictionary copied by the setup script and is *not*
  the mesh that was used.
- **outlet at 25D** and **walls at ±10D**: their own `system/blockMeshDict.main`
  (same cell spacing as production), followed by `mirrorMesh -overwrite`.

`Re400_continued` has no `0/`: it starts from the `90/` directory of the solved
production Re=400 case (`UMean`, `pMean` removed). In `Re400_walls10D`,
`system/controlDict` writes only `U` and `p` for t ≥ 50 (to save disk space);
`controlDict.full_write` is the unrestricted version.

`setup_revision_runs.sh` is the script that created the first six cases from the
production case; set `PDROM_PRODUCTION_CASE` and `PDROM_FINE_CASE` before running
it. All runs were decomposed (2–6 ranks, `scotch`) and never reconstructed;
`../../extract_snapshots_v5.py --decomposed` reads the processor directories
directly and reproduces `reconstructPar` + serial extraction exactly.

Snapshots are written every Δt = 0.03; the first 50 time units are discarded
(t = 210–250 is used at Re=100). Forces are read from
`postProcessing/forceCoeffs*` with [`08_revision/forces.py`](../../../08_revision/forces.py).
