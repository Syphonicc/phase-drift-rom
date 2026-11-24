# 01 — Generating the data

OpenFOAM cases for the two flow configurations, and the scripts that turn solver
output into the snapshot arrays everything downstream reads.

| script | does |
|---|---|
| `extract_snapshots_v2.py` | interpolates circular-cylinder solver fields onto the uniform 128 x 64 grid, writes `snapshots_Re{RE}.npy` and `metadata_Re{RE}.npy` |
| `extract_square.py` | the same for the square cylinder |
| `extract_snapshots_v5.py` | used for the additional simulations of the revision: same 128 × 64 grid as production (`--grid-from`), gzip-compressed fields, and `--decomposed` to read processor directories directly (identical to `reconstructPar` + serial extraction) |
| `interpolation_study_v3.py` | resolution study for the interpolation grid: how much of the reconstruction error is the grid rather than the autoencoder |

The interpolation grid is **not** the solver mesh, and its spacing is anisotropic
(8.9 points per diameter streamwise, 6.4 laterally). Every claim about
reconstruction fidelity in the paper is a claim about this interpolated field. See
[`../docs/DATA.md`](../docs/DATA.md).

The additional simulations of the revision (Re = 100, refined meshes at Re = 800 and
1000, outlet at 25D, lateral boundaries at ±10D (which in this O-grid also moves the inlet
and outlet), Re = 400 continued to t = 290, and force-only runs) are in
[`circular_cylinder/revision/`](circular_cylinder/revision). The modulated-inflow cases
(eps = 0.15 and 0.05) are `circular_cylinder/Re400_modulated*`. The half-time-step check of
Table II is the production Re = 400 case with `maxCo 0.4` in `system/controlDict`; it is
not stored separately.

The `_v3` suffix on the interpolation study is deliberate: earlier versions exist
in the working directory and are superseded.
