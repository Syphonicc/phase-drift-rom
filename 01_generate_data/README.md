# 01 — Generating the data

OpenFOAM cases for the two flow configurations, and the scripts that turn solver
output into the snapshot arrays everything downstream reads.

| script | does |
|---|---|
| `extract_snapshots_v2.py` | interpolates circular-cylinder solver fields onto the uniform 128 x 64 grid, writes `snapshots_Re{RE}.npy` and `metadata_Re{RE}.npy` |
| `extract_square.py` | the same for the square cylinder |
| `interpolation_study_v3.py` | resolution study for the interpolation grid: how much of the reconstruction error is the grid rather than the autoencoder |

The interpolation grid is **not** the solver mesh, and its spacing is anisotropic
(8.9 points per diameter streamwise, 6.4 laterally). Every claim about
reconstruction fidelity in the paper is a claim about this interpolated field. See
[`../docs/DATA.md`](../docs/DATA.md).

The `_v3` suffix on the interpolation study is deliberate: earlier versions exist
in the working directory and are superseded.
