# Circular-cylinder cases

| directory | Re | nu | notes |
|---|---|---|---|
| `Re400/` | 400 | 3e-4 | production case |
| `Re800/` | 800 | 1.5e-4 | production case |
| `Re400_modulated/` | 400 | 3e-4 | time-varying inflow, eps = 0.15 (`A = 0.15` in `0/U`) |
| `Re400_modulated_eps005/` | 400 | 3e-4 | time-varying inflow, eps = 0.05 (`A = 0.05` in `0/U`) |
| `Re400_finemesh/` | 400 | 3e-4 | refined mesh (`blockMeshDict.fine`), grid-convergence study |
| `revision/` | 100–1000 | | additional simulations of the revised paper (see [`revision/README.md`](revision/README.md)) |

Domain, from `system/blockMeshDict.main` with `R = 0.06` so `D = 0.12`:
5.06D upstream, 9.33D downstream, +/-5D lateral. Blockage ratio 10%. The lateral
boundaries are symmetry planes.

**`blockMeshDict.main` is the production mesh.** The `.coarse` variant,
`coarseMesh/`, `ROMfieldsDict*` and `snappyHexMeshDict` are retained from the case
template these were built from and are not used by the runs reported in the paper.

Meshes are not included in this repository. `blockMeshDict.main` describes one half
of the O-grid; regenerate the full mesh (34,764 cells) with

```bash
blockMesh -dict system/blockMeshDict.main && mirrorMesh -overwrite
```

and the refined mesh (69,296 cells) with `blockMesh -dict system/blockMeshDict.fine`
followed by `mirrorMesh -overwrite`.

The production Re = 800 `controlDict` writes the fields only (no `forceCoeffs`). For a
consistent set, the force coefficients of the production Re = 400 and 800 runs used in
Fig. 2 were obtained afterwards from the stored fields with the cases in
[`revision/forces_prod/`](revision/forces_prod) (see [`revision/README.md`](revision/README.md)).

The modulated cases write every 0.03 up to t = 110; the snapshots used in the paper
are t = 20.01–109.98 (3000 snapshots).

## Other Reynolds numbers

The circular-cylinder cases differ only in viscosity, since D = 0.12 and
U = 1.0 throughout and Re = U*D/nu. The production cases at Re = 300, 500, 600, 700
and 1000 were made by copying the Re400 case and changing nu in
`constant/transportProperties`; their case directories are not stored separately.

| Re | nu |
|---|---|
| 100 | 1.2e-3 |
| 300 | 4.0e-4 |
| 400 | 3.0e-4 |
| 500 | 2.4e-4 |
| 600 | 2.0e-4 |
| 700 | 1.714e-4 |
| 800 | 1.5e-4 |
| 1000 | 1.2e-4 |

The square-cylinder case is in [`../square_cylinder`](../square_cylinder).
