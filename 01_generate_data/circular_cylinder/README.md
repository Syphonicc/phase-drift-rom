# Circular-cylinder cases

| directory | Re | nu | notes |
|---|---|---|---|
| `Re400/` | 400 | 3e-4 | production case |
| `Re800/` | 800 | 1.5e-4 | production case |
| `Re400_modulated/` | 400 | 3e-4 | time-varying inflow, eps = 0.05 |
| `Re400_finemesh/` | 400 | 3e-4 | refined mesh, grid-convergence study |

Domain, from `system/blockMeshDict.main` with `R = 0.06` so `D = 0.12`:
5.06D upstream, 9.33D downstream, +/-5D lateral. Blockage ratio 10%.

**`blockMeshDict.main` is the production mesh.** The `.coarse` and `.fine`
variants, `coarseMesh/`, `ROMfieldsDict*` and `mirrorMeshDict` are retained from
the case template these were built from and are not all used by the runs reported
in the paper; they are kept rather than pruned because the mesh-generation chain
has not been re-verified without them.

Meshes are not included in this repository. Regenerate with `blockMesh`.

The Re=300 case and the square-cylinder case are not in this revision.
