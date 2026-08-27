# Square-cylinder case, Re = 100

Two-dimensional flow past a square cylinder at Reynolds number 100, based on
`D = 0.12`, `U = 1.0` and `nu = 0.0012`.

## Domain

| | |
|---|---|
| Upstream | 10D (x from -1.2) |
| Downstream | 30D (x to 3.6) |
| Height | 17.86D (y from -1.0714 to 1.0714) |
| Blockage | 5.6% |

The 30D downstream extent is chosen to exceed the 25D minimum reported as
necessary to preserve temporal periodicity of the solution.

## Mesh

`blockMesh` builds a uniform Cartesian background grid of 480 x 214 cells at a
spacing of 0.0100. `snappyHexMesh` then carves out the square and applies one
level of refinement, both at the body surface (`refinementSurfaces`, level 1)
and within a box enclosing the body and near wake (`refinementRegions`,
level 1). The refined spacing is 0.0050, giving 24 cells across the body face.

The result is 164,568 cells: 93,720 at the background level and 70,848 at the
refined level. Because refinement halves both in-plane directions the cell
aspect ratio is preserved, with a maximum of 1.002.

The body faces are grid-aligned, so the mesh is free of stair-stepping.

To regenerate the mesh, run `blockMesh` followed by
`snappyHexMesh -overwrite`. The resulting `constant/polyMesh` is not tracked in
this repository.

## Run settings

Solver `pimpleFoam`, laminar. `endTime = 120`, `deltaT = 0.005` adaptive with
`maxCo = 0.8`.

Snapshots are written every 0.0438 time units, which is one twentieth of the
shedding period (T = 0.876), giving about 20 per cycle. The first 60 time units
are discarded as startup transient, leaving 1370 snapshots.

## Validation

Measured over 37 cycles of the settled flow (t = 90 to 120):

| | measured | published |
|---|---|---|
| St | 0.148 | 0.137 - 0.15 |
| mean Cd | 1.502 | 1.4 - 1.5 |

The lift amplitude was steady to within 0.3% across three consecutive windows,
confirming the limit cycle had saturated before sampling began.
