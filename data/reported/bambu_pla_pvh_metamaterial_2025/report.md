# Exact-brand PVH metamaterial reference

This import preserves an independent peer-reviewed Bambu PLA Basic study:
[Bio-inspired perturbated hierarchical mechanical metamaterial for energy absorption](https://doi.org/10.1016/j.ijmecsci.2025.110847), including its checksum-pinned supplement.

## What is independently supported

- Three ASTM D638-14 tensile tests give **E = 2140 MPa** and **0.2%-offset
  yield = 36 MPa**. The visible teal trace digitizes to
  **2164.2 MPa**.
- The published validation specimen uses a 15 x 15 PVH honeycomb, 20%
  relative density, 1.0 mm vertex hexagons, 0.5 mm perturbation magnitude,
  and a nominal 0.4 mm wall printed with a 0.2 mm nozzle.
- The digitized author FEA and physical compression curves differ by
  **-7.8%** at
  the first peak and **-1.7%**
  in energy through 0.45 strain.
- The six reported optimum/baseline modulus and strength comparisons have
  FEA-to-test errors from **1.9%**
  to **16.7%**.

## Exact-replay boundary

An exact specimen replay is deliberately excluded. The publication does not
release the random perturbation directions/seed, CAD or Abaqus files, or the
post-yield PLA hardening table. Those omissions change the collapse sequence,
so inventing them would make a visually similar structure rather than the
reported specimen.

The author material card is exported for a reproducible transfer sensitivity
on this project's explicit infill geometries.
