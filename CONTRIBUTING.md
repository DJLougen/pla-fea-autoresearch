# Contributing

This project needs replication more than it needs confident claims. Useful
contributions include measured force-displacement curves, independent
reproduction attempts, additional infill patterns, solver comparisons, data
quality audits, documentation, and tests.

## Start locally

```bash
git clone https://github.com/DJLougen/pla-fea-autoresearch.git
cd pla-fea-autoresearch
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -m "not integration"
```

CalculiX is required for production FEA runs. Gmsh and MeshIO are available
through the `solid` extra:

```bash
python -m pip install -e ".[dev,solid,digitization]"
```

## Contributing physical measurements

Do not paste values from a chart into a file and describe them as raw
measurements. Submit one CSV per specimen and preserve the machine output when
available. At minimum, record:

- specimen ID, pattern, replicate, and load case;
- material brand, product, color, and lot;
- drying and conditioning protocol;
- printer, nozzle, layer height, temperatures, speed, and slicer profile;
- specimen dimensions and actual mass;
- fixture dimensions and crosshead rate;
- force, displacement, time, units, calibration IDs, and instrument
  resolution;
- failure mode and any deviations from the frozen plan.

For the first replication target, use
`results/custom_infill/graded_hourglass/first_physical_test_plan.csv`. Submit
all six curves, including results that contradict the simulation. Do not
discard outliers silently; explain any exclusion and retain the original
record.

Only contribute data you have the right to publish. Add a short provenance
note stating who measured it, when, under what license, and whether it has
already appeared elsewhere.

## Code and model changes

1. Open an issue describing the proposed experiment, model change, or defect.
2. Keep published evidence, simulated evidence, and measurements in separate
   files.
3. Add tests for changed behavior.
4. Run `pytest -m "not integration"` before opening a pull request.
5. Describe assumptions, solver/version details, and any known failure modes.

Large solver runs, generated meshes, papers, workbooks, and G-code are ignored.
Publish large reproducibility artifacts in an archival data repository and
link them with checksums instead of committing them to Git.

By contributing code, you agree that it may be distributed under the MIT
License. Data and third-party material retain their stated licenses.
