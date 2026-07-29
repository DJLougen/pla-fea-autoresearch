# Public PLA tensile reference data

Source: Al Amri, Al Azhar, “Brick Layering Experiment,” Zenodo, 2025,
DOI: [10.5281/zenodo.16935914](https://doi.org/10.5281/zenodo.16935914).
License: CC BY 4.0.

The source XLSX contains 72 specimen-level tensile stress–strain curves as
literal chart caches. `scripts/import_zenodo_16935914.py` downloads the
checksum-pinned workbook and extracts those measurements without changing
them. `manifest.csv` records source metadata and reconciles each extracted peak
stress against the specimen UTS reported in the workbook.

## Critical limitations

- This is printed PLA, but the source does not identify it as Bambu PLA Basic.
- Source `X`, `Y`, and `Z` are printing/build-placement labels. They have not
  been mapped to this repository's local material axes 1, 2, and 3.
- The dataset contains tensile curves only: no three-point-bending curves and
  no held-out component response.
- The source modulus column is labeled MPa but contains values near 3–4,
  consistent with GPa. It is preserved in the manifest but is not used.

For these reasons the extracted files remain under `data/public`, use
`partition=unassigned`, leave `print_direction` blank, and are intentionally
not copied into `data/raw`. Doing so would create false provenance and an
invalid anisotropic calibration.

