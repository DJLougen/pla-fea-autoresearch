# Public Bambu PLA Basic tensile and modal component reference data

Source: Huang and Zhang, “Experimental and numerical data from comparative
analysis of candidate structural models for active camber morphing wings,”
Mendeley Data, version 2, 2026,
DOI: [10.17632/8jxw5533py.2](https://doi.org/10.17632/8jxw5533py.2).
Dataset license: CC BY 4.0.

The associated article identifies the material as 1.75 mm Bambu PLA Basic
printed on a Bambu X1C. It reports a 0.08 mm layer height, 0.42 mm raster width,
100% rectilinear infill, a 0.4 mm nozzle at 220 °C, 350 mm/s infill speed, and
ASTM D638-22 tensile testing at 20 mm/min. The importer preserves the source
force, crosshead displacement, extensometer deformation, stress, and strain.

The same dataset also supplies physical modal measurements, measured specimen
dimensions, exact STEP geometry, and 2 mm / 1 mm source Abaqus solid meshes for
the Test2 aspect-ratio-2.5 uniform cantilever. Those files are preserved under
`modal_component/` with the following source checksums:

| File | SHA-256 |
|---|---|
| `1-1_Modal_Test_Experiment_and_Numerical_Results.xlsx` | `34545c9f937404864ad34a4d6dd5c03c0f23b20138115a5bf3dc61fea5749408` |
| `2-1_Tolerance_inspection_of_FDM_3D_printer.xlsx` | `7c7c7da128f7c19ad77a4f84a62dbd427d301c18b7ef971671dce48553ac677f` |
| `2-2_Specimen_tolerance.xlsx` | `80470d582a0bbc8cf6392352bb4583e7b468fb8fa0986fef32706744e7548c59` |
| `AR2d5_SOLID2.inp` | `048eb3b4548e292fd12cf3d98310a09d17b01a77a7fd7ed920002553e710a8bc` |
| `AR2d5_SOLID1.inp` | `a2c5e4c6215909a6639423f9e0ba107a9a39c26854d06de53ed3d38753897833` |
| `Test2_AR2.5.STEP` | `e9c816732a3a8b511dc2a68c473516a2d2deaf7fc5f1a89d0693855cd669c6f2` |
| `Figure3_print_orientations.jpg` | `ada3d306fbc63e96f07b491f41c57dca0ce478bab4cf67207ec92589dc53129a` |

Run the checksum-pinned import with:

```bash
python scripts/import_mendeley_8jxw5533py.py
```

## Data-quality finding

The article and dataset description say the raw workbook contains 21 specimens:
three replicates for each of seven print orientations. The literal raw workbook
instead contains 22 five-column curve blocks. Its labels form unbalanced groups:
five `X-F`, four `45Deg-F`, one `Y-F`, and three each for `X-O`, `Y-O`,
`45Deg-O`, and `Z-U`.

The companion summary workbook names the claimed 21 specimens, but several UTS
values differ materially from the raw curve bearing the same label. The
importer therefore:

- extracts all 22 literal raw-labeled curves without relabeling or deletion;
- records per-curve count and summary-reconciliation flags in `manifest.csv`;
- does not guess that extra curves are missing `Y-F` replicates;
- leaves `partition=unassigned` and `print_direction` blank.

The checksum-pinned reconciliation can be regenerated with:

```bash
python -m src.mendeley_tensile_reconciliation_audit
```

It proves that the mismatch is not repairable by a simple omitted-curve rule.
Only 12 of the 21 summary cells match the raw curve carrying the same label.
Across all groups, the summary reuses only 13 of the 22 raw curves, omits nine
literal curves, and reuses eight raw peaks under multiple summary labels. The
full mapping is recorded under
`results/mendeley_tensile_reconciliation_audit/`.

## Calibration limitations

The source X/Y/45/Z and flat/on-edge/upright labels describe build-platform
placement. The official Figure 3 also shows a rectilinear path, but it does not
uniquely identify the local bead direction for every tensile group. These
labels therefore are not a complete verified mapping to this repository’s
local material axes 1 (bead), 2 (in-layer transverse), and 3 (build normal).
The raw workbook has axial extensometer deformation but no lateral strain, so
it cannot identify Poisson ratios. The source also has no three-point bending.

These files remain under `data/public` and are intentionally blocked from the
production calibration loader until the label discrepancy is resolved, a
blinded partition is chosen, and the missing experimental modes are supplied.

## Allowed held-out modal use

Run:

```bash
python -m src.bambu_modal_component_validation
```

This replays the exact Test2 solid meshes in CalculiX, checks the backend
against the source authors' converged Abaqus frequencies, and compares the
pre-existing frozen isotropic material profile against five common physical
modes from each of three components. The 15 physical frequencies are locked to
external held-out validation and are prohibited from material fitting, profile
selection, or threshold tuning.

Passing this small-strain modal check does not release a directional material
card and does not substitute for static force-displacement, damage, or infill
validation.

## Allowed printer-tolerance use

Run:

```bash
python -m src.bambu_print_tolerance_infill
```

The `2-1` workbook supplies 25 Bambu PLA Basic 20 mm calibration cubes with
three measurements on each axis. The workflow preserves the measured
within-cube covariance, checks the published axis statistics, and propagates
both fixed-offset and proportional-scale interpretations through the four
frozen infill finalists. This is a bounded dimensional sensitivity for the
reported Bambu X1C process, not a universal printer tolerance or a substitute
for local bead-defect measurements.
