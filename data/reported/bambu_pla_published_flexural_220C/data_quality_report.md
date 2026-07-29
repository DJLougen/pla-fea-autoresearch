# Published Bambu PLA Basic flexural evidence: data-quality audit

## Dataset and grain

Table 3 contains 12 orientation/temperature summary rows. Figure 4 contains three plotted flexural stress–strain traces at 220 °C. The article reports five or more tested specimens per condition, but does not state whether each plotted trace is a mean, median, or a representative specimen.

## Checks performed

- SHA-256 pinning of the CC BY article XML and original TIFF.
- Exact Table 3 row count, numeric types, and domain values.
- Fixed pixel-to-axis calibration against the printed axes.
- Color-separated curve extraction with title and legend exclusion.
- Cross-check of visible-region slope and peak against Table 3.

## Cross-check findings

| Raster | Visible strain | Digitized slope (GPa) | Table modulus (GPa) | Digitized peak (MPa) | Table strength (MPa) | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 0° | 0.0051–0.0568 | 2.960 | 3.030 | 67.98 | 66.85 | pass |
| 45° | 0.0056–0.0552 | 2.911 | 3.030 | 64.81 | 64.67 | pass |
| 90° | 0.0052–0.0333 | 2.861 | 2.980 | 54.95 | 57.69 | pass |

## Findings and analytical risk

- **High severity, high confidence:** plotted-trace aggregation is undefined. These curves cannot be treated as raw replicates or used in calibration/model selection.
- **Medium severity, high confidence:** the three colored traces overlap below roughly 0.5% strain, so the extractor retains only pixels that are visibly attributable to each trace and does not fabricate the hidden origin segment.
- **Medium severity, high confidence:** 0°, 45°, and 90° are flat in-plane raster orientations, not build-Z coupons. They constrain external flexural behavior but not E3 or interlayer fracture.
- Pixel resolution is approximately 5.11e-05 strain and 0.068 MPa before line-thickness uncertainty.

## Decision

All three digitized traces pass the summary-table cross-check. They are admitted only as qualitative external validation evidence. The missing raw-flexural-data requirement remains open.
