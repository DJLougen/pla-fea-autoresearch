# Exact-brand common-infill compression evidence

## Result

Figure 4 yields nine checksum-pinned aggregate stress–strain curves for Bambu Lab purple PLA Basic: grid, gyroid, and cubic infill at 10%, 20%, and 30% nominal slicer infill. All nine digitizations pass the pixel-coverage, continuity, monotonic-strain, and nonnegative-stress gates.

The accompanying mass table contains three-replicate mean and standard deviation values for all nine conditions. Because the 30 mm cubes include 0.42 mm walls, total measured solid fraction is substantially greater than the nominal internal-infill setting.

## Data-quality boundary

- The source is a TechRxiv preprint and was not peer reviewed at capture.
- The nine traces are aggregate plotted responses, not 27 raw specimen curves.
- The stress axis is retained exactly as printed: Pa with a ×10^4 multiplier. It conflicts with the paper's MPa yield-strength text by a factor of 1,000.
- A separate ×1,000 sensitivity is exported. Its nine curve peaks cross-check the independently stated yield strengths within 4%, which strongly suggests a publication-unit error without proving which upstream field is wrong.
- The curves can challenge topology ranking and response shape, but cannot identify a transferable material failure law.

## Digitized metrics

| Infill | Pattern | Literal modulus (Pa) | ×1000 peak (MPa) | Reported yield (MPa) | Peak error | Coverage |
|---:|---|---:|---:|---:|---:|---:|
| 10% | grid | 320060 | 11.68 | 11.87 | -1.6% | 97.6% |
| 10% | gyroid | 234301 | 7.86 | 7.85 | 0.1% | 94.3% |
| 10% | cubic | 205798 | 7.45 | 7.61 | -2.1% | 99.5% |
| 20% | grid | 451297 | 18.13 | 18.70 | -3.1% | 94.3% |
| 20% | gyroid | 317761 | 11.58 | 11.62 | -0.3% | 92.5% |
| 20% | cubic | 301204 | 11.28 | 11.21 | 0.6% | 99.5% |
| 30% | grid | 548031 | 25.98 | 26.06 | -0.3% | 93.9% |
| 30% | gyroid | 384338 | 14.00 | 14.08 | -0.6% | 98.1% |
| 30% | cubic | 351650 | 14.30 | 14.32 | -0.1% | 100.0% |
