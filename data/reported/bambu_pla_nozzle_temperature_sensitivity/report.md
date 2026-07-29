# Bambu PLA Basic nozzle-temperature storage profiles

These profiles describe the as-printed 30 °C / 1 Hz storage response reported for four nozzle temperatures. Nozzle temperature is a manufacturing variable—not the service temperature.

| Nozzle | E1 | E2 | E3 proxy | G12 | Crossplot |
|---:|---:|---:|---:|---:|---|
| 210 °C | 3285.0 | 3038.2 | 2524.4 | 1184.7 | 2/3 (source_disagreement) |
| 220 °C | 3347.6 | 3113.7 | 2579.5 | 1228.4 | 3/3 (strict_pass) |
| 230 °C | 3391.8 | 3188.3 | 2626.9 | 1255.3 | 2/3 (source_disagreement) |
| 240 °C | 3413.0 | 3389.0 | 2715.5 | 1250.4 | 3/3 (strict_pass) |

## Use boundary

- E1/E2/E45 come from plotted summary markers; G12 is inferred from E45 with the separate exact-brand measured ν12.
- E3 and out-of-plane shear use the existing proportional-scaling hypothesis because the paper reports in-plane orientations only.
- The 210 °C and 230 °C 90° markers disagree between the two published cross-plots. Their profiles remain exploratory.
- These cards do not represent print-flow quality, voids, interlayer defects, service-temperature softening, loss modulus, damping, strength changes, or fracture.
