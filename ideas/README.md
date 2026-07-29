# Custom infill idea format

One JSON file describes one flat, printable centerline graph. Coordinates use
millimetres inside the declared rectangular coupon domain.

```json
{
  "schema_version": 1,
  "name": "my_pattern",
  "description": "What mechanical idea this topology is testing.",
  "units": "mm",
  "domain": {
    "width_mm": 40.0,
    "height_mm": 40.0,
    "depth_mm": 4.0
  },
  "relative_density": 0.35,
  "minimum_printable_strut_width_mm": 0.8,
  "minimum_segment_length_mm": 0.5,
  "segments": [
    [[0, 0], [40, 40]],
    [[0, 40], [40, 0]]
  ]
}
```

The name must start with a lowercase letter and contain only lowercase
letters, digits, and underscores. Segment endpoints must remain inside the
coupon.

Before simulation, the loader:

- inserts shared nodes wherever centerlines cross;
- rejects zero-length, collinearly overlapping, disconnected, or internally
  dangling struts;
- requires at least two nodes on every coupon boundary and a center bending
  load path;
- checks minimum connected-segment length and equal-mass nominal width;
- reports acute junctions for slicer inspection;
- records the source SHA-256 so results remain tied to the exact concept.

Run a fast beam-only comparison:

```bash
python -m src.custom_infill \
  --pattern ideas/graded_hourglass.json \
  --beam-only
```

Promote the same concept to equal-actual-mass 3D solids, five load modes, and a
three-level bending mesh check:

```bash
python -m src.custom_infill \
  --pattern ideas/graded_hourglass.json \
  --mesh-size 0.9
```

The refined workflow writes a printable exact-mass OpenSCAD file, matched
triangular control, solver decks and curves, comparison report, and a
three-replicate physical test plan for all five load modes. It also screens the
saved local stress tensors against reported directional tensile strengths and
two explicit theory bookends for the unpublished shear strength, then writes a
safer ranked first-displacement ramp under the pattern's
`failure_sensitivity/` folder. A custom concept does not become a validated
design until the matched coupons reproduce its predicted ordering and failure
sites.

`bowtie_fan.json` is the worked example for a bending-specific mechanism. It
uses a continuous center spine and mirrored diagonal fans to spread the center
load into the supported side rails. The full workflow keeps it only as a
bending specialist, replays its sliced bead paths, and exports a matched
six-coupon physical package rather than presenting it as a universal infill.
