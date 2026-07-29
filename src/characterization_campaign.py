from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/characterization_campaign"
RANDOMIZATION_SEED = 20260728
TENSILE_DIRECTIONS = (
    "1",
    "2",
    "3",
    "12_45",
    "13_45",
    "23_45",
)


def campaign_rows(seed: int = RANDOMIZATION_SEED) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for direction in TENSILE_DIRECTIONS:
        for replicate in range(1, 6):
            partition = "calibration" if replicate <= 3 else "validation"
            rows.append(
                {
                    "specimen_id": (
                        f"bambu-tensile-{direction}-r{replicate:02d}"
                    ),
                    "partition": partition,
                    "test_type": "tensile",
                    "print_direction": direction,
                    "replicate": replicate,
                    "source_scad": "cad/tensile_bar_50x10x4.scad",
                    "fixture": "tensile grips with bonded or clamped tabs",
                    "loading": "global X tension",
                    "required_measurements": (
                        "time_s, force_N, axial displacement_mm, "
                        "axial engineering strain, transverse strain, "
                        "actual width/thickness, failure mode"
                    ),
                    "scope": "elastic calibration; failure is observational",
                }
            )
    for direction in ("1", "3"):
        for replicate in range(1, 6):
            partition = "calibration" if replicate <= 3 else "validation"
            rows.append(
                {
                    "specimen_id": (
                        f"bambu-flexural-{direction}-r{replicate:02d}"
                    ),
                    "partition": partition,
                    "test_type": "flexural",
                    "print_direction": direction,
                    "replicate": replicate,
                    "source_scad": "cad/flexural_bar_80x10x4.scad",
                    "fixture": "three-point bending, 64 mm span",
                    "loading": "center displacement in global Z",
                    "required_measurements": (
                        "time_s, force_N, crosshead displacement_mm, "
                        "actual width/thickness/span, failure mode"
                    ),
                    "scope": "calibration/validation flexural curve",
                }
            )
    for replicate in range(1, 4):
        rows.append(
            {
                "specimen_id": f"bambu-heldout-l-r{replicate:02d}",
                "partition": "heldout",
                "test_type": "component",
                "print_direction": "mixed",
                "replicate": replicate,
                "source_scad": "cad/heldout_l_bracket.scad",
                "fixture": "clamp left face; displace top arm in global Y",
                "loading": "L-bracket cantilever opening load",
                "required_measurements": (
                    "time_s, force_N, load-point displacement_mm, "
                    "actual arm dimensions/thickness, failure mode"
                ),
                "scope": "never use for fitting or model selection",
            }
        )
    rng = random.Random(seed)
    rng.shuffle(rows)
    for order, row in enumerate(rows, start=1):
        row["randomized_print_order"] = order
    return rows


def _write_cad(cad_dir: Path) -> None:
    cad_dir.mkdir(parents=True, exist_ok=True)
    (cad_dir / "tensile_bar_50x10x4.scad").write_text(
        """// Non-standard elastic screening bar matching the coarse FEA envelope.
// Use tabs and optical gauge strain; do not use this geometry to claim ASTM strength.
cube([50, 10, 4]);
""",
        encoding="utf-8",
    )
    (cad_dir / "flexural_bar_80x10x4.scad").write_text(
        """// Rectangular three-point-bending bar matching flexural.inp.template.
cube([80, 10, 4]);
""",
        encoding="utf-8",
    )
    (cad_dir / "heldout_l_bracket.scad").write_text(
        """// Independent L-bracket matching heldout_component.inp.template.
// Horizontal arm 60x10 mm; vertical arm 20x50 mm; thickness 4 mm.
linear_extrude(height=4)
union() {
  square([60, 10]);
  translate([40, 0]) square([20, 50]);
}
""",
        encoding="utf-8",
    )


CAPTURE_FIELDS = [
    "test_id",
    "partition",
    "test_type",
    "print_direction",
    "time_s",
    "force_N",
    "displacement_mm",
    "engineering_strain",
    "engineering_stress_MPa",
    "transverse_strain",
    "specimen_width_mm",
    "specimen_thickness_mm",
    "gauge_length_mm",
    "support_span_mm",
    "specimen_mass_g",
    "raster_angle_deg",
    "layer_height_mm",
    "nozzle_temperature_C",
    "print_speed_mm_s",
    "infill_percent",
    "material_name",
    "material_lot",
    "material_color",
    "machine_id",
    "slicer_profile",
    "drying_protocol",
    "ambient_temperature_C",
    "relative_humidity_percent",
    "failure_mode",
    "operator_notes",
]


def _write_capture_templates(
    capture_dir: Path, rows: Sequence[Dict[str, object]]
) -> None:
    capture_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = capture_dir / f"{row['specimen_id']}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CAPTURE_FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "test_id": row["specimen_id"],
                    "partition": row["partition"],
                    "test_type": row["test_type"],
                    "print_direction": row["print_direction"],
                    "layer_height_mm": 0.08,
                    "nozzle_temperature_C": 220.0,
                    "infill_percent": 100.0,
                    "material_name": "Bambu PLA Basic",
                    "gauge_length_mm": (
                        50.0 if row["test_type"] == "tensile" else ""
                    ),
                    "support_span_mm": (
                        64.0 if row["test_type"] == "flexural" else ""
                    ),
                }
            )


def write_campaign(
    output_dir: Path = DEFAULT_OUTPUT,
    seed: int = RANDOMIZATION_SEED,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = campaign_rows(seed)
    _write_cad(output_dir / "cad")
    _write_capture_templates(output_dir / "capture_templates", rows)
    fields = list(rows[0])
    with (output_dir / "test_plan.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plan_path = output_dir / "test_plan.csv"
    campaign_manifest = {
        "campaign_id": "bambu_pla_basic_directional_v1",
        "material": "Bambu PLA Basic",
        "randomization_seed": seed,
        "specimens": len(rows),
        "test_plan_sha256": hashlib.sha256(
            plan_path.read_bytes()
        ).hexdigest(),
        "partition_counts": {
            partition: sum(
                row["partition"] == partition for row in rows
            )
            for partition in ("calibration", "validation", "heldout")
        },
        "partition_policy": (
            "test_plan.csv is authoritative and immutable; submitted CSV "
            "partition labels must match it exactly"
        ),
    }
    (output_dir / "campaign_manifest.json").write_text(
        json.dumps(campaign_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Bambu PLA Basic characterization campaign",
        "",
        f"Randomization seed: `{seed}`. Total specimens: {len(rows)}.",
        "",
        "| Mode | Directions | Calibration | Validation | Held out |",
        "|---|---|---:|---:|---:|",
        "| tension | 1, 2, 3, 12_45, 13_45, 23_45 | 18 | 12 | 0 |",
        "| flexural | 1, 3 | 6 | 4 | 0 |",
        "| L-bracket component | mixed | 0 | 0 | 3 |",
        "",
        "The 45-degree coupons make G12, G13, and G23 observable through "
        "off-axis elastic compliance. Record transverse strain on every "
        "tensile coupon so Poisson ratios can be estimated separately.",
        "",
        "| Direction | Required material-axis placement |",
        "|---|---|",
        "| 1 | print flat; bead direction parallel to global tensile X |",
        "| 2 | print flat; bead direction perpendicular to global tensile X |",
        "| 3 | print upright; build/layer-normal direction parallel to tensile X |",
        "| 12_45 | print flat; bead direction 45° to tensile X |",
        "| 13_45 | orient the coupon so tensile X bisects bead and build axes |",
        "| 23_45 | orient the coupon so tensile X bisects in-layer-transverse and build axes |",
        "",
        "The 13_45 and 23_45 specimens need a documented angled build fixture "
        "or support strategy. Supports must not contact the gauge section; if "
        "that cannot be achieved repeatably, use a dedicated shear fixture "
        "instead and add a matching solver template before calibration.",
        "",
        "## Print controls",
        "",
        "- Use one conditioned Bambu PLA Basic spool and record color/lot.",
        "- Randomize production in `randomized_print_order`; do not print one "
        "orientation as a single uninterrupted block.",
        "- Hold machine, nozzle, layer height, cooling, speed, wall count, "
        "and actual toolpath width fixed.",
        "- The supplied 50 × 10 × 4 mm tensile bar is a non-standard elastic "
        "screening coupon matching the current coarse FEA envelope. Use grip "
        "tabs and optical/extensometer gauge strain. It must not be used to "
        "claim ASTM ultimate strength.",
        "- Keep all held-out L-bracket results blinded until model selection "
        "is frozen.",
        "",
        "## Data capture",
        "",
        "One header-plus-metadata CSV is supplied per specimen. Add measured "
        "rows beneath the metadata row, repeating the metadata columns. For "
        "tensile tests, provide either force/displacement plus constant "
        "geometry or engineering stress/strain. Flexural and component tests "
        "should retain raw force/displacement.",
        "",
    ]
    (output_dir / "README.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the missing Bambu PLA characterization campaign"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=RANDOMIZATION_SEED)
    args = parser.parse_args()
    rows = write_campaign(Path(args.output), args.seed)
    print(
        f"wrote {len(rows)} randomized specimens and capture templates to "
        f"{Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
