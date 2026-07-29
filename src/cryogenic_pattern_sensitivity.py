from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping

from .bambu_cryogenic_reference import (
    DEFAULT_OUTPUT as DEFAULT_REFERENCE,
    write_reference,
)
from .custom_pattern_validation import CAPTURE_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLPATH = (
    PROJECT_ROOT
    / "results/custom_infill/graded_hourglass/physical_validation"
    / "toolpath_resolved_bending"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results/custom_infill/graded_hourglass"
    / "cryogenic_recovery_sensitivity"
)
DEFAULT_PHYSICAL_PACKAGE = (
    PROJECT_ROOT
    / "results/custom_infill/graded_hourglass/physical_validation"
)

PHYSICAL_CONDITIONS = (
    {
        "condition_id": "as_printed_direct",
        "conditioning": "as_printed_dry",
        "plate_id": "plate-A",
        "test_block": 1,
        "protocol": "test directly after printing",
        "prediction_factor": "baseline",
    },
    {
        "condition_id": "sealed_room_48h",
        "conditioning": "sealed_room_temperature_48h",
        "plate_id": "plate-B",
        "test_block": 2,
        "protocol": (
            "seal immediately after printing; hold at laboratory temperature "
            "for 48 h; test in the same block as the freeze/recovery plate"
        ),
        "prediction_factor": "baseline",
    },
    {
        "condition_id": "freeze_recovery",
        "conditioning": "minus80C_24h_then_room_temperature_24h",
        "plate_id": "plate-C",
        "test_block": 2,
        "protocol": (
            "seal immediately after printing; -80 C for 24 h; recover at "
            "laboratory temperature for 24 h before testing"
        ),
        "prediction_factor": "published_flexural_modulus_median",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: List[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_physical_handoff(
    output_dir: Path,
    fine_stiffness: Mapping[str, float],
    median_factor: float,
    physical_package: Path = DEFAULT_PHYSICAL_PACKAGE,
) -> Dict[str, object]:
    handoff = Path(output_dir) / "physical_handoff"
    source_plan_path = Path(physical_package) / "frozen_plan.csv"
    source_plate_path = (
        Path(physical_package)
        / "manufacturing/bambu-pla-pattern-validation.3mf"
    )
    if not source_plan_path.exists() or not source_plate_path.exists():
        raise FileNotFoundError(
            "the frozen six-object physical package is incomplete"
        )
    source_rows = _csv_rows(source_plan_path)
    source_by_pattern_replicate = {
        (row["pattern"], int(row["replicate"])): row
        for row in source_rows
    }
    plan_rows: List[Mapping[str, object]] = []
    condition_plans: Dict[str, List[Mapping[str, object]]] = {}
    for condition in PHYSICAL_CONDITIONS:
        condition_id = str(condition["condition_id"])
        condition_rows: List[Mapping[str, object]] = []
        factor = (
            median_factor
            if condition["prediction_factor"]
            == "published_flexural_modulus_median"
            else 1.0
        )
        # Alternate topology order in the freeze/recovery block so topology
        # is not tied to one fixed testing sequence.
        patterns = (
            ("triangular", "graded_hourglass")
            if condition_id != "freeze_recovery"
            else ("graded_hourglass", "triangular")
        )
        for replicate in (1, 2, 3):
            for pattern in patterns:
                source = source_by_pattern_replicate[(pattern, replicate)]
                specimen_id = (
                    f"{condition_id}-{pattern}-bending-r{replicate:02d}"
                )
                if condition_id == "as_printed_direct":
                    test_order = (
                        (replicate - 1) * 2
                        + (1 if pattern == "triangular" else 2)
                    )
                elif condition_id == "sealed_room_48h":
                    test_order = (
                        7
                        + (replicate - 1) * 4
                        + (0 if pattern == "triangular" else 2)
                    )
                else:
                    test_order = (
                        7
                        + (replicate - 1) * 4
                        + (1 if pattern == "graded_hourglass" else 3)
                    )
                row: Mapping[str, object] = {
                    **source,
                    "specimen_id": specimen_id,
                    "condition_id": condition_id,
                    "conditioning": condition["conditioning"],
                    "conditioning_protocol": condition["protocol"],
                    "plate_id": condition["plate_id"],
                    "test_block": condition["test_block"],
                    "test_order": test_order,
                    "predicted_value": fine_stiffness[pattern] * factor,
                    "prediction_factor": factor,
                    "prediction_class": (
                        "toolpath_FEA_scaled_by_published_cubic_summary"
                        if factor != 1.0
                        else "room_temperature_toolpath_FEA"
                    ),
                    "source_plate_3mf": str(source_plate_path.resolve()),
                    "required_measurements": (
                        "actual_mass_g, time_s, displacement_mm, force_N, "
                        "failure_mode, test_temperature_C, "
                        "test_relative_humidity_percent"
                    ),
                }
                plan_rows.append(row)
                condition_rows.append(row)
        condition_plans[condition_id] = condition_rows

    plan_rows.sort(key=lambda row: int(row["test_order"]))
    _write_rows(handoff / "test_plan.csv", plan_rows)
    for condition_id, rows in condition_plans.items():
        _write_rows(
            handoff / "condition_plans" / f"{condition_id}.csv",
            rows,
        )
        template_dir = handoff / "capture_templates" / condition_id
        template_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            template = {field: "" for field in CAPTURE_FIELDS}
            template.update(
                {
                    "specimen_id": row["specimen_id"],
                    "pattern": row["pattern"],
                    "load_case": "bending",
                    "replicate": row["replicate"],
                    "material": "Bambu PLA Basic",
                    "conditioning": row["conditioning"],
                    "fixture_id": "bambu-pattern-bend-v1",
                    "support_span_mm": 40.0,
                    "support_roller_diameter_mm": 4.0,
                    "loading_roller_diameter_mm": 4.0,
                    "crosshead_rate_mm_min": 1.2,
                }
            )
            _write_rows(
                template_dir / f"{row['specimen_id']}.csv",
                [template],
            )

    report = f"""# Freeze/recovery physical validation handoff

Print the existing checksum-frozen six-object plate **three times** without
rescaling, rearranging, or changing its embedded process:

1. **Plate A — as printed:** test directly after printing.
2. **Plate B — time-matched control:** seal immediately and hold at laboratory
   temperature for 48 h.
3. **Plate C — freeze/recovery:** seal immediately, hold at -80 C for 24 h,
   then recover at laboratory temperature for 24 h.

Plate B and Plate C are tested in the same block. The extra time-matched plate
is intentionally stricter than the paper: it separates freezer exposure from
the 48 h elapsed-time/storage effect that the published as-printed comparison
does not control.

Use the registered 40 mm-span, 4 mm-roller fixture and the same 1.2 mm/min
crosshead rate as the existing room-temperature package. Weigh every coupon
and record at least 20 force-displacement points through 0.4 mm corrected
midpoint displacement. Fill one matching CSV per specimen.

The 18-coupon plan is a physical validation design. Its predicted values are
the unvalidated toolpath stiffnesses, with the freeze/recovery plate scaled by
the published median cubic-infill flexural-modulus factor
({median_factor:.4f}). Do not treat those absolute values as acceptance limits.
"""
    (handoff / "README.md").write_text(report, encoding="utf-8")
    return {
        "source_plate_3mf": str(source_plate_path.resolve()),
        "source_plate_3mf_sha256": _sha256(source_plate_path),
        "plate_print_count": len(PHYSICAL_CONDITIONS),
        "condition_count": len(PHYSICAL_CONDITIONS),
        "specimen_count": len(plan_rows),
        "replicates_per_pattern_condition": 3,
        "capture_template_count": len(plan_rows),
        "time_matched_control_included": True,
        "test_plan": str((handoff / "test_plan.csv").resolve()),
        "test_plan_sha256": _sha256(handoff / "test_plan.csv"),
        "readme": str((handoff / "README.md").resolve()),
    }


def run_sensitivity(
    reference_dir: Path = DEFAULT_REFERENCE,
    toolpath_dir: Path = DEFAULT_TOOLPATH,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    reference_dir = Path(reference_dir)
    toolpath_dir = Path(toolpath_dir)
    output_dir = Path(output_dir)
    if not (reference_dir / "provenance.json").exists():
        write_reference(output_dir=reference_dir)
    reference = _json(reference_dir / "provenance.json")
    toolpath = _json(toolpath_dir / "manifest.json")
    if reference.get("quality_gate_passed") is not True:
        raise ValueError("cryogenic reference quality gate did not pass")
    if (
        toolpath.get("status")
        != "toolpath_aware_relative_bending_ranking_admitted"
    ):
        raise ValueError("toolpath-aware baseline ranking is not admitted")

    effect_rows = _csv_rows(reference_dir / "conditioning_effects.csv")
    modulus_ratios = [
        float(row["ratio"])
        for row in effect_rows
        if row["metric"] == "flexural_modulus"
    ]
    if len(modulus_ratios) != 16:
        raise ValueError("expected 16 flexural-modulus correction factors")

    result_rows = _csv_rows(toolpath_dir / "results.csv")
    fine_resolution = max(int(row["analysis_resolution_xy"]) for row in result_rows)
    fine = {
        row["pattern"]: float(row["stiffness_N_per_mm"])
        for row in result_rows
        if int(row["analysis_resolution_xy"]) == fine_resolution
        and row["status"] == "ok"
    }
    if set(fine) != {"triangular", "graded_hourglass"}:
        raise ValueError("fine toolpath stiffness pair is incomplete")

    minimum_factor = min(modulus_ratios)
    maximum_factor = max(modulus_ratios)
    baseline_ratio = (
        fine["graded_hourglass"] / fine["triangular"]
    )
    common_factor_ratio = (
        fine["graded_hourglass"] * minimum_factor
        / (fine["triangular"] * minimum_factor)
    )
    adversarial_ratio = (
        fine["graded_hourglass"] * minimum_factor
        / (fine["triangular"] * maximum_factor)
    )
    ranking_survives = adversarial_ratio > 1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    screen_rows: List[Mapping[str, object]] = []
    for pattern, stiffness in fine.items():
        screen_rows.append(
            {
                "pattern": pattern,
                "baseline_toolpath_stiffness_N_per_mm": stiffness,
                "common_factor_low_stiffness_N_per_mm": (
                    stiffness * minimum_factor
                ),
                "common_factor_high_stiffness_N_per_mm": (
                    stiffness * maximum_factor
                ),
                "published_factor_min": minimum_factor,
                "published_factor_max": maximum_factor,
                "interpretation": (
                    "scalar freeze/recovery sensitivity only; not a "
                    "topology-specific calibrated prediction"
                ),
            }
        )
    with (output_dir / "screen.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(screen_rows[0])
        )
        writer.writeheader()
        writer.writerows(screen_rows)
    ordered_factors = sorted(modulus_ratios)
    midpoint = len(ordered_factors) // 2
    median_factor = (
        ordered_factors[midpoint - 1] + ordered_factors[midpoint]
    ) / 2.0
    physical_handoff = _write_physical_handoff(
        output_dir, fine, median_factor
    )

    payload: Dict[str, object] = {
        "study_id": "graded_hourglass_cryo_recovery_transfer_screen_v1",
        "material": "Bambu PLA Basic",
        "status": (
            "ranking_survives_published_freeze_recovery_envelope"
            if ranking_survives
            else "ranking_can_reverse_within_published_freeze_recovery_envelope"
        ),
        "source_reference": str(
            (reference_dir / "provenance.json").resolve()
        ),
        "source_reference_sha256": _sha256(
            reference_dir / "provenance.json"
        ),
        "source_toolpath_manifest": str(
            (toolpath_dir / "manifest.json").resolve()
        ),
        "source_toolpath_manifest_sha256": _sha256(
            toolpath_dir / "manifest.json"
        ),
        "baseline": {
            "analysis_resolution_xy": fine_resolution,
            "triangular_stiffness_N_per_mm": fine["triangular"],
            "graded_hourglass_stiffness_N_per_mm": (
                fine["graded_hourglass"]
            ),
            "custom_to_control_ratio": baseline_ratio,
        },
        "published_flexural_modulus_correction": {
            "configuration_count": len(modulus_ratios),
            "minimum_ratio": minimum_factor,
            "maximum_ratio": maximum_factor,
            "source_matrix": (
                "0/30/60/90 degree raster angles x 20/40/60/80% cubic "
                "infill; -80 C 24 h plus 24 h room-temperature recovery"
            ),
        },
        "screens": {
            "same_factor_applied_to_both_patterns_ratio": (
                common_factor_ratio
            ),
            "adversarial_custom_min_control_max_ratio": adversarial_ratio,
            "ranking_survives_adversarial_factor_assignment": (
                ranking_survives
            ),
        },
        "physical_handoff": physical_handoff,
        "interpretation": (
            "The exact-brand paper supplies a measured room-temperature "
            "freeze/recovery correction envelope. Applying the same scalar "
            "factor preserves the linear ranking by construction. Assigning "
            "the worst published factor to the custom pattern and the best "
            "to the control is an intentionally conservative transfer test."
        ),
        "restrictions": [
            (
                "The published conditioning matrix used cubic infill, not "
                "triangular or graded-hourglass topology."
            ),
            (
                "This sensitivity changes the elastic scale only; it does "
                "not model cryogenic residual stress, interface damage, or "
                "topology-dependent conditioning."
            ),
            (
                "The specimens were tested after warming to room "
                "temperature, not while at -80 C."
            ),
            (
                "No raw curves or numeric uncertainty are available, so "
                "absolute stiffness remains unvalidated."
            ),
            (
                "Strength, fracture, fatigue, creep, and repeated thermal "
                "cycling remain unsupported."
            ),
        ],
        "gate": {
            "reference_quality_gate_passed": True,
            "toolpath_baseline_gate_passed": True,
            "sixteen_flexural_modulus_factors_present": True,
            "ranking_survives_adversarial_factor_assignment": (
                ranking_survives
            ),
            "absolute_freeze_recovery_prediction_validated": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Graded-hourglass freeze/recovery sensitivity

The fine frozen-toolpath model gives a graded-hourglass/triangular bending
stiffness ratio of **{baseline_ratio:.4f}**.

The exact-brand Bambu PLA Basic paper reports recovered/as-printed flexural
modulus ratios from **{minimum_factor:.4f} to {maximum_factor:.4f}** across
16 cubic-infill angle/density configurations after -80 C for 24 h and 24 h
room-temperature recovery.

- Applying one common measured factor to both patterns preserves the ratio at
  **{common_factor_ratio:.4f}**.
- The deliberately adversarial transfer—worst factor on graded hourglass,
  best factor on triangular—still gives **{adversarial_ratio:.4f}**.

Decision: **{payload['status']}**.

This is a bounded ranking stress test, not a calibrated cryogenic topology
model. The paper used cubic infill and room-temperature tests after recovery;
physical pattern coupons are still required before claiming the effect for
the graded-hourglass geometry.

The generated physical handoff uses the frozen six-object 3MF three times:
as-printed, sealed room-temperature 48 h, and freeze/recovery. That adds a
time-matched control missing from the published comparison and yields
**{physical_handoff['specimen_count']} coupons** with one capture template
per specimen.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen the custom bending ranking after freeze/recovery"
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--toolpath", type=Path, default=DEFAULT_TOOLPATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_sensitivity(
        args.reference, args.toolpath, args.output
    )
    print(
        json.dumps(
            {
                "study_id": result["study_id"],
                "status": result["status"],
                "baseline_ratio": result["baseline"][
                    "custom_to_control_ratio"
                ],
                "adversarial_ratio": result["screens"][
                    "adversarial_custom_min_control_max_ratio"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
