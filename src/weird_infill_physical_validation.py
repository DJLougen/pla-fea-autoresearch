from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOT = (
    PROJECT_ROOT / "results/custom_infill/weird_kite_search"
)
DEFAULT_PLAN = (
    SEARCH_ROOT
    / "physical_validation/manufacturing/physical_test_plan.csv"
)
DEFAULT_CAPTURE = (
    SEARCH_ROOT
    / "physical_validation/manufacturing/measurement_capture.csv"
)
DEFAULT_TOOLPATH = SEARCH_ROOT / "toolpath_validation/summary.json"
DEFAULT_OUTPUT = SEARCH_ROOT / "physical_validation/evaluation"
PATTERNS = (
    "triangular",
    "kite_08_08_08_08",
    "kite_08_08_12_08",
)
REPLICATES = 3
MAXIMUM_GROUP_CV = 0.10
MAXIMUM_MEAN_MASS_SPREAD = 0.03
MAXIMUM_RATIO_ERROR = 0.20


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _float(row: Mapping[str, str], field: str) -> float | None:
    text = str(row.get(field, "")).strip()
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _group_stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1))
    return {
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def build_protocol() -> Dict[str, object]:
    return {
        "protocol_id": "bambu_pla_weird_kite_bending_modal_v1",
        "material": "Bambu PLA Basic",
        "specimen_count": len(PATTERNS) * REPLICATES,
        "conditioning": (
            "same spool and drying treatment; store all coupons together "
            "for at least 24 h before testing"
        ),
        "test_order": (
            "follow physical_test_plan.csv; modal test first, then bending"
        ),
        "modal_test": {
            "boundary": (
                "coupon resting on the same two 40 mm-span support rollers "
                "used by the bending fixture; no center preload"
            ),
            "excitation": (
                "five light out-of-plane taps near coupon center, or a "
                "low-amplitude swept-sine excitation"
            ),
            "response_sensor": (
                "prefer non-contact laser vibrometry; if using an "
                "accelerometer, document its mass and attachment"
            ),
            "minimum_sample_rate_Hz": 10_000,
            "analysis_band_Hz": [500, 1_600],
            "reported_quantity": (
                "dominant out-of-plane peak and optional half-power "
                "loss factor from the averaged response spectrum"
            ),
            "repeat_requirement": (
                "five excitations per coupon; reject a peak unless it "
                "appears in at least four"
            ),
        },
        "bending_test": {
            "fixture": "existing bambu-pattern-bend-v1 three-point fixture",
            "support_span_mm": 40.0,
            "loading_location": "midspan",
            "preload_N": 0.5,
            "crosshead_rate_mm_min": 1.2,
            "corrected_endpoint_displacement_mm": 0.4,
            "stiffness_fit_window_mm": [0.05, 0.30],
            "required_capture": (
                "actual mass, corrected force-displacement curve, fitted "
                "stiffness, and visible damage flag"
            ),
        },
        "acceptance": {
            "replicates_per_pattern": REPLICATES,
            "maximum_group_coefficient_of_variation": MAXIMUM_GROUP_CV,
            "maximum_pattern_mean_mass_spread": MAXIMUM_MEAN_MASS_SPREAD,
            "maximum_candidate_to_control_ratio_error": MAXIMUM_RATIO_ERROR,
            "required_order": (
                "kite_08_08_08_08 > kite_08_08_12_08 > triangular "
                "for both bending stiffness and modal frequency"
            ),
        },
        "safety": [
            "wear eye protection and use a fragment shield during bending",
            "stop on a visible crack or unstable force drop",
            "do not treat this research comparison as product certification",
        ],
    }


def evaluate_measurements(
    plan_path: Path = DEFAULT_PLAN,
    capture_path: Path = DEFAULT_CAPTURE,
    toolpath_path: Path = DEFAULT_TOOLPATH,
) -> Dict[str, object]:
    plan = _read_csv(plan_path)
    capture = _read_csv(capture_path)
    toolpath = _json(toolpath_path)
    expected = {row["specimen_id"]: row for row in plan}
    observed = {row["specimen_id"]: row for row in capture}
    if len(expected) != len(plan):
        raise ValueError("physical plan contains duplicate specimen IDs")
    if len(observed) != len(capture):
        raise ValueError("measurement capture contains duplicate specimen IDs")
    if set(observed) != set(expected):
        raise ValueError(
            "measurement capture specimen IDs do not match the frozen plan"
        )
    required_numeric = (
        "actual_mass_g",
        "ambient_temperature_C",
        "relative_humidity_percent",
        "measured_out_of_plane_frequency_Hz",
        "bending_stiffness_N_per_mm",
    )
    missing_cells = [
        f"{specimen}:{field}"
        for specimen, row in observed.items()
        for field in required_numeric
        if _float(row, field) is None
    ]
    if missing_cells:
        return {
            "status": "awaiting_measurements",
            "expected_specimens": len(expected),
            "complete_specimens": sum(
                all(_float(row, field) is not None for field in required_numeric)
                for row in observed.values()
            ),
            "missing_numeric_cell_count": len(missing_cells),
            "first_missing_cells": missing_cells[:12],
            "physical_validation_gate_passed": False,
        }
    invalid = []
    records = []
    for specimen, row in observed.items():
        planned = expected[specimen]
        values = {field: float(_float(row, field)) for field in required_numeric}
        loss = _float(row, "measured_half_power_loss_factor")
        if not 0.5 <= values["actual_mass_g"] <= 10.0:
            invalid.append(f"{specimen}: implausible mass")
        if not 10.0 <= values["ambient_temperature_C"] <= 35.0:
            invalid.append(f"{specimen}: implausible temperature")
        if not 0.0 <= values["relative_humidity_percent"] <= 100.0:
            invalid.append(f"{specimen}: invalid humidity")
        if not 100.0 <= values["measured_out_of_plane_frequency_Hz"] <= 5_000.0:
            invalid.append(f"{specimen}: implausible modal frequency")
        if not 1.0 <= values["bending_stiffness_N_per_mm"] <= 1_000.0:
            invalid.append(f"{specimen}: implausible bending stiffness")
        if loss is not None and not 0.0 < loss < 1.0:
            invalid.append(f"{specimen}: invalid loss factor")
        records.append(
            {
                "specimen_id": specimen,
                "pattern": planned["pattern"],
                **values,
                "measured_half_power_loss_factor": loss,
                "failure_or_damage_observed": row.get(
                    "failure_or_damage_observed", ""
                ),
            }
        )
    if invalid:
        return {
            "status": "invalid_measurements",
            "validation_errors": invalid,
            "physical_validation_gate_passed": False,
        }
    groups = {}
    for pattern in PATTERNS:
        rows = [row for row in records if row["pattern"] == pattern]
        if len(rows) != REPLICATES:
            raise ValueError(f"expected three records for {pattern}")
        groups[pattern] = {
            "actual_mass_g": _group_stats(
                [row["actual_mass_g"] for row in rows]
            ),
            "bending_stiffness_N_per_mm": _group_stats(
                [row["bending_stiffness_N_per_mm"] for row in rows]
            ),
            "out_of_plane_frequency_Hz": _group_stats(
                [row["measured_out_of_plane_frequency_Hz"] for row in rows]
            ),
        }
    predicted = toolpath["fine_toolpath_predictions"]
    control = "triangular"
    comparisons = {}
    for pattern in PATTERNS:
        measured_bending_ratio = (
            groups[pattern]["bending_stiffness_N_per_mm"]["mean"]
            / groups[control]["bending_stiffness_N_per_mm"]["mean"]
        )
        measured_modal_ratio = (
            groups[pattern]["out_of_plane_frequency_Hz"]["mean"]
            / groups[control]["out_of_plane_frequency_Hz"]["mean"]
        )
        predicted_bending_ratio = (
            float(predicted[pattern]["bending_stiffness_N_per_mm"])
            / float(predicted[control]["bending_stiffness_N_per_mm"])
        )
        predicted_modal_ratio = (
            float(predicted[pattern]["out_of_plane_frequency_Hz"])
            / float(predicted[control]["out_of_plane_frequency_Hz"])
        )
        comparisons[pattern] = {
            "measured_bending_ratio": measured_bending_ratio,
            "predicted_bending_ratio": predicted_bending_ratio,
            "bending_ratio_relative_error": abs(
                measured_bending_ratio / predicted_bending_ratio - 1.0
            ),
            "measured_modal_ratio": measured_modal_ratio,
            "predicted_modal_ratio": predicted_modal_ratio,
            "modal_ratio_relative_error": abs(
                measured_modal_ratio / predicted_modal_ratio - 1.0
            ),
        }
    bending_order = sorted(
        PATTERNS,
        key=lambda pattern: groups[pattern][
            "bending_stiffness_N_per_mm"
        ]["mean"],
        reverse=True,
    )
    modal_order = sorted(
        PATTERNS,
        key=lambda pattern: groups[pattern][
            "out_of_plane_frequency_Hz"
        ]["mean"],
        reverse=True,
    )
    expected_order = [
        "kite_08_08_08_08",
        "kite_08_08_12_08",
        "triangular",
    ]
    group_cv_gate = all(
        groups[pattern][metric]["coefficient_of_variation"]
        <= MAXIMUM_GROUP_CV
        for pattern in PATTERNS
        for metric in (
            "bending_stiffness_N_per_mm",
            "out_of_plane_frequency_Hz",
        )
    )
    mean_masses = [
        groups[pattern]["actual_mass_g"]["mean"] for pattern in PATTERNS
    ]
    mass_spread = max(mean_masses) / min(mean_masses) - 1.0
    ratio_gate = all(
        comparisons[pattern]["bending_ratio_relative_error"]
        <= MAXIMUM_RATIO_ERROR
        and comparisons[pattern]["modal_ratio_relative_error"]
        <= MAXIMUM_RATIO_ERROR
        for pattern in PATTERNS
        if pattern != control
    )
    ranking_gate = (
        bending_order == expected_order and modal_order == expected_order
    )
    gate = (
        group_cv_gate
        and mass_spread <= MAXIMUM_MEAN_MASS_SPREAD
        and ratio_gate
        and ranking_gate
    )
    return {
        "status": (
            "physical_validation_passed"
            if gate
            else "physical_validation_failed"
        ),
        "expected_specimens": len(expected),
        "complete_specimens": len(records),
        "group_statistics": groups,
        "comparisons": comparisons,
        "measured_bending_order": bending_order,
        "measured_modal_order": modal_order,
        "maximum_pattern_mean_mass_spread": mass_spread,
        "gates": {
            "all_group_CVs_at_most_10_percent": group_cv_gate,
            "pattern_mean_mass_spread_at_most_3_percent": (
                mass_spread <= MAXIMUM_MEAN_MASS_SPREAD
            ),
            "candidate_ratio_errors_at_most_20_percent": ratio_gate,
            "predicted_static_and_modal_order_reproduced": ranking_gate,
            "physical_validation_gate_passed": gate,
        },
        "physical_validation_gate_passed": gate,
    }


def write_validation(
    output_dir: Path = DEFAULT_OUTPUT,
    plan_path: Path = DEFAULT_PLAN,
    capture_path: Path = DEFAULT_CAPTURE,
    toolpath_path: Path = DEFAULT_TOOLPATH,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol()
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    result = evaluate_measurements(plan_path, capture_path, toolpath_path)
    result["inputs"] = {
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": _sha256(plan_path),
        "capture": str(Path(capture_path).resolve()),
        "capture_sha256": _sha256(capture_path),
        "toolpath_reference": str(Path(toolpath_path).resolve()),
        "toolpath_reference_sha256": _sha256(toolpath_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Weird-kite physical validation intake",
        "",
        f"Status: **{result['status']}**.",
        "",
    ]
    if result["status"] == "awaiting_measurements":
        lines.extend(
            [
                (
                    f"{result['complete_specimens']}/"
                    f"{result['expected_specimens']} specimen rows are "
                    "complete."
                ),
                "",
                "Fill `measurement_capture.csv` after testing and rerun this "
                "module. Blank cells are never interpreted as simulated or "
                "passing measurements.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Physical gate: **{result['physical_validation_gate_passed']}**.",
                "",
                "See `summary.json` for group statistics, measured ordering, "
                "and candidate/control ratio errors.",
                "",
            ]
        )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the weird-kite modal and bending measurements"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--toolpath", type=Path, default=DEFAULT_TOOLPATH)
    args = parser.parse_args()
    result = write_validation(
        args.output, args.plan, args.capture, args.toolpath
    )
    print(
        f"weird-kite physical intake: {result['status']}; "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
