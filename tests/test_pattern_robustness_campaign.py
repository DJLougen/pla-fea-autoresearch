import csv
import json
from pathlib import Path

import pytest

from src.infill_benchmark import LOAD_CASES, nominal_strut_width
from src.pattern_robustness_campaign import (
    HEIGHT_MM,
    RELATIVE_DENSITY,
    WIDTH_MM,
    aggregate_robustness,
    build_perturbation_matrix,
    evaluate_finalist_measurements,
    finalist_geometries,
    rotate_quarter_turn,
)


def test_quarter_turn_preserves_centerline_length_and_domain():
    geometry = finalist_geometries()["bowtie_fan"]
    rotated = rotate_quarter_turn(geometry, "bowtie_rotated")
    assert rotated.total_length_mm == pytest.approx(
        geometry.total_length_mm
    )
    assert all(
        0.0 <= x <= WIDTH_MM and 0.0 <= y <= HEIGHT_MM
        for x, y in rotated.nodes
    )


def test_missing_strut_variants_do_not_get_mass_compensation():
    matrix = build_perturbation_matrix()
    missing = next(
        variant
        for variant, row in matrix["metadata"].items()
        if row["scenario"] == "single_missing_interior_strut"
        and row["base_pattern"] == "bowtie_fan"
    )
    base = matrix["base_geometries"]["bowtie_fan"]
    variant = matrix["geometries"][missing]
    nominal_width = nominal_strut_width(
        base, WIDTH_MM, HEIGHT_MM, RELATIVE_DENSITY
    )
    assert matrix["strut_widths"][missing] == pytest.approx(nominal_width)
    assert variant.total_length_mm < base.total_length_mm
    variant_density = (
        variant.total_length_mm
        * matrix["strut_widths"][missing]
        / (WIDTH_MM * HEIGHT_MM)
    )
    assert variant_density < RELATIVE_DENSITY


def test_robust_aggregation_penalizes_orientation_and_missing_struts():
    matrix = build_perturbation_matrix()
    base_factors = {
        "triangular": {
            "tension": 1.0,
            "shear": 1.0,
            "bending": 1.0,
            "compression": 1.0,
            "buckling": 1.0,
        },
        "kite_08_08_08_08": {
            "tension": 0.7,
            "shear": 1.1,
            "bending": 1.4,
            "compression": 0.7,
            "buckling": 0.75,
        },
        "kite_08_08_12_08": {
            "tension": 0.72,
            "shear": 1.05,
            "bending": 1.35,
            "compression": 0.72,
            "buckling": 0.78,
        },
        "bowtie_fan": {
            "tension": 0.78,
            "shear": 1.0,
            "bending": 1.45,
            "compression": 0.78,
            "buckling": 0.82,
        },
    }
    rows = []
    for variant, metadata in matrix["metadata"].items():
        for load_case in LOAD_CASES:
            value = 100.0 * base_factors[metadata["base_pattern"]][load_case]
            value *= float(metadata["bead_width_scale"])
            if metadata["orientation_degrees"] == 90:
                value *= 0.95
            if metadata["scenario"] == "single_missing_interior_strut":
                value *= 0.80
            rows.append(
                {
                    "pattern": variant,
                    "load_case": load_case,
                    "status": "ok",
                    (
                        "critical_buckling_load_N"
                        if load_case == "buckling"
                        else "stiffness_N_per_mm"
                    ): value,
                }
            )
    result = aggregate_robustness(
        rows, matrix["metadata"], matrix["structural_breaks"]
    )
    assert result["ranking"][0]["pattern"] == "triangular"
    bowtie = next(
        row
        for row in result["ranking"]
        if row["pattern"] == "bowtie_fan"
    )
    assert bowtie["conservative_bending_ratio"] > 1.0
    assert bowtie["robust_worst_load_ratio"] < 1.0
    assert len(result["physical_candidates"]) == 2


def _write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_physical_evaluator_rejects_blank_and_accepts_matched_groups(
    tmp_path: Path,
):
    patterns = (
        "triangular",
        "kite_08_08_08_08",
        "kite_08_08_12_08",
        "bowtie_fan",
    )
    predicted = {
        "triangular": 1.0,
        "kite_08_08_08_08": 1.40,
        "kite_08_08_12_08": 1.36,
        "bowtie_fan": 1.38,
    }
    plan = []
    capture = []
    for replicate in (1, 2, 3):
        for pattern in patterns:
            specimen = f"{pattern}-{replicate}"
            plan.append(
                {
                    "specimen_id": specimen,
                    "pattern": pattern,
                    "replicate": replicate,
                }
            )
            capture.append(
                {
                    "specimen_id": specimen,
                    "actual_mass_g": "",
                    "ambient_temperature_C": "",
                    "relative_humidity_percent": "",
                    "bending_stiffness_N_per_mm": "",
                }
            )
    plan_path = tmp_path / "plan.csv"
    capture_path = tmp_path / "capture.csv"
    toolpath_path = tmp_path / "toolpath.json"
    _write_rows(plan_path, plan)
    _write_rows(capture_path, capture)
    toolpath_path.write_text(
        json.dumps(
            {
                "gates": {
                    "candidate_set_admitted_for_physical_testing": True
                },
                "fine_predictions": {
                    pattern: {"to_triangular_ratio": ratio}
                    for pattern, ratio in predicted.items()
                },
            }
        ),
        encoding="utf-8",
    )
    awaiting = evaluate_finalist_measurements(
        plan_path, capture_path, toolpath_path
    )
    assert awaiting["status"] == "awaiting_measurements"
    assert awaiting["physical_validation_gate_passed"] is False

    for row in capture:
        row["actual_mass_g"] = 2.5
        row["ambient_temperature_C"] = 23.0
        row["relative_humidity_percent"] = 25.0
        row["bending_stiffness_N_per_mm"] = (
            60.0 * predicted[next(
                item["pattern"]
                for item in plan
                if item["specimen_id"] == row["specimen_id"]
            )]
        )
    _write_rows(capture_path, capture)
    validated = evaluate_finalist_measurements(
        plan_path, capture_path, toolpath_path
    )
    assert validated["status"] == "validated"
    assert validated["physical_validation_gate_passed"] is True
