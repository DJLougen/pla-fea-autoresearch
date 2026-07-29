import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.custom_pattern_validation import (
    analyze_submission,
    write_capture_package,
)
from src.manual_bending_capture import (
    MANUAL_CAPTURE_FIELDS,
    STANDARD_GRAVITY_M_S2,
    convert_manual_capture,
    initialize_manual_templates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    PROJECT_ROOT
    / "results/custom_infill/graded_hourglass/"
    "first_physical_test_plan.csv"
)
STUDY = (
    PROJECT_ROOT
    / "results/custom_infill/graded_hourglass/study.json"
)


def _package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    write_capture_package(
        package,
        PLAN,
        STUDY,
        include_manufacturing=False,
        include_fixture=False,
    )
    return package


def _write_manual_submission(
    directory: Path,
    package: Path,
    *,
    displacement_method: str = "local_indicator",
    force_calibration_error_N: float = 0.03,
) -> None:
    initialize_manual_templates(package, directory)
    with (package / "frozen_plan.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        plan = list(csv.DictReader(handle))
    slopes = {
        "graded_hourglass": (81.0, 82.0, 83.0),
        "triangular": (68.0, 69.0, 70.0),
    }
    compliance = 0.0002
    time = np.linspace(0.0, 20.0, 201)
    corrected_displacement = np.linspace(0.0, 0.4, len(time))
    for expected in plan:
        replicate = int(expected["replicate"])
        force = (
            slopes[expected["pattern"]][replicate - 1]
            * corrected_displacement
        )
        measured_displacement = corrected_displacement.copy()
        if displacement_method == "actuator_corrected":
            measured_displacement = (
                measured_displacement + compliance * force
            )
        shared = {
            "specimen_id": expected["specimen_id"],
            "pattern": expected["pattern"],
            "load_case": expected["load_case"],
            "replicate": replicate,
            "material": "Bambu PLA Basic",
            "material_lot": "lot-a",
            "material_color": "jade-white",
            "machine_id": "x1c-01",
            "slicer_profile": "pattern-validation-v1",
            "drying_protocol": "8 h at 45 C",
            "conditioning": "as_printed_dry",
            "fixture_id": "bambu-pattern-bend-v1",
            "nozzle_temperature_C": 220.0,
            "bed_temperature_C": 55.0,
            "layer_height_mm": 0.2,
            "actual_mass_g": float(expected["target_actual_mass_g"]),
            "test_temperature_C": 23.0,
            "test_relative_humidity_percent": 20.0,
            "support_span_mm": 40.0,
            "support_roller_diameter_mm": 4.0,
            "loading_roller_diameter_mm": 4.0,
            "crosshead_rate_mm_min": 1.2,
            "fixture_compliance_mm_per_N": compliance,
            "failure_mode": "none_at_endpoint",
            "notes": "",
            "force_measurement_method": "scale_mass_g",
            "force_calibration_id": "known-masses-2026-07-28",
            "force_zero_input": 0.0,
            "force_resolution_N": STANDARD_GRAVITY_M_S2 / 1000.0,
            "force_calibration_max_error_N": (
                force_calibration_error_N
            ),
            "force_capacity_N": 50.0,
            "displacement_measurement_method": displacement_method,
            "displacement_calibration_id": "gauge-block-2026-07-28",
            "displacement_zero_mm": 0.0,
            "displacement_resolution_mm": 0.005,
            "displacement_calibration_max_error_mm": 0.002,
            "fixture_compliance_calibration_id": (
                "steel-bridge-2026-07-28"
            ),
            "fixture_compliance_uncertainty_mm_per_N": 0.00005,
        }
        rows = [
            {
                **shared,
                "raw_time_s": float(time[index]),
                "raw_force_input": float(
                    force[index] * 1000.0 / STANDARD_GRAVITY_M_S2
                ),
                "raw_displacement_mm": float(
                    measured_displacement[index]
                ),
            }
            for index in range(len(time))
        ]
        pd.DataFrame(rows, columns=MANUAL_CAPTURE_FIELDS).to_csv(
            directory / f"{expected['specimen_id']}.csv", index=False
        )


def test_initializes_six_low_cost_templates(tmp_path: Path):
    package = _package(tmp_path)
    templates = tmp_path / "manual"
    manifest = initialize_manual_templates(package, templates)
    assert manifest["specimen_count"] == 6
    assert manifest["status"] == "blank_templates_only"
    assert len(list(templates.glob("*.csv"))) == 6


def test_scale_and_local_indicator_pass_unchanged_analysis(tmp_path: Path):
    package = _package(tmp_path)
    manual = tmp_path / "manual"
    _write_manual_submission(manual, package)
    converted = tmp_path / "converted"
    manifest = convert_manual_capture(manual, package, converted)
    assert manifest["status"] == "converted_and_canonical_gate_passed"
    assert manifest["canonical_validation"]["ready_for_analysis"] is True
    result = analyze_submission(
        converted,
        package / "frozen_plan.csv",
        STUDY,
        package / "manifest.json",
    )
    assert (
        result["result"]["decision"]
        == "validated_specialist_advantage"
    )


def test_actuator_compliance_correction_recovers_stiffness(tmp_path: Path):
    package = _package(tmp_path)
    manual = tmp_path / "manual"
    _write_manual_submission(
        manual,
        package,
        displacement_method="actuator_corrected",
    )
    converted = tmp_path / "converted"
    convert_manual_capture(manual, package, converted)
    result = analyze_submission(
        converted,
        package / "frozen_plan.csv",
        STUDY,
        package / "manifest.json",
    )
    custom = result["result"]["group_summaries"]["graded_hourglass"]
    assert custom["raw_mean_stiffness_N_per_mm"] == pytest.approx(82.0)


def test_excessive_force_uncertainty_is_rejected(tmp_path: Path):
    package = _package(tmp_path)
    manual = tmp_path / "manual"
    _write_manual_submission(
        manual,
        package,
        force_calibration_error_N=0.30,
    )
    with pytest.raises(ValueError, match="force uncertainty"):
        convert_manual_capture(manual, package, tmp_path / "converted")


def test_missing_frozen_specimen_is_rejected(tmp_path: Path):
    package = _package(tmp_path)
    manual = tmp_path / "manual"
    _write_manual_submission(manual, package)
    first = next(manual.glob("*.csv"))
    first.unlink()
    with pytest.raises(ValueError, match="specimen set mismatch"):
        convert_manual_capture(manual, package, tmp_path / "converted")
