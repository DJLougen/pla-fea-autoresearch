import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.custom_pattern_validation import (
    CAPTURE_FIELDS,
    analyze_submission,
    validate_submission,
    write_capture_package,
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


def _write_submission(
    directory: Path,
    package: Path,
    *,
    custom_slopes=(81.0, 82.0, 83.0),
    control_slopes=(68.0, 69.0, 70.0),
    mismatched_machine: bool = False,
) -> None:
    directory.mkdir()
    with (package / "frozen_plan.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        plan = list(csv.DictReader(handle))
    slope_lookup = {
        "graded_hourglass": custom_slopes,
        "triangular": control_slopes,
    }
    for row in plan:
        replicate = int(row["replicate"])
        slope = slope_lookup[row["pattern"]][replicate - 1]
        displacement = np.linspace(0.0, 0.4, 41)
        machine = (
            "x1c-02"
            if mismatched_machine
            and row["specimen_id"] == "graded_hourglass-bending-r03"
            else "x1c-01"
        )
        shared = {
            "specimen_id": row["specimen_id"],
            "pattern": row["pattern"],
            "load_case": row["load_case"],
            "replicate": replicate,
            "material": "Bambu PLA Basic",
            "material_lot": "lot-a",
            "material_color": "jade-white",
            "machine_id": machine,
            "slicer_profile": "pattern-validation-v1",
            "drying_protocol": "8 h at 45 C",
            "conditioning": "as_printed_dry",
            "fixture_id": "bambu-pattern-bend-v1",
            "nozzle_temperature_C": 220.0,
            "bed_temperature_C": 55.0,
            "layer_height_mm": 0.2,
            "actual_mass_g": float(row["target_actual_mass_g"]),
            "test_temperature_C": 23.0,
            "test_relative_humidity_percent": 20.0,
            "support_span_mm": 40.0,
            "support_roller_diameter_mm": 4.0,
            "loading_roller_diameter_mm": 4.0,
            "crosshead_rate_mm_min": 1.2,
            "fixture_compliance_mm_per_N": 0.0002,
            "failure_mode": "none_at_endpoint",
            "notes": "",
        }
        rows = [
            {
                **shared,
                "time_s": float(index) * 0.1,
                "displacement_mm": float(value),
                "force_N": float(slope * value),
            }
            for index, value in enumerate(displacement)
        ]
        pd.DataFrame(rows, columns=CAPTURE_FIELDS).to_csv(
            directory / f"{row['specimen_id']}.csv", index=False
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


def test_capture_package_freezes_six_templates(tmp_path: Path):
    package = _package(tmp_path)
    templates = list((package / "capture_templates").glob("*.csv"))
    assert len(templates) == 6
    assert (package / "frozen_plan.csv").exists()
    assert (package / "manifest.json").exists()


def test_blank_templates_are_not_treated_as_measurements(tmp_path: Path):
    package = _package(tmp_path)
    audit = validate_submission(
        package / "capture_templates",
        package / "frozen_plan.csv",
        STUDY,
        package / "manifest.json",
    )
    assert audit["ready_for_analysis"] is False
    assert audit["counts"] == {"valid": 0, "invalid": 6, "missing": 0}


def test_valid_curves_confirm_predicted_specialist_advantage(
    tmp_path: Path,
):
    package = _package(tmp_path)
    submission = tmp_path / "submission"
    _write_submission(submission, package)
    payload = analyze_submission(
        submission,
        package / "frozen_plan.csv",
        STUDY,
        package / "manifest.json",
    )
    assert payload["status"] == "analyzed"
    result = payload["result"]
    assert result["decision"] == "validated_specialist_advantage"
    assert result["bootstrap_95pct_ratio_interval"][0] > 1.0
    assert result["canonical_material_card_updated"] is False


def test_reversed_physical_ranking_is_rejected(tmp_path: Path):
    package = _package(tmp_path)
    submission = tmp_path / "submission"
    _write_submission(
        submission,
        package,
        custom_slopes=(58.0, 59.0, 60.0),
    )
    payload = analyze_submission(
        submission,
        package / "frozen_plan.csv",
        STUDY,
        package / "manifest.json",
    )
    assert payload["result"]["decision"] == "ranking_rejected"
    assert payload["result"]["bootstrap_95pct_ratio_interval"][1] < 1.0


def test_print_control_drift_blocks_analysis(tmp_path: Path):
    package = _package(tmp_path)
    submission = tmp_path / "submission"
    _write_submission(
        submission, package, mismatched_machine=True
    )
    audit = validate_submission(
        submission,
        package / "frozen_plan.csv",
        STUDY,
        package / "manifest.json",
    )
    assert audit["ready_for_analysis"] is False
    assert any("machine_id" in issue for issue in audit["global_issues"])


def test_frozen_plan_checksum_is_enforced(tmp_path: Path):
    package = _package(tmp_path)
    frozen = package / "frozen_plan.csv"
    frozen.write_text(
        frozen.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum"):
        validate_submission(
            package / "capture_templates",
            frozen,
            STUDY,
            package / "manifest.json",
        )
