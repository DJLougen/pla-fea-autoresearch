import csv
from pathlib import Path

from src.weird_infill_physical_validation import (
    DEFAULT_PLAN,
    evaluate_measurements,
)


def _write_capture(path: Path, rows) -> None:
    fields = [
        "specimen_id",
        "actual_mass_g",
        "ambient_temperature_C",
        "relative_humidity_percent",
        "measured_out_of_plane_frequency_Hz",
        "measured_half_power_loss_factor",
        "bending_stiffness_N_per_mm",
        "failure_or_damage_observed",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_blank_capture_is_explicitly_awaiting_measurements() -> None:
    result = evaluate_measurements()
    assert result["status"] == "awaiting_measurements"
    assert result["complete_specimens"] == 0
    assert result["physical_validation_gate_passed"] is False


def test_completed_synthetic_capture_passes_ratio_and_ranking_gate(
    tmp_path: Path,
) -> None:
    with DEFAULT_PLAN.open(encoding="utf-8", newline="") as handle:
        plan = list(csv.DictReader(handle))
    properties = {
        "triangular": (2.58, 1020.624, 62.544),
        "kite_08_08_08_08": (2.61, 1196.899, 88.089),
        "kite_08_08_12_08": (2.61, 1183.991, 86.628),
    }
    rows = []
    for planned in plan:
        mass, frequency, stiffness = properties[planned["pattern"]]
        scale = 1.0 + (int(planned["replicate"]) - 2) * 0.01
        rows.append(
            {
                "specimen_id": planned["specimen_id"],
                "actual_mass_g": mass * scale,
                "ambient_temperature_C": 18.0,
                "relative_humidity_percent": 40.0,
                "measured_out_of_plane_frequency_Hz": frequency * scale,
                "measured_half_power_loss_factor": 0.037,
                "bending_stiffness_N_per_mm": stiffness * scale,
                "failure_or_damage_observed": "no",
                "notes": "",
            }
        )
    capture = tmp_path / "capture.csv"
    _write_capture(capture, rows)
    result = evaluate_measurements(capture_path=capture)
    assert result["status"] == "physical_validation_passed"
    assert result["physical_validation_gate_passed"] is True
