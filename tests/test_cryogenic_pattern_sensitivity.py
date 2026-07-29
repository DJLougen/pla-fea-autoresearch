import csv
from pathlib import Path

from src.bambu_cryogenic_reference import write_reference
from src.cryogenic_pattern_sensitivity import run_sensitivity


def test_published_freeze_recovery_envelope_does_not_reverse_ranking(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    output = tmp_path / "screen"
    write_reference(output_dir=reference)
    result = run_sensitivity(reference_dir=reference, output_dir=output)
    assert result["gate"]["reference_quality_gate_passed"] is True
    assert result["gate"]["toolpath_baseline_gate_passed"] is True
    assert (
        result["published_flexural_modulus_correction"][
            "configuration_count"
        ]
        == 16
    )
    assert result["baseline"]["custom_to_control_ratio"] > 1.26
    assert (
        result["screens"]["adversarial_custom_min_control_max_ratio"]
        > 1.07
    )
    assert (
        result["screens"][
            "ranking_survives_adversarial_factor_assignment"
        ]
        is True
    )
    assert (
        result["gate"]["absolute_freeze_recovery_prediction_validated"]
        is False
    )
    assert result["physical_handoff"]["plate_print_count"] == 3
    assert result["physical_handoff"]["specimen_count"] == 18
    assert result["physical_handoff"]["capture_template_count"] == 18
    assert result["physical_handoff"]["time_matched_control_included"] is True
    assert (output / "summary.json").exists()
    assert (output / "screen.csv").exists()
    assert (output / "physical_handoff/test_plan.csv").exists()
    with (output / "physical_handoff/test_plan.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        plan = list(csv.DictReader(handle))
    assert [int(row["test_order"]) for row in plan] == list(range(1, 19))
    assert {
        row["condition_id"] for row in plan if int(row["test_order"]) > 6
    } == {"sealed_room_48h", "freeze_recovery"}
    assert len(
        list(
            (
                output / "physical_handoff/capture_templates"
            ).glob("*/*.csv")
        )
    ) == 18
