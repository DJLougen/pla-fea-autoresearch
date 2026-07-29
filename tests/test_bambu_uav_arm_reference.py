from src.bambu_uav_arm_reference import (
    TABLE3,
    digitize_figure7,
    verify_sources,
    write_reference,
)


def test_uav_arm_sources_and_all_twelve_physical_curves():
    assert len(verify_sources()) == 5
    curves, validation = digitize_figure7()
    assert set(curves) == set(TABLE3)
    assert all(len(specimens) == 3 for specimens in curves.values())
    assert len(validation) == 12
    assert all(row["passes_visible_trace_gate"] for row in validation)
    assert all(row["passes_table_crosscheck"] for row in validation)
    assert max(
        abs(row["average_peak_relative_error"]) for row in validation
    ) < 0.01


def test_uav_arm_writer_preserves_external_heldout_boundary(tmp_path):
    provenance = write_reference(output_dir=tmp_path)
    assert provenance["curve_quality_gate_passed"] is True
    assert provenance["physical_specimen_curve_count"] == 12
    assert "prohibited from material calibration" in provenance["allowed_use"]
    assert len(list((tmp_path / "curves").glob("*.csv"))) == 12
