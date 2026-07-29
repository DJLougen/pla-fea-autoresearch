from pathlib import Path

import pytest

from src.bambu_common_infill_reference import (
    DEFAULT_SOURCE,
    FIGURE_SHA256,
    PATTERNS,
    PLOT_BOXES,
    digitize_figure,
    import_reference,
    mass_records,
    validate_curves,
    verify_source,
)


def test_source_checksum_and_dimensions() -> None:
    result = verify_source()
    assert result["sha256"] == FIGURE_SHA256
    assert result["shape"] == [292, 850, 3]


def test_all_nine_curves_pass_digitization_gate() -> None:
    curves = digitize_figure(
        DEFAULT_SOURCE / "figure-4-density-comparison.png"
    )
    assert set(curves) == {
        (density, pattern)
        for density in PLOT_BOXES
        for pattern in PATTERNS
    }
    rows = validate_curves(curves)
    assert len(rows) == 9
    assert all(row["quality_gate_passed"] for row in rows)
    assert max(
        abs(row["x1000_peak_to_reported_yield_relative_error"])
        for row in rows
    ) <= 0.04
    for density in PLOT_BOXES:
        selected = {
            row["pattern"]: row
            for row in rows
            if row["nominal_infill_percent"] == density
        }
        assert (
            selected["grid"]["initial_effective_modulus_Pa"]
            > selected["gyroid"]["initial_effective_modulus_Pa"]
            > selected["cubic"]["initial_effective_modulus_Pa"]
        )


def test_mass_records_preserve_table_and_infer_total_density() -> None:
    rows = mass_records()
    assert len(rows) == 9
    grid_10 = next(
        row
        for row in rows
        if row["pattern"] == "grid"
        and row["nominal_infill_percent"] == 10
    )
    assert grid_10["mean_mass_g"] == pytest.approx(8.1162)
    assert 0.20 < grid_10["inferred_total_solid_fraction"] < 0.30


def test_import_writes_traceable_csvs(tmp_path: Path) -> None:
    result = import_reference(DEFAULT_SOURCE, tmp_path)
    assert result["quality_gate_passed"] is True
    assert len(list((tmp_path / "curves").glob("*.csv"))) == 9
    assert (tmp_path / "reported_mass_summary.csv").exists()
    assert (tmp_path / "curve_validation.csv").exists()
    assert (tmp_path / "provenance.json").exists()
    assert (tmp_path / "data_quality_report.md").exists()
