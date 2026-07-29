from pathlib import Path

import pytest

from src.bambu_flexural_digitization import (
    CURVES,
    DEFAULT_SOURCE,
    digitize_figure4,
    parse_table3,
    validate_digitization,
    verify_sources,
)


def test_published_sources_are_checksum_pinned():
    observed = verify_sources(DEFAULT_SOURCE)
    assert set(observed) == {
        "article.xml",
        "figure-4-flexural-curves.tif",
    }


def test_table3_has_all_temperature_orientation_cells():
    rows = parse_table3(Path(DEFAULT_SOURCE) / "article.xml")
    assert len(rows) == 12
    keys = {
        (row["nozzle_temperature_C"], row["raster_orientation_deg"])
        for row in rows
    }
    assert keys == {
        (temperature, orientation)
        for temperature in (210.0, 220.0, 230.0, 240.0)
        for orientation in (0.0, 45.0, 90.0)
    }
    row = next(
        row
        for row in rows
        if row["nozzle_temperature_C"] == 220.0
        and row["raster_orientation_deg"] == 90.0
    )
    assert row["flexural_modulus_GPa"] == pytest.approx(2.98)
    assert row["flexural_strength_MPa"] == pytest.approx(57.69)


def test_figure_digitization_crosschecks_against_table3():
    curves = digitize_figure4(
        Path(DEFAULT_SOURCE) / "figure-4-flexural-curves.tif"
    )
    assert set(curves) == set(CURVES)
    validation = validate_digitization(curves)
    assert all(row["passes_table_crosscheck"] for row in validation)
    assert all(row["digitized_points"] >= 500 for row in validation)
