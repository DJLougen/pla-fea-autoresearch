from pathlib import Path

import pytest

from src.adaptive_infill_external_validation import (
    SOURCE_SHA256,
    _adaptive_rows,
    _physical_ranking_rows,
    physical_results,
    source_checksum,
)
from src.infill_benchmark import make_pattern, nominal_strut_width


def test_checksum_pinned_source_pdf_is_present():
    assert source_checksum() == SOURCE_SHA256


def test_physical_hilbert_ranking_and_adaptive_changes_are_transcribed():
    rows = physical_results()
    assert len(rows) == 24
    rankings = _physical_ranking_rows(rows)
    assert len(rankings) == 4
    assert all(
        row["ranking"] == ["hilbert", "zigzag_90", "zigzag_45"]
        for row in rankings
    )
    changes = _adaptive_rows(rows)
    hilbert_tension_20 = next(
        row for row in changes
        if row["test_mode"] == "tension"
        and row["nominal_infill_fraction"] == .20
        and row["pattern"] == "hilbert"
    )
    assert hilbert_tension_20["peak_load_change_percent"] == pytest.approx(
        40.54, abs=.01
    )


@pytest.mark.parametrize("name", ["zigzag_90", "zigzag_45", "hilbert"])
def test_external_patterns_are_equal_nominal_mass_ready(name: str):
    geometry = make_pattern(name)
    width = nominal_strut_width(geometry, 40.0, 40.0, .20)
    assert geometry.total_length_mm * width == pytest.approx(320.0)
    assert width >= .6
    assert any(y == 0.0 for _, y in geometry.nodes)
    assert any(y == 40.0 for _, y in geometry.nodes)
