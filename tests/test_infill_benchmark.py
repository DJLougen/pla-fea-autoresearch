from pathlib import Path
import csv

import pytest

from src.calculix import CalculixRunner
from src.infill_benchmark import (
    PATTERN_NAMES,
    _parse_buckling_factor,
    make_pattern,
    nominal_strut_width,
    render_beam_deck,
    run_benchmark,
)


def test_patterns_are_connected_to_loading_boundaries_and_equal_nominal_volume():
    target_density = 0.35
    expected_area = target_density * 40.0 * 40.0
    for name in PATTERN_NAMES:
        geometry = make_pattern(name)
        assert geometry.edges
        assert any(y == 0.0 for _, y in geometry.nodes)
        assert any(y == 40.0 for _, y in geometry.nodes)
        assert all(0.0 <= x <= 40.0 and 0.0 <= y <= 40.0 for x, y in geometry.nodes)
        width = nominal_strut_width(geometry, 40.0, 40.0, target_density)
        assert geometry.total_length_mm * width == pytest.approx(expected_area)
        assert width > 0.5


def test_beam_deck_contains_reported_material_and_planar_constraints():
    geometry = make_pattern("reentrant")
    width = nominal_strut_width(geometry, 40.0, 40.0, 0.35)
    deck = render_beam_deck(
        geometry,
        width,
        4.0,
        2079.9,
        0.33,
        0.4,
        40.0,
        1.24e-9,
    )
    assert "*ELEMENT,TYPE=B31" in deck
    assert "*BEAM SECTION" in deck
    assert "2079.9, 0.33" in deck
    assert "LOAD,2,2,0.4" in deck
    assert "NALL,3,5,0." in deck


def test_printable_scad_is_clipped_to_coupon_envelope(tmp_path: Path):
    from src.infill_benchmark import render_scad

    geometry = make_pattern("grid")
    render_scad(geometry, tmp_path / "grid.scad", 1.75, 4.0, 40.0, 40.0)
    text = (tmp_path / "grid.scad").read_text()
    assert "intersection()" in text
    assert "square([40, 40]);" in text


def test_buckling_factor_parser_accepts_calculix_223_format(tmp_path: Path):
    dat = tmp_path / "model.dat"
    dat.write_text(
        """
     B U C K L I N G   F A C T O R   O U T P U T

 MODE NO       BUCKLING
                FACTOR

      1   0.2782978E+04
"""
    )
    assert _parse_buckling_factor(dat) == pytest.approx(2782.978)


@pytest.mark.integration
def test_real_calculix_runs_grid_pattern(tmp_path: Path):
    if not CalculixRunner(tmp_path).available:
        pytest.skip("CalculiX is not installed")
    results = run_benchmark(
        tmp_path / "results",
        tmp_path / "runs",
        patterns=["grid"],
        strut_width_overrides={"grid": 1.0},
    )
    assert results[0].status == "ok"
    assert results[0].strut_width_mm == pytest.approx(1.0)
    assert results[0].nominal_relative_density == pytest.approx(
        results[0].total_centerline_length_mm / (40.0 * 40.0)
    )
    assert results[0].reaction_force_N > 0.0
    assert results[0].effective_modulus_MPa > 0.0
    with (tmp_path / "results" / "physical_test_plan.csv").open() as handle:
        plan = list(csv.DictReader(handle))
    assert len(plan) == 15
    assert {row["load_case"] for row in plan} == {
        "tension",
        "shear",
        "bending",
        "compression",
        "buckling",
    }
