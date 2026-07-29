import pytest

from src.bambu_print_tolerance_infill import (
    read_cube_tolerances,
    scale_geometry,
    scenario_dimensions,
    source_axis_statistics,
)
from src.pattern_robustness_campaign import finalist_geometries


def test_extracts_25_cubes_and_matches_reported_axis_statistics():
    rows = read_cube_tolerances()
    stats = source_axis_statistics(rows)
    assert len(rows) == 25
    assert all(stats[axis]["measurement_count"] == 75 for axis in "xyz")
    assert stats["x"]["absolute_mean_deviation_mm"] == pytest.approx(
        0.01635, abs=1.0e-4
    )
    assert stats["y"]["absolute_mean_deviation_mm"] == pytest.approx(
        0.05623, abs=1.0e-4
    )
    assert stats["z"]["absolute_mean_deviation_mm"] == pytest.approx(
        0.01964, abs=1.0e-4
    )


def test_scenario_dimensions_bracket_transfer_interpretations():
    row = read_cube_tolerances()[0]
    additive = scenario_dimensions(row, "additive_offset")
    multiplicative = scenario_dimensions(row, "multiplicative_scale")
    assert additive["x"] == pytest.approx(
        40.0 + row["x_deviation_mm"]
    )
    assert multiplicative["x"] == pytest.approx(
        40.0 * row["x_mean_mm"] / 20.0
    )
    assert additive["z"] != pytest.approx(multiplicative["z"])


def test_geometry_scaling_preserves_topology():
    geometry = finalist_geometries()["kite_08_08_08_08"]
    scaled = scale_geometry(geometry, 0.99, 1.01)
    assert scaled.edges == geometry.edges
    assert len(scaled.nodes) == len(geometry.nodes)
    assert scaled.total_length_mm > 0.0
    assert max(x for x, _ in scaled.nodes) == pytest.approx(39.6)
    assert max(y for _, y in scaled.nodes) == pytest.approx(40.4)
