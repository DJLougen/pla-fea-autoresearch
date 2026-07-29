from pathlib import Path

from src.custom_pattern_fixture import (
    trace_solver_fixture,
    write_fixture_package,
)


def test_solver_bending_boundary_trace_matches_registered_fixture():
    trace = trace_solver_fixture()
    assert trace["solver_boundary_geometry_gate_passed"] is True
    assert trace["solver_endpoint_gate_passed"] is True
    for row in trace["traces"].values():
        assert row["supports"]["minimum_x_mm"] == 0.0
        assert row["supports"]["maximum_x_mm"] == 40.0
        assert row["load_strip_width_mm"] <= 1.1


def test_fixture_components_are_watertight_and_dimension_checked(
    tmp_path: Path,
):
    manifest = write_fixture_package(tmp_path)
    assert manifest["all_fixture_gates_passed"] is True
    assert len(manifest["components"]) == 2
    for component in manifest["components"]:
        assert component["watertight"] is True
        assert component["nonmanifold_edge_count"] == 0
        assert component["bounds_gate_passed"] is True
        assert Path(component["path"]).exists()
