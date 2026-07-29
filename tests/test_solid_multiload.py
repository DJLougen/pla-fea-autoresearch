from pathlib import Path

import pytest

from src.infill_benchmark import make_pattern
from src.calculix import CalculixRunner
from src.solid_infill import (
    equal_solid_mass_strut_width,
    generate_prism_mesh,
)
from src.solid_multiload import (
    render_solid_buckling_deck,
    render_solid_static_deck,
    select_boundary_sets,
    run_solid_multiload,
)


def _coarse_grid(tmp_path: Path):
    pytest.importorskip("gmsh")
    geometry = make_pattern("grid")
    width = equal_solid_mass_strut_width(
        geometry, 0.35, relative_tolerance=5.0e-4
    )
    return generate_prism_mesh(
        geometry, width, 2.0, tmp_path / "grid.msh"
    )


def test_solid_fixture_sets_cover_all_load_modes(tmp_path: Path):
    mesh = _coarse_grid(tmp_path)
    for load_case in (
        "tension",
        "shear",
        "bending",
        "compression",
        "buckling",
    ):
        sets = select_boundary_sets(
            mesh, load_case, 40.0, 40.0, 4.0, 2.0
        )
        assert sets.supports
        assert sets.loads
        assert sets.anchor in sets.supports
        assert sets.anchor_two in sets.supports


def test_solid_multiload_decks_are_validly_structured(tmp_path: Path):
    mesh = _coarse_grid(tmp_path)
    compression = render_solid_static_deck(
        mesh,
        "compression",
        2079.9,
        0.33,
        1.24e-9,
        0.4,
        mesh_size_mm=2.0,
    )
    bending = render_solid_static_deck(
        mesh,
        "bending",
        2079.9,
        0.33,
        1.24e-9,
        0.4,
        mesh_size_mm=2.0,
    )
    buckling = render_solid_buckling_deck(
        mesh, 2079.9, 0.33, 1.24e-9, mesh_size_mm=2.0
    )
    assert "*STEP,NLGEOM=YES" in compression
    assert "LOAD,2,2,-0.4" in compression
    assert "LOAD,3,3,-0.4" in bending
    assert "*BUCKLE" in buckling
    assert "*ELEMENT,TYPE=C3D6" in buckling

    plastic = render_solid_static_deck(
        mesh,
        "tension",
        2079.9,
        0.33,
        1.24e-9,
        0.8,
        mesh_size_mm=2.0,
        plastic_points=[
            {
                "true_stress_MPa": 33.0,
                "equivalent_plastic_strain": 0.0,
            },
            {
                "true_stress_MPa": 38.0,
                "equivalent_plastic_strain": 0.003,
            },
        ],
    )
    assert "*PLASTIC" in plastic
    assert "*STEP,NLGEOM=YES" in plastic

    engineering = render_solid_static_deck(
        mesh,
        "tension",
        2580.0,
        0.33,
        1.24e-9,
        0.4,
        mesh_size_mm=2.0,
        engineering_constants={
            "E1": 2580.0,
            "E2": 2580.0,
            "E3": 2060.0,
            "nu12": 0.33,
            "nu13": 0.31,
            "nu23": 0.31,
            "G12": 969.924812,
            "G13": 780.0,
            "G23": 780.0,
        },
    )
    assert "*ELASTIC,TYPE=ENGINEERING CONSTANTS" in engineering
    assert "2580,2580,2060,0.33,0.31,0.31" in engineering
    assert "*ORIENTATION,NAME=MAT_ORIENTATION" in engineering

    no_fields = render_solid_static_deck(
        mesh,
        "tension",
        2200.0,
        0.33,
        1.24e-9,
        0.4,
        incremental_nlgeom=True,
        output_element_fields=False,
    )
    assert "*EL FILE" not in no_fields
    assert "S,PEEQ" in plastic


@pytest.mark.integration
def test_real_calculix_runs_all_solid_load_modes(tmp_path: Path):
    if not CalculixRunner(tmp_path / "probe").available:
        pytest.skip("CalculiX is not installed")
    results = run_solid_multiload(
        tmp_path / "results",
        tmp_path / "runs",
        patterns=["grid"],
        mesh_size_mm=2.0,
    )
    assert len(results) == 5
    assert all(result.status == "ok" for result in results)
    for result in results:
        if result.load_case == "buckling":
            assert result.critical_buckling_load_N > 0.0
        else:
            assert result.stiffness_N_per_mm > 0.0
