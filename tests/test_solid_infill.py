from pathlib import Path

import pytest

from src.infill_benchmark import make_pattern
from src.solid_infill import (
    equal_solid_mass_strut_width,
    generate_prism_mesh,
    render_solid_tension_deck,
)


def test_equal_solid_mass_width_and_prism_mesh(tmp_path: Path):
    pytest.importorskip("gmsh")
    geometry = make_pattern("grid")
    width = equal_solid_mass_strut_width(
        geometry, 0.35, relative_tolerance=5.0e-4
    )
    mesh = generate_prism_mesh(
        geometry,
        width,
        2.0,
        tmp_path / "grid.msh",
    )
    assert mesh.actual_volume_mm3 / (40.0 * 40.0 * 4.0) == pytest.approx(
        0.35, rel=5.0e-4
    )
    assert len(mesh.node_tags) > 100
    assert len(mesh.element_tags) > 100
    assert mesh.connectivity.shape[1] == 6
    assert mesh.supports
    assert mesh.loads

    deck = render_solid_tension_deck(
        mesh, 2079.9, 0.33, 1.24e-9, 0.4
    )
    assert "*ELEMENT,TYPE=C3D6" in deck
    assert "*ORIENTATION,NAME=MAT_ORIENTATION" in deck
    assert "LOAD,2,2,0.4" in deck
    assert "*SOLID SECTION" in deck
