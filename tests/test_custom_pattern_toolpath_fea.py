import json
from pathlib import Path

import pytest

from src.custom_pattern_toolpath import (
    DEFAULT_GCODE,
    DEFAULT_MATERIAL_PROFILE,
    parse_deposited_segments,
)
from src.custom_pattern_toolpath_fea import (
    _pattern_for_object,
    build_mass_calibrated_fine_geometry,
    coarsen_to_voxel_mesh,
    render_toolpath_bending_deck,
)


def test_toolpath_object_names_support_new_valid_pattern_stems() -> None:
    assert _pattern_for_object("kite_08_08_08_08.stl_1") == (
        "kite_08_08_08_08"
    )


def _first_triangular_segments():
    parsed = parse_deposited_segments(DEFAULT_GCODE)
    return parsed["triangular.stl_1"]


def test_mass_calibrated_raster_is_connected_and_matches_extrusion():
    geometry = build_mass_calibrated_fine_geometry(
        "triangular",
        "triangular.stl_1",
        _first_triangular_segments(),
        resolution=60,
    )
    assert geometry.component_count == 1
    assert abs(geometry.mass_error_fraction) <= 0.001
    assert geometry.raster_volume_mm3 == pytest.approx(
        geometry.target_deposited_volume_mm3, rel=0.001
    )
    assert len(geometry.layer_width_scales) == 20
    assert all(0.0 < value < 2.0 for value in geometry.layer_width_scales)


def test_coarsened_mesh_preserves_mass_and_bending_boundaries():
    geometry = build_mass_calibrated_fine_geometry(
        "triangular",
        "triangular.stl_1",
        _first_triangular_segments(),
        resolution=60,
    )
    mesh = coarsen_to_voxel_mesh(geometry, 30)
    assert mesh.component_count == 1
    assert mesh.layers_z == 20
    assert mesh.represented_volume_mm3 == pytest.approx(
        geometry.raster_volume_mm3, abs=1e-9
    )
    assert len(mesh.node_tags) > 0
    assert len(mesh.element_tags) > 0
    assert len(mesh.connectivity) == len(mesh.fill_fractions)
    assert mesh.supports
    assert mesh.loads


def test_toolpath_deck_uses_fraction_scaled_c3d8_materials():
    geometry = build_mass_calibrated_fine_geometry(
        "triangular",
        "triangular.stl_1",
        _first_triangular_segments(),
        resolution=60,
    )
    mesh = coarsen_to_voxel_mesh(geometry, 30)
    profile = json.loads(
        Path(DEFAULT_MATERIAL_PROFILE).read_text(encoding="utf-8")
    )
    deck = render_toolpath_bending_deck(mesh, profile)
    assert "*ELEMENT,TYPE=C3D8" in deck
    assert "*ELASTIC,TYPE=ENGINEERING CONSTANTS" in deck
    assert deck.count("*SOLID SECTION") == len(
        set(float(value) for value in mesh.fill_fractions)
    )
    assert "LOAD,3,3,-0.4" in deck
    assert "*NODE PRINT,NSET=SUPPORTS" in deck
    assert "*EL FILE" not in deck
    field_deck = render_toolpath_bending_deck(
        mesh, profile, include_field_output=True
    )
    assert "*EL FILE,ELSET=ALL_TOOLPATH\nE" in field_deck
