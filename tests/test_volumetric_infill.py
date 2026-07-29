from pathlib import Path
import struct

import numpy as np
import pytest

from src.calculix import CalculixRunner
from src.volumetric_infill import (
    VOLUMETRIC_PATTERNS,
    _component_masks,
    _repair_missing_face_spans,
    _spans_all_domain_faces,
    connectivity_components,
    inspect_binary_stl,
    make_voxel_mesh,
    morphology_metrics,
    render_voxel_compression_deck,
    write_voxel_stl,
)


def test_platform_tie_repair_preserves_density_and_connectivity():
    occupancy = np.zeros((6, 6, 6), dtype=bool)
    occupancy[1:5, 1:5, 1:5] = True
    target_count = int(occupancy.sum())
    repaired = _repair_missing_face_spans(
        occupancy,
        np.ones_like(occupancy, dtype=float),
        target_count,
    )
    assert int(repaired.sum()) == target_count
    assert _spans_all_domain_faces(repaired)
    assert len(_component_masks(repaired)) == 1


def test_equal_mass_voxel_patterns_are_connected_and_span_all_axes():
    signatures = set()
    for pattern in VOLUMETRIC_PATTERNS:
        mesh = make_voxel_mesh(pattern, resolution=10)
        assert mesh.actual_relative_density == pytest.approx(0.5)
        assert connectivity_components(mesh.occupancy) == [
            int(mesh.occupancy.sum())
        ]
        assert len(mesh.element_tags) == 500
        assert mesh.connectivity.shape == (500, 8)
        assert all(mesh.supports_by_axis[axis] for axis in "xyz")
        assert all(mesh.loads_by_axis[axis] for axis in "xyz")
        metrics = morphology_metrics(mesh)
        signatures.add(
            (
                round(metrics["line_of_sight_standard_deviation"], 6),
                round(
                    metrics[
                        "surface_area_proxy_per_solid_volume_mm_inv"
                    ],
                    6,
                ),
            )
        )
    assert len(signatures) == 4


def test_voxel_deck_supports_elastic_and_predamage_plastic():
    mesh = make_voxel_mesh("gyroid", resolution=8)
    elastic = render_voxel_compression_deck(mesh, "x", 0.01)
    plastic = render_voxel_compression_deck(
        mesh,
        "z",
        0.05,
        plastic_points=[
            {"stress_MPa": 50.0, "equivalent_plastic_strain": 0.0},
            {"stress_MPa": 60.0, "equivalent_plastic_strain": 0.05},
        ],
    )
    assert "*ELEMENT,TYPE=C3D8" in elastic
    assert "LOAD,1,1,-0.2" in elastic
    assert "*STATIC,DIRECT" in elastic
    assert "*EL FILE" not in elastic
    assert "*PLASTIC" in plastic
    assert "LOAD,3,3,-1" in plastic
    assert "*STEP,NLGEOM=YES" in plastic
    assert "*EL FILE" not in plastic


def test_binary_stl_is_watertight_and_volume_matched(
    tmp_path: Path,
):
    mesh = make_voxel_mesh("gyroid", resolution=8)
    path = tmp_path / "gyroid.stl"
    triangle_count = write_voxel_stl(mesh, path)
    payload = path.read_bytes()
    assert len(payload) == 84 + 50 * triangle_count
    assert struct.unpack("<I", payload[80:84])[0] == triangle_count
    assert triangle_count > 0
    metrics = inspect_binary_stl(path)
    assert metrics["watertight"] is True
    assert metrics["volume_mm3"] == pytest.approx(
        mesh.actual_volume_mm3, rel=1.0e-5
    )


@pytest.mark.integration
def test_real_calculix_runs_voxel_compression(tmp_path: Path):
    runner = CalculixRunner(tmp_path / "runs", executable="ccx")
    if not runner.available:
        pytest.skip("CalculiX is not installed")
    mesh = make_voxel_mesh("cubic", resolution=8)
    deck = render_voxel_compression_deck(mesh, "x", 0.01)
    path = tmp_path / "cubic.inp"
    path.write_text(deck, encoding="utf-8")
    result = runner.run(path, {}, run_id="cubic")
    assert result.ok, result.error
    assert result.curve["reaction_force"][-1] > 0.0
