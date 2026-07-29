from pathlib import Path

import pytest

from src.custom_pattern_slicer import (
    DEFAULT_STL_DIR,
    find_orca_slicer,
    parse_object_extrusion,
    write_manufacturing_package,
)


def test_object_extrusion_parser_separates_labeled_toolpaths(
    tmp_path: Path,
):
    gcode = tmp_path / "tiny.gcode"
    gcode.write_text(
        "\n".join(
            [
                "; start printing object, unique label id: 1",
                "G1 X1 Y1 E1.0",
                "G1 E0.8",
                "G1 X2 E0.5",
                "; stop printing object triangular.stl_1 id:0 copy 0",
                "; start printing object, unique label id: 2",
                "G1 X1 Y1 E2.0",
                "; stop printing object graded_hourglass.stl_1 id:0 copy 0",
            ]
        ),
        encoding="utf-8",
    )
    rows = parse_object_extrusion(gcode)
    assert (
        rows["triangular.stl_1"]["deposited_filament_length_mm"]
        == 1.5
    )
    assert (
        rows["graded_hourglass.stl_1"][
            "deposited_filament_length_mm"
        ]
        == 2.0
    )


def test_manufacturing_mass_tolerance_is_bounded(tmp_path: Path):
    with pytest.raises(ValueError, match="must lie"):
        write_manufacturing_package(
            tmp_path,
            DEFAULT_STL_DIR,
            maximum_toolpath_mass_ratio_error=0.031,
        )


@pytest.mark.integration
def test_registered_orca_plate_passes_all_manufacturing_gates(
    tmp_path: Path,
):
    try:
        orca = find_orca_slicer()
    except FileNotFoundError:
        pytest.skip("OrcaSlicer is not installed")
    manifest = write_manufacturing_package(
        tmp_path, DEFAULT_STL_DIR, orca
    )
    assert manifest["all_manufacturing_gates_passed"] is True
    assert manifest["three_mf_inspection"]["build_item_count"] == 6
    assert (
        manifest["replicates_have_identical_toolpath_volume"]
        is True
    )
    assert abs(manifest["toolpath_mass_ratio_error"]) <= 0.01
    assert Path(manifest["plate_3mf"]).exists()
    assert Path(manifest["gcode"]).exists()

    refreshed = write_manufacturing_package(
        tmp_path, DEFAULT_STL_DIR, orca
    )
    assert refreshed["all_manufacturing_gates_passed"] is True
    assert Path(refreshed["plate_3mf"]).exists()
    assert Path(refreshed["gcode"]).exists()
