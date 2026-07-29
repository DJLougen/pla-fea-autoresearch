import json
import math
from pathlib import Path

import pytest

from src.custom_pattern_toolpath import (
    DEFAULT_GCODE,
    DEFAULT_MANUFACTURING_MANIFEST,
    DEFAULT_MATERIAL_PROFILE,
    DEFAULT_PRINT_MESH_MANIFEST,
    _pattern_for_object,
    parse_deposited_segments,
    write_toolpath_audit,
)


def test_object_pattern_parser_accepts_a_registered_custom_stem():
    assert _pattern_for_object(
        "bowtie_fan.stl_2", "bowtie_fan"
    ) == "bowtie_fan"
    with pytest.raises(ValueError, match="unexpected"):
        _pattern_for_object("other.stl_1", "bowtie_fan")


def test_segment_parser_tracks_layers_features_and_orientation(
    tmp_path: Path,
):
    gcode = tmp_path / "tiny.gcode"
    gcode.write_text(
        "\n".join(
            [
                "G90",
                "M83",
                "; Z_HEIGHT: 0.2",
                "; FEATURE: Outer wall",
                "; LINE_WIDTH: 0.45",
                "G0 X0 Y0 Z0.2",
                "; start printing object, unique label id: 1",
                "G1 X10 Y0 E1.0",
                "; FEATURE: Inner wall",
                "G1 X10 Y10 E2.0",
                "G1 X0 Y10 E-0.5",
                "; stop printing object triangular.stl_1 id:0 copy 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = parse_deposited_segments(gcode)["triangular.stl_1"]
    assert len(rows) == 2
    assert [row["feature"] for row in rows] == [
        "Outer wall",
        "Inner wall",
    ]
    assert all(row["layer_z_mm"] == 0.2 for row in rows)
    assert rows[0]["line_width_mm"] == 0.45
    assert rows[0]["angle_rad_mod_pi"] == 0.0
    assert math.isclose(rows[1]["angle_rad_mod_pi"], math.pi / 2.0)
    assert sum(float(row["filament_length_mm"]) for row in rows) == 3.0


def test_segment_parser_supports_absolute_extrusion(tmp_path: Path):
    gcode = tmp_path / "absolute.gcode"
    gcode.write_text(
        "\n".join(
            [
                "G90",
                "M82",
                "G92 E0",
                "; Z_HEIGHT: 0.2",
                "; start printing object, unique label id: 1",
                "G1 X1 E0.5",
                "G1 X2 E0.9",
                "; stop printing object triangular.stl_1 id:0 copy 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = parse_deposited_segments(gcode)["triangular.stl_1"]
    assert [row["filament_length_mm"] for row in rows] == pytest.approx(
        [0.5, 0.4]
    )


def test_frozen_plate_toolpath_audit_admits_only_relative_ranking(
    tmp_path: Path,
):
    manifest = write_toolpath_audit(
        tmp_path,
        DEFAULT_GCODE,
        DEFAULT_MANUFACTURING_MANIFEST,
        DEFAULT_PRINT_MESH_MANIFEST,
        DEFAULT_MATERIAL_PROFILE,
    )
    assert (
        manifest["status"]
        == "relative_ranking_admitted_with_restrictions"
    )
    assert manifest["object_count"] == 6
    assert (
        manifest["gates"]["relative_linear_stiffness_ranking_admitted"]
        is True
    )
    assert manifest["gates"]["absolute_stiffness_validated"] is False
    assert (
        manifest["gates"]["replicate_toolpath_volumes_identical"] is True
    )
    assert manifest["gates"]["expected_20_layers_per_object"] is True
    assert (
        manifest["comparison"]["toolpath_mass_ratio_error"]
        == pytest.approx(0.0088290885621063)
    )
    assert (
        manifest["pattern_summaries"]["triangular"][
            "effective_underfill_fraction"
        ]
        == pytest.approx(0.0706442249115331)
    )
    assert (
        manifest["pattern_summaries"]["graded_hourglass"][
            "effective_underfill_fraction"
        ]
        == pytest.approx(0.0624118646086647)
    )
    written = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert written["outputs"]["object_summary.csv"]["sha256"]
    assert (tmp_path / "report.md").exists()
