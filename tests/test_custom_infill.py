import json
from pathlib import Path

import pytest

from src.custom_infill import admit_pattern_study, load_custom_pattern
from src.infill_benchmark import make_geometry_from_segments


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_crossing_custom_struts_receive_a_shared_beam_node():
    geometry = make_geometry_from_segments(
        "crossed",
        [
            ((0.0, 0.0), (40.0, 40.0)),
            ((0.0, 40.0), (40.0, 0.0)),
        ],
        40.0,
    )
    crossing = geometry.nodes.index((20.0, 20.0)) + 1
    incident = [
        edge for edge in geometry.edges if crossing in edge
    ]
    assert len(incident) == 4


def test_example_custom_pattern_passes_manufacturing_gate():
    definition, geometry, validation = load_custom_pattern(
        PROJECT_ROOT / "ideas/graded_hourglass.json"
    )
    assert definition.name == "graded_hourglass"
    assert validation["status"] == "valid"
    assert validation["graph_connected"]
    assert validation["nominal_equal_mass_strut_width_mm"] >= 0.8
    assert len(geometry.edges) == validation["normalized_edges"]


def test_bowtie_fan_example_is_connected_and_printable():
    definition, geometry, validation = load_custom_pattern(
        PROJECT_ROOT / "ideas/bowtie_fan.json"
    )
    assert definition.name == "bowtie_fan"
    assert validation["status"] == "valid"
    assert validation["graph_connected"]
    assert validation["nominal_equal_mass_strut_width_mm"] >= 0.8
    assert validation["boundary_node_counts"]["bending_load_line"] == 5
    assert len(geometry.edges) == 36


def test_disconnected_custom_pattern_is_rejected(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "name": "disconnected",
        "units": "mm",
        "domain": {
            "width_mm": 40,
            "height_mm": 40,
            "depth_mm": 4,
        },
        "relative_density": 0.35,
        "segments": [
            [[0, 0], [0, 40]],
            [[40, 0], [40, 40]],
        ],
    }
    path = tmp_path / "disconnected.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disconnected"):
        load_custom_pattern(path)


def test_collinear_overlap_is_rejected_as_ambiguous():
    with pytest.raises(ValueError, match="collinear overlap"):
        make_geometry_from_segments(
            "overlap",
            [
                ((0.0, 0.0), (30.0, 0.0)),
                ((20.0, 0.0), (40.0, 0.0)),
            ],
            40.0,
        )


def test_pattern_study_is_bound_to_canonical_bambu_twin():
    admission = admit_pattern_study("test_pattern")
    assert admission["gate_passed"] is True
    assert admission["twin_id"] == "bambu_pla_basic_evidence_twin_v1"
    assert (
        admission["selection"]["model_id"]
        == "tds_transverse_elastic_sensitivity_v2"
    )
    assert admission["canonical_card_sha256"]


def test_pattern_study_rejects_hot_or_humid_condition():
    with pytest.raises(ValueError, match="rejected"):
        admit_pattern_study("hot", temperature_C=54.0)
    with pytest.raises(ValueError, match="unsupported"):
        admit_pattern_study(
            "humid",
            temperature_C=25.0,
            relative_humidity_percent=55.0,
            conditioning="saturated",
        )


def test_pattern_study_rejects_noncanonical_properties(tmp_path: Path):
    changed = tmp_path / "changed.json"
    changed.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        admit_pattern_study("changed", changed)
