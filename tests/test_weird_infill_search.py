import pytest

from src.weird_infill_search import (
    build_candidate_geometries,
    make_kite_geometry,
    render_solid_modal_deck,
    select_beam_promotions,
)


def test_kite_family_is_deterministic_and_connected() -> None:
    geometries = build_candidate_geometries()
    assert len(geometries) == 83
    geometry = geometries["kite_08_12_16_08"]
    assert geometry.name == "kite_08_12_16_08"
    assert len(geometry.nodes) > 20
    assert len(geometry.edges) > 30
    assert geometry.total_length_mm > 300.0


def test_kite_apex_validation() -> None:
    with pytest.raises(ValueError, match="four"):
        make_kite_geometry((8, 12, 16))
    with pytest.raises(ValueError, match="\\[5, 17\\]"):
        make_kite_geometry((4, 12, 12, 12))


def test_beam_promotion_keeps_distinct_objectives_and_baseline() -> None:
    rows = []
    for index, pattern in enumerate(
        (
            "kite_08_08_08_08",
            "kite_12_12_12_12",
            "kite_16_16_16_16",
            "kite_08_12_16_08",
            "kite_16_12_08_16",
            "graded_hourglass",
        )
    ):
        rows.append(
            {
                "pattern": pattern,
                "weighted_static_score": 0.9 - index * 0.02,
                "balanced_geometric_mean": 0.7 + index * 0.02,
                "worst_case_normalized": 0.5 + index * 0.01,
                "bending_normalized": 0.6 + index * 0.03,
                "buckling_normalized": 0.8 - index * 0.01,
            }
        )
    promoted = select_beam_promotions(rows, count=6)
    assert len(promoted) == 6
    assert len(set(promoted)) == 6
    assert "graded_hourglass" in promoted
