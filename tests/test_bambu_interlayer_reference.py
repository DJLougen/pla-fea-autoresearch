import pytest

from src.bambu_interlayer_reference import (
    SOURCE_SHA256,
    build_profiles,
    reported_rows,
    source_checksum,
)
from src.interlayer_initiation_screen import pairwise_matches


def test_peer_reviewed_interlayer_source_is_checksum_pinned():
    assert source_checksum() == SOURCE_SHA256


def test_exact_table_values_build_positive_uncertainty_profiles():
    assert len(reported_rows()) == 4
    profiles = build_profiles()["profiles"]
    seam = profiles["traditional_multi_head_seam"]
    continuous = profiles["smlp_continuous_layers"]
    assert seam["normal_tensile_mean_MPa"] == pytest.approx(35.07)
    assert seam["normal_tensile_lower_1sd_MPa"] == pytest.approx(32.52)
    assert continuous["normal_tensile_mean_MPa"] == pytest.approx(47.49)
    assert continuous["apparent_tear_lower_1sd_MPa"] == pytest.approx(10.98)


def test_pairwise_ranking_score_preserves_reference_order():
    assert pairwise_matches(["hilbert", "zigzag_90", "zigzag_45"]) == 3
    assert pairwise_matches(["zigzag_90", "hilbert", "zigzag_45"]) == 2
    assert pairwise_matches(["zigzag_90", "zigzag_45", "hilbert"]) == 1
