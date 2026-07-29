import pytest

from src.infill_benchmark import make_staggered_diamond_variant
from src.parametric_infill_search import build_search_geometries


def test_parametric_geometries_cover_search_grid_and_are_equal_mass_ready():
    geometries = build_search_geometries()
    assert len(geometries) == 23
    assert "sd_c3_o50" in geometries
    assert "triangular" in geometries
    for geometry in geometries.values():
        assert geometry.edges
        assert geometry.total_length_mm > 0.0
        assert any(y == 0.0 for _, y in geometry.nodes)
        assert any(y == 40.0 for _, y in geometry.nodes)


def test_staggered_variant_validates_parameters():
    with pytest.raises(ValueError, match="at least 2"):
        make_staggered_diamond_variant("bad", cells=1)
    with pytest.raises(ValueError, match="offset_fraction"):
        make_staggered_diamond_variant("bad", offset_fraction=0.95)
