from pathlib import Path

import pytest

from src.bambu_plastic_calibration import aggregate_in_plane_prepeak


def test_real_bambu_prepeak_curve_is_admissible():
    dataset = Path("data/public/mendeley-8jxw5533py-v2")
    if not dataset.exists():
        pytest.skip("public Bambu curves are not imported")
    aggregate, plastic, metadata = aggregate_in_plane_prepeak(
        dataset, 2204.1774610119137
    )
    assert len(aggregate) > 100
    assert metadata["specimens"] == 19
    assert metadata["yield_engineering_stress_MPa"] > 25.0
    assert metadata["maximum_supported_engineering_strain"] > 0.02
    strains = [
        point["equivalent_plastic_strain"] for point in plastic
    ]
    stresses = [point["true_stress_MPa"] for point in plastic]
    assert strains[0] == pytest.approx(0.0)
    assert all(b > a for a, b in zip(strains, strains[1:]))
    assert all(b >= a for a, b in zip(stresses, stresses[1:]))
