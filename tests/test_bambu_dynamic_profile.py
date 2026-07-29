import pytest

from src.bambu_dynamic_profile import (
    build_dynamic_profile,
    infer_in_plane_shear_modulus,
)
from src.materials import check_admissibility


def test_off_axis_storage_modulus_identifies_positive_g12():
    g12 = infer_in_plane_shear_modulus(
        3347.6059, 3113.7201, 3259.2081, 0.3459
    )
    assert g12 == pytest.approx(1227.6, rel=2e-3)


def test_dynamic_storage_profile_is_admissible_and_bounded():
    profile = build_dynamic_profile()
    constants = profile["screening_engineering_constants"]
    assert check_admissibility(constants).valid
    assert constants["E1"] > constants["E2"] > constants["E3"]
    assert constants["nu12"] == pytest.approx(0.3459)
    assert profile["dynamic_reference"][
        "minimum_DMA_tan_delta_peak_Tg_C"
    ] == pytest.approx(62.50, abs=0.02)
    assert "No loss modulus" in profile["limitations"][3]


@pytest.mark.parametrize("temperature", [210, 220, 230, 240])
def test_all_nozzle_temperature_storage_profiles_are_admissible(temperature):
    profile = build_dynamic_profile(nozzle_temperature_C=temperature)
    assert check_admissibility(
        profile["screening_engineering_constants"]
    ).valid
    assert profile["dynamic_reference"]["nozzle_temperature_C"] == temperature
    expected_passes = 3 if temperature in (220, 240) else 2
    assert (
        profile["dynamic_reference"][
            "independent_crossplot_rows_passed"
        ]
        == expected_passes
    )
