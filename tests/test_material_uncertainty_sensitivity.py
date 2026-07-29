from src.materials import check_admissibility
from src.material_uncertainty_sensitivity import (
    SCENARIOS,
    build_sensitivity_profiles,
)


def test_sensitivity_profiles_are_admissible_and_provenance_labeled():
    profiles = build_sensitivity_profiles()
    assert set(profiles) == set(SCENARIOS)
    for name, profile in profiles.items():
        constants = profile["screening_engineering_constants"]
        assert check_admissibility(constants).valid
        assert profile["uncertainty_scenario"]["name"] == name
        assert "not a probability" in profile["uncertainty_scenario"][
            "interpretation"
        ]
    assert profiles["measured_low"]["screening_engineering_constants"][
        "E1"
    ] == 2360.0
    assert profiles["shear_high"]["screening_engineering_constants"][
        "G13"
    ] == 975.0
