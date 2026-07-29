import pytest

from src.bambu_tds_profile import build_tds_profiles
from src.materials import check_admissibility


def test_tds_profiles_separate_reported_constants_from_priors():
    isotropic, transverse = build_tds_profiles()
    assert isotropic["model_class"] == "isotropic"
    assert transverse["model_class"] == "transverse_isotropic"
    constants = transverse["screening_engineering_constants"]
    assert constants["E1"] == pytest.approx(2580.0)
    assert constants["E2"] == pytest.approx(2580.0)
    assert constants["E3"] == pytest.approx(2060.0)
    assert constants["nu12"] == pytest.approx(0.3459)
    assert constants["G12"] == pytest.approx(
        2580.0 / (2.0 * (1.0 + 0.3459))
    )
    assert "direct exact-brand" in transverse[
        "engineering_constants_evidence"
    ]["nu12"]
    assert transverse["profile_id"].endswith("_v2")
    assert transverse["engineering_constants_evidence"]["G13"].endswith(
        "prior"
    )
    assert check_admissibility(constants).valid
