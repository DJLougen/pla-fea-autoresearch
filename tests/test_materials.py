import numpy as np
import pytest

from src.materials import (
    PRIORS,
    check_admissibility,
    compliance_matrix,
    material_card,
    off_axis_youngs_modulus,
    reciprocal_poisson,
    vector_to_constants,
)


def test_off_axis_modulus_recovers_isotropic_modulus():
    constants = vector_to_constants("isotropic", [2200.0, 0.33])
    for plane in ("12", "13", "23"):
        assert off_axis_youngs_modulus(constants, plane) == pytest.approx(
            2200.0
        )


def test_compliance_is_symmetric_positive_definite():
    matrix = compliance_matrix(PRIORS)
    np.testing.assert_allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() > 0
    assert check_admissibility(PRIORS).valid


def test_reciprocal_poisson_relations():
    reciprocal = reciprocal_poisson(PRIORS)
    assert reciprocal["nu21"] == pytest.approx(
        PRIORS["nu12"] * PRIORS["E2"] / PRIORS["E1"]
    )


def test_inadmissible_material_is_rejected_before_card_render():
    constants = dict(PRIORS)
    constants["G13"] = -1.0
    assert not check_admissibility(constants).valid
    with pytest.raises(ValueError, match="inadmissible"):
        material_card(constants)


def test_material_card_has_calculix_engineering_constants_layout():
    card = material_card(PRIORS)
    assert "*ELASTIC, TYPE=ENGINEERING CONSTANTS" in card
    lines = card.splitlines()
    elastic = lines.index("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
    assert len(lines[elastic + 1].split(",")) == 8
    assert float(lines[elastic + 2]) == pytest.approx(PRIORS["G23"])


def test_model_parameter_tying():
    iso = vector_to_constants("isotropic", [2400.0, 0.3])
    assert iso["E1"] == iso["E2"] == iso["E3"]
    ti = vector_to_constants(
        "transverse_isotropic", [2500.0, 1800.0, 650.0, 0.32, 0.28]
    )
    assert ti["E1"] == ti["E2"]
    assert ti["G13"] == ti["G23"]
    tied = vector_to_constants(
        "orthotropic",
        [2500, 1800, 900, 650, 0.32, 0.28],
        parameter_ties={"E2": "E1", "G23": "G13", "nu23": "nu13"},
    )
    assert tied["E2"] == tied["E1"]
    assert tied["G23"] == tied["G13"]
    assert tied["nu23"] == tied["nu13"]
