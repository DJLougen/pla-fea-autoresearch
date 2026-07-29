from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.characterization_campaign import write_campaign
from src.characterization_identifiability import DIRECTIONS
from src.directional_material_calibration import (
    calibrate_submission,
    fit_orthotropic_constants,
    predict_directional_observables,
    score_coupon_partition,
    synthetic_recovery,
    tensile_observables,
)
from src.materials import check_admissibility


TRUTH = {
    "E1": 2850.0,
    "E2": 2475.0,
    "E3": 1780.0,
    "nu12": 0.34,
    "nu13": 0.29,
    "nu23": 0.27,
    "G12": 880.0,
    "G13": 610.0,
    "G23": 540.0,
}


def _observations(validation_scale: float = 1.0):
    rows = []
    for direction in DIRECTIONS:
        modulus, poisson = predict_directional_observables(TRUTH, direction)
        for replicate in range(1, 6):
            partition = "calibration" if replicate <= 3 else "validation"
            rows.append(
                {
                    "test_id": f"{direction}-{replicate}",
                    "partition": partition,
                    "test_type": "tensile",
                    "print_direction": direction,
                    "axial_modulus_MPa": (
                        modulus
                        if partition == "calibration"
                        else modulus * validation_scale
                    ),
                    "effective_global_xy_poisson": poisson,
                }
            )
    return rows


def test_tensile_curve_extracts_axial_and_lateral_slopes():
    strain = np.linspace(0.0, 0.004, 41)
    result = tensile_observables(
        pd.DataFrame(
            {
                "engineering_strain": strain,
                "engineering_stress_MPa": 2500.0 * strain + 0.01,
                "transverse_strain": -0.32 * strain + 1.0e-6,
            }
        )
    )
    assert result["axial_modulus_MPa"] == pytest.approx(2500.0)
    assert result["effective_global_xy_poisson"] == pytest.approx(0.32)
    assert result["stress_fit_r_squared"] == pytest.approx(1.0)
    assert result["lateral_fit_r_squared"] == pytest.approx(1.0)


def test_nine_constant_fit_recovers_synthetic_truth():
    fit = fit_orthotropic_constants(_observations())
    assert fit["quality_gate_passed"] is True
    assert check_admissibility(fit["constants"]).valid
    for name, truth in TRUTH.items():
        assert fit["constants"][name] == pytest.approx(truth, rel=1.0e-6)
    validation = score_coupon_partition(
        fit["constants"], _observations(), "validation"
    )
    assert validation["quality_gate_passed"] is True
    assert validation["specimens"] == 12


def test_validation_partition_can_veto_an_otherwise_valid_fit():
    observations = _observations(validation_scale=1.30)
    fit = fit_orthotropic_constants(observations)
    validation = score_coupon_partition(
        fit["constants"], observations, "validation"
    )
    assert fit["quality_gate_passed"] is True
    assert validation["modulus_gate_passed"] is False
    assert validation["quality_gate_passed"] is False


def test_fit_rejects_missing_off_axis_direction():
    observations = [
        row
        for row in _observations()
        if row["print_direction"] != "23_45"
    ]
    with pytest.raises(ValueError, match="all six directions"):
        fit_orthotropic_constants(observations)


def test_synthetic_recovery_exercises_identifiability_and_release_gates():
    result = synthetic_recovery()
    assert result["recovery_gate_passed"] is True
    assert result["validation_gate_passed"] is True
    assert result["maximum_parameter_relative_error"] < 0.01


def test_blank_campaign_releases_no_directional_card(tmp_path: Path):
    campaign = tmp_path / "campaign"
    output = tmp_path / "calibration"
    write_campaign(campaign)
    result = calibrate_submission(
        campaign / "capture_templates", campaign, output
    )
    assert result["status"] == "awaiting_measurements"
    assert result["card_released"] is False
    assert result["synthetic_recovery"]["recovery_gate_passed"] is True
    assert not (output / "material.inp").exists()
    assert not (output / "fitted_constants.json").exists()
    assert (output / "calibration_manifest.json").exists()
