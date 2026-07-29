from pathlib import Path

from src.characterization_identifiability import (
    DIRECTIONS,
    PARAMETERS,
    matrix_diagnostics,
    numerical_jacobian,
    run_identifiability,
)
from src.materials import PRIORS


def test_full_campaign_identifies_all_nine_constants() -> None:
    labels, _baseline, jacobian = numerical_jacobian(PRIORS)
    diagnostics = matrix_diagnostics(jacobian)
    assert len(labels) == 12
    assert jacobian.shape == (12, 9)
    assert diagnostics["rank"] == len(PARAMETERS)
    assert diagnostics["condition_number"] < 10.0


def test_lateral_and_off_axis_measurements_are_both_required() -> None:
    _, _, axial_only = numerical_jacobian(
        PRIORS, include_lateral=False
    )
    _, _, principal_only = numerical_jacobian(
        PRIORS, ("1", "2", "3"), include_lateral=True
    )
    _, _, no_build_coupled = numerical_jacobian(
        PRIORS,
        ("1", "2", "3", "12_45"),
        include_lateral=True,
    )
    assert matrix_diagnostics(axial_only)["rank"] < 9
    assert matrix_diagnostics(principal_only)["rank"] < 9
    assert matrix_diagnostics(no_build_coupled)["rank"] < 9


def test_identifiability_artifacts_are_exported(
    tmp_path: Path,
) -> None:
    result = run_identifiability(tmp_path)
    assert result["full_rank_gate_passed"] is True
    assert result["condition_gate_passed"] is True
    assert result["ablation_gate_passed"] is True
    assert result["all_parameters_have_sensitivity"] is True
    assert (tmp_path / "observable_parameter_sensitivity.csv").exists()
    assert (tmp_path / "campaign_ablation.csv").exists()
    assert (tmp_path / "linearized_uncertainty_forecast.csv").exists()
