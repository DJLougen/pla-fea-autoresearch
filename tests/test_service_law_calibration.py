from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.service_law_calibration import (
    _retention_registry,
    calibrate_submission,
    fit_bounded_dma_surface,
    fit_censored_basquin,
    fit_positive_relaxation,
    generalized_maxwell_relaxation,
    predict_dma_surface,
    tensile_curve_metrics,
)
from src.service_law_campaign import write_campaign


def test_dma_surface_recovers_bounded_synthetic_response():
    storage_coefficients = np.array(
        [7.75, -0.24, 0.018, -0.08, 0.004, -0.01]
    )
    loss_coefficients = np.array(
        [4.30, 0.12, 0.09, 0.02, -0.01, 0.015]
    )
    rows = []
    for orientation, scale in (("XY", 1.0), ("Z", 0.72)):
        for replicate in (1, 2, 3):
            replicate_scale = 1.0 + (replicate - 2) * 0.003
            for temperature in (20, 30, 40, 50, 60, 70):
                for frequency in (0.1, 1.0, 10.0):
                    storage = predict_dma_surface(
                        storage_coefficients,
                        np.array([temperature]),
                        np.array([frequency]),
                    )[0]
                    loss = predict_dma_surface(
                        loss_coefficients,
                        np.array([temperature]),
                        np.array([frequency]),
                    )[0]
                    rows.append(
                        {
                            "orientation": orientation,
                            "replicate": replicate,
                            "test_temperature_C": temperature,
                            "frequency_Hz": frequency,
                            "storage_modulus_MPa": (
                                storage * scale * replicate_scale
                            ),
                            "loss_modulus_MPa": (
                                loss * scale * replicate_scale
                            ),
                        }
                    )
    result = fit_bounded_dma_surface(pd.DataFrame(rows))
    assert result["quality_gate_passed"] is True
    assert set(result["orientations"]) == {"XY", "Z"}
    assert (
        result["orientations"]["XY"]["storage"][
            "leave_one_replicate_out_normalized_rmse"
        ]
        < 0.01
    )


def test_positive_relaxation_fit_recovers_a_positive_spectrum():
    time = np.geomspace(1.0, 14_400.0, 81)
    relaxation_times = np.geomspace(1.0 / 3.0, 43_200.0, 6)
    observed = generalized_maxwell_relaxation(
        time,
        850.0,
        np.array([500.0, 350.0, 240.0, 160.0, 90.0, 40.0]),
        relaxation_times,
    )
    result = fit_positive_relaxation(time, observed, branch_count=6)
    assert result["positive_spectrum_gate_passed"] is True
    assert result["fit_gate_passed"] is True
    assert result["fit_normalized_rmse"] < 1.0e-8
    assert all(value >= 0.0 for value in result["branch_modulus_MPa"])


def test_censored_basquin_fit_keeps_runouts_as_survival_data():
    rng = np.random.default_rng(20260729)
    stress = np.repeat([0.40, 0.60, 0.75], 5)
    true_log_life = 3.0 - 8.0 * np.log10(stress)
    realized = 10.0 ** (
        true_log_life + rng.normal(0.0, 0.10, len(stress))
    )
    runout = 100_000.0
    failed = realized < runout
    observed = np.minimum(realized, runout)
    result = fit_censored_basquin(stress, observed, failed)
    assert result["optimizer_success"] is True
    assert result["quality_gate_passed"] is True
    assert result["failure_count"] >= 3
    assert result["runout_count"] >= 1
    assert result["log10_stress_fraction_slope"] < 0.0
    anchors = result["median_cycles_at_stress_fraction"]
    assert anchors["0.40"] > anchors["0.60"] > anchors["0.75"]
    assert "log-survival" in result["censoring_contract"]


def test_tensile_curve_metrics_extract_modulus_and_strength():
    strain = np.linspace(0.0, 0.04, 161)
    stress = np.minimum(2400.0 * strain, 48.0)
    metrics = tensile_curve_metrics(
        pd.DataFrame(
            {
                "engineering_strain": strain,
                "engineering_stress_MPa": stress,
            }
        )
    )
    assert metrics["modulus_MPa"] == pytest.approx(2400.0, rel=1e-6)
    assert metrics["ultimate_strength_MPa"] == pytest.approx(48.0)
    assert metrics["terminal_strain"] == pytest.approx(0.04)


def test_retention_registry_keeps_orientation_and_condition_bounds():
    rows = []
    for orientation in ("XY", "Z"):
        for condition, scale in (("dry", 1.0), ("wet", 0.8)):
            for replicate in (1, 2, 3):
                rows.append(
                    {
                        "orientation": orientation,
                        "conditioning": condition,
                        "pre_condition_mass_g": 2.0,
                        "post_condition_mass_g": (
                            2.0 if condition == "dry" else 2.01
                        ),
                        "modulus_MPa": 2400.0 * scale,
                        "ultimate_strength_MPa": 50.0 * scale,
                        "terminal_strain": 0.04 * scale,
                    }
                )
    result = _retention_registry(rows, "dry")
    assert result["quality_gate_passed"] is True
    wet = next(
        row
        for row in result["groups"]
        if row["orientation"] == "XY" and row["conditioning"] == "wet"
    )
    assert wet["modulus_MPa_retention_factor"] == pytest.approx(0.8)
    assert wet["ultimate_strength_MPa_retention_factor"] == pytest.approx(0.8)
    assert wet["mean_mass_change_fraction"] == pytest.approx(0.005)


def test_blank_campaign_calibration_releases_no_model_files(tmp_path: Path):
    campaign = tmp_path / "campaign"
    output = tmp_path / "calibration"
    write_campaign(campaign)
    payload = calibrate_submission(
        campaign / "capture_templates", campaign, output
    )
    assert payload["status"] == "awaiting_measurements"
    assert payload["released_stages"] == []
    assert (
        payload["synthetic_recovery"][
            "all_synthetic_recovery_gates_passed"
        ]
        is True
    )
    assert all(
        result["status"] == "not_released"
        for result in payload["stage_results"].values()
    )
    assert list(output.glob("stage_*.json")) == []


def test_complete_dma_stage_emits_only_the_bounded_dma_model(
    tmp_path: Path,
):
    campaign = tmp_path / "campaign"
    output = tmp_path / "calibration"
    write_campaign(campaign)
    plan = pd.read_csv(campaign / "frozen_plan.csv")
    for test_id in plan.loc[plan["stage"] == 1, "test_id"]:
        path = campaign / "capture_templates" / f"{test_id}.csv"
        frame = pd.read_csv(path)
        frame["material_lot"] = "lot-a"
        frame["material_color"] = "white"
        frame["machine_id"] = "dma-01"
        frame["print_profile"] = "service-law-v1"
        frame["drying_protocol"] = "8 h at 45 C"
        frame["ambient_temperature_C"] = 23.0
        frame["ambient_relative_humidity_percent"] = 20.0
        temperature = frame["test_temperature_C"].to_numpy(float)
        frequency = frame["frequency_Hz"].to_numpy(float)
        storage = (
            2500.0
            * np.exp(-0.009 * (temperature - 20.0))
            * (1.0 + 0.006 * np.log10(frequency))
        )
        loss = (
            70.0
            * np.exp(0.004 * (temperature - 20.0))
            * (1.0 + 0.03 * np.log10(frequency))
        )
        frame["storage_modulus_MPa"] = storage
        frame["loss_modulus_MPa"] = loss
        frame["tan_delta"] = loss / storage
        frame.to_csv(path, index=False)
    payload = calibrate_submission(
        campaign / "capture_templates", campaign, output
    )
    assert payload["status"] == "partial_service_laws_released"
    assert payload["released_stages"] == [1]
    assert payload["stage_results"]["1"]["quality_gate_passed"] is True
    assert Path(payload["stage_results"]["1"]["model_file"]).exists()
    assert all(
        payload["stage_results"][str(stage)]["status"] == "not_released"
        for stage in (2, 3, 4, 5)
    )
