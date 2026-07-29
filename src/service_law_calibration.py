from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.special import log_ndtr

from .service_law_campaign import (
    DEFAULT_OUTPUT as DEFAULT_CAMPAIGN,
    STAGE_LAWS,
    validate_submission,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/bambu_pla_service_law_calibration"
DMA_REFERENCE_TEMPERATURE_C = 23.0
RELAXATION_BRANCHES = 6


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dma_design(
    temperature_C: np.ndarray, frequency_Hz: np.ndarray
) -> np.ndarray:
    delta_temperature = (
        np.asarray(temperature_C, dtype=float)
        - DMA_REFERENCE_TEMPERATURE_C
    ) / 25.0
    log_frequency = np.log10(np.asarray(frequency_Hz, dtype=float))
    return np.column_stack(
        (
            np.ones_like(delta_temperature),
            delta_temperature,
            log_frequency,
            delta_temperature**2,
            log_frequency**2,
            delta_temperature * log_frequency,
        )
    )


def predict_dma_surface(
    coefficients: Sequence[float],
    temperature_C: Sequence[float] | np.ndarray,
    frequency_Hz: Sequence[float] | np.ndarray,
) -> np.ndarray:
    temperature = np.asarray(temperature_C, dtype=float)
    frequency = np.asarray(frequency_Hz, dtype=float)
    if temperature.shape != frequency.shape:
        raise ValueError("temperature and frequency arrays must have one shape")
    design = _dma_design(temperature, frequency)
    return np.exp(design @ np.asarray(coefficients, dtype=float))


def fit_bounded_dma_surface(frame: pd.DataFrame) -> Dict[str, object]:
    required = (
        "orientation",
        "replicate",
        "test_temperature_C",
        "frequency_Hz",
        "storage_modulus_MPa",
        "loss_modulus_MPa",
    )
    if not set(required).issubset(frame):
        raise ValueError("DMA frame is missing required columns")
    outputs: Dict[str, object] = {}
    for orientation in sorted(frame["orientation"].astype(str).unique()):
        selected = frame.loc[
            frame["orientation"].astype(str) == orientation
        ].copy()
        temperature = selected["test_temperature_C"].to_numpy(float)
        frequency = selected["frequency_Hz"].to_numpy(float)
        design = _dma_design(temperature, frequency)
        orientation_result: Dict[str, object] = {}
        for label, column in (
            ("storage", "storage_modulus_MPa"),
            ("loss", "loss_modulus_MPa"),
        ):
            observed = selected[column].to_numpy(float)
            if np.any(observed <= 0.0):
                raise ValueError("DMA moduli must be positive")
            coefficients = np.linalg.lstsq(
                design, np.log(observed), rcond=None
            )[0]
            prediction = np.exp(design @ coefficients)
            nrmse = float(
                np.sqrt(np.mean((prediction - observed) ** 2))
                / np.mean(observed)
            )
            fold_errors = []
            replicates = sorted(selected["replicate"].astype(int).unique())
            for replicate in replicates:
                validation = selected["replicate"].astype(int) == replicate
                training_design = design[~validation]
                training = np.log(observed[~validation])
                fold_coefficients = np.linalg.lstsq(
                    training_design, training, rcond=None
                )[0]
                fold_prediction = np.exp(
                    design[validation] @ fold_coefficients
                )
                fold_errors.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (
                                    fold_prediction
                                    - observed[validation]
                                )
                                ** 2
                            )
                        )
                        / np.mean(observed[validation])
                    )
                )
            orientation_result[label] = {
                "coefficients_log_MPa": coefficients.tolist(),
                "fit_normalized_rmse": nrmse,
                "leave_one_replicate_out_normalized_rmse": float(
                    np.mean(fold_errors)
                ),
            }
        storage_gate = (
            orientation_result["storage"][
                "leave_one_replicate_out_normalized_rmse"
            ]
            <= 0.10
        )
        loss_gate = (
            orientation_result["loss"][
                "leave_one_replicate_out_normalized_rmse"
            ]
            <= 0.15
        )
        outputs[orientation] = {
            **orientation_result,
            "temperature_range_C": [
                float(np.min(temperature)),
                float(np.max(temperature)),
            ],
            "frequency_range_Hz": [
                float(np.min(frequency)),
                float(np.max(frequency)),
            ],
            "row_count": len(selected),
            "quality_gate_passed": storage_gate and loss_gate,
        }
    return {
        "model": "bounded_log_quadratic_DMA_response_surface_v1",
        "reference_temperature_C": DMA_REFERENCE_TEMPERATURE_C,
        "features": [
            "1",
            "(T-23)/25",
            "log10(f)",
            "((T-23)/25)^2",
            "log10(f)^2",
            "((T-23)/25)*log10(f)",
        ],
        "orientations": outputs,
        "quality_gate_passed": bool(
            outputs
            and all(
                result["quality_gate_passed"]
                for result in outputs.values()
            )
        ),
        "restriction": (
            "interpolation only inside the sampled temperature-frequency "
            "rectangle; not a WLF master curve or arbitrary-rate law"
        ),
    }


def generalized_maxwell_relaxation(
    time_s: Sequence[float] | np.ndarray,
    equilibrium_modulus_MPa: float,
    branch_modulus_MPa: Sequence[float] | np.ndarray,
    relaxation_time_s: Sequence[float] | np.ndarray,
) -> np.ndarray:
    time = np.asarray(time_s, dtype=float)
    branches = np.asarray(branch_modulus_MPa, dtype=float)
    relaxation = np.asarray(relaxation_time_s, dtype=float)
    if np.any(time < 0.0) or np.any(branches < 0.0):
        raise ValueError("Maxwell inputs must be nonnegative")
    if equilibrium_modulus_MPa < 0.0 or np.any(relaxation <= 0.0):
        raise ValueError("Maxwell equilibrium/time inputs are inadmissible")
    return float(equilibrium_modulus_MPa) + np.sum(
        branches[np.newaxis, :]
        * np.exp(-time[:, np.newaxis] / relaxation[np.newaxis, :]),
        axis=1,
    )


def fit_positive_relaxation(
    time_s: Sequence[float] | np.ndarray,
    relaxation_modulus_MPa: Sequence[float] | np.ndarray,
    branch_count: int = RELAXATION_BRANCHES,
) -> Dict[str, object]:
    time = np.asarray(time_s, dtype=float)
    observed = np.asarray(relaxation_modulus_MPa, dtype=float)
    valid = np.isfinite(time) & np.isfinite(observed)
    time = time[valid]
    observed = observed[valid]
    order = np.argsort(time)
    time = time[order]
    observed = observed[order]
    if len(time) < 20 or np.min(time) <= 0.0 or np.any(observed <= 0.0):
        raise ValueError("relaxation fit needs >=20 positive time/modulus points")
    if branch_count < 2:
        raise ValueError("positive relaxation fit needs at least two branches")
    relaxation_times = np.geomspace(
        max(float(np.min(time)) / 3.0, 1.0e-6),
        float(np.max(time)) * 3.0,
        branch_count,
    )
    design = np.column_stack(
        (
            np.ones_like(time),
            np.exp(
                -time[:, np.newaxis]
                / relaxation_times[np.newaxis, :]
            ),
        )
    )
    scale = float(np.mean(observed))
    coefficients, residual_norm = nnls(design / scale, observed / scale)
    prediction = design @ coefficients
    equilibrium = float(coefficients[0])
    branches = coefficients[1:]
    nrmse = float(
        np.sqrt(np.mean((prediction - observed) ** 2))
        / np.mean(observed)
    )
    instantaneous = equilibrium + float(np.sum(branches))
    positive = bool(
        equilibrium >= 0.0
        and np.all(branches >= 0.0)
        and instantaneous > 0.0
    )
    return {
        "model": "positive_fixed_time_generalized_Maxwell_v1",
        "equilibrium_modulus_MPa": equilibrium,
        "branch_modulus_MPa": branches.tolist(),
        "relaxation_time_s": relaxation_times.tolist(),
        "instantaneous_modulus_MPa": instantaneous,
        "normalized_branch_modulus": (
            branches / instantaneous
        ).tolist(),
        "normalized_equilibrium_modulus": equilibrium / instantaneous,
        "fit_normalized_rmse": nrmse,
        "nnls_residual_norm": float(residual_norm),
        "positive_spectrum_gate_passed": positive,
        "fit_gate_passed": positive and nrmse <= 0.05,
        "time_range_s": [float(np.min(time)), float(np.max(time))],
        "point_count": len(time),
    }


def fit_censored_basquin(
    maximum_stress_fraction: Sequence[float] | np.ndarray,
    cycles: Sequence[float] | np.ndarray,
    failed: Sequence[bool] | np.ndarray,
) -> Dict[str, object]:
    stress = np.asarray(maximum_stress_fraction, dtype=float)
    life = np.asarray(cycles, dtype=float)
    failure = np.asarray(failed, dtype=bool)
    valid = (
        np.isfinite(stress)
        & np.isfinite(life)
        & (stress > 0.0)
        & (stress < 1.0)
        & (life > 0.0)
    )
    stress = stress[valid]
    life = life[valid]
    failure = failure[valid]
    if len(stress) < 10 or np.sum(failure) < 3:
        raise ValueError("censored Basquin fit needs >=10 tests and >=3 failures")
    x = np.log10(stress)
    y = np.log10(life)
    if len(np.unique(np.round(stress, 8))) < 3:
        raise ValueError("censored Basquin fit needs three stress levels")
    if np.sum(failure) >= 2:
        slope, intercept = np.polyfit(x[failure], y[failure], 1)
    else:
        slope, intercept = -6.0, float(np.mean(y))
    slope = min(float(slope), -0.1)
    residual_scale = max(
        float(np.std(y[failure] - (intercept + slope * x[failure]), ddof=1))
        if np.sum(failure) >= 3
        else 0.25,
        0.08,
    )

    def unpack(parameters: np.ndarray) -> tuple[float, float, float]:
        current_intercept = float(parameters[0])
        current_slope = -math.exp(float(parameters[1]))
        sigma = math.exp(float(parameters[2]))
        return current_intercept, current_slope, sigma

    def objective(parameters: np.ndarray) -> float:
        current_intercept, current_slope, sigma = unpack(parameters)
        mean = current_intercept + current_slope * x
        z = (y - mean) / sigma
        log_likelihood = np.empty(len(y), dtype=float)
        log_likelihood[failure] = (
            -0.5 * z[failure] ** 2
            - math.log(sigma)
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_likelihood[~failure] = log_ndtr(-z[~failure])
        return float(-np.sum(log_likelihood))

    initial = np.asarray(
        [intercept, math.log(-slope), math.log(residual_scale)],
        dtype=float,
    )
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=((-20.0, 30.0), (-5.0, 6.0), (-5.0, 2.0)),
    )
    fitted_intercept, fitted_slope, sigma = unpack(result.x)
    anchors = (0.40, 0.60, 0.75)
    median_life = {
        f"{fraction:.2f}": float(
            10.0
            ** (
                fitted_intercept
                + fitted_slope * math.log10(fraction)
            )
        )
        for fraction in anchors
    }
    monotonic = (
        median_life["0.40"] > median_life["0.60"] > median_life["0.75"]
    )
    return {
        "model": "right_censored_lognormal_Basquin_v1",
        "log10_life_intercept": fitted_intercept,
        "log10_stress_fraction_slope": fitted_slope,
        "log10_life_scatter_sigma": sigma,
        "specimen_count": len(stress),
        "failure_count": int(np.sum(failure)),
        "runout_count": int(np.sum(~failure)),
        "stress_fraction_range": [
            float(np.min(stress)),
            float(np.max(stress)),
        ],
        "median_cycles_at_stress_fraction": median_life,
        "negative_log_likelihood": float(result.fun),
        "optimizer_success": bool(result.success),
        "monotonic_life_gate_passed": monotonic,
        "quality_gate_passed": bool(
            result.success
            and fitted_slope < 0.0
            and 0.01 < sigma < 2.0
            and monotonic
        ),
        "censoring_contract": (
            "runout cycle counts contribute log-survival likelihood and are "
            "never treated as failures"
        ),
    }


def tensile_curve_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    strain = pd.to_numeric(
        frame["engineering_strain"], errors="coerce"
    ).to_numpy(float)
    stress = pd.to_numeric(
        frame["engineering_stress_MPa"], errors="coerce"
    ).to_numpy(float)
    valid = np.isfinite(strain) & np.isfinite(stress)
    strain = strain[valid]
    stress = stress[valid]
    order = np.argsort(strain)
    strain = strain[order]
    stress = stress[order]
    elastic = (strain >= 0.0005) & (strain <= 0.0030)
    if np.sum(elastic) < 5:
        elastic = (strain >= 0.0) & (strain <= 0.004)
    if np.sum(elastic) < 5:
        raise ValueError("tensile curve has too few elastic-region points")
    design = np.column_stack(
        (strain[elastic], np.ones(int(np.sum(elastic))))
    )
    modulus, intercept = np.linalg.lstsq(
        design, stress[elastic], rcond=None
    )[0]
    prediction = modulus * strain[elastic] + intercept
    residual = float(np.sum((stress[elastic] - prediction) ** 2))
    total = float(
        np.sum((stress[elastic] - np.mean(stress[elastic])) ** 2)
    )
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    if modulus <= 0.0:
        raise ValueError("tensile curve produced nonpositive modulus")
    return {
        "modulus_MPa": float(modulus),
        "elastic_intercept_MPa": float(intercept),
        "elastic_fit_r_squared": r_squared,
        "ultimate_strength_MPa": float(np.max(stress)),
        "terminal_strain": float(np.max(strain)),
    }


def _load_stage_frames(
    input_dir: Path,
    audit: Mapping[str, object],
    stage: int,
) -> List[tuple[Mapping[str, object], pd.DataFrame]]:
    selected = [
        row
        for row in audit["specimens"]
        if int(row["stage"]) == stage and row["status"] == "valid"
    ]
    return [
        (
            row,
            pd.read_csv(Path(input_dir) / f"{row['test_id']}.csv"),
        )
        for row in selected
    ]


def _calibrate_dma(
    frames: Sequence[tuple[Mapping[str, object], pd.DataFrame]]
) -> Dict[str, object]:
    combined = pd.concat(
        [frame.assign(_test_id=row["test_id"]) for row, frame in frames],
        ignore_index=True,
    )
    return fit_bounded_dma_surface(combined)


def _calibrate_relaxation(
    frames: Sequence[tuple[Mapping[str, object], pd.DataFrame]]
) -> Dict[str, object]:
    groups: Dict[tuple[str, float], List[pd.DataFrame]] = {}
    for row, frame in frames:
        temperature = float(frame["test_temperature_C"].iloc[0])
        groups.setdefault((str(row["orientation"]), temperature), []).append(
            frame
        )
    models = {}
    for (orientation, temperature), group_frames in sorted(groups.items()):
        reference_times = np.geomspace(1.0, 14_400.0, 81)
        interpolated = []
        for frame in group_frames:
            time = frame["time_s"].to_numpy(float)
            strain = frame["target_strain"].to_numpy(float)
            stress = frame["engineering_stress_MPa"].to_numpy(float)
            modulus = stress / strain
            interpolated.append(
                np.interp(
                    np.log(reference_times),
                    np.log(time),
                    modulus,
                )
            )
        mean_modulus = np.mean(interpolated, axis=0)
        model = fit_positive_relaxation(
            reference_times, mean_modulus, RELAXATION_BRANCHES
        )
        model["replicates"] = len(group_frames)
        model["orientation"] = orientation
        model["temperature_C"] = temperature
        models[f"{orientation}_{temperature:g}C"] = model
    return {
        "model": "orientation_temperature_positive_relaxation_registry_v1",
        "models": models,
        "quality_gate_passed": bool(
            models
            and all(model["fit_gate_passed"] for model in models.values())
        ),
        "restriction": (
            "interpolate only at the measured orientations and 23/40/50 C; "
            "no time-temperature superposition is assumed"
        ),
    }


def _curve_metric_rows(
    frames: Sequence[tuple[Mapping[str, object], pd.DataFrame]]
) -> List[Dict[str, object]]:
    rows = []
    for audit_row, frame in frames:
        metrics = tensile_curve_metrics(frame)
        rows.append(
            {
                "test_id": audit_row["test_id"],
                "orientation": str(frame["orientation"].iloc[0]),
                "conditioning": str(frame["conditioning"].iloc[0]),
                "replicate": int(frame["replicate"].iloc[0]),
                "pre_condition_mass_g": float(
                    frame["pre_condition_mass_g"].iloc[0]
                ),
                "post_condition_mass_g": float(
                    frame["post_condition_mass_g"].iloc[0]
                ),
                **metrics,
            }
        )
    return rows


def _retention_registry(
    metric_rows: Sequence[Mapping[str, object]],
    control_condition: str,
) -> Dict[str, object]:
    metrics = (
        "modulus_MPa",
        "ultimate_strength_MPa",
        "terminal_strain",
    )
    outputs = []
    orientations = sorted(
        {str(row["orientation"]) for row in metric_rows}
    )
    for orientation in orientations:
        selected = [
            row for row in metric_rows if row["orientation"] == orientation
        ]
        controls = [
            row
            for row in selected
            if row["conditioning"] == control_condition
        ]
        if not controls:
            raise ValueError(f"missing {control_condition} controls")
        for condition in sorted(
            {str(row["conditioning"]) for row in selected}
        ):
            current = [
                row for row in selected if row["conditioning"] == condition
            ]
            result = {
                "orientation": orientation,
                "conditioning": condition,
                "specimens": len(current),
                "control_conditioning": control_condition,
                "mean_mass_change_fraction": float(
                    np.mean(
                        [
                            float(row["post_condition_mass_g"])
                            / float(row["pre_condition_mass_g"])
                            - 1.0
                            for row in current
                        ]
                    )
                ),
            }
            for metric in metrics:
                control_mean = float(
                    np.mean([float(row[metric]) for row in controls])
                )
                current_mean = float(
                    np.mean([float(row[metric]) for row in current])
                )
                result[f"mean_{metric}"] = current_mean
                result[f"{metric}_retention_factor"] = (
                    current_mean / control_mean
                )
            outputs.append(result)
    return {
        "model": "matched_group_mechanical_retention_factors_v1",
        "control_condition": control_condition,
        "groups": outputs,
        "quality_gate_passed": bool(
            outputs and all(row["specimens"] >= 3 for row in outputs)
        ),
        "restriction": (
            "multiplicative screening factors only for the tested material "
            "lot, orientation, exposure, and room-temperature tensile method"
        ),
    }


def _calibrate_fatigue(
    frames: Sequence[tuple[Mapping[str, object], pd.DataFrame]]
) -> Dict[str, object]:
    specimen_rows = []
    for audit_row, frame in frames:
        specimen_rows.append(
            {
                "test_id": audit_row["test_id"],
                "orientation": str(frame["orientation"].iloc[0]),
                "stress_fraction": float(
                    frame[
                        "maximum_stress_fraction_of_directional_UTS"
                    ].iloc[0]
                ),
                "cycles": float(frame["cycle_count"].iloc[-1]),
                "failed": str(frame["outcome"].iloc[0]) == "failure",
            }
        )
    models = {}
    for orientation in sorted(
        {str(row["orientation"]) for row in specimen_rows}
    ):
        selected = [
            row
            for row in specimen_rows
            if row["orientation"] == orientation
        ]
        models[orientation] = fit_censored_basquin(
            [row["stress_fraction"] for row in selected],
            [row["cycles"] for row in selected],
            [row["failed"] for row in selected],
        )
    return {
        "model": "directional_right_censored_Basquin_registry_v1",
        "models": models,
        "quality_gate_passed": bool(
            models
            and all(model["quality_gate_passed"] for model in models.values())
        ),
        "restriction": (
            "R=0.1, 5 Hz, dry 23 C tension-tension life only; not a "
            "multiaxial damage accumulation or variable-amplitude law"
        ),
    }


def synthetic_recovery_check() -> Dict[str, object]:
    storage_coefficients = np.asarray(
        [7.75, -0.24, 0.018, -0.08, 0.004, -0.01]
    )
    loss_coefficients = np.asarray(
        [4.30, 0.12, 0.09, 0.02, -0.01, 0.015]
    )
    dma_rows = []
    for orientation, scale in (("XY", 1.0), ("Z", 0.72)):
        for replicate in (1, 2, 3):
            replicate_scale = 1.0 + (replicate - 2) * 0.003
            for temperature in (20, 30, 40, 50, 60, 70):
                for frequency in (0.1, 1.0, 10.0):
                    storage = predict_dma_surface(
                        storage_coefficients,
                        np.asarray([temperature]),
                        np.asarray([frequency]),
                    )[0]
                    loss = predict_dma_surface(
                        loss_coefficients,
                        np.asarray([temperature]),
                        np.asarray([frequency]),
                    )[0]
                    dma_rows.append(
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
    dma = fit_bounded_dma_surface(pd.DataFrame(dma_rows))

    time = np.geomspace(1.0, 14_400.0, 81)
    relaxation_times = np.geomspace(
        1.0 / 3.0, 43_200.0, RELAXATION_BRANCHES
    )
    relaxation_observed = generalized_maxwell_relaxation(
        time,
        850.0,
        np.asarray([500.0, 350.0, 240.0, 160.0, 90.0, 40.0]),
        relaxation_times,
    )
    relaxation = fit_positive_relaxation(
        time, relaxation_observed, RELAXATION_BRANCHES
    )

    rng = np.random.default_rng(20260729)
    stress = np.repeat([0.40, 0.60, 0.75], 5)
    latent_life = 10.0 ** (
        3.0
        - 8.0 * np.log10(stress)
        + rng.normal(0.0, 0.10, len(stress))
    )
    failed = latent_life < 100_000.0
    fatigue = fit_censored_basquin(
        stress, np.minimum(latent_life, 100_000.0), failed
    )

    strain = np.linspace(0.0, 0.04, 161)
    curve = tensile_curve_metrics(
        pd.DataFrame(
            {
                "engineering_strain": strain,
                "engineering_stress_MPa": np.minimum(
                    2400.0 * strain, 48.0
                ),
            }
        )
    )
    curve_gate = (
        abs(curve["modulus_MPa"] / 2400.0 - 1.0) <= 1.0e-6
        and abs(curve["ultimate_strength_MPa"] / 48.0 - 1.0)
        <= 1.0e-12
    )
    gates = {
        "DMA_heldout_replicate_recovery": dma["quality_gate_passed"],
        "positive_relaxation_spectrum_recovery": (
            relaxation["fit_gate_passed"]
        ),
        "right_censored_fatigue_recovery": (
            fatigue["quality_gate_passed"]
            and fatigue["runout_count"] >= 1
        ),
        "tensile_metric_recovery": curve_gate,
    }
    return {
        "gates": gates,
        "all_synthetic_recovery_gates_passed": all(gates.values()),
        "metrics": {
            "maximum_DMA_leave_one_replicate_out_NRMSE": max(
                float(response[metric][
                    "leave_one_replicate_out_normalized_rmse"
                ])
                for response in dma["orientations"].values()
                for metric in ("storage", "loss")
            ),
            "relaxation_fit_NRMSE": relaxation[
                "fit_normalized_rmse"
            ],
            "fatigue_failures": fatigue["failure_count"],
            "fatigue_runouts": fatigue["runout_count"],
            "fatigue_slope": fatigue[
                "log10_stress_fraction_slope"
            ],
            "tensile_modulus_relative_error": abs(
                curve["modulus_MPa"] / 2400.0 - 1.0
            ),
        },
    }


def calibrate_submission(
    input_dir: Path,
    campaign_root: Path = DEFAULT_CAMPAIGN,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    input_dir = Path(input_dir)
    campaign_root = Path(campaign_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intake = validate_submission(
        input_dir,
        campaign_root / "frozen_plan.csv",
        campaign_root / "campaign_manifest.json",
    )
    stage_results: Dict[str, Dict[str, object]] = {}
    calibrators = {
        1: _calibrate_dma,
        2: _calibrate_relaxation,
        3: lambda frames: _retention_registry(
            _curve_metric_rows(frames), "dry_<20RH"
        ),
        4: _calibrate_fatigue,
        5: lambda frames: _retention_registry(
            _curve_metric_rows(frames), "unaged_control"
        ),
    }
    for stage, law in STAGE_LAWS.items():
        intake_stage = intake["stage_status"][str(stage)]
        if intake_stage["status"] != "ready_for_calibration":
            stage_results[str(stage)] = {
                "law": law,
                "status": "not_released",
                "reason": intake_stage["status"],
                "quality_gate_passed": False,
                "model_file": None,
            }
            continue
        model = calibrators[stage](
            _load_stage_frames(input_dir, intake, stage)
        )
        model_path = output_dir / f"stage_{stage}_{law}.json"
        model_path.write_text(
            json.dumps(model, indent=2) + "\n", encoding="utf-8"
        )
        stage_results[str(stage)] = {
            "law": law,
            "status": (
                "released_within_tested_bounds"
                if model.get("quality_gate_passed") is True
                else "fit_quality_gate_failed"
            ),
            "quality_gate_passed": model.get("quality_gate_passed") is True,
            "model_file": str(model_path.resolve()),
            "model_file_sha256": _sha256(model_path),
        }
        if stage in {3, 5}:
            _write_csv(
                output_dir / f"stage_{stage}_retention_factors.csv",
                model["groups"],
            )
    released = [
        int(stage)
        for stage, result in stage_results.items()
        if result["status"] == "released_within_tested_bounds"
    ]
    payload = {
        "calibration_id": "bambu_pla_basic_service_law_calibration_v1",
        "material": "Bambu PLA Basic",
        "status": (
            "all_service_laws_released"
            if len(released) == len(STAGE_LAWS)
            else "partial_service_laws_released"
            if released
            else "awaiting_measurements"
        ),
        "campaign_manifest": str(
            (campaign_root / "campaign_manifest.json").resolve()
        ),
        "campaign_manifest_sha256": _sha256(
            campaign_root / "campaign_manifest.json"
        ),
        "input_directory": str(input_dir.resolve()),
        "intake_status": intake["status"],
        "released_stages": released,
        "stage_results": stage_results,
        "synthetic_recovery": synthetic_recovery_check(),
        "selector_contract": {
            "no_fallback": True,
            "release_requires_intake_and_fit_quality_gates": True,
            "fatigue_is_postprocessing_not_a_CalculiX_material_card": True,
            "relaxation_requires_frequency_or_time_specific_proxy_use": True,
            "retention_factors_are_bounded_sensitivities": True,
        },
    }
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Bambu PLA Basic service-law calibration",
        "",
        f"Status: **{payload['status']}**.",
        "",
        "| Stage | Law | Status |",
        "|---:|---|---|",
    ]
    for stage, result in stage_results.items():
        lines.append(
            f"| {stage} | {result['law']} | {result['status']} |"
        )
    lines.extend(
        [
            "",
            "No incomplete or failed stage produces a model file. Released "
            "models remain bounded to their frozen test orientation, "
            "temperature, humidity, frequency, stress ratio, and exposure.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate bounded Bambu PLA service laws"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CAMPAIGN / "capture_templates",
    )
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = calibrate_submission(args.input, args.campaign, args.output)
    print(
        f"service-law calibration: status={payload['status']}; "
        f"released={len(payload['released_stages'])}/5; "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
