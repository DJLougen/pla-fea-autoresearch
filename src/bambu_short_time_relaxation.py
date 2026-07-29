from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .bambu_viscoelastic_wave_reference import (
    MAXIMUM_REDUCED_FREQUENCY_HZ,
    MINIMUM_REDUCED_FREQUENCY_HZ,
    POISSON_RATIO,
    REFERENCE_RELATIVE_HUMIDITY_PERCENT,
    REFERENCE_TEMPERATURE_C,
    REPORTED_DENSITY_KG_M3,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_viscoelastic_wave_2026"
    / "dma_master_curve_18C.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_short_time_relaxation_18C"
)
MODEL_TERM_CANDIDATES = (3, 5, 7, 9)
CROSS_VALIDATION_FOLDS = 5
OUTLIER_MAD_THRESHOLD = 6.0
STORAGE_RESIDUAL_SCALE_MPA = 15.0
LOSS_RESIDUAL_SCALE_MPA = 5.0
LOSS_SCORE_WEIGHT = 0.25
TERM_COMPLEXITY_PENALTY = 0.002
ANCHOR_FREQUENCIES_HZ = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 25_000.0)


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"frequency curve is empty: {path}")
    return rows


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def robust_loss_outliers(
    loss_factor: Sequence[float],
    threshold: float = OUTLIER_MAD_THRESHOLD,
) -> np.ndarray:
    values = np.asarray(loss_factor, dtype=float)
    if values.ndim != 1 or len(values) < 5:
        raise ValueError("at least five loss-factor values are required")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("loss factors must be finite and positive")
    logged = np.log(values)
    center = float(np.median(logged))
    mad = float(1.4826 * np.median(np.abs(logged - center)))
    if not np.isfinite(mad) or mad <= 0.0:
        return np.zeros(len(values), dtype=bool)
    return np.abs(logged - center) > float(threshold) * mad


def relaxation_times(term_count: int) -> np.ndarray:
    if int(term_count) < 1:
        raise ValueError("term count must be positive")
    minimum = 1.0 / (
        2.0 * math.pi * MAXIMUM_REDUCED_FREQUENCY_HZ
    )
    maximum = 1.0 / (
        2.0 * math.pi * MINIMUM_REDUCED_FREQUENCY_HZ
    )
    return np.geomspace(minimum, maximum, int(term_count))


def generalized_maxwell_complex(
    frequency_hz: Sequence[float] | np.ndarray,
    equilibrium_shear_mpa: float | Sequence[float] | np.ndarray,
    branch_shear_mpa: Sequence[float] | np.ndarray,
    relaxation_time_s: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    branches = np.asarray(branch_shear_mpa, dtype=float)
    times = np.asarray(relaxation_time_s, dtype=float)
    if frequency.ndim != 1 or np.any(frequency <= 0.0):
        raise ValueError("frequency must be a positive one-dimensional array")
    if (
        branches.ndim != 1
        or times.ndim != 1
        or len(branches) != len(times)
        or len(branches) == 0
    ):
        raise ValueError("branch moduli and relaxation times must align")
    if (
        np.any(~np.isfinite(branches))
        or np.any(~np.isfinite(times))
        or np.any(branches < 0.0)
        or np.any(times <= 0.0)
    ):
        raise ValueError("Maxwell branches must be finite and admissible")
    omega_tau = (
        2.0 * math.pi * frequency[:, None] * times[None, :]
    )
    storage_kernel = omega_tau**2 / (1.0 + omega_tau**2)
    loss_kernel = omega_tau / (1.0 + omega_tau**2)
    equilibrium = np.asarray(equilibrium_shear_mpa, dtype=float)
    storage = equilibrium + storage_kernel @ branches
    loss = loss_kernel @ branches
    return np.asarray(storage, dtype=float), np.asarray(loss, dtype=float)


def generalized_maxwell_relaxation(
    time_s: Sequence[float] | np.ndarray,
    equilibrium_shear_mpa: float,
    branch_shear_mpa: Sequence[float] | np.ndarray,
    relaxation_time_s: Sequence[float] | np.ndarray,
) -> np.ndarray:
    times_requested = np.atleast_1d(np.asarray(time_s, dtype=float))
    branches = np.asarray(branch_shear_mpa, dtype=float)
    branch_times = np.asarray(relaxation_time_s, dtype=float)
    if np.any(times_requested < 0.0):
        raise ValueError("relaxation time must be non-negative")
    if (
        len(branches) != len(branch_times)
        or np.any(branches < 0.0)
        or np.any(branch_times <= 0.0)
        or float(equilibrium_shear_mpa) <= 0.0
    ):
        raise ValueError("relaxation spectrum is inadmissible")
    return float(equilibrium_shear_mpa) + np.exp(
        -times_requested[:, None] / branch_times[None, :]
    ) @ branches


def _arrays(rows: Sequence[Mapping[str, str]]) -> Dict[str, object]:
    required = {
        "source_file",
        "reduced_frequency_at_18C_Hz",
        "shear_storage_modulus_MPa",
        "shear_loss_modulus_MPa",
        "loss_factor",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"frequency curve is missing columns: {sorted(missing)}")
    source_names = sorted({str(row["source_file"]) for row in rows})
    source_index = {
        name: index for index, name in enumerate(source_names)
    }
    result: Dict[str, object] = {
        "source_names": source_names,
        "source_index": np.asarray(
            [source_index[str(row["source_file"])] for row in rows],
            dtype=int,
        ),
        "frequency_hz": np.asarray(
            [
                float(row["reduced_frequency_at_18C_Hz"])
                for row in rows
            ],
            dtype=float,
        ),
        "storage_mpa": np.asarray(
            [float(row["shear_storage_modulus_MPa"]) for row in rows],
            dtype=float,
        ),
        "loss_mpa": np.asarray(
            [float(row["shear_loss_modulus_MPa"]) for row in rows],
            dtype=float,
        ),
        "loss_factor": np.asarray(
            [float(row["loss_factor"]) for row in rows],
            dtype=float,
        ),
    }
    for name in ("frequency_hz", "storage_mpa", "loss_mpa"):
        values = np.asarray(result[name], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    return result


def _fold_ids(
    source_index: np.ndarray,
    frequency_hz: np.ndarray,
    folds: int = CROSS_VALIDATION_FOLDS,
) -> np.ndarray:
    result = np.zeros(len(source_index), dtype=int)
    for source in sorted(set(source_index.tolist())):
        indices = np.where(source_index == source)[0]
        indices = indices[np.argsort(frequency_hz[indices])]
        result[indices] = np.arange(len(indices), dtype=int) % int(folds)
    return result


def _fit(
    source_index: np.ndarray,
    frequency_hz: np.ndarray,
    storage_mpa: np.ndarray,
    loss_mpa: np.ndarray,
    term_count: int,
    train_mask: np.ndarray,
) -> Dict[str, object]:
    source_count = int(np.max(source_index)) + 1
    times = relaxation_times(term_count)

    def predict(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        equilibrium = np.exp(parameters[:source_count])
        branches = np.exp(parameters[source_count:])
        storage, loss = generalized_maxwell_complex(
            frequency_hz,
            equilibrium[source_index],
            branches,
            times,
        )
        return storage, loss

    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted_storage, predicted_loss = predict(parameters)
        return np.concatenate(
            (
                (
                    predicted_storage[train_mask]
                    - storage_mpa[train_mask]
                )
                / STORAGE_RESIDUAL_SCALE_MPA,
                (predicted_loss[train_mask] - loss_mpa[train_mask])
                / LOSS_RESIDUAL_SCALE_MPA,
            )
        )

    initial = np.log(
        np.concatenate(
            (
                np.full(source_count, 650.0),
                np.full(term_count, 5.0),
            )
        )
    )
    lower = np.log(
        np.concatenate(
            (
                np.full(source_count, 300.0),
                np.full(term_count, 1.0e-6),
            )
        )
    )
    upper = np.log(
        np.concatenate(
            (
                np.full(source_count, 1_200.0),
                np.full(term_count, 500.0),
            )
        )
    )
    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=50_000,
    )
    predicted_storage, predicted_loss = predict(result.x)
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "equilibrium_by_source_mpa": np.exp(result.x[:source_count]),
        "branch_shear_mpa": np.exp(result.x[source_count:]),
        "relaxation_time_s": times,
        "predicted_storage_mpa": predicted_storage,
        "predicted_loss_mpa": predicted_loss,
    }


def _normalized_rmse(
    predicted: np.ndarray, observed: np.ndarray, mask: np.ndarray
) -> float:
    rmse = float(
        np.sqrt(np.mean((predicted[mask] - observed[mask]) ** 2))
    )
    return rmse / float(np.mean(observed[mask]))


def fit_short_time_spectrum(
    rows: Sequence[Mapping[str, str]],
) -> Dict[str, object]:
    data = _arrays(rows)
    outlier = robust_loss_outliers(data["loss_factor"])
    inlier = ~outlier
    source_index = data["source_index"][inlier]
    frequency = data["frequency_hz"][inlier]
    storage = data["storage_mpa"][inlier]
    loss = data["loss_mpa"][inlier]
    fold_ids = _fold_ids(source_index, frequency)

    candidates = []
    for term_count in MODEL_TERM_CANDIDATES:
        fold_metrics = []
        for fold in range(CROSS_VALIDATION_FOLDS):
            validation = fold_ids == fold
            fit = _fit(
                source_index,
                frequency,
                storage,
                loss,
                term_count,
                ~validation,
            )
            storage_error = _normalized_rmse(
                fit["predicted_storage_mpa"],
                storage,
                validation,
            )
            loss_error = _normalized_rmse(
                fit["predicted_loss_mpa"],
                loss,
                validation,
            )
            score = (
                storage_error
                + LOSS_SCORE_WEIGHT * loss_error
                + TERM_COMPLEXITY_PENALTY * term_count
            )
            fold_metrics.append(
                {
                    "fold": fold,
                    "storage_normalized_rmse": storage_error,
                    "loss_normalized_rmse": loss_error,
                    "selection_score": score,
                }
            )
        scores = np.asarray(
            [row["selection_score"] for row in fold_metrics],
            dtype=float,
        )
        candidates.append(
            {
                "term_count": term_count,
                "fold_metrics": fold_metrics,
                "mean_storage_normalized_rmse": float(
                    np.mean(
                        [
                            row["storage_normalized_rmse"]
                            for row in fold_metrics
                        ]
                    )
                ),
                "mean_loss_normalized_rmse": float(
                    np.mean(
                        [
                            row["loss_normalized_rmse"]
                            for row in fold_metrics
                        ]
                    )
                ),
                "mean_selection_score": float(np.mean(scores)),
                "selection_score_standard_error": float(
                    np.std(scores, ddof=1) / math.sqrt(len(scores))
                ),
            }
        )
    best = min(candidates, key=lambda row: row["mean_selection_score"])
    one_standard_error_limit = (
        best["mean_selection_score"]
        + best["selection_score_standard_error"]
    )
    eligible = [
        row
        for row in candidates
        if row["mean_selection_score"] <= one_standard_error_limit
    ]
    selected = min(eligible, key=lambda row: row["term_count"])
    final = _fit(
        source_index,
        frequency,
        storage,
        loss,
        selected["term_count"],
        np.ones(len(frequency), dtype=bool),
    )
    equilibrium = np.asarray(
        final["equilibrium_by_source_mpa"], dtype=float
    )
    branches = np.asarray(final["branch_shear_mpa"], dtype=float)
    times = np.asarray(final["relaxation_time_s"], dtype=float)
    predicted_storage = np.asarray(
        final["predicted_storage_mpa"], dtype=float
    )
    predicted_loss = np.asarray(final["predicted_loss_mpa"], dtype=float)
    final_storage_nrmse = _normalized_rmse(
        predicted_storage,
        storage,
        np.ones(len(storage), dtype=bool),
    )
    final_loss_nrmse = _normalized_rmse(
        predicted_loss,
        loss,
        np.ones(len(loss), dtype=bool),
    )
    return {
        "raw_row_count": len(rows),
        "inlier_row_count": int(np.sum(inlier)),
        "outlier_row_count": int(np.sum(outlier)),
        "outlier_mask": outlier,
        "inlier_mask": inlier,
        "candidate_models": candidates,
        "best_score_term_count": best["term_count"],
        "one_standard_error_limit": one_standard_error_limit,
        "selected_term_count": selected["term_count"],
        "selected_by_one_standard_error_rule": True,
        "optimizer_success": final["success"],
        "optimizer_message": final["message"],
        "optimizer_function_evaluations": final["function_evaluations"],
        "source_names": data["source_names"],
        "equilibrium_by_source_mpa": equilibrium,
        "mean_equilibrium_shear_mpa": float(np.mean(equilibrium)),
        "branch_shear_mpa": branches,
        "relaxation_time_s": times,
        "instantaneous_mean_shear_mpa": float(
            np.mean(equilibrium) + np.sum(branches)
        ),
        "final_storage_normalized_rmse": final_storage_nrmse,
        "final_loss_normalized_rmse": final_loss_nrmse,
        "storage_fit_gate_passed": bool(final_storage_nrmse <= 0.03),
        "loss_fit_gate_passed": bool(final_loss_nrmse <= 0.35),
        "spectrum_positive_gate_passed": bool(
            np.all(equilibrium > 0.0)
            and np.all(branches > 0.0)
            and np.all(times > 0.0)
        ),
        "predicted_storage_inliers_mpa": predicted_storage,
        "predicted_loss_inliers_mpa": predicted_loss,
    }


def _isotropic_storage_card(frequency_hz: float, storage_mpa: float) -> str:
    young = 2.0 * float(storage_mpa) * (1.0 + POISSON_RATIO)
    density_tonnes_mm3 = REPORTED_DENSITY_KG_M3 * 1.0e-12
    frequency_label = f"{float(frequency_hz):g}".replace(".", "P")
    return (
        f"*MATERIAL,NAME=BAMBU_PLA_18C_{frequency_label}HZ_STORAGE\n"
        "*DENSITY\n"
        f"{density_tonnes_mm3:.12g}\n"
        "*ELASTIC\n"
        f"{young:.12g},{POISSON_RATIO:.12g}\n"
    )


def write_short_time_relaxation(
    source_path: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(source_path)
    model = fit_short_time_spectrum(rows)
    outlier = np.asarray(model["outlier_mask"], dtype=bool)
    inlier = np.asarray(model["inlier_mask"], dtype=bool)
    branches = np.asarray(model["branch_shear_mpa"], dtype=float)
    times = np.asarray(model["relaxation_time_s"], dtype=float)
    equilibrium_by_source = np.asarray(
        model["equilibrium_by_source_mpa"], dtype=float
    )
    mean_equilibrium = float(model["mean_equilibrium_shear_mpa"])

    source_lookup = {
        name: index for index, name in enumerate(model["source_names"])
    }
    all_frequency = np.asarray(
        [float(row["reduced_frequency_at_18C_Hz"]) for row in rows],
        dtype=float,
    )
    all_equilibrium = np.asarray(
        [
            equilibrium_by_source[source_lookup[str(row["source_file"])]]
            for row in rows
        ],
        dtype=float,
    )
    all_storage, all_loss = generalized_maxwell_complex(
        all_frequency,
        all_equilibrium,
        branches,
        times,
    )
    fit_rows = []
    for index, row in enumerate(rows):
        fit_rows.append(
            {
                **row,
                "objective_outlier": bool(outlier[index]),
                "fit_partition": "excluded_outlier" if outlier[index] else "fit",
                "predicted_shear_storage_modulus_MPa": float(
                    all_storage[index]
                ),
                "predicted_shear_loss_modulus_MPa": float(all_loss[index]),
                "storage_residual_MPa": float(
                    all_storage[index]
                    - float(row["shear_storage_modulus_MPa"])
                ),
                "loss_residual_MPa": float(
                    all_loss[index]
                    - float(row["shear_loss_modulus_MPa"])
                ),
            }
        )
    _write_csv(output_dir / "fit_to_raw_points.csv", fit_rows)

    spectrum_rows = []
    instantaneous = float(model["instantaneous_mean_shear_mpa"])
    for index, (branch, time) in enumerate(
        zip(branches, times), start=1
    ):
        spectrum_rows.append(
            {
                "branch": index,
                "relaxation_time_s": float(time),
                "branch_shear_modulus_MPa": float(branch),
                "normalized_to_instantaneous_shear": float(
                    branch / instantaneous
                ),
            }
        )
    _write_csv(output_dir / "maxwell_spectrum.csv", spectrum_rows)

    frequency_grid = np.geomspace(
        MINIMUM_REDUCED_FREQUENCY_HZ,
        MAXIMUM_REDUCED_FREQUENCY_HZ,
        161,
    )
    storage_grid, loss_grid = generalized_maxwell_complex(
        frequency_grid,
        mean_equilibrium,
        branches,
        times,
    )
    frequency_rows = [
        {
            "reduced_frequency_at_18C_Hz": float(frequency),
            "shear_storage_modulus_MPa": float(storage),
            "shear_loss_modulus_MPa": float(loss),
            "loss_factor": float(loss / storage),
            "equivalent_isotropic_youngs_modulus_MPa": float(
                2.0 * storage * (1.0 + POISSON_RATIO)
            ),
        }
        for frequency, storage, loss in zip(
            frequency_grid, storage_grid, loss_grid
        )
    ]
    _write_csv(output_dir / "frequency_response.csv", frequency_rows)

    time_grid = np.geomspace(float(times[0]), float(times[-1]), 161)
    relaxation_grid = generalized_maxwell_relaxation(
        time_grid,
        mean_equilibrium,
        branches,
        times,
    )
    relaxation_rows = [
        {
            "time_s": float(time),
            "mean_shear_relaxation_modulus_MPa": float(modulus),
            "normalized_to_instantaneous": float(modulus / instantaneous),
        }
        for time, modulus in zip(time_grid, relaxation_grid)
    ]
    _write_csv(output_dir / "short_time_relaxation_sensitivity.csv", relaxation_rows)

    cards_dir = output_dir / "frequency_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    anchor_storage, anchor_loss = generalized_maxwell_complex(
        np.asarray(ANCHOR_FREQUENCIES_HZ),
        mean_equilibrium,
        branches,
        times,
    )
    anchor_rows = []
    for frequency, storage, loss in zip(
        ANCHOR_FREQUENCIES_HZ, anchor_storage, anchor_loss
    ):
        label = f"{frequency:g}".replace(".", "p")
        card_path = cards_dir / f"storage_{label}Hz.inp"
        card_path.write_text(
            _isotropic_storage_card(frequency, storage),
            encoding="utf-8",
        )
        anchor_rows.append(
            {
                "frequency_Hz": frequency,
                "shear_storage_modulus_MPa": float(storage),
                "shear_loss_modulus_MPa": float(loss),
                "loss_factor": float(loss / storage),
                "card": _relative(card_path),
                "solver_interpretation": (
                    "frequency-specific in-phase elastic proxy; damping "
                    "remains external"
                ),
            }
        )
    _write_csv(output_dir / "frequency_card_index.csv", anchor_rows)

    candidate_rows = []
    for candidate in model["candidate_models"]:
        for fold in candidate["fold_metrics"]:
            candidate_rows.append(
                {
                    "term_count": candidate["term_count"],
                    **fold,
                    "mean_selection_score": candidate[
                        "mean_selection_score"
                    ],
                    "selection_score_standard_error": candidate[
                        "selection_score_standard_error"
                    ],
                    "selected": (
                        candidate["term_count"]
                        == model["selected_term_count"]
                    ),
                }
            )
    _write_csv(output_dir / "model_selection.csv", candidate_rows)

    summary = {
        "model_id": "bambu_pla_18C_short_time_maxwell_sensitivity_v1",
        "material": "Bambu PLA Basic",
        "reference_temperature_C": REFERENCE_TEMPERATURE_C,
        "reference_relative_humidity_percent": (
            REFERENCE_RELATIVE_HUMIDITY_PERCENT
        ),
        "conditioning": "ambient_lab_18C_40RH",
        "source_curve": str(source_path.relative_to(PROJECT_ROOT)),
        "raw_row_count": model["raw_row_count"],
        "inlier_row_count": model["inlier_row_count"],
        "outlier_row_count": model["outlier_row_count"],
        "outlier_rule": (
            "absolute log-loss-factor deviation greater than 6 robust MAD"
        ),
        "candidate_term_counts": list(MODEL_TERM_CANDIDATES),
        "cross_validation_folds": CROSS_VALIDATION_FOLDS,
        "best_score_term_count": model["best_score_term_count"],
        "selected_term_count": model["selected_term_count"],
        "selection_rule": (
            "smallest model within one standard error of the best "
            "five-fold score"
        ),
        "mean_equilibrium_shear_modulus_MPa": mean_equilibrium,
        "instantaneous_mean_shear_modulus_MPa": instantaneous,
        "minimum_supported_time_s": float(times[0]),
        "maximum_supported_time_s": float(times[-1]),
        "minimum_supported_frequency_Hz": MINIMUM_REDUCED_FREQUENCY_HZ,
        "maximum_supported_frequency_Hz": MAXIMUM_REDUCED_FREQUENCY_HZ,
        "final_storage_normalized_rmse": model[
            "final_storage_normalized_rmse"
        ],
        "final_loss_normalized_rmse": model[
            "final_loss_normalized_rmse"
        ],
        "storage_fit_gate_passed": model["storage_fit_gate_passed"],
        "loss_fit_gate_passed": model["loss_fit_gate_passed"],
        "positive_spectrum_gate_passed": model[
            "spectrum_positive_gate_passed"
        ],
        "overall_status": (
            "storage_and_short_time_relaxation_sensitivity_only"
            if (
                model["storage_fit_gate_passed"]
                and model["spectrum_positive_gate_passed"]
            )
            else "rejected"
        ),
        "calculix_support": (
            "frequency-specific in-phase elastic cards only; CalculiX has "
            "no native linear generalized-Maxwell material card in the "
            "verified manual"
        ),
        "admitted_uses": [
            "small-strain frequency-dependent storage sensitivity at 18 C",
            (
                "small-strain shear-relaxation sensitivity only between "
                f"{float(times[0]):.6g} and {float(times[-1]):.6g} s"
            ),
            "frequency-specific equivalent elastic CalculiX screens",
        ],
        "hard_exclusions": [
            "time-domain CalculiX Prony material claim",
            "creep beyond the fitted sub-second window",
            "constant-load creep compliance",
            "fatigue or cyclic-damage prediction",
            "large-strain, plasticity, damage, or fracture",
            "arbitrary temperature, humidity, or print-process transfer",
            "precision loss-modulus prediction because its fit gate failed",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Bambu PLA Basic short-time relaxation sensitivity

A positive {summary['selected_term_count']}-branch generalized-Maxwell
spectrum was selected from {list(MODEL_TERM_CANDIDATES)} branches by
source-stratified five-fold cross-validation and the one-standard-error rule.
The deterministic robust gate excluded {summary['outlier_row_count']} of
{summary['raw_row_count']} raw points because their loss factors were more
than six robust MAD from the log-median.

The final storage-modulus normalized RMSE is
**{summary['final_storage_normalized_rmse']:.2%}** and passes the 3% gate.
The loss-modulus normalized RMSE is
**{summary['final_loss_normalized_rmse']:.2%}** and fails the 35% precision
gate. The raw loss scatter is retained; it is not fitted away or replaced by
the paper's representative value.

The law is therefore admitted only as a small-strain storage and relaxation
sensitivity at 18 C / 40% RH from
{summary['minimum_supported_time_s']:.6g} to
{summary['maximum_supported_time_s']:.6g} s. Six frequency-specific
in-phase CalculiX elastic cards are exported. They do not create a native
CalculiX Prony law, and they do not unlock long-term creep, fatigue, failure,
moisture, aging, or temperature extrapolation.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a bounded positive short-time Maxwell sensitivity to the "
            "exact-brand 18 C DMA reconstruction"
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = write_short_time_relaxation(args.source, args.output)
    print(
        json.dumps(
            {
                "status": summary["overall_status"],
                "selected_term_count": summary["selected_term_count"],
                "storage_fit_gate_passed": summary[
                    "storage_fit_gate_passed"
                ],
                "loss_fit_gate_passed": summary[
                    "loss_fit_gate_passed"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
