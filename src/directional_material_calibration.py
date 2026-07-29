from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .calculix import CalculixRunner, ORIENTATIONS
from .campaign_ingestion import validate_campaign_submission
from .characterization_campaign import DEFAULT_OUTPUT as DEFAULT_CAMPAIGN
from .characterization_identifiability import DIRECTIONS, PARAMETERS
from .evaluate import _effective_global_xy_poisson
from .materials import (
    PRIORS,
    check_admissibility,
    material_card,
    off_axis_youngs_modulus,
)
from .prepare import preprocess_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/directional_material_calibration"
ELASTIC_WINDOW = (0.0005, 0.0030)
MODULUS_SCALE = 0.03
POISSON_SCALE = 0.01
PARAMETER_BOUNDS = {
    "E1": (500.0, 6000.0),
    "E2": (400.0, 6000.0),
    "E3": (300.0, 5000.0),
    "nu12": (0.05, 0.48),
    "nu13": (0.02, 0.47),
    "nu23": (0.02, 0.47),
    "G12": (100.0, 2500.0),
    "G13": (100.0, 2500.0),
    "G23": (100.0, 2500.0),
}


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


def _affine_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack((x, np.ones(len(x))))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = slope * x + intercept
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return float(slope), float(intercept), float(r_squared)


def tensile_observables(frame: pd.DataFrame) -> Dict[str, float]:
    strain = pd.to_numeric(
        frame["engineering_strain"], errors="coerce"
    ).to_numpy(float)
    stress = pd.to_numeric(
        frame["engineering_stress_MPa"], errors="coerce"
    ).to_numpy(float)
    lateral = pd.to_numeric(
        frame["transverse_strain"], errors="coerce"
    ).to_numpy(float)
    valid = np.isfinite(strain) & np.isfinite(stress) & np.isfinite(lateral)
    strain, stress, lateral = strain[valid], stress[valid], lateral[valid]
    order = np.argsort(strain)
    strain, stress, lateral = strain[order], stress[order], lateral[order]
    selected = (
        (strain >= ELASTIC_WINDOW[0]) & (strain <= ELASTIC_WINDOW[1])
    )
    if int(np.sum(selected)) < 6:
        raise ValueError(
            "tensile curve needs at least six complete axial/lateral points "
            "inside the preregistered 0.05%-0.30% elastic window"
        )
    modulus, stress_intercept, stress_r_squared = _affine_slope(
        strain[selected], stress[selected]
    )
    transverse_slope, lateral_intercept, lateral_r_squared = _affine_slope(
        strain[selected], lateral[selected]
    )
    poisson = -transverse_slope
    if not 100.0 <= modulus <= 10_000.0:
        raise ValueError("elastic modulus is outside the fixed plausibility range")
    if not -0.2 < poisson < 0.5:
        raise ValueError("effective Poisson ratio is outside plausible limits")
    return {
        "axial_modulus_MPa": modulus,
        "effective_global_xy_poisson": poisson,
        "stress_intercept_MPa": stress_intercept,
        "transverse_strain_intercept": lateral_intercept,
        "stress_fit_r_squared": stress_r_squared,
        "lateral_fit_r_squared": lateral_r_squared,
        "elastic_points": int(np.sum(selected)),
    }


def predict_directional_observables(
    constants: Mapping[str, float], direction: str
) -> tuple[float, float]:
    if direction not in DIRECTIONS:
        raise ValueError(f"unsupported tensile direction: {direction}")
    if direction.endswith("_45"):
        modulus = off_axis_youngs_modulus(
            constants, direction.split("_", 1)[0], 45.0
        )
    else:
        modulus = float(constants[f"E{direction}"])
    poisson = _effective_global_xy_poisson(constants, direction)
    return float(modulus), float(poisson)


def fit_orthotropic_constants(
    observations: Sequence[Mapping[str, object]],
    initial: Mapping[str, float] = PRIORS,
) -> Dict[str, object]:
    calibration = [
        row
        for row in observations
        if str(row.get("partition")) == "calibration"
        and str(row.get("test_type", "tensile")) == "tensile"
    ]
    directions = {str(row["print_direction"]) for row in calibration}
    if directions != set(DIRECTIONS):
        raise ValueError("calibration observations must cover all six directions")
    if any(
        sum(str(row["print_direction"]) == direction for row in calibration) < 3
        for direction in DIRECTIONS
    ):
        raise ValueError("calibration requires three replicates per direction")

    lower = np.asarray([PARAMETER_BOUNDS[name][0] for name in PARAMETERS])
    upper = np.asarray([PARAMETER_BOUNDS[name][1] for name in PARAMETERS])
    initial_vector = np.asarray([float(initial[name]) for name in PARAMETERS])

    def unpack(vector: np.ndarray) -> Dict[str, float]:
        return {
            name: float(value) for name, value in zip(PARAMETERS, vector)
        }

    def residual(vector: np.ndarray) -> np.ndarray:
        constants = unpack(vector)
        values: List[float] = []
        for row in calibration:
            predicted_modulus, predicted_poisson = (
                predict_directional_observables(
                    constants, str(row["print_direction"])
                )
            )
            observed_modulus = float(row["axial_modulus_MPa"])
            observed_poisson = float(
                row["effective_global_xy_poisson"]
            )
            values.extend(
                (
                    np.log(predicted_modulus / observed_modulus)
                    / MODULUS_SCALE,
                    (predicted_poisson - observed_poisson) / POISSON_SCALE,
                )
            )
        admissibility = check_admissibility(constants)
        values.append(0.0 if admissibility.valid else 1.0e3)
        return np.asarray(values, dtype=float)

    result = least_squares(
        residual,
        initial_vector,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=5000,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
    )
    constants = unpack(result.x)
    admissibility = check_admissibility(constants)
    return {
        "model": "nine_constant_linear_orthotropic_v1",
        "constants": constants,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "function_evaluations": int(result.nfev),
        "weighted_residual_norm": float(np.linalg.norm(result.fun[:-1])),
        "admissible": admissibility.valid,
        "minimum_compliance_eigenvalue": admissibility.minimum_eigenvalue,
        "admissibility_reasons": list(admissibility.reasons),
        "calibration_specimens": len(calibration),
        "calibration_directions": sorted(directions),
        "quality_gate_passed": bool(result.success and admissibility.valid),
        "partition_contract": (
            "only frozen calibration tensile replicates r01-r03 enter the fit"
        ),
    }


def score_coupon_partition(
    constants: Mapping[str, float],
    observations: Sequence[Mapping[str, object]],
    partition: str,
) -> Dict[str, object]:
    selected = [
        row
        for row in observations
        if str(row.get("partition")) == partition
        and str(row.get("test_type", "tensile")) == "tensile"
    ]
    residual_rows = []
    for row in selected:
        predicted_modulus, predicted_poisson = predict_directional_observables(
            constants, str(row["print_direction"])
        )
        observed_modulus = float(row["axial_modulus_MPa"])
        observed_poisson = float(row["effective_global_xy_poisson"])
        residual_rows.append(
            {
                **dict(row),
                "predicted_axial_modulus_MPa": predicted_modulus,
                "modulus_relative_error": (
                    predicted_modulus / observed_modulus - 1.0
                ),
                "predicted_effective_global_xy_poisson": predicted_poisson,
                "poisson_absolute_error": (
                    predicted_poisson - observed_poisson
                ),
            }
        )
    if not residual_rows:
        return {
            "partition": partition,
            "quality_gate_passed": False,
            "reason": "no tensile observations",
            "residual_rows": [],
        }
    modulus_errors = np.abs(
        [float(row["modulus_relative_error"]) for row in residual_rows]
    )
    poisson_errors = np.abs(
        [float(row["poisson_absolute_error"]) for row in residual_rows]
    )
    direction_counts = {
        direction: sum(
            str(row["print_direction"]) == direction for row in residual_rows
        )
        for direction in DIRECTIONS
    }
    required_replicates = 2 if partition == "validation" else 3
    coverage = all(
        direction_counts[direction] >= required_replicates
        for direction in DIRECTIONS
    )
    metrics = {
        "partition": partition,
        "specimens": len(residual_rows),
        "direction_counts": direction_counts,
        "median_absolute_modulus_relative_error": float(
            np.median(modulus_errors)
        ),
        "maximum_absolute_modulus_relative_error": float(
            np.max(modulus_errors)
        ),
        "mean_absolute_poisson_error": float(np.mean(poisson_errors)),
        "maximum_absolute_poisson_error": float(np.max(poisson_errors)),
        "coverage_gate_passed": coverage,
        "modulus_gate_passed": bool(
            np.median(modulus_errors) <= 0.10
            and np.max(modulus_errors) <= 0.20
        ),
        "poisson_gate_passed": bool(
            np.mean(poisson_errors) <= 0.05
            and np.max(poisson_errors) <= 0.10
        ),
        "residual_rows": residual_rows,
    }
    metrics["quality_gate_passed"] = bool(
        metrics["coverage_gate_passed"]
        and metrics["modulus_gate_passed"]
        and metrics["poisson_gate_passed"]
    )
    return metrics


def _load_tensile_observations(
    input_dir: Path, audit: Mapping[str, object]
) -> List[Dict[str, object]]:
    rows = []
    for audit_row in audit["specimens"]:
        if audit_row["status"] != "valid" or audit_row["test_type"] != "tensile":
            continue
        path = Path(input_dir) / f"{audit_row['specimen_id']}.csv"
        processed, metadata = preprocess_frame(pd.read_csv(path), path.name)
        rows.append(
            {
                "test_id": audit_row["specimen_id"],
                "partition": audit_row["expected_partition"],
                "test_type": audit_row["test_type"],
                "print_direction": audit_row["print_direction"],
                "source_file": str(path.resolve()),
                "source_sha256": _sha256(path),
                **tensile_observables(processed),
            }
        )
    return rows


def _component_linear_stiffness(frame: pd.DataFrame) -> Dict[str, float]:
    displacement = np.abs(
        pd.to_numeric(frame["displacement_mm"], errors="coerce").to_numpy(float)
    )
    force = np.abs(
        pd.to_numeric(frame["force_N"], errors="coerce").to_numpy(float)
    )
    valid = np.isfinite(displacement) & np.isfinite(force)
    displacement, force = displacement[valid], force[valid]
    order = np.argsort(displacement)
    displacement, force = displacement[order], force[order]
    upper = 0.30 * float(np.max(displacement))
    selected = (displacement >= 0.0) & (displacement <= upper)
    if int(np.sum(selected)) < 6 or upper <= 0.0:
        raise ValueError("component curve needs six points in its first 30%")
    slope, intercept, r_squared = _affine_slope(
        displacement[selected], force[selected]
    )
    if slope <= 0.0:
        raise ValueError("component curve produced nonpositive stiffness")
    return {
        "stiffness_N_per_mm": slope,
        "force_intercept_N": intercept,
        "fit_r_squared": r_squared,
        "maximum_displacement_mm": float(np.max(displacement)),
    }


def validate_heldout_components(
    constants: Mapping[str, float],
    input_dir: Path,
    audit: Mapping[str, object],
    runs_dir: Path,
) -> Dict[str, object]:
    runner = CalculixRunner(runs_dir, keep_successful=True)
    if not runner.available:
        return {
            "status": "solver_unavailable",
            "quality_gate_passed": False,
            "heldout_used_for_fitting": False,
            "specimens": [],
        }
    rows = []
    for audit_row in audit["specimens"]:
        if audit_row["expected_partition"] != "heldout":
            continue
        path = Path(input_dir) / f"{audit_row['specimen_id']}.csv"
        processed, _ = preprocess_frame(pd.read_csv(path), path.name)
        measured = _component_linear_stiffness(processed)
        result = runner.run(
            PROJECT_ROOT / "meshes/heldout_component.inp.template",
            {
                "ORIENTATION": ORIENTATIONS["mixed"],
                "MATERIAL_CARD": material_card(
                    constants, "BAMBU_PLA_BASIC_DIRECTIONAL_V1"
                ),
                "MAX_DISPLACEMENT": measured["maximum_displacement_mm"],
            },
            run_id=str(audit_row["specimen_id"]),
        )
        row = {
            "test_id": audit_row["specimen_id"],
            "source_sha256": _sha256(path),
            **measured,
            "solver_status": result.status,
        }
        if result.ok:
            endpoint_displacement = float(
                np.max(np.abs(result.curve["displacement"]))
            )
            endpoint_force = float(
                np.max(np.abs(result.curve["reaction_force"]))
            )
            predicted = endpoint_force / endpoint_displacement
            row.update(
                {
                    "predicted_stiffness_N_per_mm": predicted,
                    "stiffness_relative_error": (
                        predicted / measured["stiffness_N_per_mm"] - 1.0
                    ),
                }
            )
        rows.append(row)
    successful = [
        row for row in rows if row.get("predicted_stiffness_N_per_mm") is not None
    ]
    if len(successful) != 3:
        return {
            "status": "solver_or_curve_failure",
            "quality_gate_passed": False,
            "heldout_used_for_fitting": False,
            "specimens": rows,
        }
    errors = np.abs(
        [float(row["stiffness_relative_error"]) for row in successful]
    )
    passed = bool(np.median(errors) <= 0.20 and np.max(errors) <= 0.30)
    return {
        "status": "validated" if passed else "validation_failed",
        "quality_gate_passed": passed,
        "heldout_used_for_fitting": False,
        "median_absolute_stiffness_relative_error": float(np.median(errors)),
        "maximum_absolute_stiffness_relative_error": float(np.max(errors)),
        "specimens": rows,
    }


def synthetic_recovery(seed: int = 20260728) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    truth = {
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
    observations = []
    for direction in DIRECTIONS:
        modulus, poisson = predict_directional_observables(truth, direction)
        for replicate in range(1, 6):
            partition = "calibration" if replicate <= 3 else "validation"
            observations.append(
                {
                    "test_id": f"synthetic-{direction}-{replicate}",
                    "partition": partition,
                    "test_type": "tensile",
                    "print_direction": direction,
                    "axial_modulus_MPa": modulus
                    * (1.0 + rng.normal(0.0, 0.002)),
                    "effective_global_xy_poisson": poisson
                    + rng.normal(0.0, 0.0007),
                }
            )
    fit = fit_orthotropic_constants(observations)
    validation = score_coupon_partition(
        fit["constants"], observations, "validation"
    )
    relative_errors = {
        name: abs(float(fit["constants"][name]) / truth[name] - 1.0)
        for name in PARAMETERS
    }
    return {
        "truth": truth,
        "fitted": fit["constants"],
        "maximum_parameter_relative_error": float(
            max(relative_errors.values())
        ),
        "parameter_relative_errors": relative_errors,
        "validation_gate_passed": validation["quality_gate_passed"],
        "recovery_gate_passed": bool(
            fit["quality_gate_passed"]
            and validation["quality_gate_passed"]
            and max(relative_errors.values()) <= 0.03
        ),
    }


def calibrate_submission(
    input_dir: Path,
    campaign_dir: Path = DEFAULT_CAMPAIGN,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    input_dir, campaign_dir, output_dir = (
        Path(input_dir),
        Path(campaign_dir),
        Path(output_dir),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in (
        "material.inp",
        "fitted_constants.json",
        "coupon_observations.csv",
        "validation_residuals.csv",
        "heldout_component_residuals.csv",
    ):
        path = output_dir / stale
        if path.exists():
            path.unlink()
    audit = validate_campaign_submission(
        input_dir,
        campaign_dir / "test_plan.csv",
        campaign_dir / "campaign_manifest.json",
    )
    recovery = synthetic_recovery()
    payload: Dict[str, object] = {
        "calibration_id": "bambu_pla_basic_directional_calibration_v1",
        "material": "Bambu PLA Basic",
        "input_directory": str(input_dir.resolve()),
        "campaign_plan_sha256": audit["plan_sha256"],
        "campaign_ready": audit["ready_for_fixed_preprocessing"],
        "campaign_counts": audit["counts"],
        "synthetic_recovery": recovery,
        "card_released": False,
        "heldout_used_for_fitting": False,
    }
    if not audit["ready_for_fixed_preprocessing"]:
        payload.update(
            {
                "status": "awaiting_measurements",
                "coupon_validation": None,
                "heldout_component_validation": {
                    "status": "not_run_before_card_release",
                    "quality_gate_passed": False,
                    "heldout_used_for_fitting": False,
                },
            }
        )
    else:
        observations = _load_tensile_observations(input_dir, audit)
        fit = fit_orthotropic_constants(observations)
        validation = score_coupon_partition(
            fit["constants"], observations, "validation"
        )
        release = bool(
            fit["quality_gate_passed"]
            and validation["quality_gate_passed"]
            and recovery["recovery_gate_passed"]
        )
        _write_csv(output_dir / "coupon_observations.csv", observations)
        _write_csv(
            output_dir / "validation_residuals.csv",
            validation.pop("residual_rows"),
        )
        heldout: Dict[str, object] = {
            "status": "not_run_before_card_release",
            "quality_gate_passed": False,
            "heldout_used_for_fitting": False,
        }
        if release:
            constants_path = output_dir / "fitted_constants.json"
            constants_path.write_text(
                json.dumps(fit, indent=2) + "\n", encoding="utf-8"
            )
            card_path = output_dir / "material.inp"
            card_path.write_text(
                material_card(
                    fit["constants"], "BAMBU_PLA_BASIC_DIRECTIONAL_V1"
                ),
                encoding="utf-8",
            )
            heldout = validate_heldout_components(
                fit["constants"],
                input_dir,
                audit,
                output_dir / "heldout_runs",
            )
            _write_csv(
                output_dir / "heldout_component_residuals.csv",
                heldout.get("specimens", []),
            )
            payload.update(
                {
                    "card_released": True,
                    "material_card": str(card_path.resolve()),
                    "material_card_sha256": _sha256(card_path),
                    "fitted_constants": str(constants_path.resolve()),
                    "fitted_constants_sha256": _sha256(constants_path),
                }
            )
        payload.update(
            {
                "status": (
                    "coupon_validated_card_released"
                    if release
                    else "coupon_quality_gate_failed"
                ),
                "fit": fit,
                "coupon_validation": validation,
                "heldout_component_validation": heldout,
                "source_hashes": {
                    row["specimen_id"]: row["source_sha256"]
                    for row in audit["specimens"]
                },
            }
        )
    manifest_path = output_dir / "calibration_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Directional Bambu PLA Basic calibration",
        "",
        f"Status: **{payload['status']}**.",
        "",
        "The nine-constant card uses only the frozen r01-r03 tensile "
        "calibration partition. The r04-r05 tensile coupons are independent "
        "release gates. The three L-brackets are evaluated only after the "
        "card is fixed and never enter fitting or model selection.",
        "",
        f"Card released: **{payload['card_released']}**.",
        (
            "Blank, incomplete, invalid, or coupon-validation-failing "
            "submissions emit no material card."
        ),
    ]
    (output_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a validation-gated nine-constant Bambu PLA Basic card"
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_CAMPAIGN / "capture_templates"),
    )
    parser.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload = calibrate_submission(
        Path(args.input), Path(args.campaign), Path(args.output)
    )
    print(
        f"{payload['status']}; card released={payload['card_released']}; "
        f"manifest={Path(args.output).resolve() / 'calibration_manifest.json'}"
    )


if __name__ == "__main__":
    main()
