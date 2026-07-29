from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .evaluate import _effective_global_xy_poisson
from .materials import PRIORS, off_axis_youngs_modulus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/characterization_identifiability"
)
PARAMETERS = (
    "E1",
    "E2",
    "E3",
    "nu12",
    "nu13",
    "nu23",
    "G12",
    "G13",
    "G23",
)
DIRECTIONS = ("1", "2", "3", "12_45", "13_45", "23_45")


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def observables(
    constants: Mapping[str, float],
    directions: Sequence[str] = DIRECTIONS,
    *,
    include_lateral: bool = True,
) -> Tuple[List[str], np.ndarray]:
    labels: List[str] = []
    values: List[float] = []
    for direction in directions:
        if direction.endswith("_45"):
            modulus = off_axis_youngs_modulus(
                constants, direction.split("_", 1)[0], 45.0
            )
        else:
            modulus = float(constants[f"E{direction}"])
        labels.append(f"{direction}_axial_modulus_MPa")
        values.append(float(modulus))
        if include_lateral:
            labels.append(f"{direction}_global_xy_poisson")
            values.append(
                _effective_global_xy_poisson(constants, direction)
            )
    return labels, np.asarray(values, dtype=float)


def numerical_jacobian(
    constants: Mapping[str, float],
    directions: Sequence[str] = DIRECTIONS,
    *,
    include_lateral: bool = True,
    normalized: bool = True,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    labels, baseline = observables(
        constants,
        directions,
        include_lateral=include_lateral,
    )
    jacobian = np.zeros((len(baseline), len(PARAMETERS)), dtype=float)
    for column, parameter in enumerate(PARAMETERS):
        scale = abs(float(constants[parameter]))
        step = max(scale * 1.0e-5, 1.0e-7)
        upper = dict(constants)
        lower = dict(constants)
        upper[parameter] = float(upper[parameter]) + step
        lower[parameter] = float(lower[parameter]) - step
        _, upper_values = observables(
            upper,
            directions,
            include_lateral=include_lateral,
        )
        _, lower_values = observables(
            lower,
            directions,
            include_lateral=include_lateral,
        )
        jacobian[:, column] = (
            upper_values - lower_values
        ) / (2.0 * step)
    if normalized:
        parameter_scale = np.asarray(
            [float(constants[name]) for name in PARAMETERS],
            dtype=float,
        )
        observable_scale = np.maximum(np.abs(baseline), 1.0e-10)
        jacobian = (
            jacobian
            * parameter_scale[np.newaxis, :]
            / observable_scale[:, np.newaxis]
        )
    return labels, baseline, jacobian


def matrix_diagnostics(jacobian: np.ndarray) -> Dict[str, object]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian, tol=1.0e-8))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == jacobian.shape[1]
        else None
    )
    return {
        "rows": int(jacobian.shape[0]),
        "columns": int(jacobian.shape[1]),
        "rank": rank,
        "condition_number": condition,
        "singular_values": singular_values.tolist(),
    }


def _ablation_rows(
    constants: Mapping[str, float],
) -> List[Dict[str, object]]:
    cases = (
        ("full_campaign", DIRECTIONS, True),
        ("axial_only_all_six", DIRECTIONS, False),
        ("principal_only_with_lateral", ("1", "2", "3"), True),
        (
            "no_build_coupled_off_axis",
            ("1", "2", "3", "12_45"),
            True,
        ),
        (
            "no_in_plane_off_axis",
            ("1", "2", "3", "13_45", "23_45"),
            True,
        ),
    )
    rows = []
    for name, directions, include_lateral in cases:
        _, _, jacobian = numerical_jacobian(
            constants,
            directions,
            include_lateral=include_lateral,
        )
        diagnostics = matrix_diagnostics(jacobian)
        rows.append(
            {
                "case": name,
                "directions": ",".join(directions),
                "includes_lateral_strain": include_lateral,
                "observables": diagnostics["rows"],
                "parameters": diagnostics["columns"],
                "rank": diagnostics["rank"],
                "condition_number": diagnostics["condition_number"],
                "all_nine_constants_identifiable": (
                    diagnostics["rank"] == len(PARAMETERS)
                ),
            }
        )
    return rows


def _planning_uncertainty(
    constants: Mapping[str, float],
    absolute_jacobian: np.ndarray,
    baseline: np.ndarray,
    replicates: int = 5,
    modulus_relative_noise: float = 0.03,
    poisson_absolute_noise: float = 0.01,
) -> List[Dict[str, object]]:
    standard_error = np.asarray(
        [
            (
                abs(value) * modulus_relative_noise
                if index % 2 == 0
                else poisson_absolute_noise
            )
            / np.sqrt(replicates)
            for index, value in enumerate(baseline)
        ],
        dtype=float,
    )
    weighted = absolute_jacobian / standard_error[:, np.newaxis]
    information = weighted.T @ weighted
    covariance = np.linalg.inv(information)
    parameter_standard_error = np.sqrt(np.diag(covariance))
    rows = []
    for index, parameter in enumerate(PARAMETERS):
        value = float(constants[parameter])
        rows.append(
            {
                "parameter": parameter,
                "planning_value": value,
                "linearized_standard_error": float(
                    parameter_standard_error[index]
                ),
                "linearized_relative_standard_error": float(
                    parameter_standard_error[index] / abs(value)
                ),
                "assumed_replicates_per_direction": replicates,
                "assumed_modulus_relative_noise": (
                    modulus_relative_noise
                ),
                "assumed_poisson_absolute_noise": (
                    poisson_absolute_noise
                ),
                "interpretation": (
                    "planning forecast only; not a confidence interval"
                ),
            }
        )
    return rows


def run_identifiability(
    output_dir: Path = DEFAULT_OUTPUT,
    constants: Mapping[str, float] = PRIORS,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels, baseline, normalized = numerical_jacobian(constants)
    _, _, absolute = numerical_jacobian(
        constants, normalized=False
    )
    diagnostics = matrix_diagnostics(normalized)

    sensitivity_rows = []
    for row_index, label in enumerate(labels):
        for column_index, parameter in enumerate(PARAMETERS):
            sensitivity_rows.append(
                {
                    "observable": label,
                    "parameter": parameter,
                    "baseline_observable": float(baseline[row_index]),
                    "normalized_sensitivity": float(
                        normalized[row_index, column_index]
                    ),
                    "absolute_sensitivity": float(
                        absolute[row_index, column_index]
                    ),
                }
            )
    _write_csv(
        output_dir / "observable_parameter_sensitivity.csv",
        sensitivity_rows,
    )
    singular_rows = [
        {
            "index": index + 1,
            "singular_value": value,
            "relative_to_largest": (
                value / diagnostics["singular_values"][0]
            ),
        }
        for index, value in enumerate(diagnostics["singular_values"])
    ]
    _write_csv(output_dir / "singular_values.csv", singular_rows)
    ablations = _ablation_rows(constants)
    _write_csv(output_dir / "campaign_ablation.csv", ablations)
    planning = _planning_uncertainty(
        constants, absolute, baseline
    )
    _write_csv(
        output_dir / "linearized_uncertainty_forecast.csv", planning
    )
    column_norms = np.linalg.norm(normalized, axis=0)
    parameter_rows = [
        {
            "parameter": parameter,
            "normalized_sensitivity_norm": float(column_norms[index]),
            "observable_in_full_campaign": bool(
                column_norms[index] > 0.1
            ),
        }
        for index, parameter in enumerate(PARAMETERS)
    ]
    _write_csv(
        output_dir / "parameter_observability.csv", parameter_rows
    )
    full_rank = diagnostics["rank"] == len(PARAMETERS)
    condition_ok = (
        diagnostics["condition_number"] is not None
        and float(diagnostics["condition_number"]) < 25.0
    )
    ablation_gate = all(
        (
            row["all_nine_constants_identifiable"]
            if row["case"] == "full_campaign"
            else not row["all_nine_constants_identifiable"]
        )
        for row in ablations
    )
    payload = {
        "material": "Bambu PLA Basic",
        "campaign": "bambu_pla_basic_directional_v1",
        "directions": list(DIRECTIONS),
        "parameters": list(PARAMETERS),
        "observables": labels,
        "normalized_jacobian": diagnostics,
        "full_rank_gate_passed": full_rank,
        "condition_gate_passed": condition_ok,
        "ablation_gate_passed": ablation_gate,
        "all_parameters_have_sensitivity": bool(
            np.all(column_norms > 0.1)
        ),
        "planning_noise_assumptions": {
            "replicates_per_direction": 5,
            "modulus_relative_noise": 0.03,
            "poisson_absolute_noise": 0.01,
        },
        "maximum_forecast_relative_standard_error": max(
            float(row["linearized_relative_standard_error"])
            for row in planning
        ),
        "interpretation": (
            "The planned measurements can identify the nine constants "
            "locally around the planning card. This proves design "
            "identifiability, not that measurements have been acquired."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Directional campaign identifiability

The exact equations used by the fixed evaluator were differentiated around
the Bambu PLA Basic planning card. The six planned tensile orientations
provide twelve observables: one axial modulus and one global transverse
Poisson response per direction.

- Sensitivity matrix: **{diagnostics['rows']} x {diagnostics['columns']}**
- Rank: **{diagnostics['rank']}/9**
- Normalized condition number: **{float(diagnostics['condition_number']):.3f}**
- Largest linearized relative standard-error forecast:
  **{payload['maximum_forecast_relative_standard_error']:.1%}**

The full campaign is locally identifiable. Every ablation is deliberately
retained: axial curves without lateral strain, principal directions without
off-axis coupons, removal of build-coupled off-axis coupons, and removal of
the in-plane off-axis coupon all lose full nine-parameter identifiability.

The uncertainty forecast assumes five repeats per direction, 3% modulus noise,
and 0.01 absolute Poisson noise. It is a planning calculation, not measured
uncertainty. The three missing shear moduli and Poisson ratios remain
unmeasured until the campaign CSVs are filled.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit directional campaign parameter identifiability"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_identifiability(args.output)
    print(
        json.dumps(
            {
                "rank": result["normalized_jacobian"]["rank"],
                "condition_number": result["normalized_jacobian"][
                    "condition_number"
                ],
                "full_rank_gate_passed": result["full_rank_gate_passed"],
                "ablation_gate_passed": result["ablation_gate_passed"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
