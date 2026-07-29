from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .bambu_ballistic_reference import (
    BALLISTIC_TEST,
    DEFAULT_OUTPUT as DEFAULT_REFERENCE,
    PATTERN_RESULTS,
    PRINT_PROCESS,
    write_reference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/ballistic_pattern_validation"
PHYSICAL_RANKING = ("honeycomb", "cubic", "grid", "gyroid")


def analytical_penetration_mm(
    velocity_m_s: float = 355.0,
    pattern_coefficient: float = 1.0,
    infill_coefficient: float = 0.5,
    dynamic_coefficient: float = 1.5,
    compressive_strength_MPa: float = 68.0,
) -> float:
    mass = float(BALLISTIC_TEST["projectile_mass_kg"])
    diameter = float(BALLISTIC_TEST["projectile_diameter_m"])
    area = math.pi * diameter ** 2 / 4.0
    penetration_m = (
        mass * velocity_m_s ** 2
        / (
            2.0
            * infill_coefficient
            * pattern_coefficient
            * dynamic_coefficient
            * compressive_strength_MPa
            * 1.0e6
            * area
        )
    )
    return penetration_m * 1000.0


def summarize_physical_results() -> List[Dict[str, object]]:
    baseline = analytical_penetration_mm()
    kinetic_energy_J = (
        0.5
        * float(BALLISTIC_TEST["projectile_mass_kg"])
        * float(BALLISTIC_TEST["mean_velocity_m_s"]) ** 2
    )
    rows: List[Dict[str, object]] = []
    for pattern, result in PATTERN_RESULTS.items():
        common: Dict[str, object] = {
            "pattern": pattern,
            "target_mass_g": result["mass_g"],
            "replicates": BALLISTIC_TEST["replicates_per_pattern"],
            "projectile_kinetic_energy_J_at_mean_velocity": kinetic_energy_J,
            "nominal_coefficient_analytical_penetration_mm": baseline,
        }
        values = result.get("penetration_mm")
        if values is None:
            lower_bound = float(result["penetration_lower_bound_mm"])
            common.update(
                {
                    "censoring": "right_censored_all_three",
                    "mean_penetration_mm": None,
                    "sample_standard_deviation_mm": None,
                    "penetration_lower_bound_mm": lower_bound,
                    "effective_pattern_coefficient": None,
                    "effective_pattern_coefficient_upper_bound": (
                        baseline / lower_bound
                    ),
                    "average_resistance_force_N": None,
                    "average_resistance_force_upper_bound_N": (
                        kinetic_energy_J / (lower_bound / 1000.0)
                    ),
                    "specific_resistance_N_per_kg": None,
                    "specific_resistance_upper_bound_N_per_kg": (
                        kinetic_energy_J
                        / (lower_bound / 1000.0)
                        / (float(result["mass_g"]) / 1000.0)
                    ),
                    "stopped_within_200mm": False,
                }
            )
        else:
            numeric = [float(value) for value in values]
            mean = statistics.mean(numeric)
            standard_deviation = statistics.stdev(numeric)
            resistance = kinetic_energy_J / (mean / 1000.0)
            common.update(
                {
                    "censoring": "none",
                    "mean_penetration_mm": mean,
                    "sample_standard_deviation_mm": standard_deviation,
                    "penetration_lower_bound_mm": None,
                    "effective_pattern_coefficient": baseline / mean,
                    "effective_pattern_coefficient_upper_bound": None,
                    "average_resistance_force_N": resistance,
                    "average_resistance_force_upper_bound_N": None,
                    "specific_resistance_N_per_kg": (
                        resistance / (float(result["mass_g"]) / 1000.0)
                    ),
                    "specific_resistance_upper_bound_N_per_kg": None,
                    "stopped_within_200mm": mean < 200.0,
                }
            )
        rows.append(common)
    grid_specific = next(
        float(row["specific_resistance_N_per_kg"])
        for row in rows
        if row["pattern"] == "grid"
    )
    for row in rows:
        specific = row["specific_resistance_N_per_kg"]
        upper = row["specific_resistance_upper_bound_N_per_kg"]
        row["specific_resistance_ratio_to_grid"] = (
            None if specific is None else float(specific) / grid_specific
        )
        row["specific_resistance_ratio_to_grid_upper_bound"] = (
            None if upper is None else float(upper) / grid_specific
        )
    return rows


def velocity_sensitivity(
    physical_rows: Sequence[Mapping[str, object]],
    velocities_m_s: Sequence[float] = (354.0, 355.0, 368.0),
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    baseline_velocity = float(BALLISTIC_TEST["mean_velocity_m_s"])
    for source in physical_rows:
        for velocity in velocities_m_s:
            scale = (float(velocity) / baseline_velocity) ** 2
            measured = source["mean_penetration_mm"]
            lower_bound = source["penetration_lower_bound_mm"]
            standard_deviation = source["sample_standard_deviation_mm"]
            rows.append(
                {
                    "pattern": source["pattern"],
                    "velocity_m_s": velocity,
                    "empirical_mean_penetration_mm": (
                        None
                        if measured is None
                        else float(measured) * scale
                    ),
                    "empirical_one_sd_upper_penetration_mm": (
                        None
                        if measured is None or standard_deviation is None
                        else (
                            float(measured) + float(standard_deviation)
                        )
                        * scale
                    ),
                    "right_censored_penetration_lower_bound_mm": (
                        None
                        if lower_bound is None
                        else float(lower_bound) * scale
                    ),
                    "target_length_mm": BALLISTIC_TEST[
                        "target_length_mm"
                    ],
                    "mean_prediction_stops_within_target": (
                        False
                        if measured is None
                        else float(measured) * scale
                        < float(BALLISTIC_TEST["target_length_mm"])
                    ),
                    "model_class": (
                        "same-setup empirical v-squared sensitivity; "
                        "not validated velocity extrapolation"
                    ),
                }
            )
    return rows


def ranking(rows: Sequence[Mapping[str, object]]) -> List[str]:
    def resistance(row: Mapping[str, object]) -> float:
        value = row["average_resistance_force_N"]
        if value is not None:
            return float(value)
        return float(row["average_resistance_force_upper_bound_N"])

    return [
        str(row["pattern"])
        for row in sorted(rows, key=resistance, reverse=True)
    ]


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_validation(
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    write_reference(output_dir=DEFAULT_REFERENCE)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    physical = summarize_physical_results()
    sensitivity = velocity_sensitivity(physical)
    _write_csv(output_dir / "physical_pattern_results.csv", physical)
    _write_csv(output_dir / "velocity_sensitivity.csv", sensitivity)
    predicted_ranking = ranking(physical)
    grid = next(row for row in physical if row["pattern"] == "grid")
    nominal_grid_error = (
        float(grid["nominal_coefficient_analytical_penetration_mm"])
        / float(grid["mean_penetration_mm"])
        - 1.0
    )
    honeycomb = next(
        row for row in physical if row["pattern"] == "honeycomb"
    )
    summary = {
        "material": "Bambu PLA Basic black",
        "source": "peer-reviewed exact-brand physical experiment",
        "patterns": list(PATTERN_RESULTS),
        "replicates_per_pattern": 3,
        "physical_target_tests": 12,
        "physical_ranking": list(PHYSICAL_RANKING),
        "derived_resistance_ranking": predicted_ranking,
        "ranking_match": predicted_ranking == list(PHYSICAL_RANKING),
        "nominal_analytical_penetration_mm": (
            analytical_penetration_mm()
        ),
        "nominal_grid_relative_error": nominal_grid_error,
        "honeycomb_specific_resistance_ratio_to_grid": honeycomb[
            "specific_resistance_ratio_to_grid"
        ],
        "author_FE_grid_penetration_mm": 184.0,
        "measured_grid_mean_penetration_mm": grid[
            "mean_penetration_mm"
        ],
        "author_FE_grid_relative_error": (
            184.0 / float(grid["mean_penetration_mm"]) - 1.0
        ),
        "transfer_status": (
            "empirical same-setup pattern/velocity sensitivity and author "
            "homogenized damage-model validation"
        ),
        "limitations": [
            "Pattern coefficients are inferred from the same physical data "
            "they reproduce; they are calibration, not held-out validation.",
            "The analytical v-squared scaling is unvalidated outside the "
            "reported 354-368 m/s velocity window.",
            "Gyroid results are right-censored beyond the 200 mm target.",
            "Targets have different actual masses despite equal slicer "
            "infill percentages.",
            "The author FE model homogenized grid infill and cannot predict "
            "the four-pattern ranking from geometry.",
            "This research-scale evidence is not armor certification or a "
            "basis for personal safety decisions.",
        ],
        "print_process": PRINT_PROCESS,
        "ballistic_test": BALLISTIC_TEST,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Exact-brand ballistic infill validation

## Physical result

Twelve black Bambu PLA Basic targets were tested: cubic, grid, gyroid, and
honeycomb, with three targets per pattern. All used 50% slicer infill and the
same 200 x 60 x 60 mm envelope. The physical stopping ranking is:

1. honeycomb - 110.72 mm mean penetration;
2. cubic - 161.52 mm;
3. grid - 169.21 mm;
4. gyroid - beyond the 200 mm target in all three tests.

Honeycomb retains a
{float(summary['honeycomb_specific_resistance_ratio_to_grid']):.2f}x
specific-resistance advantage over grid after dividing the energy-per-depth
measure by actual target mass. Equal slicer infill did not produce equal mass,
so both raw and mass-normalized views are retained.

## Models

The paper's nominal energy-balance model predicts
{float(summary['nominal_analytical_penetration_mm']):.2f} mm without a fitted
pattern coefficient. Its grid error is
{abs(float(summary['nominal_grid_relative_error'])):.1%}. Coefficients inferred
from each pattern exactly encode the observed ordering and are exported for
same-setup sensitivity; they are calibration, not independent validation.

The author Abaqus model predicts 184 mm for a homogenized grid target versus
{float(summary['measured_grid_mean_penetration_mm']):.2f} mm measured, an
{abs(float(summary['author_FE_grid_relative_error'])):.1%} overprediction.
It validates a bounded damage/penetration response, but it does not explicitly
model the internal grid and cannot predict the pattern ranking from geometry.

## Transfer boundary

The velocity table applies the energy model's P proportional to v^2 scaling
only across the paper's observed 354-368 m/s window. It is not a general
high-rate constitutive law. Gyroid remains right-censored. These results are
research validation evidence, not armor certification or a basis for personal
safety decisions.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build exact-brand ballistic infill validation"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = write_validation(args.output)
    print(
        f"wrote {summary['physical_target_tests']} physical-target results; "
        f"ranking_match={summary['ranking_match']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
