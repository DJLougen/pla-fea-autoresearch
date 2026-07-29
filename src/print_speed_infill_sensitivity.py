from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .bambu_print_speed_reference import (
    DEFAULT_OUTPUT as DEFAULT_PROFILE_DIR,
    SAMPLE_MAP,
    write_reference,
)
from .interlayer_initiation_screen import (
    DEFAULT_SUMMARIES,
    PHYSICAL_ORDER,
    pairwise_matches,
)
from .parsing import parse_frd
from .strength_screening import stress_rows_to_local


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/print_speed_infill_sensitivity"
BASELINE_PROFILE = (
    PROJECT_ROOT / "data/reported/bambu_pla_raw_curve_elastic.json"
)


def _read_profiles(
    path: Path = DEFAULT_PROFILE_DIR / "process_groups.csv",
) -> List[Dict[str, object]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _baseline_modulus(path: Path = BASELINE_PROFILE) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(payload["screening_isotropic"]["E_MPa"])


def screen_process_profiles(
    summaries: Sequence[tuple[str, Path]] = DEFAULT_SUMMARIES,
    profile_path: Path = DEFAULT_PROFILE_DIR / "process_groups.csv",
    baseline_profile: Path = BASELINE_PROFILE,
) -> List[Dict[str, object]]:
    profiles = _read_profiles(profile_path)
    baseline_modulus = _baseline_modulus(baseline_profile)
    rows: List[Dict[str, object]] = []
    for study, summary_path in summaries:
        sources = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        for source in sources:
            if source.get("load_case") == "buckling":
                continue
            common: Dict[str, object] = {
                "study": study,
                "pattern": source["pattern"],
                "load_case": source["load_case"],
                "source_status": source["status"],
                "run_directory": source["run_directory"],
                "applied_displacement_mm": source.get(
                    "applied_displacement_mm"
                ),
                "baseline_reaction_force_N": source.get(
                    "reaction_force_N"
                ),
                "baseline_modulus_MPa": baseline_modulus,
            }
            local = None
            if source["status"] == "ok":
                curve = parse_frd(
                    Path(str(source["run_directory"])) / "model.frd"
                )
                stress = curve.get("frd_stress")
                if stress is not None:
                    local = stress_rows_to_local(stress)
            for profile in profiles:
                row = {
                    **common,
                    "sample_group": int(profile["sample_group"]),
                    "print_speed_mm_s": int(profile["print_speed_mm_s"]),
                    "build_plate_axis": profile["build_plate_axis"],
                    "ironing": (
                        str(profile["ironing"]).lower() == "true"
                    ),
                    "tensile_strength_mean_MPa": float(
                        profile["tensile_strength_mean_MPa"]
                    ),
                    "tensile_strength_standard_deviation_MPa": float(
                        profile[
                            "tensile_strength_standard_deviation_MPa"
                        ]
                    ),
                    "youngs_modulus_mean_MPa": float(
                        profile["youngs_modulus_mean_MPa"]
                    ),
                }
                if source["status"] != "ok":
                    row["screen_status"] = "source_solver_failed"
                    rows.append(row)
                    continue
                if local is None or len(local) == 0:
                    row["screen_status"] = "stress_field_missing"
                    rows.append(row)
                    continue
                tensile = np.maximum(local[:, :2], 0.0)
                point_stress = np.max(tensile, axis=1)
                maximum = float(np.max(point_stress))
                p99 = float(np.percentile(point_stress, 99.0))
                strength = float(row["tensile_strength_mean_MPa"])
                lower_strength = (
                    strength
                    - float(
                        row[
                            "tensile_strength_standard_deviation_MPa"
                        ]
                    )
                )
                modulus_scale = (
                    float(row["youngs_modulus_mean_MPa"])
                    / baseline_modulus
                )
                row.update(
                    {
                        "screen_status": "ok",
                        "maximum_inplane_tensile_stress_MPa": maximum,
                        "p99_inplane_tensile_stress_MPa": p99,
                        "modulus_scale_to_saved_field": modulus_scale,
                        "scaled_maximum_stress_at_test_displacement_MPa": (
                            maximum * modulus_scale
                        ),
                        "scaled_reaction_force_at_test_displacement_N": (
                            None
                            if source.get("reaction_force_N") is None
                            else abs(float(source["reaction_force_N"]))
                            * modulus_scale
                        ),
                        "maximum_mean_strength_utilization": (
                            maximum * modulus_scale / strength
                        ),
                        "p99_mean_strength_utilization": (
                            p99 * modulus_scale / strength
                        ),
                        "maximum_lower_1sd_strength_utilization": (
                            maximum * modulus_scale / lower_strength
                        ),
                        "linear_peak_initiation_force_N": (
                            None
                            if source.get("reaction_force_N") is None
                            or maximum <= 0.0
                            else abs(float(source["reaction_force_N"]))
                            * strength / maximum
                        ),
                        "linear_p99_initiation_force_N": (
                            None
                            if source.get("reaction_force_N") is None
                            or p99 <= 0.0
                            else abs(float(source["reaction_force_N"]))
                            * strength / p99
                        ),
                        "linear_lower_1sd_peak_initiation_force_N": (
                            None
                            if source.get("reaction_force_N") is None
                            or maximum <= 0.0
                            else abs(float(source["reaction_force_N"]))
                            * lower_strength / maximum
                        ),
                    }
                )
                rows.append(row)
    return rows


def heldout_rankings(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for study, density in (
        ("external_density_20", 0.20),
        ("external_density_40", 0.40),
    ):
        for sample in SAMPLE_MAP:
            subset = [
                row for row in rows
                if row["study"] == study
                and int(row["sample_group"]) == sample
                and row["screen_status"] == "ok"
            ]
            for basis, metric in (
                ("peak", "linear_peak_initiation_force_N"),
                ("p99", "linear_p99_initiation_force_N"),
            ):
                ranking = [
                    str(row["pattern"])
                    for row in sorted(
                        subset,
                        key=lambda row: float(row[metric]),
                        reverse=True,
                    )
                ]
                result.append(
                    {
                        "nominal_infill_fraction": density,
                        "sample_group": sample,
                        **SAMPLE_MAP[sample],
                        "field_basis": basis,
                        "predicted_ranking": ranking,
                        "physical_ranking": list(PHYSICAL_ORDER),
                        "full_ranking_match": ranking
                        == list(PHYSICAL_ORDER),
                        "pairwise_matches": pairwise_matches(ranking),
                        "pairwise_comparisons": 3,
                        "hilbert_beats_zigzag_90": (
                            ranking.index("hilbert")
                            < ranking.index("zigzag_90")
                        ),
                        "hilbert_beats_zigzag_45": (
                            ranking.index("hilbert")
                            < ranking.index("zigzag_45")
                        ),
                    }
                )
    return result


def process_effect_checks(
    profiles: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    lookup = {
        int(row["sample_group"]): row for row in profiles
    }
    comparisons = {
        "low_speed_over_high_speed": ((1, 3), (2, 4), (5, 7), (6, 8)),
        "Y_axis_over_X_axis": ((5, 1), (6, 2), (7, 3), (8, 4)),
        "ironing_over_no_ironing": ((1, 2), (3, 4), (5, 6), (7, 8)),
    }
    rows: List[Dict[str, object]] = []
    for effect, pairs in comparisons.items():
        for numerator, denominator in pairs:
            for response, field in (
                ("tensile_strength", "tensile_strength_mean_MPa"),
                ("youngs_modulus", "youngs_modulus_mean_MPa"),
            ):
                first = float(lookup[numerator][field])
                second = float(lookup[denominator][field])
                rows.append(
                    {
                        "effect": effect,
                        "response": response,
                        "numerator_sample_group": numerator,
                        "denominator_sample_group": denominator,
                        "numerator_value": first,
                        "denominator_value": second,
                        "ratio": first / second,
                        "expected_direction_observed": first > second,
                    }
                )
    return rows


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_sensitivity(
    output_dir: Path = DEFAULT_OUTPUT,
    summaries: Sequence[tuple[str, Path]] = DEFAULT_SUMMARIES,
) -> Dict[str, object]:
    reference = write_reference()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = _read_profiles()
    rows = screen_process_profiles(summaries=summaries)
    rankings = heldout_rankings(rows)
    effects = process_effect_checks(profiles)
    _write_csv(output_dir / "process_field_rows.csv", rows)
    _write_csv(output_dir / "heldout_rankings.csv", rankings)
    _write_csv(output_dir / "matched_process_effects.csv", effects)
    (output_dir / "process_field_rows.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "heldout_rankings.json").write_text(
        json.dumps(rankings, indent=2) + "\n", encoding="utf-8"
    )
    valid = [row for row in rows if row["screen_status"] == "ok"]
    strength_profiles = {
        int(row["sample_group"]): float(
            row["tensile_strength_mean_MPa"]
        )
        for row in profiles
    }
    minimum_sample = min(strength_profiles, key=strength_profiles.get)
    maximum_sample = max(strength_profiles, key=strength_profiles.get)
    summary = {
        "material": reference["material"],
        "source_static_fields": len(valid) // len(profiles),
        "process_profiles": len(profiles),
        "valid_field_profile_evaluations": len(valid),
        "heldout_ranking_cases": len(rankings),
        "heldout_full_ranking_matches": sum(
            bool(row["full_ranking_match"]) for row in rankings
        ),
        "heldout_pairwise_matches": sum(
            int(row["pairwise_matches"]) for row in rankings
        ),
        "heldout_pairwise_comparisons": 3 * len(rankings),
        "hilbert_over_zigzag_90_matches": sum(
            bool(row["hilbert_beats_zigzag_90"]) for row in rankings
        ),
        "hilbert_over_zigzag_45_matches": sum(
            bool(row["hilbert_beats_zigzag_45"]) for row in rankings
        ),
        "lowest_mean_strength_sample_group": minimum_sample,
        "highest_mean_strength_sample_group": maximum_sample,
        "mean_strength_spread_percent": (
            (strength_profiles[maximum_sample]
             / strength_profiles[minimum_sample] - 1.0)
            * 100.0
        ),
        "matched_effect_successes": {
            effect: {
                response: sum(
                    bool(row["expected_direction_observed"])
                    for row in effects
                    if row["effect"] == effect
                    and row["response"] == response
                )
                for response in ("tensile_strength", "youngs_modulus")
            }
            for effect in (
                "low_speed_over_high_speed",
                "Y_axis_over_X_axis",
                "ironing_over_no_ironing",
            )
        },
        "interpretation": (
            "modulus scales prescribed-displacement stress and reaction "
            "together, so it cancels from the linearized initiation load; "
            "the measured strength shifts the predicted initiation load"
        ),
        "limitations": [
            "The process law is applied to local in-plane normal tension only.",
            "No measured speed-conditioned shear or compression strength is "
            "available.",
            "The saved fields are linearly rescaled isotropic elastic fields.",
            "Sparse-infill toolpaths are outside the source coupon process.",
            "Post-peak curve shapes are not propagated through a 3D damage "
            "solver because no validated damage/fracture law exists.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Print-speed/process sensitivity for infill

## Result

Eight measured Bambu PLA Basic process profiles screened
{summary['source_static_fields']} saved 3D static fields, producing
{summary['valid_field_profile_evaluations']} valid field/profile evaluations.
The measured mean-strength spread is
{summary['mean_strength_spread_percent']:.1f}% from the weakest to strongest
process group.

For the held-out Hilbert/zigzag experiment, the in-plane initiation screen
matches {summary['heldout_full_ranking_matches']}/{summary['heldout_ranking_cases']}
complete rankings and
{summary['heldout_pairwise_matches']}/{summary['heldout_pairwise_comparisons']}
pairwise orderings. Hilbert beats zigzag 45 degrees in
{summary['hilbert_over_zigzag_45_matches']}/{summary['heldout_ranking_cases']}
screens and zigzag 90 degrees in
{summary['hilbert_over_zigzag_90_matches']}/{summary['heldout_ranking_cases']}.

## Interpretation

At a fixed imposed displacement, the higher measured modulus raises both
stress and reaction. In the linearized force at first in-plane tensile
initiation those two changes cancel, so process-conditioned strength controls
the initiation-load shift. This is useful for setting conservative coupon test
loads and checking whether a topology result survives plausible process
variation.

The source reports that X/Y are build-plate axes while the deposited layer
pattern remains oriented the same way relative to the specimen. The axis
effect is therefore retained as an empirical printer/process profile, not
misidentified as a new material-axis constant.

## Model boundary

The eight profiles come from solid 100% rectilinear ISO 527-2 coupons on a
Bambu Lab A1 mini. Only local in-plane tensile initiation is screened.
Speed-conditioned shear, compression, sparse-infill void morphology, damage
evolution, crack growth, and fracture energy remain unmeasured. The
representative post-peak CSVs are available for bounded material-point
sensitivity, but they are not passed off as a validated 3D failure law.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Propagate Bambu print-speed profiles through infill fields"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = write_sensitivity(args.output)
    print(
        f"screened {summary['source_static_fields']} fields x "
        f"{summary['process_profiles']} process profiles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
