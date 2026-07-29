from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .bambu_interlayer_reference import (
    DEFAULT_OUTPUT as DEFAULT_PROFILE_DIR,
    write_reference,
)
from .parsing import parse_frd
from .strength_screening import screen_interface_initiation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/interlayer_initiation_screen"
DEFAULT_SUMMARIES = (
    (
        "external_density_20",
        PROJECT_ROOT
        / "results/adaptive_infill_external_validation"
        / "strength_fields_density_20"
        / "summary.json",
    ),
    (
        "external_density_40",
        PROJECT_ROOT
        / "results/adaptive_infill_external_validation"
        / "strength_fields_density_40"
        / "summary.json",
    ),
    (
        "parametric_search",
        PROJECT_ROOT / "results/parametric_infill_search/solid/summary.json",
    ),
    (
        "custom_graded_hourglass",
        PROJECT_ROOT
        / "results/custom_infill/graded_hourglass/solid/summary.json",
    ),
)
PHYSICAL_ORDER = ("hilbert", "zigzag_90", "zigzag_45")


def _profile_path() -> Path:
    return DEFAULT_PROFILE_DIR / "interface_initiation_profiles.json"


def audit_summaries(
    summaries: Sequence[tuple[str, Path]] = DEFAULT_SUMMARIES,
    profile_path: Path | None = None,
) -> List[Dict[str, object]]:
    profile_path = _profile_path() if profile_path is None else Path(profile_path)
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))["profiles"]
    rows: List[Dict[str, object]] = []
    for study, summary_path in summaries:
        source_rows = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        for source in source_rows:
            if source.get("load_case") == "buckling":
                continue
            for profile_name, profile in profiles.items():
                row: Dict[str, object] = {
                    "study": study,
                    "pattern": source["pattern"],
                    "load_case": source["load_case"],
                    "process_profile": profile_name,
                    "source_status": source["status"],
                    "run_directory": source["run_directory"],
                    "reaction_force_N": source.get("reaction_force_N"),
                    "applied_displacement_mm": source.get(
                        "applied_displacement_mm"
                    ),
                }
                if source["status"] != "ok":
                    row["screen_status"] = "source_solver_failed"
                    rows.append(row)
                    continue
                curve = parse_frd(
                    Path(str(source["run_directory"])) / "model.frd"
                )
                stress = curve.get("frd_stress")
                if stress is None:
                    row["screen_status"] = "stress_field_missing"
                    rows.append(row)
                    continue
                screen = screen_interface_initiation(
                    stress,
                    float(profile["normal_tensile_mean_MPa"]),
                    float(profile["apparent_tear_mean_MPa"]),
                    float(profile["normal_tensile_lower_1sd_MPa"]),
                    float(profile["apparent_tear_lower_1sd_MPa"]),
                )
                if screen is None:
                    row["screen_status"] = "stress_field_empty"
                    rows.append(row)
                    continue
                row["screen_status"] = "ok"
                row.update(screen.to_dict())
                force = source.get("reaction_force_N")
                if force is not None and screen.maximum_utilization > 0.0:
                    row["linear_peak_initiation_force_N"] = (
                        abs(float(force)) / screen.maximum_utilization
                    )
                    row["linear_p99_initiation_force_N"] = (
                        abs(float(force)) / screen.p99_utilization
                    )
                    row["linear_lower_1sd_peak_initiation_force_N"] = (
                        abs(float(force))
                        / screen.maximum_lower_1sd_utilization
                    )
                else:
                    row["linear_peak_initiation_force_N"] = None
                    row["linear_p99_initiation_force_N"] = None
                    row["linear_lower_1sd_peak_initiation_force_N"] = None
                rows.append(row)
    return rows


def _ranking(rows: Sequence[Mapping[str, object]], metric: str) -> List[str]:
    return [
        str(row["pattern"])
        for row in sorted(
            rows, key=lambda row: float(row[metric]), reverse=True
        )
    ]


def pairwise_matches(
    predicted: Sequence[str],
    reference: Sequence[str] = PHYSICAL_ORDER,
) -> int:
    matches = 0
    for first_index, first in enumerate(reference):
        for second in reference[first_index + 1:]:
            matches += predicted.index(first) < predicted.index(second)
    return matches


def heldout_rankings(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for study, density in (
        ("external_density_20", .20),
        ("external_density_40", .40),
    ):
        for profile in (
            "traditional_multi_head_seam",
            "smlp_continuous_layers",
        ):
            subset = [
                row for row in rows
                if row["study"] == study
                and row["process_profile"] == profile
                and row["screen_status"] == "ok"
            ]
            for field_basis, metric in (
                ("peak", "linear_peak_initiation_force_N"),
                ("p99", "linear_p99_initiation_force_N"),
            ):
                ranking = _ranking(subset, metric)
                result.append(
                    {
                        "nominal_infill_fraction": density,
                        "process_profile": profile,
                        "field_basis": field_basis,
                        "predicted_ranking": ranking,
                        "physical_ranking": list(PHYSICAL_ORDER),
                        "full_ranking_match": ranking == list(PHYSICAL_ORDER),
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


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_screen(
    output_dir: Path = DEFAULT_OUTPUT,
    summaries: Sequence[tuple[str, Path]] = DEFAULT_SUMMARIES,
) -> Dict[str, object]:
    write_reference()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = audit_summaries(summaries)
    valid = [row for row in rows if row["screen_status"] == "ok"]
    rankings = heldout_rankings(rows)
    _write_csv(output_dir / "interface_initiation_rows.csv", rows)
    _write_csv(output_dir / "heldout_rankings.csv", rankings)
    (output_dir / "interface_initiation_rows.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "heldout_rankings.json").write_text(
        json.dumps(rankings, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "valid_static_profile_fields": len(valid),
        "source_static_fields": len(valid) // 2,
        "process_profiles": 2,
        "heldout_ranking_cases": len(rankings),
        "heldout_full_ranking_matches": sum(
            row["full_ranking_match"] for row in rankings
        ),
        "heldout_pairwise_matches": sum(
            int(row["pairwise_matches"]) for row in rankings
        ),
        "heldout_pairwise_comparisons": 3 * len(rankings),
        "hilbert_over_zigzag_90_matches": sum(
            row["hilbert_beats_zigzag_90"] for row in rankings
        ),
        "hilbert_over_zigzag_45_matches": sum(
            row["hilbert_beats_zigzag_45"] for row in rankings
        ),
        "criterion_status": "measured-strength initiation sensitivity",
        "damage_evolution_status": "not identified",
        "fracture_energy_status": "not available",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    worst = max(
        valid,
        key=lambda row: float(row["maximum_lower_1sd_utilization"]),
    )
    report = f"""# Bambu PLA Basic interlayer-initiation screen

## Result

The peer-reviewed exact-brand interface strengths screened
{summary['source_static_fields']} saved 3D static fields under two process
profiles ({summary['valid_static_profile_fields']} field/profile evaluations).
The worst endpoint is `{worst['study']}` / `{worst['pattern']}` /
`{worst['load_case']}` under `{worst['process_profile']}`, at
{float(worst['maximum_lower_1sd_utilization']):.3f} of the one-SD-lower
initiation envelope.

For the held-out Hilbert/zigzag experiment, the linearized interface-initiation
loads match {summary['heldout_full_ranking_matches']}/{summary['heldout_ranking_cases']}
complete rankings and
{summary['heldout_pairwise_matches']}/{summary['heldout_pairwise_comparisons']}
pairwise orderings. Hilbert beats zigzag 45° in
{summary['hilbert_over_zigzag_45_matches']}/{summary['heldout_ranking_cases']}
screens, but it never beats the idealized load-aligned zigzag 90°.

## Interpretation

This measured-strength screen improves on the earlier theoretical shear
bookends and partially recovers Hilbert's advantage over the 45° raster. It
still fails the decisive Hilbert-over-90° comparison. That residual mismatch
should not be fitted away: the external physical specimen's exact walls,
toolpaths, build orientation, and geometry are unavailable, while the
simulated square coupon has ideal continuous load-aligned ribs.

The predicted initiation forces are linear extrapolations from elastic stress
fields. They are suitable for ranking sensitivity and choosing initial test
loads, not for final-load claims.

## Model boundary

- Local-3 opening uses the paper's vertical interlayer tensile strength.
- `sqrt(tau23^2 + tau31^2)` uses apparent tear strength as a process-sensitive
  interface proxy; the experiment is not a pure shear test.
- The quadratic interaction has no traction-separation displacement, fracture
  energy, damage evolution, crack growth, unloading, or post-peak response.
- The source used a modified dual-extruder printer, 208 C nozzle, 0.3 mm
  layers, and 0.4 mm single-bead walls. Both process profiles are retained
  because strength depends strongly on deposition history and seam formation.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen Bambu PLA fields against measured interface strengths"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = write_screen(args.output)
    print(
        f"screened {summary['source_static_fields']} static fields; "
        f"held-out pairwise matches "
        f"{summary['heldout_pairwise_matches']}/"
        f"{summary['heldout_pairwise_comparisons']}; "
        f"report: {(args.output / 'report.md').resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
