from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .bambu_tds_profile import DEFAULT_TRANSVERSE_PROFILE
from .parsing import parse_frd
from .strength_screening import (
    directional_tensile_strengths,
    screen_curve_bounded_tension_shear_interaction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/multiaxial_failure_sensitivity"
DEFAULT_SUMMARIES = (
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


def audit_summaries(
    summaries: Sequence[tuple[str, Path]] = DEFAULT_SUMMARIES,
    properties_path: Path = DEFAULT_TRANSVERSE_PROFILE,
) -> List[Dict[str, object]]:
    properties = json.loads(Path(properties_path).read_text(encoding="utf-8"))
    audited: List[Dict[str, object]] = []
    for study, summary_path in summaries:
        source_rows = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        for source in source_rows:
            row: Dict[str, object] = {
                "study": study,
                "pattern": source["pattern"],
                "load_case": source["load_case"],
                "source_status": source["status"],
                "summary_path": str(Path(summary_path).resolve()),
                "run_directory": source["run_directory"],
                "applied_displacement_mm": source.get(
                    "applied_displacement_mm"
                ),
                "maximum_von_mises_MPa": source.get(
                    "maximum_von_mises_MPa"
                ),
            }
            if source["load_case"] == "buckling":
                row["screen_status"] = "not_applicable_eigenvalue_buckling"
                audited.append(row)
                continue
            if source["status"] != "ok":
                row["screen_status"] = "source_solver_failed"
                audited.append(row)
                continue
            curve = parse_frd(
                Path(str(source["run_directory"])) / "model.frd"
            )
            screen = screen_curve_bounded_tension_shear_interaction(
                curve, properties
            )
            if screen is None:
                row["screen_status"] = "stress_field_missing"
            else:
                row["screen_status"] = "ok"
                row.update(screen.to_dict())
                row["normal_compression_assessment"] = (
                    "unassessed"
                    if source["load_case"] == "compression"
                    else "not_primary_load"
                )
                row["peak_crosses_von_mises_shear_bookend"] = (
                    screen.maximum_von_mises_shear_bookend_utilization > 1.0
                )
                row["peak_crosses_equal_tensile_shear_bookend"] = (
                    screen.maximum_equal_tensile_shear_bookend_utilization
                    > 1.0
                )
            audited.append(row)
    return audited


def _pattern_envelope(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    keys = sorted(
        {
            (str(row["study"]), str(row["pattern"]))
            for row in rows
            if row.get("screen_status") == "ok"
        }
    )
    envelope: List[Dict[str, object]] = []
    for study, pattern in keys:
        valid = [
            row
            for row in rows
            if row.get("screen_status") == "ok"
            and row["study"] == study
            and row["pattern"] == pattern
        ]
        peak = max(
            valid,
            key=lambda row: float(
                row["maximum_von_mises_shear_bookend_utilization"]
            ),
        )
        p99 = max(
            valid,
            key=lambda row: float(
                row["p99_von_mises_shear_bookend_utilization"]
            ),
        )
        envelope.append(
            {
                "study": study,
                "pattern": pattern,
                "worst_peak_load": peak["load_case"],
                "worst_peak_von_mises_shear_bookend": peak[
                    "maximum_von_mises_shear_bookend_utilization"
                ],
                "same_point_family_equal_tensile_shear_bookend": peak[
                    "maximum_equal_tensile_shear_bookend_utilization"
                ],
                "worst_p99_load": p99["load_case"],
                "worst_p99_von_mises_shear_bookend": p99[
                    "p99_von_mises_shear_bookend_utilization"
                ],
                "governing_component": peak["governing_component"],
            }
        )
    envelope.sort(
        key=lambda row: float(
            row["worst_p99_von_mises_shear_bookend"]
        )
    )
    return envelope


def _physical_priorities(
    rows: Sequence[Mapping[str, object]],
    target_initial_utilization: float = 0.6,
    count: int = 12,
) -> List[Dict[str, object]]:
    valid = [
        row for row in rows if row.get("screen_status") == "ok"
    ]
    valid.sort(
        key=lambda row: float(
            row["p99_von_mises_shear_bookend_utilization"]
        ),
        reverse=True,
    )
    priorities: List[Dict[str, object]] = []
    for rank, row in enumerate(valid[:count], start=1):
        p99 = float(row["p99_von_mises_shear_bookend_utilization"])
        scale = min(1.0, target_initial_utilization / p99) if p99 > 0 else 1.0
        endpoint = row.get("applied_displacement_mm")
        recommended = (
            None if endpoint is None else float(endpoint) * scale
        )
        priorities.append(
            {
                "priority": rank,
                "study": row["study"],
                "pattern": row["pattern"],
                "load_case": row["load_case"],
                "replicates": 3,
                "screen_p99_conservative": p99,
                "screen_peak_conservative": row[
                    "maximum_von_mises_shear_bookend_utilization"
                ],
                "screen_peak_optimistic": row[
                    "maximum_equal_tensile_shear_bookend_utilization"
                ],
                "endpoint_displacement_mm": endpoint,
                "recommended_first_displacement_mm": recommended,
                "linear_screen_scale": scale,
                "governing_component": row["governing_component"],
                "required_observation": (
                    "record crushing plus crack site; compression strength "
                    "is not represented"
                    if row["load_case"] == "compression"
                    else "record first crack site, peak load, and full "
                    "force-displacement curve"
                ),
            }
        )
    return priorities


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_audit(
    output_dir: Path = DEFAULT_OUTPUT,
    summaries: Sequence[tuple[str, Path]] = DEFAULT_SUMMARIES,
    properties_path: Path = DEFAULT_TRANSVERSE_PROFILE,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = audit_summaries(summaries, properties_path)
    valid = [row for row in rows if row.get("screen_status") == "ok"]
    envelope = _pattern_envelope(rows)
    priorities = _physical_priorities(rows)
    _write_csv(output_dir / "failure_sensitivity_rows.csv", rows)
    _write_csv(output_dir / "pattern_envelope.csv", envelope)
    _write_csv(output_dir / "physical_validation_priority.csv", priorities)
    (output_dir / "failure_sensitivity_rows.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    properties = json.loads(
        Path(properties_path).read_text(encoding="utf-8")
    )
    means, lower = directional_tensile_strengths(properties)
    source = properties["source"]
    conservative_peak_crossings = sum(
        float(row["maximum_von_mises_shear_bookend_utilization"]) > 1.0
        for row in valid
    )
    optimistic_peak_crossings = sum(
        float(row["maximum_equal_tensile_shear_bookend_utilization"]) > 1.0
        for row in valid
    )
    conservative_p99_crossings = sum(
        float(row["p99_von_mises_shear_bookend_utilization"]) > 1.0
        for row in valid
    )
    topology_robust = sum(
        (
            float(row["worst_peak_von_mises_shear_bookend"]) <= 1.0
            and float(
                row["same_point_family_equal_tensile_shear_bookend"]
            )
            <= 1.0
        )
        for row in envelope
    )
    study_count = len({str(row["study"]) for row in valid})
    study_label = "study" if study_count == 1 else "studies"
    metadata = {
        "material": properties["material"],
        "properties_file": str(Path(properties_path).resolve()),
        "source": source,
        "reported_directional_tensile_mean_MPa": means.tolist(),
        "reported_directional_tensile_lower_1sd_MPa": (
            None if lower is None else lower.tolist()
        ),
        "shear_strength_bookends": {
            "conservative": "Sij=min(Xti,Xtj)/sqrt(3)",
            "optimistic": "Sij=min(Xti,Xtj)",
            "status": "theory sensitivity, not measured Bambu PLA Basic shear",
        },
        "criterion": (
            "sqrt(sum((positive_sigma_i/Xti)^2) + "
            "sum((abs_tau_ij)/Sij)^2)"
        ),
        "normal_compression": "omitted; no matched compression strength",
        "damage_and_fracture": "not represented",
        "valid_static_fields": len(valid),
        "conservative_peak_crossings": conservative_peak_crossings,
        "optimistic_peak_crossings": optimistic_peak_crossings,
        "conservative_p99_crossings": conservative_p99_crossings,
        "patterns_below_both_peak_bookends": topology_robust,
    }
    (output_dir / "method.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Bambu PLA Basic bounded multiaxial failure sensitivity",
        "",
        "## Decision",
        "",
        (
            f"Screened {len(valid)} saved static 3D stress fields across the "
            f"{study_count} configured {study_label}. At the simulated "
            "endpoint, "
            f"{conservative_peak_crossings} cases cross the conservative "
            f"shear bookend and {optimistic_peak_crossings} still cross the "
            "optimistic bookend. The less hotspot-sensitive p99 diagnostic "
            f"crosses the conservative bookend in {conservative_p99_crossings} "
            "cases."
        ),
        "",
        (
            "This is a test-prioritization screen, not a calibrated failure "
            "model. A pattern that looks good under both bookends is a stronger "
            "candidate for printing; a crossing means reduce the first test "
            "displacement and measure where failure actually begins."
        ),
        "",
        "## Pattern envelope",
        "",
        "| Study | Pattern | Worst peak load | Conservative peak | "
        "Optimistic peak | Worst p99 | Governing term |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in envelope:
        lines.append(
            f"| {row['study']} | {row['pattern']} | "
            f"{row['worst_peak_load']} | "
            f"{float(row['worst_peak_von_mises_shear_bookend']):.3f} | "
            f"{float(row['same_point_family_equal_tensile_shear_bookend']):.3f} | "
            f"{float(row['worst_p99_von_mises_shear_bookend']):.3f} "
            f"({row['worst_p99_load']}) | {row['governing_component']} |"
        )
    lines.extend(
        [
            "",
            "## First physical tests",
            "",
            (
                "The companion `physical_validation_priority.csv` ranks up to "
                "12 cases, assigns three replicates each, and scales the first "
                "displacement toward a conservative p99 utilization of 0.6. "
                "The scale is linear-elastic guidance only; use machine "
                "limits and stop on visible cracking."
            ),
            "",
            "| Priority | Pattern | Load | First displacement (mm) | "
            "p99 screen | Observation |",
            "|---:|---|---|---:|---:|---|",
        ]
    )
    for row in priorities:
        first = row["recommended_first_displacement_mm"]
        first_text = "n/a" if first is None else f"{float(first):.3f}"
        lines.append(
            f"| {row['priority']} | {row['pattern']} | {row['load_case']} | "
            f"{first_text} | {float(row['screen_p99_conservative']):.3f} | "
            f"{row['required_observation']} |"
        )
    lines.extend(
        [
            "",
            "## Model boundary",
            "",
            (
                f"- Directional tensile means are "
                f"{means[0]:.1f}/{means[1]:.1f}/{means[2]:.1f} MPa in local "
                "1/2/3, from the Bambu PLA Basic TDS."
            ),
            "- Conservative shear bookend: "
            "`Sij=min(Xti,Xtj)/sqrt(3)`. Optimistic bookend: "
            "`Sij=min(Xti,Xtj)`. These are theory choices, not confidence "
            "bounds or measured shear strengths.",
            "- Positive local normal stresses and all three local shear "
            "components interact quadratically. Negative normal stress is "
            "ignored, so compression cases assess shear/tensile splitting "
            "only—not crushing.",
            "- Peak extrapolated stresses can be singular or mesh-sensitive; "
            "p99 is a node-sampled diagnostic, not a volume percentile.",
            "- The threshold does not predict crack growth, delamination, "
            "post-peak response, fatigue life, or impact failure.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bound Bambu PLA tension-shear initiation sensitivity"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--properties", default=str(DEFAULT_TRANSVERSE_PROFILE))
    args = parser.parse_args()
    rows = write_audit(
        output_dir=Path(args.output),
        properties_path=Path(args.properties),
    )
    valid = sum(row.get("screen_status") == "ok" for row in rows)
    print(
        f"bounded multiaxial screen completed {valid} static fields; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
