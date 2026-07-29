from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .bambu_tds_profile import DEFAULT_TRANSVERSE_PROFILE
from .parsing import parse_frd
from .strength_screening import screen_curve_directional_normal_strength


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "results/parametric_infill_search/solid/summary.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/directional_strength_screening"


def audit_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    properties_path: Path = DEFAULT_TRANSVERSE_PROFILE,
) -> List[Dict[str, object]]:
    source_rows = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    properties = json.loads(
        Path(properties_path).read_text(encoding="utf-8")
    )
    audited: List[Dict[str, object]] = []
    for source in source_rows:
        row: Dict[str, object] = {
            "pattern": source["pattern"],
            "load_case": source["load_case"],
            "source_status": source["status"],
            "run_directory": source["run_directory"],
            "maximum_von_mises_MPa": source.get(
                "maximum_von_mises_MPa"
            ),
        }
        if source["load_case"] == "buckling":
            row["screen_status"] = "not_applicable_buckling"
            audited.append(row)
            continue
        if source["status"] != "ok":
            row["screen_status"] = "source_solver_failed"
            audited.append(row)
            continue
        frd_path = Path(str(source["run_directory"])) / "model.frd"
        curve = parse_frd(frd_path)
        screen = screen_curve_directional_normal_strength(
            curve,
            properties,
        )
        if screen is None:
            row["screen_status"] = "stress_field_missing"
        else:
            row["screen_status"] = "ok"
            row.update(screen.to_dict())
            row["exceeds_mean_tensile_reference"] = (
                screen.maximum_mean_strength_utilization > 1.0
            )
            row["exceeds_lower_1sd_tensile_reference"] = (
                screen.maximum_lower_1sd_strength_utilization is not None
                and screen.maximum_lower_1sd_strength_utilization > 1.0
            )
        audited.append(row)
    return audited


def _pattern_envelope(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    patterns = sorted({str(row["pattern"]) for row in rows})
    envelope: List[Dict[str, object]] = []
    for pattern in patterns:
        valid = [
            row
            for row in rows
            if row["pattern"] == pattern and row["screen_status"] == "ok"
        ]
        if not valid:
            continue
        peak = max(
            valid,
            key=lambda row: float(
                row["maximum_mean_strength_utilization"]
            ),
        )
        p99 = max(
            valid,
            key=lambda row: float(row["p99_mean_strength_utilization"]),
        )
        envelope.append(
            {
                "pattern": pattern,
                "worst_load_case": peak["load_case"],
                "worst_maximum_mean_utilization": peak[
                    "maximum_mean_strength_utilization"
                ],
                "worst_maximum_lower_1sd_utilization": max(
                    float(row["maximum_lower_1sd_strength_utilization"])
                    for row in valid
                ),
                "worst_p99_load_case": p99["load_case"],
                "worst_p99_mean_utilization": p99[
                    "p99_mean_strength_utilization"
                ],
                "governing_direction": peak["governing_direction"],
            }
        )
    envelope.sort(
        key=lambda row: float(row["worst_maximum_mean_utilization"])
    )
    return envelope


def write_audit(
    output_dir: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    properties_path: Path = DEFAULT_TRANSVERSE_PROFILE,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = audit_summary(summary_path, properties_path)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "directional_strength_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "directional_strength_rows.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    envelope = _pattern_envelope(rows)
    with (output_dir / "pattern_envelope.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(envelope[0]))
        writer.writeheader()
        writer.writerows(envelope)

    valid = [row for row in rows if row["screen_status"] == "ok"]
    peak_exceedances = sum(
        bool(row["exceeds_mean_tensile_reference"]) for row in valid
    )
    p99_exceedances = sum(
        float(row["p99_mean_strength_utilization"]) > 1.0
        for row in valid
    )
    worst = max(
        valid,
        key=lambda row: float(row["maximum_mean_strength_utilization"]),
    )
    properties = json.loads(
        Path(properties_path).read_text(encoding="utf-8")
    )
    strengths = properties["reported_strength_reference"]
    lines = [
        "# Bambu PLA Basic directional normal-strength screen",
        "",
        "## Result",
        "",
        (
            f"Recovered and screened {len(valid)} static 3D stress fields from "
            f"`{Path(summary_path).name}`. {peak_exceedances} endpoint cases "
            "cross the reported mean tensile reference at at least one "
            f"extrapolated stress point; {p99_exceedances} cases also cross it "
            "at the node-sampled 99th percentile."
        ),
        "",
        (
            f"The largest screen value is `{worst['pattern']}` / "
            f"`{worst['load_case']}` at "
            f"{float(worst['maximum_mean_strength_utilization']):.3f} of the "
            f"mean reference, governed by {worst['governing_direction']}."
        ),
        "",
        "## Per-pattern envelope",
        "",
        "| Pattern | Worst load | Peak utilization | Mean−1SD utilization | "
        "Worst p99 | Governing axis |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in envelope:
        lines.append(
            f"| {row['pattern']} | {row['worst_load_case']} | "
            f"{float(row['worst_maximum_mean_utilization']):.3f} | "
            f"{float(row['worst_maximum_lower_1sd_utilization']):.3f} | "
            f"{float(row['worst_p99_mean_utilization']):.3f} "
            f"({row['worst_p99_load_case']}) | "
            f"{row['governing_direction']} |"
        )
    lines.extend(
        [
            "",
            "## Static endpoint details",
            "",
            "| Pattern | Load | σ1+ (MPa) | σ2+ (MPa) | σ3+ (MPa) | "
            "Peak utilization | p99 utilization | Axis |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in valid:
        lines.append(
            f"| {row['pattern']} | {row['load_case']} | "
            f"{float(row['maximum_tensile_sigma1_MPa']):.3f} | "
            f"{float(row['maximum_tensile_sigma2_MPa']):.3f} | "
            f"{float(row['maximum_tensile_sigma3_MPa']):.3f} | "
            f"{float(row['maximum_mean_strength_utilization']):.3f} | "
            f"{float(row['p99_mean_strength_utilization']):.3f} | "
            f"{row['governing_direction']} |"
        )
    lines.extend(
        [
            "",
            "## What this means",
            "",
            (
                "- Local 1 and 2 are the flat-print XY plane and use "
                f"{float(strengths['overall_tensile_UTS_MPa']):.1f} MPa. "
                "Local 3 is the build direction and uses "
                f"{float(strengths['z_upright_tensile_UTS_MPa']):.1f} MPa."
            ),
            "- Only positive normal stress is screened. Compression and all "
            "three shear components are deliberately unjudged because the "
            "profile has no matched directional compression/shear strengths.",
            "- Peak values come from extrapolated CalculiX stress points and "
            "can be mesh/hotspot sensitive. The p99 column is an unweighted "
            "node-sampled diagnostic, not a volume percentile.",
            "- Crossing 1.0 does not predict crack initiation or final load. "
            "The TDS values are summary UTS references, not a calibrated "
            "multiaxial damage law.",
            "- A candidate that crosses the screen should be physically tested "
            "at a lower initial displacement and monitored for the actual "
            "failure site before any load-bearing use.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen saved 3D stress fields against directional TDS tension"
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--properties", default=str(DEFAULT_TRANSVERSE_PROFILE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    rows = write_audit(
        Path(args.output),
        Path(args.summary),
        Path(args.properties),
    )
    valid = sum(row["screen_status"] == "ok" for row in rows)
    print(
        f"directional strength screen completed {valid} static cases; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
