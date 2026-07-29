from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .bambu_dynamic_profile import DEFAULT_OUTPUT as DYNAMIC_PROFILE
from .infill_benchmark import (
    LOAD_CASES,
    PatternGeometry,
    make_pattern,
    make_staggered_diamond_variant,
)
from .solid_multiload import SolidLoadResult, run_solid_multiload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/dma_storage_sensitivity"
DEFAULT_RUNS = PROJECT_ROOT / "runs/dma_storage_sensitivity"
DEFAULT_STATIC = (
    PROJECT_ROOT / "results/parametric_infill_search/solid/summary.json"
)
PATTERNS = ("triangular", "sd_c2_o50")


def _metric(row: Mapping[str, object] | SolidLoadResult) -> float | None:
    if isinstance(row, SolidLoadResult):
        if row.load_case == "buckling":
            return row.critical_buckling_load_N
        return row.stiffness_N_per_mm
    if row["load_case"] == "buckling":
        value = row.get("critical_buckling_load_N")
    else:
        value = row.get("stiffness_N_per_mm")
    return None if value is None else float(value)


def compare_storage_to_static(
    storage_results: Sequence[SolidLoadResult],
    static_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    storage = {
        (row.pattern, row.load_case): row for row in storage_results
    }
    static = {
        (str(row["pattern"]), str(row["load_case"])): row
        for row in static_rows
        if row.get("pattern") in PATTERNS
    }
    rows = []
    for pattern in PATTERNS:
        for load_case in LOAD_CASES:
            dynamic_row = storage[(pattern, load_case)]
            static_row = static[(pattern, load_case)]
            dynamic_value = _metric(dynamic_row)
            static_value = _metric(static_row)
            ratio = (
                None
                if dynamic_value is None
                or static_value is None
                or static_value <= 0.0
                else dynamic_value / static_value
            )
            rows.append(
                {
                    "pattern": pattern,
                    "load_case": load_case,
                    "metric": (
                        "critical_buckling_load_N"
                        if load_case == "buckling"
                        else "stiffness_N_per_mm"
                    ),
                    "static_value": static_value,
                    "storage_30C_1Hz_value": dynamic_value,
                    "storage_to_static_ratio": ratio,
                    "static_status": static_row.get("status"),
                    "storage_status": dynamic_row.status,
                }
            )
    return rows


def run_dma_storage_sensitivity(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    properties_path: Path = DYNAMIC_PROFILE,
    static_summary_path: Path = DEFAULT_STATIC,
    mesh_size_mm: float = 0.7,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometries: Dict[str, PatternGeometry] = {
        "triangular": make_pattern("triangular"),
        "sd_c2_o50": make_staggered_diamond_variant(
            "sd_c2_o50", cells=2, offset_fraction=0.5
        ),
    }
    storage_results = run_solid_multiload(
        output_dir / "solid",
        Path(runs_dir),
        properties_path=Path(properties_path),
        patterns=PATTERNS,
        load_cases=LOAD_CASES,
        mesh_size_mm=mesh_size_mm,
        geometries=geometries,
    )
    static_rows = json.loads(
        Path(static_summary_path).read_text(encoding="utf-8")
    )
    comparison = compare_storage_to_static(storage_results, static_rows)
    fields = list(comparison[0])
    with (output_dir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)

    ranking_checks = {}
    for load_case in LOAD_CASES:
        rows = [
            row for row in comparison if row["load_case"] == load_case
        ]
        static_winner = max(
            rows, key=lambda row: float(row["static_value"])
        )["pattern"]
        storage_winner = max(
            rows, key=lambda row: float(row["storage_30C_1Hz_value"])
        )["pattern"]
        ranking_checks[load_case] = {
            "static_winner": static_winner,
            "storage_winner": storage_winner,
            "preserved": static_winner == storage_winner,
        }
    payload = {
        "material": "Bambu PLA Basic",
        "scope": "30 C, 1 Hz in-phase storage-stiffness sensitivity",
        "properties_path": str(Path(properties_path).resolve()),
        "static_baseline_path": str(
            Path(static_summary_path).resolve()
        ),
        "mesh_size_mm": mesh_size_mm,
        "successful_cases": sum(
            row.status == "ok" for row in storage_results
        ),
        "ranking_checks": ranking_checks,
        "comparison": comparison,
        "limitations": [
            "Static CalculiX loads use E-prime as an in-phase stiffness proxy.",
            "No loss modulus, phase lag, damping, inertia, or resonance is "
            "modeled.",
            "Strength and failure references are not rate-adjusted.",
            "This is not a service-temperature or viscoelastic constitutive law.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Bambu PLA Basic 30 °C / 1 Hz storage-stiffness sensitivity",
        "",
        "The digitized DMA storage moduli were inserted as an admissible "
        "orthotropic elastic sensitivity. CalculiX static responses represent "
        "only the in-phase stiffness proxy at the published 30 °C and 1 Hz "
        "condition.",
        "",
        "| Pattern | Load | Static | Storage proxy | Ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['pattern']} | {row['load_case']} | "
            f"{float(row['static_value']):.3f} | "
            f"{float(row['storage_30C_1Hz_value']):.3f} | "
            f"{float(row['storage_to_static_ratio']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Ranking check",
            "",
            "| Load | Static winner | Storage-proxy winner | Preserved |",
            "|---|---|---|---|",
        ]
    )
    for load_case, check in ranking_checks.items():
        lines.append(
            f"| {load_case} | {check['static_winner']} | "
            f"{check['storage_winner']} | {check['preserved']} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- No loss modulus was published, so damping and phase lag are "
            "not simulated.",
            "- No frequency sweep is available, so the result must not be "
            "used for resonance or another frequency.",
            "- The lowest digitized DMA tan-delta peak is 62.50 °C. That is an "
            "exclusion boundary, not an allowable service temperature.",
            "- Strength is not rate-adjusted; these results do not establish a "
            "dynamic failure load.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Bambu PLA 30 C / 1 Hz storage sensitivity"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--properties", default=str(DYNAMIC_PROFILE))
    parser.add_argument("--static-summary", default=str(DEFAULT_STATIC))
    parser.add_argument("--mesh-size", type=float, default=0.7)
    args = parser.parse_args()
    payload = run_dma_storage_sensitivity(
        Path(args.output),
        Path(args.runs_dir),
        Path(args.properties),
        Path(args.static_summary),
        args.mesh_size,
    )
    print(
        f"DMA storage sensitivity: {payload['successful_cases']}/10 cases; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
