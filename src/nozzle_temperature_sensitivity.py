from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .bambu_dma_reference import NOZZLE_TEMPERATURES_C
from .bambu_nozzle_temperature_profiles import (
    DEFAULT_OUTPUT as PROFILE_ROOT,
    write_nozzle_temperature_profiles,
)
from .infill_benchmark import (
    LOAD_CASES,
    PatternGeometry,
    make_pattern,
    make_staggered_diamond_variant,
)
from .solid_multiload import SolidLoadResult, run_solid_multiload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/nozzle_temperature_sensitivity"
DEFAULT_RUNS = PROJECT_ROOT / "runs/nozzle_temperature_sensitivity"
PATTERNS = ("triangular", "sd_c2_o50")


def _metric(row: Mapping[str, object] | SolidLoadResult) -> float | None:
    load_case = (
        row.load_case if isinstance(row, SolidLoadResult) else row["load_case"]
    )
    if load_case == "buckling":
        value = (
            row.critical_buckling_load_N
            if isinstance(row, SolidLoadResult)
            else row.get("critical_buckling_load_N")
        )
    else:
        value = (
            row.stiffness_N_per_mm
            if isinstance(row, SolidLoadResult)
            else row.get("stiffness_N_per_mm")
        )
    return None if value is None else float(value)


def summarize_temperature_results(
    results_by_temperature: Mapping[int, Sequence[SolidLoadResult]],
    quality_by_temperature: Mapping[int, str],
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    lookup = {
        (temperature, row.pattern, row.load_case): row
        for temperature, results in results_by_temperature.items()
        for row in results
    }
    rows: List[Dict[str, object]] = []
    for temperature in NOZZLE_TEMPERATURES_C:
        for pattern in PATTERNS:
            for load_case in LOAD_CASES:
                row = lookup[(temperature, pattern, load_case)]
                value = _metric(row)
                baseline = _metric(lookup[(220, pattern, load_case)])
                rows.append(
                    {
                        "nozzle_temperature_C": temperature,
                        "source_quality": quality_by_temperature[temperature],
                        "pattern": pattern,
                        "load_case": load_case,
                        "metric": (
                            "critical_buckling_load_N"
                            if load_case == "buckling"
                            else "stiffness_N_per_mm"
                        ),
                        "value": value,
                        "ratio_to_220C": (
                            None
                            if value is None
                            or baseline is None
                            or baseline <= 0.0
                            else value / baseline
                        ),
                        "status": row.status,
                        "directional_normal_utilization_mean": (
                            row.directional_normal_utilization_mean
                        ),
                        "directional_normal_utilization_p99_mean": (
                            row.directional_normal_utilization_p99_mean
                        ),
                    }
                )

    ranking_checks: Dict[str, object] = {}
    for load_case in LOAD_CASES:
        by_temperature = {}
        for temperature in NOZZLE_TEMPERATURES_C:
            candidates = [
                row
                for row in rows
                if row["nozzle_temperature_C"] == temperature
                and row["load_case"] == load_case
                and row["value"] is not None
            ]
            by_temperature[str(temperature)] = max(
                candidates, key=lambda row: float(row["value"])
            )["pattern"]
        ranking_checks[load_case] = {
            "winner_by_nozzle_temperature_C": by_temperature,
            "preserved": len(set(by_temperature.values())) == 1,
        }

    spreads = []
    for pattern in PATTERNS:
        for load_case in LOAD_CASES:
            current = [
                row
                for row in rows
                if row["pattern"] == pattern
                and row["load_case"] == load_case
                and row["value"] is not None
            ]
            minimum = min(current, key=lambda row: float(row["value"]))
            maximum = max(current, key=lambda row: float(row["value"]))
            spreads.append(
                {
                    "pattern": pattern,
                    "load_case": load_case,
                    "minimum_temperature_C": minimum[
                        "nozzle_temperature_C"
                    ],
                    "maximum_temperature_C": maximum[
                        "nozzle_temperature_C"
                    ],
                    "minimum_value": minimum["value"],
                    "maximum_value": maximum["value"],
                    "max_to_min_ratio": (
                        float(maximum["value"]) / float(minimum["value"])
                    ),
                }
            )
    return rows, {
        "ranking_checks": ranking_checks,
        "spreads": spreads,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_nozzle_temperature_sensitivity(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    profile_root: Path = PROFILE_ROOT,
    mesh_size_mm: float = 0.7,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    profile_root = Path(profile_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_nozzle_temperature_profiles(output_dir=profile_root)
    quality = {
        int(row["nozzle_temperature_C"]): str(row["crossplot_quality"])
        for row in manifest
    }
    geometries: Dict[str, PatternGeometry] = {
        "triangular": make_pattern("triangular"),
        "sd_c2_o50": make_staggered_diamond_variant(
            "sd_c2_o50", cells=2, offset_fraction=0.5
        ),
    }
    results_by_temperature: Dict[int, Sequence[SolidLoadResult]] = {}
    for temperature in NOZZLE_TEMPERATURES_C:
        results_by_temperature[temperature] = run_solid_multiload(
            output_dir / f"print_{temperature}C" / "solid",
            runs_dir / f"print_{temperature}C",
            properties_path=profile_root / f"print_{temperature}C.json",
            patterns=PATTERNS,
            load_cases=LOAD_CASES,
            mesh_size_mm=mesh_size_mm,
            geometries=geometries,
        )
    rows, aggregate = summarize_temperature_results(
        results_by_temperature, quality
    )
    _write_csv(output_dir / "comparison.csv", rows)
    _write_csv(output_dir / "metric_spreads.csv", aggregate["spreads"])

    strength_crossings = {
        str(temperature): sum(
            row["nozzle_temperature_C"] == temperature
            and row["directional_normal_utilization_mean"] is not None
            and float(row["directional_normal_utilization_mean"]) > 1.0
            for row in rows
        )
        for temperature in NOZZLE_TEMPERATURES_C
    }
    payload = {
        "material": "Bambu PLA Basic",
        "scope": (
            "as-printed 30 C / 1 Hz storage-stiffness sensitivity across "
            "published nozzle temperatures"
        ),
        "measurement_temperature_C": 30.0,
        "frequency_Hz": 1.0,
        "nozzle_temperatures_C": list(NOZZLE_TEMPERATURES_C),
        "patterns": list(PATTERNS),
        "mesh_size_mm": mesh_size_mm,
        "profile_manifest": str(
            (profile_root / "manifest.json").resolve()
        ),
        "successful_cases": sum(
            row["status"] == "ok" for row in rows
        ),
        "source_quality_by_temperature": {
            str(key): value for key, value in quality.items()
        },
        "directional_normal_peak_crossings": strength_crossings,
        **aggregate,
        "comparison": rows,
        "limitations": [
            "Nozzle temperature is a manufacturing parameter, not service "
            "temperature.",
            "The 210 C and 230 C profiles retain a published 90-degree "
            "crossplot disagreement.",
            "Static CalculiX loads use E-prime as an in-phase stiffness proxy.",
            "Print-flow quality, void formation, loss modulus, damping, and "
            "fracture are not modeled. Flexural strength is evaluated only "
            "by the separate process-strength screen.",
            "The published 240 C condition is not automatically a safe or "
            "recommended printer setting.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Bambu PLA Basic nozzle-temperature infill sensitivity",
        "",
        "## Decision",
        "",
        (
            f"All {payload['successful_cases']}/40 CalculiX cases completed. "
            "The study changes the as-printed 30 °C / 1 Hz storage profile "
            "according to the published nozzle-temperature matrix; it does "
            "not heat the part in service."
        ),
        "",
        "## Topology ranking",
        "",
        "| Load | 210 °C | 220 °C | 230 °C | 240 °C | Preserved |",
        "|---|---|---|---|---|---|",
    ]
    for load_case, check in aggregate["ranking_checks"].items():
        winners = check["winner_by_nozzle_temperature_C"]
        lines.append(
            f"| {load_case} | {winners['210']} | {winners['220']} | "
            f"{winners['230']} | {winners['240']} | "
            f"{check['preserved']} |"
        )
    lines.extend(
        [
            "",
            "## Process sensitivity range",
            "",
            "| Pattern | Load | Minimum nozzle | Maximum nozzle | "
            "Maximum/minimum |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in aggregate["spreads"]:
        lines.append(
            f"| {row['pattern']} | {row['load_case']} | "
            f"{row['minimum_temperature_C']} °C | "
            f"{row['maximum_temperature_C']} °C | "
            f"{float(row['max_to_min_ratio']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Source-quality and strength flags",
            "",
            "| Nozzle | Figure crossplot | Endpoint directional-normal "
            "crossings |",
            "|---:|---|---:|",
        ]
    )
    for temperature in NOZZLE_TEMPERATURES_C:
        lines.append(
            f"| {temperature} °C | {quality[temperature]} | "
            f"{strength_crossings[str(temperature)]} |"
        )
    lines.extend(
        [
            "",
            "At every nozzle temperature, the two crossings are triangular "
            "tension and `sd_c2_o50` tension at the 1% displacement endpoint. "
            "The storage-profile strength references are not rate- or "
            "process-adjusted, so this is a ramp-reduction flag rather than a "
            "failure prediction.",
            "",
            "The 220 °C and 240 °C matrices pass all three independent "
            "Figure 7/Figure 9 checks. The 210 °C and 230 °C 90° values do "
            "not, so those cases are exploratory. A stiffer storage response "
            "does not prove better print quality or higher strength.",
            "",
            "## Boundary",
            "",
            "- This is a manufacturing-process sensitivity at a fixed 30 °C "
            "measurement temperature and 1 Hz.",
            "- It does not model nozzle flow, thermal history, crystallization, "
            "voids, interlayer defects, service-temperature softening, "
            "damping, or fracture. The companion flexural process-strength "
            "screen checks bending references only.",
            "- Do not select 240 °C from stiffness alone; confirm the filament "
            "and printer limits and inspect the actual extrusion.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Bambu PLA nozzle-temperature infill sensitivity"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--profiles", default=str(PROFILE_ROOT))
    parser.add_argument("--mesh-size", type=float, default=0.7)
    args = parser.parse_args()
    payload = run_nozzle_temperature_sensitivity(
        Path(args.output),
        Path(args.runs_dir),
        Path(args.profiles),
        args.mesh_size,
    )
    print(
        f"nozzle-temperature sensitivity: "
        f"{payload['successful_cases']}/40 cases; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
