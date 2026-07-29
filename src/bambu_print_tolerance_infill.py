from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
from openpyxl import load_workbook

from .infill_benchmark import (
    LOAD_CASES,
    PatternGeometry,
    PatternResult,
    run_benchmark,
)
from .pattern_robustness_campaign import PATTERNS, finalist_geometries
from .solid_multiload import SolidLoadResult, _balanced_rows, run_solid_multiload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKBOOK = (
    PROJECT_ROOT
    / "data/public/mendeley-8jxw5533py-v2/modal_component/"
    "2-1_Tolerance_inspection_of_FDM_3D_printer.xlsx"
)
SOURCE_SHA256 = (
    "7c7c7da128f7c19ad77a4f84a62dbd427d301c18b7ef971671dce48553ac677f"
)
MATERIAL_PROFILE = (
    PROJECT_ROOT / "data/reported/bambu_pla_raw_curve_elastic.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/bambu_print_tolerance_infill"
)
DEFAULT_RUNS = PROJECT_ROOT / "runs/bambu_print_tolerance_infill"
NOMINAL_CUBE_MM = 20.0
NOMINAL_DOMAIN = (40.0, 40.0, 4.0)
INTERPRETATIONS = ("additive_offset", "multiplicative_scale")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_cube_tolerances(
    workbook_path: Path = SOURCE_WORKBOOK,
) -> List[Dict[str, object]]:
    workbook_path = Path(workbook_path)
    if _sha256(workbook_path) != SOURCE_SHA256:
        raise ValueError("Bambu cube-tolerance workbook checksum mismatch")
    workbook = load_workbook(
        workbook_path, read_only=True, data_only=True
    )
    sheet = workbook[workbook.sheetnames[0]]
    rows = []
    for row_number in range(3, 28):
        cube_id = int(sheet.cell(row_number, 1).value)
        measurements = {
            "x": [
                float(sheet.cell(row_number, column).value)
                for column in (2, 3, 4)
            ],
            "y": [
                float(sheet.cell(row_number, column).value)
                for column in (5, 6, 7)
            ],
            "z": [
                float(sheet.cell(row_number, column).value)
                for column in (8, 9, 10)
            ],
        }
        record: Dict[str, object] = {
            "cube_id": cube_id,
            "mass_g": float(sheet.cell(row_number, 11).value),
        }
        for axis, values in measurements.items():
            mean = float(np.mean(values))
            record[f"{axis}_mean_mm"] = mean
            record[f"{axis}_deviation_mm"] = mean - NOMINAL_CUBE_MM
            record[f"{axis}_measurement_sd_mm"] = float(
                np.std(values, ddof=1)
            )
            for replicate, value in enumerate(values, start=1):
                record[f"{axis}_measurement_{replicate}_mm"] = value
        rows.append(record)
    workbook.close()
    if len(rows) != 25:
        raise ValueError(f"expected 25 cubes, found {len(rows)}")
    return rows


def source_axis_statistics(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, Dict[str, float]]:
    stats = {}
    for axis in ("x", "y", "z"):
        values = np.asarray(
            [
                float(row[f"{axis}_measurement_{replicate}_mm"])
                - NOMINAL_CUBE_MM
                for row in rows
                for replicate in (1, 2, 3)
            ],
            dtype=float,
        )
        stats[axis] = {
            "measurement_count": int(values.size),
            "signed_mean_deviation_mm": float(np.mean(values)),
            "absolute_mean_deviation_mm": float(abs(np.mean(values))),
            "standard_deviation_mm": float(np.std(values, ddof=1)),
            "minimum_deviation_mm": float(np.min(values)),
            "maximum_deviation_mm": float(np.max(values)),
        }
    return stats


def scenario_dimensions(
    row: Mapping[str, object],
    interpretation: str,
) -> Dict[str, float]:
    if interpretation not in INTERPRETATIONS:
        raise ValueError(f"unknown interpretation {interpretation!r}")
    dimensions = {}
    for axis, nominal in zip(("x", "y", "z"), NOMINAL_DOMAIN):
        mean = float(row[f"{axis}_mean_mm"])
        deviation = float(row[f"{axis}_deviation_mm"])
        if interpretation == "additive_offset":
            value = nominal + deviation
        else:
            value = nominal * mean / NOMINAL_CUBE_MM
        if value <= 0.0:
            raise ValueError("scenario dimensions must remain positive")
        dimensions[axis] = value
    return dimensions


def scale_geometry(
    geometry: PatternGeometry,
    scale_x: float,
    scale_y: float,
) -> PatternGeometry:
    nodes = tuple(
        (float(x) * scale_x, float(y) * scale_y)
        for x, y in geometry.nodes
    )
    length = sum(
        math.hypot(
            nodes[second - 1][0] - nodes[first - 1][0],
            nodes[second - 1][1] - nodes[first - 1][1],
        )
        for first, second in geometry.edges
    )
    return PatternGeometry(
        geometry.name, nodes, geometry.edges, float(length)
    )


def _scenario_geometries(
    dimensions: Mapping[str, float],
) -> Dict[str, PatternGeometry]:
    scale_x = float(dimensions["x"]) / NOMINAL_DOMAIN[0]
    scale_y = float(dimensions["y"]) / NOMINAL_DOMAIN[1]
    return {
        name: scale_geometry(geometry, scale_x, scale_y)
        for name, geometry in finalist_geometries().items()
    }


def run_beam_scenarios(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    cube_rows = read_cube_tolerances()
    all_rows = []
    for cube in cube_rows:
        for interpretation in INTERPRETATIONS:
            dimensions = scenario_dimensions(cube, interpretation)
            scenario_id = (
                f"cube_{int(cube['cube_id']):02d}_{interpretation}"
            )
            results = run_benchmark(
                output_dir / "beam" / scenario_id,
                runs_dir / "beam" / scenario_id,
                properties_path=MATERIAL_PROFILE,
                patterns=PATTERNS,
                load_cases=("bending",),
                domain_width_mm=dimensions["x"],
                domain_height_mm=dimensions["y"],
                depth_mm=dimensions["z"],
                relative_density=0.35,
                geometries=_scenario_geometries(dimensions),
            )
            for result in results:
                all_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "cube_id": cube["cube_id"],
                        "interpretation": interpretation,
                        "domain_x_mm": dimensions["x"],
                        "domain_y_mm": dimensions["y"],
                        "depth_z_mm": dimensions["z"],
                        **asdict(result),
                    }
                )
    _write_csv(output_dir / "beam_bending_scenarios.csv", all_rows)
    return all_rows


def select_solid_scenarios(
    beam_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    by_scenario: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for row in beam_rows:
        by_scenario.setdefault(str(row["scenario_id"]), {})[
            str(row["pattern"])
        ] = row
    scored = []
    for scenario_id, rows in by_scenario.items():
        if set(rows) != set(PATTERNS):
            raise ValueError(f"incomplete beam scenario {scenario_id}")
        control = float(rows["triangular"]["stiffness_N_per_mm"])
        candidate = float(
            rows["kite_08_08_08_08"]["stiffness_N_per_mm"]
        )
        source = rows["triangular"]
        scored.append(
            {
                "scenario_id": scenario_id,
                "cube_id": int(source["cube_id"]),
                "interpretation": source["interpretation"],
                "domain_x_mm": float(source["domain_x_mm"]),
                "domain_y_mm": float(source["domain_y_mm"]),
                "depth_z_mm": float(source["depth_z_mm"]),
                "beam_uniform_kite_to_triangular_bending_ratio": (
                    candidate / control
                ),
            }
        )
    scored.sort(
        key=lambda row: float(
            row["beam_uniform_kite_to_triangular_bending_ratio"]
        )
    )
    selected = [scored[0], scored[len(scored) // 2], scored[-1]]
    for role, row in zip(("minimum", "median", "maximum"), selected):
        row["selection_role"] = role
    return selected


def run_solid_scenarios(
    selected: Sequence[Mapping[str, object]],
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    mesh_size_mm: float = 0.7,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    all_rows = []
    for scenario in selected:
        dimensions = {
            "x": float(scenario["domain_x_mm"]),
            "y": float(scenario["domain_y_mm"]),
            "z": float(scenario["depth_z_mm"]),
        }
        scenario_id = str(scenario["scenario_id"])
        results = run_solid_multiload(
            output_dir / "solid" / scenario_id,
            runs_dir / "solid" / scenario_id,
            properties_path=MATERIAL_PROFILE,
            patterns=PATTERNS,
            load_cases=LOAD_CASES,
            mesh_size_mm=mesh_size_mm,
            domain_width_mm=dimensions["x"],
            domain_height_mm=dimensions["y"],
            depth_mm=dimensions["z"],
            relative_density=0.35,
            geometries=_scenario_geometries(dimensions),
        )
        for result in results:
            all_rows.append(
                {
                    "scenario_id": scenario_id,
                    "selection_role": scenario["selection_role"],
                    "cube_id": scenario["cube_id"],
                    "interpretation": scenario["interpretation"],
                    "domain_x_mm": dimensions["x"],
                    "domain_y_mm": dimensions["y"],
                    "depth_z_mm": dimensions["z"],
                    **asdict(result),
                }
            )
    _write_csv(output_dir / "solid_selected_scenarios.csv", all_rows)
    return all_rows


def _specific_metric(row: Mapping[str, object]) -> float:
    if row["load_case"] == "buckling":
        return float(row["critical_buckling_load_N_per_g"])
    return float(row["specific_stiffness_N_per_mm_g"])


def aggregate(
    beam_rows: Sequence[Mapping[str, object]],
    selected: Sequence[Mapping[str, object]],
    solid_rows: Sequence[Mapping[str, object]],
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    cube_rows = read_cube_tolerances()
    _write_csv(output_dir / "cube_measurements.csv", cube_rows)
    axis_stats = source_axis_statistics(cube_rows)
    _write_csv(
        output_dir / "axis_statistics.csv",
        [
            {"axis": axis, **values}
            for axis, values in axis_stats.items()
        ],
    )
    _write_csv(output_dir / "selected_solid_scenarios.csv", selected)

    beam_by_scenario: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for row in beam_rows:
        beam_by_scenario.setdefault(str(row["scenario_id"]), {})[
            str(row["pattern"])
        ] = row
    beam_ratios = [
        float(rows["kite_08_08_08_08"]["stiffness_N_per_mm"])
        / float(rows["triangular"]["stiffness_N_per_mm"])
        for rows in beam_by_scenario.values()
    ]

    solid_by_scenario: Dict[str, List[Mapping[str, object]]] = {}
    for row in solid_rows:
        solid_by_scenario.setdefault(str(row["scenario_id"]), []).append(
            row
        )
    ranking_rows = []
    load_ratio_rows = []
    for scenario_id, rows in solid_by_scenario.items():
        objects = [
            SolidLoadResult(
                **{
                    key: value
                    for key, value in row.items()
                    if key in SolidLoadResult.__dataclass_fields__
                }
            )
            for row in rows
        ]
        balanced = _balanced_rows(objects, PATTERNS, LOAD_CASES)
        for rank, row in enumerate(balanced, start=1):
            ranking_rows.append(
                {
                    "scenario_id": scenario_id,
                    "rank": rank,
                    **row,
                }
            )
        lookup = {
            (str(row["pattern"]), str(row["load_case"])): row
            for row in rows
        }
        for load_case in LOAD_CASES:
            control = _specific_metric(
                lookup[("triangular", load_case)]
            )
            for pattern in PATTERNS:
                load_ratio_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "load_case": load_case,
                        "pattern": pattern,
                        "ratio_to_triangular": _specific_metric(
                            lookup[(pattern, load_case)]
                        )
                        / control,
                    }
                )
    _write_csv(output_dir / "solid_balanced_rankings.csv", ranking_rows)
    _write_csv(output_dir / "solid_load_ratios.csv", load_ratio_rows)

    winners = [
        row["pattern"] for row in ranking_rows if int(row["rank"]) == 1
    ]
    uniform_bending = [
        float(row["ratio_to_triangular"])
        for row in load_ratio_rows
        if row["load_case"] == "bending"
        and row["pattern"] == "kite_08_08_08_08"
    ]
    reported_stats_match = (
        abs(axis_stats["x"]["absolute_mean_deviation_mm"] - 0.016)
        <= 0.001
        and abs(axis_stats["y"]["absolute_mean_deviation_mm"] - 0.056)
        <= 0.001
        and abs(axis_stats["z"]["absolute_mean_deviation_mm"] - 0.020)
        <= 0.001
        and abs(axis_stats["x"]["standard_deviation_mm"] - 0.028)
        <= 0.001
        and abs(axis_stats["y"]["standard_deviation_mm"] - 0.029)
        <= 0.001
        and abs(axis_stats["z"]["standard_deviation_mm"] - 0.015)
        <= 0.001
    )
    gates = {
        "source_checksum_passed": _sha256(SOURCE_WORKBOOK)
        == SOURCE_SHA256,
        "twenty_five_cube_measurements_passed": len(cube_rows) == 25,
        "reported_axis_statistics_crosscheck_passed": (
            reported_stats_match
        ),
        "beam_scenario_matrix_passed": len(beam_rows) == 200
        and all(row["status"] == "ok" for row in beam_rows),
        "solid_selected_matrix_passed": len(solid_rows) == 60
        and all(row["status"] == "ok" for row in solid_rows),
        "balanced_winner_stable": set(winners) == {"triangular"},
        "absolute_failure_prediction_allowed": False,
    }
    status = (
        "measured_print_tolerance_sensitivity_passed"
        if all(
            gates[key]
            for key in (
                "source_checksum_passed",
                "twenty_five_cube_measurements_passed",
                "reported_axis_statistics_crosscheck_passed",
                "beam_scenario_matrix_passed",
                "solid_selected_matrix_passed",
                "balanced_winner_stable",
            )
        )
        else "print_tolerance_gate_failed"
    )
    payload = {
        "study_id": "bambu_x1c_pla_print_tolerance_infill_v1",
        "status": status,
        "material": "Bambu PLA Basic",
        "printer": "Bambu X1C",
        "source_dataset_doi": "10.17632/8jxw5533py.2",
        "source_article_doi": "10.1016/j.dib.2026.113015",
        "source_workbook_sha256": SOURCE_SHA256,
        "cube_count": len(cube_rows),
        "measurements_per_axis": 75,
        "axis_statistics": axis_stats,
        "geometry_error_interpretations": list(INTERPRETATIONS),
        "solver_cases": {
            "beam_bending_cases": len(beam_rows),
            "selected_solid_multiload_cases": len(solid_rows),
        },
        "beam_uniform_kite_to_triangular_bending_ratio": {
            "minimum": float(np.min(beam_ratios)),
            "median": float(np.median(beam_ratios)),
            "maximum": float(np.max(beam_ratios)),
        },
        "solid_uniform_kite_to_triangular_bending_ratio": {
            "minimum": float(np.min(uniform_bending)),
            "median": float(np.median(uniform_bending)),
            "maximum": float(np.max(uniform_bending)),
        },
        "solid_balanced_winners": winners,
        "gates": gates,
        "allowed_use": (
            "Small-strain sensitivity of the four frozen equal-relative-density "
            "finalists to empirical X/Y/Z dimensional error from 20 mm cubes."
        ),
        "limitations": [
            "Cube measurements characterize one Bambu X1C process and do not establish population tolerances across printers, colors, or lots.",
            "The additive-offset and multiplicative-scale models bracket two transfer interpretations; neither proves how 20 mm cube error transfers to a 40 x 40 x 4 mm lattice.",
            "Whole-geometry scaling does not reproduce local bead-width, corner-rounding, void, warping, or missing-strut defects.",
            "The study predicts small-strain stiffness and eigenvalue buckling only, not failure, fatigue, or fracture.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Bambu X1C measured print-tolerance infill sensitivity",
        "",
        f"Status: **{status}**.",
        "",
        (
            "The source contributes 25 Bambu PLA Basic calibration cubes and "
            "75 measurements per axis. Signed mean deviations are "
            f"X={axis_stats['x']['signed_mean_deviation_mm']:+.4f} mm, "
            f"Y={axis_stats['y']['signed_mean_deviation_mm']:+.4f} mm, and "
            f"Z={axis_stats['z']['signed_mean_deviation_mm']:+.4f} mm."
        ),
        "",
        (
            f"Two transfer interpretations produce 200 beam bending cases. "
            f"Three observed minimum/median/maximum ratio scenarios advance "
            f"to 60 solid multi-load cases. Triangular remains the balanced "
            f"winner in {len(winners)}/{len(winners)} solid scenarios."
        ),
        "",
        (
            "The solid uniform-kite bending ratio spans "
            f"{np.min(uniform_bending):.3f}–{np.max(uniform_bending):.3f}x "
            "triangular across the selected measured-tolerance scenarios."
        ),
        "",
        "This is a bounded CAD-to-print dimensional sensitivity. It does not "
        "replace coupon force-displacement validation or model local bead "
        "defects.",
        "",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propagate measured Bambu X1C dimensional error through infill finalists"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--mesh-size", type=float, default=0.7)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only:
        with (args.output / "beam_bending_scenarios.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            beam_rows = list(csv.DictReader(handle))
        with (args.output / "selected_solid_scenarios.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            selected = list(csv.DictReader(handle))
        with (args.output / "solid_selected_scenarios.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            solid_rows = list(csv.DictReader(handle))
    else:
        beam_rows = run_beam_scenarios(args.output, args.runs_dir)
        selected = select_solid_scenarios(beam_rows)
        solid_rows = run_solid_scenarios(
            selected, args.output, args.runs_dir, args.mesh_size
        )
    payload = aggregate(beam_rows, selected, solid_rows, args.output)
    print(
        f"{payload['status']}; beam cases="
        f"{payload['solver_cases']['beam_bending_cases']}; solid cases="
        f"{payload['solver_cases']['selected_solid_multiload_cases']}"
    )


if __name__ == "__main__":
    main()
