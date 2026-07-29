from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .bambu_tds_profile import build_tds_profiles
from .infill_benchmark import LOAD_CASES, PatternGeometry, make_pattern
from .materials import check_admissibility
from .parametric_infill_search import build_search_geometries
from .solid_multiload import (
    SolidLoadResult,
    _balanced_rows,
    run_solid_multiload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/material_uncertainty_sensitivity"
DEFAULT_RUNS = PROJECT_ROOT / "runs/material_uncertainty_sensitivity"
CENTRAL_SUMMARY = (
    PROJECT_ROOT / "results/parametric_infill_search/solid/summary.json"
)
PATTERNS = (
    "triangular",
    "sd_c2_o50",
    "sd_c3_o50",
    "x_braced",
    "grid",
)
SCENARIOS = ("measured_low", "measured_high", "shear_low", "shear_high")


def build_sensitivity_profiles() -> Dict[str, Dict[str, object]]:
    _, base = build_tds_profiles()
    definitions = {
        "measured_low": {
            "E1": 2360.0,
            "E2": 2360.0,
            "E3": 1890.0,
            "description": "TDS XY and Z Young moduli at mean minus 1 SD",
        },
        "measured_high": {
            "E1": 2800.0,
            "E2": 2800.0,
            "E3": 2230.0,
            "description": "TDS XY and Z Young moduli at mean plus 1 SD",
        },
        "shear_low": {
            "G13": 585.0,
            "G23": 585.0,
            "description": "Unmeasured G13/G23 prior reduced by 25%",
        },
        "shear_high": {
            "G13": 975.0,
            "G23": 975.0,
            "description": "Unmeasured G13/G23 prior increased by 25%",
        },
    }
    profiles: Dict[str, Dict[str, object]] = {}
    for name, changes in definitions.items():
        profile = copy.deepcopy(base)
        constants = profile["screening_engineering_constants"]
        for key, value in changes.items():
            if key == "description":
                continue
            constants[key] = float(value)
        if "E1" in changes:
            constants["G12"] = constants["E1"] / (
                2.0 * (1.0 + constants["nu12"])
            )
            profile["screening_isotropic"]["E_MPa"] = constants["E1"]
        admissibility = check_admissibility(constants)
        if not admissibility.valid:
            raise ValueError(
                f"{name} is inadmissible: "
                + "; ".join(admissibility.reasons)
            )
        profile["profile_id"] = f"bambu_pla_tds_{name}_sensitivity_v1"
        profile["uncertainty_scenario"] = {
            "name": name,
            "description": changes["description"],
            "interpretation": (
                "structured sensitivity bound, not a probability distribution"
            ),
        }
        profile["admissibility"] = {
            "valid": True,
            "minimum_compliance_eigenvalue": (
                admissibility.minimum_eigenvalue
            ),
        }
        profiles[name] = profile
    return profiles


def write_profiles(output_dir: Path = DEFAULT_OUTPUT) -> Dict[str, Path]:
    profiles = build_sensitivity_profiles()
    profile_dir = Path(output_dir) / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, profile in profiles.items():
        path = profile_dir / f"{name}.json"
        path.write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8"
        )
        paths[name] = path
    return paths


def _geometries() -> Dict[str, PatternGeometry]:
    search = build_search_geometries()
    return {
        "triangular": make_pattern("triangular"),
        "sd_c2_o50": search["sd_c2_o50"],
        "sd_c3_o50": search["sd_c3_o50"],
        "x_braced": make_pattern("x_braced"),
        "grid": make_pattern("grid"),
    }


def run_scenario(
    scenario: str,
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    mesh_size_mm: float = 0.7,
) -> Sequence[SolidLoadResult]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown sensitivity scenario {scenario!r}")
    profile_path = Path(output_dir) / "profiles" / f"{scenario}.json"
    if not profile_path.exists():
        write_profiles(output_dir)
    geometries = _geometries()
    return run_solid_multiload(
        Path(output_dir) / "scenarios" / scenario,
        Path(runs_dir) / scenario,
        properties_path=profile_path,
        patterns=list(PATTERNS),
        load_cases=LOAD_CASES,
        mesh_size_mm=mesh_size_mm,
        geometries=geometries,
    )


def _metric(row: Mapping[str, object]) -> float | None:
    if row["load_case"] == "buckling":
        value = row.get("critical_buckling_load_N")
    else:
        value = row.get("stiffness_N_per_mm")
    return None if value is None else float(value)


def aggregate_scenarios(
    output_dir: Path = DEFAULT_OUTPUT,
    central_summary: Path = CENTRAL_SUMMARY,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    scenario_rows: Dict[str, List[Mapping[str, object]]] = {
        "central": json.loads(Path(central_summary).read_text(encoding="utf-8"))
    }
    for scenario in SCENARIOS:
        path = output_dir / "scenarios" / scenario / "summary.json"
        if not path.exists():
            raise ValueError(f"missing completed scenario summary: {path}")
        scenario_rows[scenario] = json.loads(
            path.read_text(encoding="utf-8")
        )
    selected_rows = {
        scenario: [
            row for row in rows if row["pattern"] in PATTERNS
        ]
        for scenario, rows in scenario_rows.items()
    }
    ranking_rows: List[Dict[str, object]] = []
    score_by_pattern: Dict[str, Dict[str, float]] = {
        pattern: {} for pattern in PATTERNS
    }
    rank_by_pattern: Dict[str, List[int]] = {
        pattern: [] for pattern in PATTERNS
    }
    for scenario, rows in selected_rows.items():
        objects = [SolidLoadResult(**row) for row in rows]
        ranking = _balanced_rows(objects, PATTERNS, LOAD_CASES)
        for rank, row in enumerate(ranking, start=1):
            score = float(row["balanced_geometric_mean"])
            pattern = str(row["pattern"])
            ranking_rows.append(
                {
                    "scenario": scenario,
                    "rank": rank,
                    "pattern": pattern,
                    "balanced_score": score,
                }
            )
            score_by_pattern[pattern][scenario] = score
            rank_by_pattern[pattern].append(rank)
    with (output_dir / "scenario_rankings.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(ranking_rows[0])
        )
        writer.writeheader()
        writer.writerows(ranking_rows)

    lookup = {
        scenario: {
            (str(row["pattern"]), str(row["load_case"])): row
            for row in rows
        }
        for scenario, rows in selected_rows.items()
    }
    envelope_rows: List[Dict[str, object]] = []
    for pattern in PATTERNS:
        for load_case in LOAD_CASES:
            values = {
                scenario: _metric(rows[(pattern, load_case)])
                for scenario, rows in lookup.items()
            }
            if any(value is None for value in values.values()):
                continue
            numeric = {key: float(value) for key, value in values.items()}
            minimum_scenario = min(numeric, key=numeric.get)
            maximum_scenario = max(numeric, key=numeric.get)
            central = numeric["central"]
            envelope_rows.append(
                {
                    "pattern": pattern,
                    "load_case": load_case,
                    "central_metric": central,
                    "minimum_metric": numeric[minimum_scenario],
                    "maximum_metric": numeric[maximum_scenario],
                    "minimum_scenario": minimum_scenario,
                    "maximum_scenario": maximum_scenario,
                    "minimum_change_percent": (
                        100.0
                        * (numeric[minimum_scenario] / central - 1.0)
                    ),
                    "maximum_change_percent": (
                        100.0
                        * (numeric[maximum_scenario] / central - 1.0)
                    ),
                }
            )
    with (output_dir / "metric_envelopes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(envelope_rows[0])
        )
        writer.writeheader()
        writer.writerows(envelope_rows)

    robustness = []
    best_by_scenario = {
        scenario: max(
            score_by_pattern[pattern][scenario] for pattern in PATTERNS
        )
        for scenario in scenario_rows
    }
    for pattern in PATTERNS:
        scores = score_by_pattern[pattern]
        robustness.append(
            {
                "pattern": pattern,
                "minimum_balanced_score": min(scores.values()),
                "maximum_balanced_score": max(scores.values()),
                "maximum_regret": max(
                    best_by_scenario[scenario] - score
                    for scenario, score in scores.items()
                ),
                "best_rank": min(rank_by_pattern[pattern]),
                "worst_rank": max(rank_by_pattern[pattern]),
            }
        )
    robustness.sort(
        key=lambda row: float(row["minimum_balanced_score"]),
        reverse=True,
    )
    with (output_dir / "robustness_ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(robustness[0])
        )
        writer.writeheader()
        writer.writerows(robustness)
    report = [
        "# Bambu PLA Basic material-envelope infill sensitivity",
        "",
        "This is a structured robustness envelope, not a Monte Carlo "
        "probability distribution. It combines the TDS XY/Z Young-modulus "
        "mean ±1 SD cases with ±25% sensitivity on unmeasured G13/G23 priors.",
        "",
        "| Robust rank | Pattern | Min balanced | Max balanced | "
        "Max regret | Rank range |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(robustness, start=1):
        report.append(
            f"| {rank} | {row['pattern']} | "
            f"{row['minimum_balanced_score']:.3f} | "
            f"{row['maximum_balanced_score']:.3f} | "
            f"{row['maximum_regret']:.4f} | "
            f"{row['best_rank']}–{row['worst_rank']} |"
        )
    winner = robustness[0]["pattern"]
    report.extend(
        [
            "",
            "## Decision",
            "",
            f"`{winner}` has the highest worst-case balanced score across the "
            "five scenarios. Scenario-specific rankings and every raw metric "
            "bound are saved in `scenario_rankings.csv` and "
            "`metric_envelopes.csv`.",
            "",
            "These bounds do not cover print defects, batch/color variation, "
            "moisture, temperature, rate, damage, or fracture.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    payload = {
        "scenarios": ["central", *SCENARIOS],
        "robustness": robustness,
        "winner": winner,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or aggregate Bambu PLA material-envelope scenarios"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--mesh-size", type=float, default=0.7)
    args = parser.parse_args()
    if args.prepare_only:
        paths = write_profiles(Path(args.output))
        print(f"wrote {len(paths)} admissible sensitivity profiles")
        return
    if args.aggregate:
        payload = aggregate_scenarios(Path(args.output))
        print(
            f"aggregated {len(payload['scenarios'])} scenarios; "
            f"winner={payload['winner']}; "
            f"report: {(Path(args.output) / 'report.md').resolve()}"
        )
        return
    if args.scenario is None:
        parser.error("choose --scenario, --prepare-only, or --aggregate")
    results = run_scenario(
        args.scenario,
        Path(args.output),
        Path(args.runs_dir),
        mesh_size_mm=args.mesh_size,
    )
    successful = sum(row.status == "ok" for row in results)
    print(
        f"{args.scenario}: completed {successful}/{len(results)} cases"
    )


if __name__ == "__main__":
    main()
