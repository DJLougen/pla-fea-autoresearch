from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .infill_benchmark import LOAD_CASES, PatternGeometry, make_pattern
from .materials import check_admissibility
from .parametric_infill_search import build_search_geometries
from .solid_multiload import run_solid_multiload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/static_poisson_propagation"
DEFAULT_RUNS = PROJECT_ROOT / "runs/static_poisson_propagation"
CANONICAL_PROFILE = (
    PROJECT_ROOT / "data/reported/bambu_pla_tds_transverse_sensitivity.json"
)
PATTERNS = (
    "triangular",
    "sd_c2_o50",
    "sd_c3_o50",
    "x_braced",
    "grid",
)
SCENARIOS = ("prior_nu12_0p33", "measured_nu12_0p3459")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_propagation_profiles(
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    measured = json.loads(CANONICAL_PROFILE.read_text(encoding="utf-8"))
    if (
        measured.get("profile_id")
        != "bambu_pla_tds_transverse_sensitivity_v2"
        or measured.get("engineering_constants_evidence", {}).get("nu12", "")
        .startswith("direct exact-brand")
        is not True
    ):
        raise ValueError("canonical measured-nu12 profile v2 is not available")
    prior = copy.deepcopy(measured)
    prior["profile_id"] = "bambu_pla_tds_transverse_prior_nu12_replay_v1"
    prior["screening_isotropic"]["nu"] = 0.33
    prior["screening_isotropic"]["nu_provenance"] = (
        "historical project initialization prior replay"
    )
    prior_constants = prior["screening_engineering_constants"]
    prior_constants["nu12"] = 0.33
    prior_constants["G12"] = prior_constants["E1"] / (2.0 * (1.0 + 0.33))
    prior["engineering_constants_evidence"]["nu12"] = (
        "historical project initialization prior replay"
    )
    prior["engineering_constants_evidence"]["G12"] = (
        "derived from E1 and historical nu12 by in-plane isotropy"
    )
    for profile in (prior, measured):
        admissibility = check_admissibility(
            profile["screening_engineering_constants"]
        )
        if not admissibility.valid:
            raise ValueError(
                "Poisson propagation profile is inadmissible: "
                + "; ".join(admissibility.reasons)
            )
    profiles = {
        "prior_nu12_0p33": prior,
        "measured_nu12_0p3459": measured,
    }
    paths = {}
    for scenario, profile in profiles.items():
        path = profile_dir / f"{scenario}.json"
        path.write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8"
        )
        paths[scenario] = path
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


def _metric(row: Mapping[str, object]) -> float | None:
    field = (
        "critical_buckling_load_N"
        if row["load_case"] == "buckling"
        else "stiffness_N_per_mm"
    )
    value = row.get(field)
    return None if value is None else float(value)


def aggregate_scenario_rows(
    scenario_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, object]:
    if set(scenario_rows) != set(SCENARIOS):
        raise ValueError("both prior and measured Poisson scenarios are required")
    lookups = {
        scenario: {
            (str(row["pattern"]), str(row["load_case"])): row
            for row in rows
        }
        for scenario, rows in scenario_rows.items()
    }
    expected = {
        (pattern, load_case)
        for pattern in PATTERNS
        for load_case in LOAD_CASES
    }
    if any(set(lookup) != expected for lookup in lookups.values()):
        raise ValueError("Poisson propagation matrix is incomplete")
    metric_rows = []
    rankings = {}
    all_solved = True
    for scenario, lookup in lookups.items():
        normalized_by_load: Dict[str, Dict[str, float]] = {}
        for load_case in LOAD_CASES:
            values = {}
            for pattern in PATTERNS:
                row = lookup[(pattern, load_case)]
                all_solved = all_solved and row.get("status") == "ok"
                value = _metric(row)
                if row.get("status") != "ok" or value is None or value <= 0.0:
                    raise ValueError(
                        f"invalid {scenario}/{pattern}/{load_case} metric"
                    )
                values[pattern] = value
            maximum = max(values.values())
            normalized_by_load[load_case] = {
                pattern: value / maximum
                for pattern, value in values.items()
            }
        scores = {
            pattern: float(
                math.exp(
                    np.mean(
                        [
                            math.log(
                                normalized_by_load[load_case][pattern]
                            )
                            for load_case in LOAD_CASES
                        ]
                    )
                )
            )
            for pattern in PATTERNS
        }
        rankings[scenario] = sorted(
            PATTERNS, key=lambda pattern: scores[pattern], reverse=True
        )
    prior = lookups["prior_nu12_0p33"]
    measured = lookups["measured_nu12_0p3459"]
    for pattern in PATTERNS:
        for load_case in LOAD_CASES:
            prior_value = float(_metric(prior[(pattern, load_case)]))
            measured_value = float(_metric(measured[(pattern, load_case)]))
            metric_rows.append(
                {
                    "pattern": pattern,
                    "load_case": load_case,
                    "prior_metric": prior_value,
                    "measured_nu12_metric": measured_value,
                    "relative_change": measured_value / prior_value - 1.0,
                }
            )
    maximum_change = max(
        abs(float(row["relative_change"])) for row in metric_rows
    )
    rank_preserved = (
        rankings["prior_nu12_0p33"]
        == rankings["measured_nu12_0p3459"]
    )
    return {
        "all_solver_cases_passed": all_solved,
        "rankings": rankings,
        "rank_order_preserved": rank_preserved,
        "maximum_absolute_metric_change": maximum_change,
        "metric_rows": metric_rows,
        "propagation_gate_passed": bool(all_solved and rank_preserved),
    }


def run_propagation(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    mesh_size_mm: float = 0.9,
) -> Dict[str, object]:
    output_dir, runs_dir = Path(output_dir), Path(runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_paths = build_propagation_profiles(output_dir)
    geometries = _geometries()
    scenario_rows = {}
    for scenario in SCENARIOS:
        results = run_solid_multiload(
            output_dir / "scenarios" / scenario,
            runs_dir / scenario,
            properties_path=profile_paths[scenario],
            patterns=list(PATTERNS),
            load_cases=LOAD_CASES,
            mesh_size_mm=mesh_size_mm,
            geometries=geometries,
        )
        scenario_rows[scenario] = [
            {
                key: value
                for key, value in vars(result).items()
            }
            for result in results
        ]
    aggregate = aggregate_scenario_rows(scenario_rows)
    _write_csv(
        output_dir / "metric_changes.csv", aggregate.pop("metric_rows")
    )
    payload = {
        "study_id": "bambu_pla_static_inplane_poisson_propagation_v1",
        "material": "Bambu PLA Basic",
        "canonical_profile_id": (
            "bambu_pla_tds_transverse_sensitivity_v2"
        ),
        "historical_nu12": 0.33,
        "measured_nu12": 0.3459,
        "patterns": list(PATTERNS),
        "load_cases": list(LOAD_CASES),
        "mesh_size_mm": mesh_size_mm,
        "solver_case_count": sum(len(rows) for rows in scenario_rows.values()),
        **aggregate,
        "interpretation": (
            "propagation check for replacing one historical in-plane "
            "Poisson prior with a separate exact-brand process measurement"
        ),
        "restrictions": [
            "nu13, nu23, G13, and G23 remain priors.",
            "G12 is derived under in-plane isotropy and is not a direct "
            "quasistatic shear measurement.",
            "The measured nu12 comes from a separate Bambu-X1C process at "
            "18 C / 40% RH and is not an environmental law.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Static in-plane Poisson propagation",
        "",
        "The canonical room directional profile now replaces the historical "
        "nu12=0.33 prior with the exact-brand measured nu12=0.3459. This "
        "study reruns five representative patterns in five load modes under "
        "both values.",
        "",
        f"Solver cases: **{payload['solver_case_count']}**.",
        f"Rank order preserved: **{payload['rank_order_preserved']}**.",
        (
            "Maximum absolute individual metric change: "
            f"**{payload['maximum_absolute_metric_change']:.3%}**."
        ),
        "",
        "The update supports the in-plane coupling term only. It does not "
        "supply out-of-plane Poisson ratios or directly measured "
        "quasistatic shear moduli.",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate the exact-brand measured in-plane Poisson ratio "
            "through representative static infill simulations"
        )
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--mesh-size", type=float, default=0.9)
    args = parser.parse_args()
    payload = run_propagation(
        Path(args.output), Path(args.runs_dir), args.mesh_size
    )
    print(
        f"{payload['solver_case_count']} cases; "
        f"rank preserved={payload['rank_order_preserved']}; "
        f"max change={payload['maximum_absolute_metric_change']:.3%}"
    )


if __name__ == "__main__":
    main()
