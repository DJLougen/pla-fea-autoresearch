from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.stats import qmc

from .bambu_tds_profile import build_tds_profiles
from .infill_benchmark import LOAD_CASES, PatternResult, run_benchmark
from .materials import check_admissibility
from .pattern_robustness_campaign import PATTERNS, finalist_geometries
from .solid_multiload import SolidLoadResult, run_solid_multiload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/directional_interval_robustness"
DEFAULT_RUNS = PROJECT_ROOT / "runs/directional_interval_robustness"
PROFILE_ID = "bambu_pla_directional_interval_sample_v1"
SEED = 20260728
LHS_SAMPLES = 24
MAX_PARALLEL_SCENARIOS = 4

# The Young-modulus intervals are the Bambu TDS mean +/- one reported
# standard deviation. The remaining intervals are deliberately broad
# sensitivity bounds around unresolved constants; they are not confidence
# intervals and do not imply a probability distribution.
BOUNDS: Dict[str, tuple[float, float]] = {
    "E1": (2360.0, 2800.0),
    "E2": (2360.0, 2800.0),
    "E3": (1890.0, 2230.0),
    "nu13": (0.20, 0.40),
    "nu23": (0.20, 0.40),
    "G12": (670.0, 1250.0),
    "G13": (390.0, 1170.0),
    "G23": (390.0, 1170.0),
}
FIXED_NU12 = 0.3459


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_constants() -> Dict[str, float]:
    _, transverse = build_tds_profiles()
    return {
        key: float(value)
        for key, value in transverse[
            "screening_engineering_constants"
        ].items()
    }


def _sample_rows(
    lhs_samples: int = LHS_SAMPLES,
    seed: int = SEED,
) -> list[Dict[str, object]]:
    if lhs_samples < 1:
        raise ValueError("lhs_samples must be positive")
    central = _canonical_constants()
    names = tuple(BOUNDS)
    rows: list[Dict[str, object]] = [
        {
            "sample_id": "canonical",
            "sample_kind": "canonical",
            **central,
        }
    ]
    for name, (lower, upper) in BOUNDS.items():
        for label, value in (("low", lower), ("high", upper)):
            constants = dict(central)
            constants[name] = value
            rows.append(
                {
                    "sample_id": f"axis_{name}_{label}",
                    "sample_kind": "axis_bound",
                    **constants,
                }
            )
    for label, selector in (
        ("all_low", 0),
        ("all_high", 1),
    ):
        constants = dict(central)
        for name, bounds in BOUNDS.items():
            constants[name] = bounds[selector]
        rows.append(
            {
                "sample_id": label,
                "sample_kind": "joint_anchor",
                **constants,
            }
        )
    coupling = dict(central)
    for name, (lower, upper) in BOUNDS.items():
        coupling[name] = (
            upper if name in {"nu13", "nu23"} else lower
        )
    rows.append(
        {
            "sample_id": "low_stiffness_high_coupling",
            "sample_kind": "joint_anchor",
            **coupling,
        }
    )

    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n=lhs_samples)
    lower = np.asarray([BOUNDS[name][0] for name in names])
    upper = np.asarray([BOUNDS[name][1] for name in names])
    scaled = qmc.scale(unit, lower, upper)
    for index, values in enumerate(scaled, start=1):
        constants = dict(central)
        constants.update(
            {
                name: float(value)
                for name, value in zip(names, values)
            }
        )
        rows.append(
            {
                "sample_id": f"lhs_{index:02d}",
                "sample_kind": "latin_hypercube",
                **constants,
            }
        )

    accepted = []
    rejected = []
    for row in rows:
        constants = {
            key: float(row[key])
            for key in (
                "E1",
                "E2",
                "E3",
                "nu12",
                "nu13",
                "nu23",
                "G12",
                "G13",
                "G23",
            )
        }
        constants["nu12"] = FIXED_NU12
        row["nu12"] = FIXED_NU12
        admissibility = check_admissibility(constants)
        row["minimum_compliance_eigenvalue"] = (
            admissibility.minimum_eigenvalue
        )
        if admissibility.valid:
            accepted.append(row)
        else:
            rejected.append(
                {
                    "sample_id": row["sample_id"],
                    "reasons": "; ".join(admissibility.reasons),
                }
            )
    if rejected:
        identifiers = ", ".join(
            str(row["sample_id"]) for row in rejected
        )
        raise ValueError(
            "declared directional envelope generated inadmissible anchors: "
            + identifiers
        )
    return accepted


def write_profiles(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    lhs_samples: int = LHS_SAMPLES,
    seed: int = SEED,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    _, base = build_tds_profiles()
    rows = _sample_rows(lhs_samples, seed)
    paths: Dict[str, Path] = {}
    manifest_rows = []
    for row in rows:
        profile = copy.deepcopy(base)
        sample_id = str(row["sample_id"])
        constants = profile["screening_engineering_constants"]
        for name in (
            "E1",
            "E2",
            "E3",
            "nu12",
            "nu13",
            "nu23",
            "G12",
            "G13",
            "G23",
        ):
            constants[name] = float(row[name])
        profile["profile_id"] = f"{PROFILE_ID}_{sample_id}"
        profile["model_class"] = "orthotropic_interval_sensitivity"
        profile["directional_interval_sample"] = {
            "sample_id": sample_id,
            "sample_kind": row["sample_kind"],
            "seed": seed,
            "interpretation": (
                "deterministic non-probabilistic sensitivity sample; "
                "not a fitted or measured material card"
            ),
            "fixed_nu12": FIXED_NU12,
            "bounds": {
                name: list(bounds) for name, bounds in BOUNDS.items()
            },
        }
        profile["limitations"] = list(profile.get("limitations", [])) + [
            "The interval sample is not a confidence interval or probability "
            "distribution.",
            "Only E1/E2/E3 bounds come from TDS mean +/- one standard "
            "deviation; out-of-plane Poisson and static shear ranges are "
            "declared sensitivity bounds.",
        ]
        path = profile_dir / f"{sample_id}.json"
        path.write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8"
        )
        paths[sample_id] = path
        manifest_rows.append(dict(row))
    _write_csv(output_dir / "material_samples.csv", manifest_rows)
    manifest = {
        "study_id": "bambu_pla_directional_interval_robustness_v1",
        "material": "Bambu PLA Basic",
        "sample_count": len(rows),
        "latin_hypercube_sample_count": lhs_samples,
        "seed": seed,
        "fixed_constants": {"nu12": FIXED_NU12},
        "bounds": {
            name: list(bounds) for name, bounds in BOUNDS.items()
        },
        "interpretation": (
            "deterministic interval coverage, not a probability distribution"
        ),
        "evidence_partition": {
            "E1_E2_E3": "Bambu TDS mean +/- one reported standard deviation",
            "nu12": (
                "fixed exact-brand in-plane measurement; source reports no "
                "measurement uncertainty"
            ),
            "nu13_nu23_G12_G13_G23": (
                "declared broad sensitivity bounds, not measurements"
            ),
        },
    }
    (output_dir / "profile_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def _metric(row: Mapping[str, object]) -> float:
    field = (
        "critical_buckling_load_N"
        if row["load_case"] == "buckling"
        else "stiffness_N_per_mm"
    )
    value = row.get(field)
    if row.get("status") != "ok" or value is None or float(value) <= 0.0:
        raise ValueError(
            f"invalid metric for {row.get('pattern')}/"
            f"{row.get('load_case')}"
        )
    return float(value)


def aggregate_interval_rows(
    scenario_rows: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    fidelity: str,
) -> Dict[str, object]:
    if not scenario_rows:
        raise ValueError("at least one material scenario is required")
    expected = {
        (pattern, load_case)
        for pattern in PATTERNS
        for load_case in LOAD_CASES
    }
    detail_rows = []
    sample_rows = []
    adversarial: set[str] = {"canonical"}
    for sample_id, rows in scenario_rows.items():
        lookup = {
            (str(row["pattern"]), str(row["load_case"])): row
            for row in rows
        }
        if set(lookup) != expected:
            raise ValueError(
                f"material sample {sample_id} has an incomplete result matrix"
            )
        balanced_by_pattern = {}
        for pattern in PATTERNS:
            ratios = []
            for load_case in LOAD_CASES:
                value = _metric(lookup[(pattern, load_case)])
                control = _metric(lookup[("triangular", load_case)])
                ratio = value / control
                ratios.append(ratio)
                detail_rows.append(
                    {
                        "fidelity": fidelity,
                        "sample_id": sample_id,
                        "pattern": pattern,
                        "load_case": load_case,
                        "metric": value,
                        "triangular_metric": control,
                        "pattern_to_triangular_ratio": ratio,
                    }
                )
            balanced_by_pattern[pattern] = float(
                math.exp(np.mean(np.log(ratios)))
            )
        winner = max(
            PATTERNS, key=lambda name: balanced_by_pattern[name]
        )
        sample_rows.append(
            {
                "fidelity": fidelity,
                "sample_id": sample_id,
                "winner": winner,
                **{
                    f"{pattern}_balanced_ratio": value
                    for pattern, value in balanced_by_pattern.items()
                },
            }
        )

    candidate_summary = []
    for pattern in PATTERNS:
        pattern_details = [
            row for row in detail_rows if row["pattern"] == pattern
        ]
        balanced_values = [
            float(row[f"{pattern}_balanced_ratio"])
            for row in sample_rows
        ]
        minimum_balanced_index = int(np.argmin(balanced_values))
        minimum_balanced_sample = str(
            sample_rows[minimum_balanced_index]["sample_id"]
        )
        adversarial.add(minimum_balanced_sample)
        load_envelopes = {}
        for load_case in LOAD_CASES:
            selected = [
                row
                for row in pattern_details
                if row["load_case"] == load_case
            ]
            minimum = min(
                selected,
                key=lambda row: float(
                    row["pattern_to_triangular_ratio"]
                ),
            )
            maximum = max(
                selected,
                key=lambda row: float(
                    row["pattern_to_triangular_ratio"]
                ),
            )
            if pattern != "triangular":
                adversarial.add(str(minimum["sample_id"]))
            load_envelopes[load_case] = {
                "minimum_ratio": float(
                    minimum["pattern_to_triangular_ratio"]
                ),
                "minimum_sample": str(minimum["sample_id"]),
                "maximum_ratio": float(
                    maximum["pattern_to_triangular_ratio"]
                ),
                "maximum_sample": str(maximum["sample_id"]),
            }
        candidate_summary.append(
            {
                "pattern": pattern,
                "minimum_balanced_ratio": min(balanced_values),
                "maximum_balanced_ratio": max(balanced_values),
                "minimum_balanced_sample": minimum_balanced_sample,
                "best_sample_rank": min(
                    sorted(
                        PATTERNS,
                        key=lambda name: float(
                            next(
                                row for row in sample_rows
                                if row["sample_id"]
                                == sample["sample_id"]
                            )[f"{name}_balanced_ratio"]
                        ),
                        reverse=True,
                    ).index(pattern)
                    + 1
                    for sample in sample_rows
                ),
                "worst_sample_rank": max(
                    sorted(
                        PATTERNS,
                        key=lambda name: float(
                            next(
                                row for row in sample_rows
                                if row["sample_id"]
                                == sample["sample_id"]
                            )[f"{name}_balanced_ratio"]
                        ),
                        reverse=True,
                    ).index(pattern)
                    + 1
                    for sample in sample_rows
                ),
                "load_envelopes": load_envelopes,
            }
        )
    candidate_summary.sort(
        key=lambda row: float(row["minimum_balanced_ratio"]),
        reverse=True,
    )
    return {
        "fidelity": fidelity,
        "sample_count": len(scenario_rows),
        "all_solver_cases_passed": True,
        "solver_case_count": len(detail_rows),
        "candidate_summary": candidate_summary,
        "sample_rows": sample_rows,
        "detail_rows": detail_rows,
        "adversarial_sample_ids": sorted(adversarial),
    }


def _run_beam_matrix(
    profile_paths: Mapping[str, Path],
    output_dir: Path,
    runs_dir: Path,
) -> Dict[str, Sequence[Mapping[str, object]]]:
    geometries = finalist_geometries()
    scenario_rows = {}
    for sample_id, profile_path in profile_paths.items():
        results = run_benchmark(
            output_dir / "beam" / sample_id,
            runs_dir / "beam" / sample_id,
            properties_path=profile_path,
            patterns=list(PATTERNS),
            load_cases=LOAD_CASES,
            geometries=geometries,
        )
        scenario_rows[sample_id] = [
            vars(row) if isinstance(row, PatternResult) else dict(row)
            for row in results
        ]
    return scenario_rows


def _run_solid_sample(
    sample_id: str,
    profile_path: Path,
    output_dir: Path,
    runs_dir: Path,
    mesh_size_mm: float,
    stage: str,
) -> tuple[str, Sequence[Mapping[str, object]]]:
    results = run_solid_multiload(
        output_dir / stage / sample_id,
        runs_dir / stage / sample_id,
        properties_path=profile_path,
        patterns=list(PATTERNS),
        load_cases=LOAD_CASES,
        mesh_size_mm=mesh_size_mm,
        geometries=finalist_geometries(),
    )
    rows = [
        vars(row) if isinstance(row, SolidLoadResult) else dict(row)
        for row in results
    ]
    return sample_id, rows


def _run_solid_matrix(
    sample_ids: Sequence[str],
    profile_paths: Mapping[str, Path],
    output_dir: Path,
    runs_dir: Path,
    mesh_size_mm: float,
    *,
    stage: str,
    max_workers: int = MAX_PARALLEL_SCENARIOS,
) -> Dict[str, Sequence[Mapping[str, object]]]:
    expected_cases = len(PATTERNS) * len(LOAD_CASES)
    scenario_rows: Dict[str, Sequence[Mapping[str, object]]] = {}
    pending = []
    for sample_id in sample_ids:
        summary_path = output_dir / stage / sample_id / "summary.json"
        if summary_path.exists():
            rows = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                len(rows) == expected_cases
                and all(row.get("status") == "ok" for row in rows)
            ):
                scenario_rows[sample_id] = rows
                continue
        pending.append(sample_id)

    if pending:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        with ProcessPoolExecutor(
            max_workers=min(max_workers, len(pending))
        ) as executor:
            futures = {
                executor.submit(
                    _run_solid_sample,
                    sample_id,
                    profile_paths[sample_id],
                    output_dir,
                    runs_dir,
                    mesh_size_mm,
                    stage,
                ): sample_id
                for sample_id in pending
            }
            for future in as_completed(futures):
                sample_id, rows = future.result()
                scenario_rows[sample_id] = rows
    return {
        sample_id: scenario_rows[sample_id] for sample_id in sample_ids
    }


def _load_completed_matrix(
    root: Path,
    sample_ids: Sequence[str],
) -> Dict[str, Sequence[Mapping[str, object]]] | None:
    rows = {}
    for sample_id in sample_ids:
        path = Path(root) / sample_id / "summary.json"
        if not path.exists():
            return None
        rows[sample_id] = json.loads(path.read_text(encoding="utf-8"))
    return rows


def _refinement_change(
    coarse: Mapping[str, object],
    refined: Mapping[str, object],
) -> Dict[str, object]:
    coarse_lookup = {
        (
            str(row["sample_id"]),
            str(row["pattern"]),
            str(row["load_case"]),
        ): float(row["pattern_to_triangular_ratio"])
        for row in coarse["detail_rows"]
    }
    changes = []
    for row in refined["detail_rows"]:
        key = (
            str(row["sample_id"]),
            str(row["pattern"]),
            str(row["load_case"]),
        )
        coarse_ratio = coarse_lookup[key]
        refined_ratio = float(row["pattern_to_triangular_ratio"])
        changes.append(
            {
                "sample_id": key[0],
                "pattern": key[1],
                "load_case": key[2],
                "coarse_ratio": coarse_ratio,
                "refined_ratio": refined_ratio,
                "relative_change": refined_ratio / coarse_ratio - 1.0,
            }
        )
    maximum = max(abs(float(row["relative_change"])) for row in changes)
    return {
        "maximum_absolute_ratio_change": maximum,
        "ratio_refinement_gate_passed": maximum <= 0.05,
        "rows": changes,
    }


def run_campaign(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    *,
    lhs_samples: int = LHS_SAMPLES,
    seed: int = SEED,
    mesh_size_mm: float = 0.9,
) -> Dict[str, object]:
    output_dir, runs_dir = Path(output_dir), Path(runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_paths = write_profiles(
        output_dir, lhs_samples=lhs_samples, seed=seed
    )
    sample_ids = list(profile_paths)
    beam_rows = _load_completed_matrix(
        output_dir / "beam", sample_ids
    )
    if beam_rows is None:
        beam_rows = _run_beam_matrix(
            profile_paths, output_dir, runs_dir
        )
    beam = aggregate_interval_rows(beam_rows, fidelity="beam_B31")

    coarse_mesh_size_mm = 1.2
    coarse_rows = _load_completed_matrix(
        output_dir / "solid_coarse", sample_ids
    )
    if coarse_rows is None:
        coarse_rows = _run_solid_matrix(
            sample_ids,
            profile_paths,
            output_dir,
            runs_dir,
            coarse_mesh_size_mm,
            stage="solid_coarse",
        )
    coarse = aggregate_interval_rows(
        coarse_rows, fidelity="solid_C3D6_coarse"
    )
    promoted = coarse["adversarial_sample_ids"]
    refined_rows = _run_solid_matrix(
        promoted,
        profile_paths,
        output_dir,
        runs_dir,
        mesh_size_mm,
        stage="solid_refined",
    )
    refined = aggregate_interval_rows(
        refined_rows, fidelity="solid_C3D6_refined"
    )
    refinement = _refinement_change(coarse, refined)

    _write_csv(output_dir / "beam_sample_rankings.csv", beam["sample_rows"])
    _write_csv(output_dir / "beam_ratios.csv", beam["detail_rows"])
    _write_csv(
        output_dir / "coarse_solid_sample_rankings.csv",
        coarse["sample_rows"],
    )
    _write_csv(
        output_dir / "coarse_solid_ratios.csv", coarse["detail_rows"]
    )
    _write_csv(
        output_dir / "refined_solid_sample_rankings.csv",
        refined["sample_rows"],
    )
    _write_csv(
        output_dir / "refined_solid_ratios.csv", refined["detail_rows"]
    )
    _write_csv(
        output_dir / "refinement_ratio_changes.csv", refinement["rows"]
    )

    payload = {
        "study_id": "bambu_pla_directional_interval_robustness_v1",
        "material": "Bambu PLA Basic",
        "status": "bounded_sensitivity_only",
        "material_sample_count": len(profile_paths),
        "beam_solver_case_count": beam["solver_case_count"],
        "beam_interval_response_status": (
            "invariant_not_used_for_material_promotion"
        ),
        "coarse_solid_mesh_size_mm": coarse_mesh_size_mm,
        "coarse_solid_solver_case_count": coarse["solver_case_count"],
        "promoted_adversarial_sample_count": len(promoted),
        "promoted_adversarial_sample_ids": promoted,
        "refined_solid_mesh_size_mm": mesh_size_mm,
        "refined_solid_solver_case_count": refined["solver_case_count"],
        "all_solver_cases_passed": (
            beam["all_solver_cases_passed"]
            and coarse["all_solver_cases_passed"]
            and refined["all_solver_cases_passed"]
        ),
        "beam_candidate_summary": beam["candidate_summary"],
        "coarse_solid_candidate_summary": coarse["candidate_summary"],
        "refined_solid_candidate_summary": refined["candidate_summary"],
        "maximum_absolute_refinement_ratio_change": refinement[
            "maximum_absolute_ratio_change"
        ],
        "ratio_refinement_gate_passed": refinement[
            "ratio_refinement_gate_passed"
        ],
        "interpretation": (
            "deterministic non-probabilistic interval robustness screen"
        ),
        "claims_not_supported": [
            "probability that one pattern wins",
            "confidence interval on any unmeasured constant",
            "global mathematical worst case between sampled cards",
            "failure load, fracture, fatigue, creep, or environmental life",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    refined_by_pattern = {
        str(row["pattern"]): row
        for row in refined["candidate_summary"]
    }
    report = [
        "# Directional material-interval robustness",
        "",
        (
            f"{len(profile_paths)} admissible Bambu PLA Basic material cards "
            f"were screened in {beam['solver_case_count']} beam cases. "
            f"Because the beam ratios are invariant to this interval, all "
            f"cards were run in {coarse['solver_case_count']} coarse "
            f"equal-mass C3D6 cases. {len(promoted)} adversarial cards were "
            f"then promoted to {refined['solver_case_count']} refined cases."
        ),
        "",
        "This is deterministic interval coverage, not a Monte Carlo "
        "probability distribution. E1/E2/E3 use the TDS mean ±1 SD. "
        "nu12 is fixed to the separate exact-brand measurement. The "
        "unmeasured out-of-plane Poisson and static shear constants use "
        "declared broad sensitivity bounds.",
        "",
        "| Robust rank | Pattern | Coarse min balanced | Coarse max balanced | "
        "Rank range | Refined adverse bending |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(
        coarse["candidate_summary"], start=1
    ):
        bending = refined_by_pattern[str(row["pattern"])][
            "load_envelopes"
        ]["bending"]
        report.append(
            f"| {rank} | {row['pattern']} | "
            f"{row['minimum_balanced_ratio']:.3f} | "
            f"{row['maximum_balanced_ratio']:.3f} | "
            f"{row['best_sample_rank']}–{row['worst_sample_rank']} | "
            f"{bending['minimum_ratio']:.3f} |"
        )
    report.extend(
        [
            "",
            "The sampled minimum is not a proof of the continuous global "
            "minimum. It is a stronger material-card sensitivity screen than "
            "the previous four one-factor scenarios, while preserving the "
            "same physical-evidence exclusions.",
            "",
            (
                "Maximum coarse-to-refined candidate/control ratio change: "
                f"{refinement['maximum_absolute_ratio_change']:.2%}; "
                f"5% gate: {refinement['ratio_refinement_gate_passed']}."
            ),
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate a deterministic orthotropic material interval through "
            "the four printable infill finalists"
        )
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--lhs-samples", type=int, default=LHS_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mesh-size", type=float, default=0.9)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        paths = write_profiles(
            Path(args.output),
            lhs_samples=args.lhs_samples,
            seed=args.seed,
        )
        print(f"wrote {len(paths)} admissible interval profiles")
        return
    payload = run_campaign(
        Path(args.output),
        Path(args.runs_dir),
        lhs_samples=args.lhs_samples,
        seed=args.seed,
        mesh_size_mm=args.mesh_size,
    )
    print(
        f"{payload['material_sample_count']} material cards; "
        f"{payload['beam_solver_case_count']} beam cases; "
        f"{payload['coarse_solid_solver_case_count']} coarse solid cases; "
        f"{payload['refined_solid_solver_case_count']} refined solid cases; "
        f"status={payload['status']}"
    )


if __name__ == "__main__":
    main()
