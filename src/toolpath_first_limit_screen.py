from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .bambu_tds_profile import DEFAULT_TRANSVERSE_PROFILE
from .calculix import CalculixRunner
from .custom_pattern_toolpath import parse_deposited_segments
from .custom_pattern_toolpath_fea import (
    DISPLACEMENT_MM,
    FINE_RASTER_RESOLUTION,
    build_mass_calibrated_fine_geometry,
    coarsen_to_voxel_mesh,
    render_toolpath_bending_deck,
)
from .strength_screening import (
    orthotropic_stress_from_tensor_strain,
    screen_bounded_tension_shear_interaction,
    screen_interface_initiation,
)
from .weird_infill_toolpath_validation import (
    DEFAULT_GCODE,
    PATTERNS,
    SEARCH_ROOT,
    select_representative_objects,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = SEARCH_ROOT / "toolpath_first_limit_screen"
DEFAULT_RUNS = (
    PROJECT_ROOT
    / "runs/custom_infill/weird_kite_search/toolpath_first_limit_screen"
)
DEFAULT_INTERFACE_PROFILE = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_interlayer_strength_2026"
    / "interface_initiation_profiles.json"
)
DEFAULT_RESOLUTIONS = (60, 80)
MAXIMUM_LIMIT_MESH_CHANGE = 0.10
TARGET_FIRST_RAMP_UTILIZATION = 0.60


def _json(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def first_limit_metrics(
    stress_rows: np.ndarray,
    properties: Mapping[str, object],
    interface_profiles: Mapping[str, Mapping[str, object]],
    endpoint_displacement_mm: float = DISPLACEMENT_MM,
) -> Dict[str, object]:
    bounded = screen_bounded_tension_shear_interaction(
        stress_rows,
        [
            float(
                properties["reported_strength_reference"][
                    "overall_tensile_UTS_MPa"
                ]
            ),
            float(
                properties["reported_strength_reference"][
                    "overall_tensile_UTS_MPa"
                ]
            ),
            float(
                properties["reported_strength_reference"][
                    "z_upright_tensile_UTS_MPa"
                ]
            ),
        ],
        [
            float(
                properties["reported_strength_reference"][
                    "overall_tensile_UTS_MPa"
                ]
                - properties["reported_strength_reference"][
                    "overall_tensile_UTS_standard_deviation_MPa"
                ]
            ),
            float(
                properties["reported_strength_reference"][
                    "overall_tensile_UTS_MPa"
                ]
                - properties["reported_strength_reference"][
                    "overall_tensile_UTS_standard_deviation_MPa"
                ]
            ),
            float(
                properties["reported_strength_reference"][
                    "z_upright_tensile_UTS_MPa"
                ]
                - properties["reported_strength_reference"][
                    "z_upright_tensile_UTS_standard_deviation_MPa"
                ]
            ),
        ],
    )
    if bounded is None:
        raise ValueError("stress field contains no usable rows")
    interface_results: Dict[str, object] = {}
    for name, profile in interface_profiles.items():
        screen = screen_interface_initiation(
            stress_rows,
            float(profile["normal_tensile_mean_MPa"]),
            float(profile["apparent_tear_mean_MPa"]),
            float(profile["normal_tensile_lower_1sd_MPa"]),
            float(profile["apparent_tear_lower_1sd_MPa"]),
        )
        if screen is None:
            raise ValueError(f"interface screen {name} is empty")
        interface_results[name] = screen

    theory_conservative = float(
        bounded.p99_lower_1sd_von_mises_shear_bookend_utilization
    )
    theory_optimistic = float(
        bounded.p99_equal_tensile_shear_bookend_utilization
    )
    conservative_utilization = max(
        [theory_conservative]
        + [
            float(screen.p99_lower_1sd_utilization)
            for screen in interface_results.values()
        ]
    )
    conservative_limit = (
        float(endpoint_displacement_mm) / conservative_utilization
    )
    optimistic_scenario_limits = []
    for screen in interface_results.values():
        scenario_utilization = max(
            theory_optimistic,
            float(screen.p99_utilization),
        )
        optimistic_scenario_limits.append(
            float(endpoint_displacement_mm) / scenario_utilization
        )
    optimistic_limit = max(optimistic_scenario_limits)
    if conservative_limit > optimistic_limit:
        raise ValueError("first-limit bounds are reversed")
    result: Dict[str, object] = {
        **bounded.to_dict(),
        "conservative_p99_combined_utilization": conservative_utilization,
        "conservative_p99_first_limit_displacement_mm": conservative_limit,
        "optimistic_p99_first_limit_displacement_mm": optimistic_limit,
        "recommended_first_ramp_displacement_mm": min(
            float(endpoint_displacement_mm),
            TARGET_FIRST_RAMP_UTILIZATION * conservative_limit,
        ),
        "endpoint_exceeds_conservative_p99_limit": (
            conservative_utilization >= 1.0
        ),
    }
    for name, screen in interface_results.items():
        for key, value in screen.to_dict().items():
            result[f"{name}_{key}"] = value
    return result


def evaluate_first_limit_gate(
    rows: Sequence[Mapping[str, object]],
    patterns: Sequence[str] = PATTERNS,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
) -> Dict[str, object]:
    lookup = {
        (str(row["pattern"]), int(row["resolution_xy"])): row
        for row in rows
    }
    expected = {
        (pattern, int(resolution))
        for pattern in patterns
        for resolution in resolutions
    }
    if set(lookup) != expected:
        raise ValueError("first-limit rows do not cover the requested matrix")
    all_ok = all(row["status"] == "ok" for row in rows)
    all_fields = all(
        int(row["sampled_strain_points"]) >= 1 for row in rows
    )
    all_bounds = all(
        0.0
        < float(row["conservative_p99_first_limit_displacement_mm"])
        <= float(row["optimistic_p99_first_limit_displacement_mm"])
        for row in rows
    )
    fine = max(int(value) for value in resolutions)
    previous = (
        sorted(int(value) for value in resolutions)[-2]
        if len(resolutions) > 1
        else fine
    )
    mesh_changes = {}
    for pattern in patterns:
        mesh_changes[pattern] = {}
        for field in (
            "conservative_p99_first_limit_displacement_mm",
            "optimistic_p99_first_limit_displacement_mm",
        ):
            coarse = float(lookup[(pattern, previous)][field])
            fine_value = float(lookup[(pattern, fine)][field])
            mesh_changes[pattern][field] = abs(
                coarse / fine_value - 1.0
            )
    mesh_converged = all(
        change <= MAXIMUM_LIMIT_MESH_CHANGE
        for by_field in mesh_changes.values()
        for change in by_field.values()
    )
    admitted = all_ok and all_fields and all_bounds and mesh_converged
    return {
        "fine_resolution_xy": fine,
        "first_limit_mesh_changes": mesh_changes,
        "all_calculix_cases_passed": all_ok,
        "all_endpoint_strain_fields_present": all_fields,
        "all_first_limit_bands_ordered_and_positive": all_bounds,
        "all_60_to_80_limit_changes_at_most_10_percent": mesh_converged,
        "bounded_first_limit_screen_admitted": admitted,
        "absolute_failure_physically_validated": False,
    }


def run_toolpath_first_limit_screen(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    gcode_path: Path = DEFAULT_GCODE,
    properties_path: Path = DEFAULT_TRANSVERSE_PROFILE,
    interface_profile_path: Path = DEFAULT_INTERFACE_PROFILE,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    gcode_path = Path(gcode_path)
    resolutions = tuple(sorted(set(int(value) for value in resolutions)))
    if len(resolutions) < 2:
        raise ValueError("at least two resolutions are required")
    if any(FINE_RASTER_RESOLUTION % value != 0 for value in resolutions):
        raise ValueError("resolutions must divide the fine raster")
    frozen = _json(SEARCH_ROOT / "toolpath_validation/summary.json")
    if (
        frozen.get("gate", {}).get("toolpath_candidate_gate_passed")
        is not True
        or frozen.get("source_gcode_sha256") != _sha256(gcode_path)
    ):
        raise ValueError("frozen weird-infill toolpath gate is not admitted")
    properties = _json(properties_path)
    interface_source = _json(interface_profile_path)
    interface_profiles = interface_source["profiles"]
    constants = properties["screening_engineering_constants"]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decks").mkdir(parents=True, exist_ok=True)
    parsed = parse_deposited_segments(gcode_path)
    selected = select_representative_objects(parsed)
    runner = CalculixRunner(
        runs_dir,
        executable="ccx",
        timeout_seconds=600.0,
        keep_successful=True,
    )
    rows: list[Dict[str, object]] = []
    for pattern, object_name in selected.items():
        geometry = build_mass_calibrated_fine_geometry(
            pattern,
            object_name,
            parsed[object_name],
            resolution=FINE_RASTER_RESOLUTION,
        )
        for resolution in resolutions:
            mesh = coarsen_to_voxel_mesh(geometry, resolution)
            deck = output_dir / "decks" / f"{pattern}-{resolution}.inp"
            deck.write_text(
                render_toolpath_bending_deck(
                    mesh,
                    properties,
                    displacement_mm=DISPLACEMENT_MM,
                    include_field_output=True,
                ),
                encoding="utf-8",
            )
            solver = runner.run(
                deck,
                {},
                run_id=f"toolpath-first-limit-{pattern}-{resolution}",
            )
            row: Dict[str, object] = {
                "pattern": pattern,
                "resolution_xy": resolution,
                "status": solver.status,
                "nodes": len(mesh.node_tags),
                "elements": len(mesh.element_tags),
                "represented_volume_mm3": mesh.represented_volume_mm3,
                "sampled_strain_points": 0,
                "endpoint_displacement_mm": DISPLACEMENT_MM,
                "reaction_force_N": None,
                "bending_stiffness_N_per_mm": None,
                "run_directory": str(solver.run_dir.resolve()),
                "deck": str(deck.resolve()),
                "deck_sha256": _sha256(deck),
                "error": solver.error,
            }
            if solver.ok:
                strain = solver.curve.get("frd_tostrain")
                if strain is None:
                    row["status"] = "strain_field_missing"
                    row["error"] = "CalculiX FRD has no TOSTRAIN field"
                else:
                    stress = orthotropic_stress_from_tensor_strain(
                        strain, constants
                    )
                    metrics = first_limit_metrics(
                        stress,
                        properties,
                        interface_profiles,
                        DISPLACEMENT_MM,
                    )
                    displacement = float(
                        abs(solver.curve["displacement"][-1])
                    )
                    force = float(
                        abs(solver.curve["reaction_force"][-1])
                    )
                    row.update(metrics)
                    row["sampled_strain_points"] = len(strain)
                    row["reaction_force_N"] = force
                    row["bending_stiffness_N_per_mm"] = (
                        force / displacement
                    )
                    row["conservative_p99_first_limit_force_N"] = (
                        force
                        * float(
                            metrics[
                                "conservative_p99_first_limit_displacement_mm"
                            ]
                        )
                        / DISPLACEMENT_MM
                    )
                    row["optimistic_p99_first_limit_force_N"] = (
                        force
                        * float(
                            metrics[
                                "optimistic_p99_first_limit_displacement_mm"
                            ]
                        )
                        / DISPLACEMENT_MM
                    )
            rows.append(row)
    _write_csv(output_dir / "results.csv", rows)
    gate = evaluate_first_limit_gate(rows, PATTERNS, resolutions)
    fine = int(gate["fine_resolution_xy"])
    fine_rows = [
        row for row in rows if int(row["resolution_xy"]) == fine
    ]
    fine_rows.sort(key=lambda row: PATTERNS.index(str(row["pattern"])))
    safe_ramp_rows = [
        {
            "pattern": row["pattern"],
            "replicates": 3,
            "simulated_endpoint_displacement_mm": DISPLACEMENT_MM,
            "recommended_first_ramp_displacement_mm": row[
                "recommended_first_ramp_displacement_mm"
            ],
            "conservative_p99_first_limit_displacement_mm": row[
                "conservative_p99_first_limit_displacement_mm"
            ],
            "optimistic_p99_first_limit_displacement_mm": row[
                "optimistic_p99_first_limit_displacement_mm"
            ],
            "required_observation": (
                "stop on audible or visible cracking; record crack site and "
                "full force-displacement curve"
            ),
        }
        for row in fine_rows
    ]
    _write_csv(output_dir / "safe_first_ramp.csv", safe_ramp_rows)

    payload: Dict[str, object] = {
        "study_id": "bambu_pla_weird_kite_toolpath_first_limit_screen_v1",
        "material": "Bambu PLA Basic",
        "status": (
            "bounded_toolpath_first_limit_screen_admitted"
            if gate["bounded_first_limit_screen_admitted"]
            else "toolpath_first_limit_screen_not_admitted"
        ),
        "source_gcode": str(gcode_path.resolve()),
        "source_gcode_sha256": _sha256(gcode_path),
        "properties": str(Path(properties_path).resolve()),
        "properties_sha256": _sha256(properties_path),
        "interface_profiles": str(
            Path(interface_profile_path).resolve()
        ),
        "interface_profiles_sha256": _sha256(interface_profile_path),
        "patterns": list(PATTERNS),
        "resolutions_xy": list(resolutions),
        "endpoint_displacement_mm": DISPLACEMENT_MM,
        "fine_results": {
            row["pattern"]: {
                "bending_stiffness_N_per_mm": row[
                    "bending_stiffness_N_per_mm"
                ],
                "conservative_p99_first_limit_displacement_mm": row[
                    "conservative_p99_first_limit_displacement_mm"
                ],
                "optimistic_p99_first_limit_displacement_mm": row[
                    "optimistic_p99_first_limit_displacement_mm"
                ],
                "conservative_p99_first_limit_force_N": row[
                    "conservative_p99_first_limit_force_N"
                ],
                "optimistic_p99_first_limit_force_N": row[
                    "optimistic_p99_first_limit_force_N"
                ],
                "recommended_first_ramp_displacement_mm": row[
                    "recommended_first_ramp_displacement_mm"
                ],
                "governing_component": row["governing_component"],
                "sampled_strain_points": row[
                    "sampled_strain_points"
                ],
            }
            for row in fine_rows
        },
        "gate": gate,
        "solver_case_count": len(rows),
        "restrictions": [
            "The voxel stiffness uses fill-fraction scaling; solid-material "
            "stress is reconstructed from the saved strain field.",
            "The p99 metric is node sampled and is not a volume percentile.",
            "The conservative shear boundary is theoretical because matched "
            "Bambu quasistatic shear strength is unavailable.",
            "Interface opening/tear strengths come from a modified-printer "
            "thin-wall process and are retained as process sensitivities.",
            "Linear scaling estimates first-limit bands only; there is no "
            "damage evolution, crack growth, post-peak, fatigue, or lifetime.",
            "Absolute first failure requires the registered physical coupons.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Frozen-toolpath first-limit bending screen",
        "",
        "The sliced paths were replayed to 0.4 mm, endpoint strain fields "
        "were converted back to solid-material local stress, and p99 first-"
        "limit bands were formed from the directional tensile/shear and "
        "interface-initiation sensitivities.",
        "",
        "| Pattern | Stiffness (N/mm) | Conservative limit (mm) | "
        "Optimistic limit (mm) | First ramp (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fine_rows:
        report.append(
            f"| {row['pattern']} | "
            f"{float(row['bending_stiffness_N_per_mm']):.3f} | "
            f"{float(row['conservative_p99_first_limit_displacement_mm']):.3f} | "
            f"{float(row['optimistic_p99_first_limit_displacement_mm']):.3f} | "
            f"{float(row['recommended_first_ramp_displacement_mm']):.3f} |"
        )
    report.extend(
        [
            "",
            f"Decision: **{payload['status']}**.",
            "",
            "The band is a linear test-planning sensitivity, not predicted "
            "fracture or final load. The generated safe-ramp table should be "
            "used before advancing the physical coupons to 0.4 mm.",
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
            "Screen frozen weird-infill toolpaths for bounded first material "
            "limits under bending"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--gcode", type=Path, default=DEFAULT_GCODE)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
    )
    args = parser.parse_args()
    result = run_toolpath_first_limit_screen(
        output_dir=args.output,
        runs_dir=args.runs,
        gcode_path=args.gcode,
        resolutions=args.resolutions,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "solver_case_count": result["solver_case_count"],
                "fine_results": result["fine_results"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
