from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .custom_infill import load_custom_pattern
from .custom_pattern_slicer import (
    FILAMENT_OVERRIDE,
    OFFICIAL_MACHINE,
    PROCESS_OVERRIDE,
    _config_from_gcode,
    _expected_config_gate,
    _run as run_orca,
    find_orca_slicer,
    parse_object_extrusion,
)
from .custom_pattern_toolpath import (
    DEFAULT_MATERIAL_PROFILE as DIRECTIONAL_MATERIAL_PROFILE,
    parse_deposited_segments,
)
from .custom_pattern_toolpath_fea import (
    build_mass_calibrated_fine_geometry,
    coarsen_to_voxel_mesh,
    render_toolpath_bending_deck,
)
from .calculix import CalculixRunner
from .infill_benchmark import (
    LOAD_CASES,
    PatternGeometry,
    make_geometry_from_segments,
    make_pattern,
    nominal_strut_width,
    run_benchmark,
)
from .weird_infill_search import (
    _three_mf_build_count,
    make_kite_geometry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/pattern_robustness_campaign"
DEFAULT_RUNS = PROJECT_ROOT / "runs/pattern_robustness_campaign"
PROPERTIES = PROJECT_ROOT / "data/reported/bambu_pla_screening.json"
WIDTH_MM = 40.0
HEIGHT_MM = 40.0
DEPTH_MM = 4.0
RELATIVE_DENSITY = 0.35
BEAD_WIDTH_SCALES = (0.90, 1.00, 1.10)
PATTERNS = (
    "triangular",
    "kite_08_08_08_08",
    "kite_08_08_12_08",
    "bowtie_fan",
)
SHORT_NAMES = {
    "triangular": "tri",
    "kite_08_08_08_08": "ku",
    "kite_08_08_12_08": "kg",
    "bowtie_fan": "bf",
}
SOURCE_STLS = {
    "triangular": (
        PROJECT_ROOT
        / "results/custom_infill/weird_kite_search/"
        "physical_validation/stl/triangular.stl"
    ),
    "kite_08_08_08_08": (
        PROJECT_ROOT
        / "results/custom_infill/weird_kite_search/"
        "physical_validation/stl/kite_08_08_08_08.stl"
    ),
    "kite_08_08_12_08": (
        PROJECT_ROOT
        / "results/custom_infill/weird_kite_search/"
        "physical_validation/stl/kite_08_08_12_08.stl"
    ),
    "bowtie_fan": (
        PROJECT_ROOT
        / "results/custom_infill/bowtie_fan/"
        "physical_validation/printable_stl/bowtie_fan.stl"
    ),
}
TOOLPATH_RESOLUTIONS = (60, 80, 120)
FINE_RASTER_RESOLUTION = 240
MAXIMUM_TOOLPATH_CONVERGENCE_CHANGE = 0.05
MAXIMUM_PHYSICAL_GROUP_CV = 0.10
MAXIMUM_PHYSICAL_RATIO_ERROR = 0.20


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def finalist_geometries() -> Dict[str, PatternGeometry]:
    bowtie = load_custom_pattern(PROJECT_ROOT / "ideas/bowtie_fan.json")[1]
    return {
        "triangular": make_pattern("triangular"),
        "kite_08_08_08_08": make_kite_geometry(
            (8.0, 8.0, 8.0, 8.0), "kite_08_08_08_08"
        ),
        "kite_08_08_12_08": make_kite_geometry(
            (8.0, 8.0, 12.0, 8.0), "kite_08_08_12_08"
        ),
        "bowtie_fan": bowtie,
    }


def _segments(geometry: PatternGeometry):
    return [
        (
            geometry.nodes[first - 1],
            geometry.nodes[second - 1],
        )
        for first, second in geometry.edges
    ]


def rotate_quarter_turn(
    geometry: PatternGeometry, name: str
) -> PatternGeometry:
    return make_geometry_from_segments(
        name,
        [
            (
                (HEIGHT_MM - first[1], first[0]),
                (HEIGHT_MM - second[1], second[0]),
            )
            for first, second in _segments(geometry)
        ],
        WIDTH_MM,
    )


def _graph_connected(geometry: PatternGeometry) -> bool:
    adjacency = {
        index: set() for index in range(1, len(geometry.nodes) + 1)
    }
    for first, second in geometry.edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    visited = set()
    pending = [1]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency[current] - visited)
    return len(visited) == len(adjacency)


def _interior_edge_indices(geometry: PatternGeometry) -> list[int]:
    indices = []
    for index, (first, second) in enumerate(geometry.edges):
        start = geometry.nodes[first - 1]
        end = geometry.nodes[second - 1]
        same_perimeter = (
            start[0] == end[0] and start[0] in (0.0, WIDTH_MM)
        ) or (
            start[1] == end[1] and start[1] in (0.0, HEIGHT_MM)
        )
        if not same_perimeter:
            indices.append(index)
    return indices


def build_perturbation_matrix() -> Dict[str, object]:
    bases = finalist_geometries()
    geometries: Dict[str, PatternGeometry] = {}
    widths: Dict[str, float] = {}
    metadata: Dict[str, Dict[str, object]] = {}
    structural_breaks = []
    for base_name, base in bases.items():
        short = SHORT_NAMES[base_name]
        nominal_width = nominal_strut_width(
            base, WIDTH_MM, HEIGHT_MM, RELATIVE_DENSITY
        )
        orientations = {
            0: make_geometry_from_segments(
                f"{short}_o0_nom", _segments(base), WIDTH_MM
            ),
            90: rotate_quarter_turn(base, f"{short}_o90_nom"),
        }
        for angle, oriented in orientations.items():
            for scale in BEAD_WIDTH_SCALES:
                suffix = (
                    "nom"
                    if math.isclose(scale, 1.0)
                    else f"w{int(round(scale * 100)):03d}"
                )
                variant = f"{short}_o{angle}_{suffix}"
                geometry = make_geometry_from_segments(
                    variant, _segments(oriented), WIDTH_MM
                )
                geometries[variant] = geometry
                widths[variant] = nominal_width * scale
                metadata[variant] = {
                    "base_pattern": base_name,
                    "orientation_degrees": angle,
                    "scenario": (
                        "nominal"
                        if math.isclose(scale, 1.0)
                        else "uniform_bead_width"
                    ),
                    "bead_width_scale": scale,
                    "missing_edge_index": None,
                    "nominal_base_strut_width_mm": nominal_width,
                }
        original_segments = _segments(base)
        for edge_index in _interior_edge_indices(base):
            variant = f"{short}_o0_m{edge_index:03d}"
            remaining = [
                segment
                for index, segment in enumerate(original_segments)
                if index != edge_index
            ]
            geometry = make_geometry_from_segments(
                variant, remaining, WIDTH_MM
            )
            row = {
                "variant": variant,
                "base_pattern": base_name,
                "orientation_degrees": 0,
                "scenario": "single_missing_interior_strut",
                "bead_width_scale": 1.0,
                "missing_edge_index": edge_index,
                "nominal_base_strut_width_mm": nominal_width,
            }
            if not _graph_connected(geometry):
                structural_breaks.append(row)
                continue
            geometries[variant] = geometry
            widths[variant] = nominal_width
            metadata[variant] = {key: value for key, value in row.items() if key != "variant"}
    return {
        "base_geometries": bases,
        "geometries": geometries,
        "strut_widths": widths,
        "metadata": metadata,
        "structural_breaks": structural_breaks,
    }


def _metric(row: Mapping[str, object]) -> float:
    if row.get("status") != "ok":
        return 0.0
    field = (
        "critical_buckling_load_N"
        if row["load_case"] == "buckling"
        else "stiffness_N_per_mm"
    )
    value = row.get(field)
    return 0.0 if value is None else float(value)


def aggregate_robustness(
    result_rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, Mapping[str, object]],
    structural_breaks: Sequence[Mapping[str, object]] = (),
) -> Dict[str, object]:
    lookup = {
        (str(row["pattern"]), str(row["load_case"])): _metric(row)
        for row in result_rows
    }
    for row in structural_breaks:
        for load_case in LOAD_CASES:
            lookup[(str(row["variant"]), load_case)] = 0.0
    base_variants: Dict[str, list[str]] = {name: [] for name in PATTERNS}
    for variant, row in metadata.items():
        base_variants[str(row["base_pattern"])].append(variant)
    for row in structural_breaks:
        base_variants[str(row["base_pattern"])].append(str(row["variant"]))

    summaries = []
    conservative_by_pattern: Dict[str, Dict[str, float]] = {}
    for base in PATTERNS:
        short = SHORT_NAMES[base]
        conservative: Dict[str, float] = {}
        details: Dict[str, object] = {}
        for load_case in LOAD_CASES:
            nominal_0 = lookup[(f"{short}_o0_nom", load_case)]
            nominal_90 = lookup[(f"{short}_o90_nom", load_case)]
            bead_90 = [
                lookup[(f"{short}_o{angle}_w090", load_case)]
                for angle in (0, 90)
            ]
            bead_110 = [
                lookup[(f"{short}_o{angle}_w110", load_case)]
                for angle in (0, 90)
            ]
            missing_variants = [
                variant
                for variant in base_variants[base]
                if (
                    metadata.get(variant, {}).get("scenario")
                    == "single_missing_interior_strut"
                    or any(
                        str(row["variant"]) == variant
                        for row in structural_breaks
                    )
                )
            ]
            missing_values = np.asarray(
                [lookup[(variant, load_case)] for variant in missing_variants],
                dtype=float,
            )
            missing_p10 = float(np.percentile(missing_values, 10.0))
            missing_worst = float(np.min(missing_values))
            floor = min(nominal_0, nominal_90, *bead_90, missing_p10)
            conservative[load_case] = floor
            details[load_case] = {
                "nominal_orientation_0": nominal_0,
                "nominal_orientation_90": nominal_90,
                "orientation_min_to_max_ratio": (
                    min(nominal_0, nominal_90)
                    / max(nominal_0, nominal_90)
                ),
                "bead_90_min_retention": (
                    min(bead_90)
                    / min(nominal_0, nominal_90)
                ),
                "bead_110_max_change": (
                    max(bead_110)
                    / max(nominal_0, nominal_90)
                    - 1.0
                ),
                "single_missing_strut_scenarios": len(missing_values),
                "single_missing_strut_p10_retention": (
                    missing_p10 / nominal_0
                ),
                "single_missing_strut_worst_retention": (
                    missing_worst / nominal_0
                ),
                "conservative_metric": floor,
            }
        conservative_by_pattern[base] = conservative
        summaries.append(
            {
                "pattern": base,
                "load_details": details,
            }
        )

    control = conservative_by_pattern["triangular"]
    control_summary = next(
        summary
        for summary in summaries
        if summary["pattern"] == "triangular"
    )
    control_nominal = {
        load_case: float(
            control_summary["load_details"][load_case][
                "nominal_orientation_0"
            ]
        )
        for load_case in LOAD_CASES
    }
    ranking = []
    for summary in summaries:
        base = str(summary["pattern"])
        ratios = {
            load_case: (
                conservative_by_pattern[base][load_case]
                / control[load_case]
            )
            for load_case in LOAD_CASES
        }
        nominal_ratios = {
            load_case: (
                float(
                    summary["load_details"][load_case][
                        "nominal_orientation_0"
                    ]
                )
                / control_nominal[load_case]
            )
            for load_case in LOAD_CASES
        }
        balanced = float(
            math.exp(
                np.mean([math.log(max(value, 1.0e-12)) for value in ratios.values()])
            )
        )
        summary["conservative_to_triangular"] = ratios
        summary["nominal_orientation_0_to_triangular"] = nominal_ratios
        summary["robust_balanced_ratio"] = balanced
        summary["robust_worst_load_ratio"] = min(ratios.values())
        summary["conservative_bending_ratio"] = ratios["bending"]
        ranking.append(
            {
                "pattern": base,
                "robust_balanced_ratio": balanced,
                "robust_worst_load_ratio": min(ratios.values()),
                "nominal_orientation_0_bending_ratio": nominal_ratios[
                    "bending"
                ],
                "conservative_bending_ratio": ratios["bending"],
                **{
                    f"{load_case}_conservative_to_triangular": value
                    for load_case, value in ratios.items()
                },
            }
        )
    ranking.sort(
        key=lambda row: float(row["robust_balanced_ratio"]),
        reverse=True,
    )
    noncontrol = [row for row in ranking if row["pattern"] != "triangular"]
    balanced_specialist = str(noncontrol[0]["pattern"])
    robust_bending_specialist = str(
        max(
            noncontrol,
            key=lambda row: float(row["conservative_bending_ratio"]),
        )["pattern"]
    )
    bending_specialist = str(
        max(
            noncontrol,
            key=lambda row: float(
                row["nominal_orientation_0_bending_ratio"]
            ),
        )["pattern"]
    )
    selected = [balanced_specialist]
    for pattern in (robust_bending_specialist, bending_specialist):
        if pattern not in selected:
            selected.append(pattern)
    for row in sorted(
        noncontrol,
        key=lambda item: float(item["conservative_bending_ratio"]),
        reverse=True,
    ):
        if len(selected) >= 2:
            break
        if row["pattern"] not in selected:
            selected.append(str(row["pattern"]))
    return {
        "ranking": ranking,
        "pattern_summaries": summaries,
        "balanced_specialist": balanced_specialist,
        "robust_bending_specialist": robust_bending_specialist,
        "bending_specialist": bending_specialist,
        "physical_candidates": selected,
        "selection_contract": (
            "minimax envelope over 0/90 placement, 10% bead-width loss, "
            "and the 10th-percentile single-missing-interior-strut response; "
            "balanced score is the geometric mean across five load cases"
        ),
    }


def write_finalist_plate(
    output_dir: Path,
    patterns: Sequence[str],
    ranking: Sequence[Mapping[str, object]],
    replicates: int = 3,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    stl_dir = output_dir / "stl"
    stl_dir.mkdir(parents=True, exist_ok=True)
    stls = []
    source_rows = []
    for pattern in patterns:
        source = SOURCE_STLS[pattern]
        if not source.exists():
            raise ValueError(f"missing admitted source STL: {source}")
        target = stl_dir / f"{pattern}.stl"
        shutil.copy2(source, target)
        stls.append(target)
        source_rows.append(
            {
                "pattern": pattern,
                "source_stl": str(source.resolve()),
                "source_stl_sha256": _sha256(source),
                "packaged_stl": str(target.resolve()),
                "packaged_stl_sha256": _sha256(target),
            }
        )
    orca = find_orca_slicer()
    plate = output_dir / "bambu-pla-robust-finalist-showdown.3mf"
    run_orca(
        (
            str(orca),
            "--debug",
            "2",
            "--load-settings",
            f"{OFFICIAL_MACHINE};{PROCESS_OVERRIDE}",
            "--load-filaments",
            str(FILAMENT_OVERRIDE),
            "--clone-objects",
            ",".join(str(replicates) for _ in stls),
            "--arrange",
            "1",
            "--ensure-on-bed",
            "--export-3mf",
            str(plate),
            *(str(path) for path in stls),
        ),
        output_dir / "export.log",
    )
    sliced = output_dir / "sliced"
    sliced.mkdir(parents=True, exist_ok=True)
    run_orca(
        (
            str(orca),
            "--debug",
            "2",
            "--slice",
            "0",
            "--outputdir",
            str(sliced),
            str(plate),
        ),
        output_dir / "slice.log",
    )
    gcode = sliced / "plate_1.gcode"
    if not gcode.exists():
        raise RuntimeError("OrcaSlicer did not produce finalist G-code")
    extrusion = parse_object_extrusion(gcode)
    object_rows: Dict[str, list[Mapping[str, object]]] = {}
    for pattern in patterns:
        matches = [
            row
            for name, row in extrusion.items()
            if name.startswith(f"{pattern}.stl")
        ]
        if len(matches) != replicates:
            raise ValueError(
                f"expected {replicates} sliced {pattern} objects, found "
                f"{len(matches)}"
            )
        object_rows[pattern] = matches
    means = {
        pattern: float(
            np.mean(
                [
                    float(row["predicted_deposited_mass_g"])
                    for row in rows
                ]
            )
        )
        for pattern, rows in object_rows.items()
    }
    mass_spread = max(means.values()) / min(means.values()) - 1.0
    config = _config_from_gcode(gcode)
    build_count = _three_mf_build_count(plate)
    gate = bool(
        build_count == len(patterns) * replicates
        and _expected_config_gate(config)
        and mass_spread <= 0.03
    )
    ranking_lookup = {str(row["pattern"]): row for row in ranking}
    plan = []
    capture = []
    order = 0
    for replicate in range(1, replicates + 1):
        for pattern in patterns:
            order += 1
            specimen = f"RF-{order:02d}-{pattern}-R{replicate}"
            plan.append(
                {
                    "test_order": order,
                    "specimen_id": specimen,
                    "pattern": pattern,
                    "replicate": replicate,
                    "predicted_deposited_mass_g": object_rows[pattern][
                        replicate - 1
                    ]["predicted_deposited_mass_g"],
                    "robust_balanced_ratio": ranking_lookup[pattern][
                        "robust_balanced_ratio"
                    ],
                    "nominal_orientation_0_bending_ratio": ranking_lookup[
                        pattern
                    ]["nominal_orientation_0_bending_ratio"],
                    "conservative_bending_ratio": ranking_lookup[pattern][
                        "conservative_bending_ratio"
                    ],
                    "test": "three_point_bending_to_0p4mm",
                    "material": "Bambu PLA Basic",
                }
            )
            capture.append(
                {
                    "specimen_id": specimen,
                    "actual_mass_g": "",
                    "ambient_temperature_C": "",
                    "relative_humidity_percent": "",
                    "bending_stiffness_N_per_mm": "",
                    "failure_or_damage_observed": "",
                    "notes": "",
                }
            )
    _write_csv(output_dir / "source_stls.csv", source_rows)
    _write_csv(output_dir / "physical_test_plan.csv", plan)
    _write_csv(output_dir / "measurement_capture.csv", capture)
    manifest = {
        "package_id": "bambu_pla_robust_finalist_showdown_v1",
        "material": "Bambu PLA Basic",
        "patterns": list(patterns),
        "replicates_per_pattern": replicates,
        "build_item_count": build_count,
        "plate_3mf": str(plate.resolve()),
        "plate_3mf_sha256": _sha256(plate),
        "gcode": str(gcode.resolve()),
        "gcode_sha256": _sha256(gcode),
        "predicted_deposited_mass_g": means,
        "maximum_pattern_mean_mass_spread": mass_spread,
        "slicer_config_gate_passed": _expected_config_gate(config),
        "manufacturing_gate_passed": gate,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not gate:
        raise RuntimeError("robust-finalist physical plate gate failed")
    return manifest


def run_finalist_toolpath_replay(
    output_dir: Path,
    runs_dir: Path,
    physical_manifest: Mapping[str, object],
    resolutions: Sequence[int] = TOOLPATH_RESOLUTIONS,
) -> Dict[str, object]:
    output_dir, runs_dir = Path(output_dir), Path(runs_dir)
    gcode = Path(str(physical_manifest["gcode"]))
    if (
        physical_manifest.get("manufacturing_gate_passed") is not True
        or physical_manifest.get("gcode_sha256") != _sha256(gcode)
    ):
        raise ValueError("finalist physical G-code gate is not valid")
    patterns = tuple(str(value) for value in physical_manifest["patterns"])
    profile = json.loads(
        DIRECTIONAL_MATERIAL_PROFILE.read_text(encoding="utf-8")
    )
    parsed = parse_deposited_segments(gcode)
    selected_objects = {
        pattern: next(
            name
            for name in sorted(parsed)
            if name.startswith(f"{pattern}.stl_")
        )
        for pattern in patterns
    }
    for directory in ("fine_geometry", "voxel_meshes", "decks", "curves"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    runner = CalculixRunner(
        runs_dir,
        executable="ccx",
        timeout_seconds=600.0,
        keep_successful=True,
    )
    rows = []
    geometries = {}
    for pattern, object_name in selected_objects.items():
        geometry = build_mass_calibrated_fine_geometry(
            pattern,
            object_name,
            parsed[object_name],
            resolution=FINE_RASTER_RESOLUTION,
        )
        geometry_path = (
            output_dir
            / "fine_geometry"
            / f"{pattern}-{FINE_RASTER_RESOLUTION}.npz"
        )
        np.savez_compressed(
            geometry_path,
            occupancy=geometry.occupancy,
            raster_size_mm=np.asarray([geometry.raster_size_mm]),
        )
        geometries[pattern] = {
            "object_name": object_name,
            "target_deposited_volume_mm3": (
                geometry.target_deposited_volume_mm3
            ),
            "raster_volume_mm3": geometry.raster_volume_mm3,
            "mass_error_fraction": geometry.mass_error_fraction,
            "component_count": geometry.component_count,
            "fine_geometry": str(geometry_path.resolve()),
            "fine_geometry_sha256": _sha256(geometry_path),
        }
        for resolution in resolutions:
            mesh = coarsen_to_voxel_mesh(geometry, int(resolution))
            mesh_path = (
                output_dir
                / "voxel_meshes"
                / f"{pattern}-{resolution}.npz"
            )
            np.savez_compressed(
                mesh_path,
                node_tags=mesh.node_tags,
                coordinates=mesh.coordinates,
                element_tags=mesh.element_tags,
                connectivity=mesh.connectivity,
                fill_fractions=mesh.fill_fractions,
            )
            deck_path = output_dir / "decks" / f"{pattern}-{resolution}.inp"
            deck_path.write_text(
                render_toolpath_bending_deck(mesh, profile),
                encoding="utf-8",
            )
            solver = runner.run(
                deck_path,
                {},
                run_id=f"robust-finalist-{pattern}-{resolution}",
            )
            stiffness = None
            if solver.ok:
                displacement = float(
                    abs(solver.curve["displacement"][-1])
                )
                force = float(abs(solver.curve["reaction_force"][-1]))
                stiffness = force / displacement
                curve_path = (
                    output_dir
                    / "curves"
                    / f"{pattern}-{resolution}.csv"
                )
                _write_csv(
                    curve_path,
                    [
                        {
                            "displacement_mm": float(current),
                            "force_N": float(current_force),
                        }
                        for current, current_force in zip(
                            np.abs(solver.curve["displacement"]),
                            np.abs(solver.curve["reaction_force"]),
                        )
                    ],
                )
            rows.append(
                {
                    "pattern": pattern,
                    "resolution_xy": int(resolution),
                    "status": solver.status,
                    "nodes": len(mesh.node_tags),
                    "elements": len(mesh.element_tags),
                    "component_count": mesh.component_count,
                    "represented_volume_mm3": mesh.represented_volume_mm3,
                    "target_deposited_volume_mm3": (
                        geometry.target_deposited_volume_mm3
                    ),
                    "mass_error_fraction": (
                        mesh.represented_volume_mm3
                        / geometry.target_deposited_volume_mm3
                        - 1.0
                    ),
                    "bending_stiffness_N_per_mm": stiffness,
                    "run_directory": str(solver.run_dir.resolve()),
                    "deck": str(deck_path.resolve()),
                    "deck_sha256": _sha256(deck_path),
                    "voxel_mesh": str(mesh_path.resolve()),
                    "voxel_mesh_sha256": _sha256(mesh_path),
                    "error": solver.error,
                }
            )
    _write_csv(output_dir / "results.csv", rows)
    ordered = sorted(set(int(value) for value in resolutions))
    medium, fine = ordered[-2], ordered[-1]
    lookup = {
        (str(row["pattern"]), int(row["resolution_xy"])): row
        for row in rows
    }
    convergence = {}
    fine_predictions = {}
    ratio_by_resolution = {}
    for resolution in ordered:
        control_value = float(
            lookup[("triangular", resolution)][
                "bending_stiffness_N_per_mm"
            ]
        )
        ratio_by_resolution[str(resolution)] = {
            pattern: (
                float(
                    lookup[(pattern, resolution)][
                        "bending_stiffness_N_per_mm"
                    ]
                )
                / control_value
            )
            for pattern in patterns
        }
    control = float(
        lookup[("triangular", fine)]["bending_stiffness_N_per_mm"]
    )
    for pattern in patterns:
        medium_value = lookup[(pattern, medium)][
            "bending_stiffness_N_per_mm"
        ]
        fine_value = lookup[(pattern, fine)][
            "bending_stiffness_N_per_mm"
        ]
        convergence[pattern] = (
            None
            if medium_value is None or fine_value is None
            else abs(float(medium_value) / float(fine_value) - 1.0)
        )
        fine_predictions[pattern] = {
            "bending_stiffness_N_per_mm": fine_value,
            "to_triangular_ratio": (
                None
                if fine_value is None
                else float(fine_value) / control
            ),
        }
    all_solved = all(row["status"] == "ok" for row in rows)
    connected = all(
        row["component_count"] == 1 for row in rows
    ) and all(row["component_count"] == 1 for row in geometries.values())
    mass_gate = all(
        abs(float(row["mass_error_fraction"])) <= 0.001
        for row in rows
    )
    convergence_gate = all(
        value is not None
        and float(value) <= MAXIMUM_TOOLPATH_CONVERGENCE_CHANGE
        for value in convergence.values()
    )
    candidate_gain = all(
        float(fine_predictions[pattern]["to_triangular_ratio"]) > 1.0
        for pattern in patterns
        if pattern != "triangular"
    )
    ratio_convergence = {
        pattern: abs(
            ratio_by_resolution[str(medium)][pattern]
            / ratio_by_resolution[str(fine)][pattern]
            - 1.0
        )
        for pattern in patterns
    }
    ratio_convergence_gate = all(
        value <= MAXIMUM_TOOLPATH_CONVERGENCE_CHANGE
        for value in ratio_convergence.values()
    )
    gain_at_every_resolution = all(
        ratio_by_resolution[str(resolution)][pattern] > 1.0
        for resolution in ordered
        for pattern in patterns
        if pattern != "triangular"
    )
    absolute_gate = all(
        (all_solved, connected, mass_gate, convergence_gate, candidate_gain)
    )
    candidate_set_gate = all(
        (
            all_solved,
            connected,
            mass_gate,
            ratio_convergence_gate,
            gain_at_every_resolution,
        )
    )
    payload = {
        "study_id": "bambu_pla_robust_finalist_frozen_toolpath_v1",
        "material": "Bambu PLA Basic",
        "source_gcode": str(gcode.resolve()),
        "source_gcode_sha256": _sha256(gcode),
        "patterns": list(patterns),
        "resolutions_xy": ordered,
        "fine_raster_resolution": FINE_RASTER_RESOLUTION,
        "solver_case_count": len(rows),
        "fine_geometries": geometries,
        "convergence": convergence,
        "ratio_by_resolution": ratio_by_resolution,
        "last_two_ratio_changes": ratio_convergence,
        "fine_predictions": fine_predictions,
        "gates": {
            "all_solver_cases_passed": all_solved,
            "all_geometries_connected": connected,
            "all_mass_errors_at_most_0p1_percent": mass_gate,
            "all_last_two_resolution_changes_at_most_5_percent": (
                convergence_gate
            ),
            "all_last_two_ratio_changes_at_most_5_percent": (
                ratio_convergence_gate
            ),
            "all_noncontrols_beat_triangular": candidate_gain,
            "all_noncontrols_beat_triangular_at_every_resolution": (
                gain_at_every_resolution
            ),
            "absolute_toolpath_replay_gate_passed": absolute_gate,
            "candidate_set_admitted_for_physical_testing": (
                candidate_set_gate
            ),
        },
        "status": (
            (
                "absolute_and_relative_toolpath_replay_admitted"
                if absolute_gate
                else "relative_candidate_set_admitted_absolute_not_converged"
            )
            if candidate_set_gate
            else "toolpath_candidate_set_gate_failed"
        ),
        "restrictions": [
            "Voxel fill fractions scale stiffness and density linearly.",
            "Beads and layers are perfectly bonded.",
            "Absolute stiffness and ranking still require the matched "
            "physical measurements.",
            "Candidate-set admission means every exotic pattern beats the "
            "control at every resolution and its matched ratio converges; "
            "it does not rank the exotic patterns against one another.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _group_statistics(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1))
    return {
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean,
    }


def evaluate_finalist_measurements(
    plan_path: Path,
    capture_path: Path,
    toolpath_manifest_path: Path,
) -> Dict[str, object]:
    with Path(plan_path).open(encoding="utf-8", newline="") as handle:
        plan = list(csv.DictReader(handle))
    with Path(capture_path).open(encoding="utf-8", newline="") as handle:
        capture = list(csv.DictReader(handle))
    expected = {row["specimen_id"]: row for row in plan}
    observed = {row["specimen_id"]: row for row in capture}
    if len(expected) != len(plan) or len(observed) != len(capture):
        raise ValueError("physical plan/capture contains duplicate IDs")
    if set(expected) != set(observed):
        raise ValueError("physical capture IDs do not match the frozen plan")
    required = (
        "actual_mass_g",
        "ambient_temperature_C",
        "relative_humidity_percent",
        "bending_stiffness_N_per_mm",
    )
    parsed = {}
    missing = []
    for specimen, row in observed.items():
        parsed[specimen] = {}
        for field in required:
            try:
                value = float(str(row.get(field, "")).strip())
            except ValueError:
                value = float("nan")
            if not math.isfinite(value):
                missing.append(f"{specimen}:{field}")
            parsed[specimen][field] = value
    if missing:
        return {
            "status": "awaiting_measurements",
            "expected_specimens": len(expected),
            "missing_numeric_cell_count": len(missing),
            "first_missing_cells": missing[:12],
            "physical_validation_gate_passed": False,
        }
    invalid = []
    for specimen, values in parsed.items():
        if not 0.5 <= values["actual_mass_g"] <= 10.0:
            invalid.append(f"{specimen}: implausible mass")
        if not 10.0 <= values["ambient_temperature_C"] <= 35.0:
            invalid.append(f"{specimen}: implausible temperature")
        if not 0.0 <= values["relative_humidity_percent"] <= 100.0:
            invalid.append(f"{specimen}: invalid humidity")
        if not 1.0 <= values["bending_stiffness_N_per_mm"] <= 1000.0:
            invalid.append(f"{specimen}: implausible stiffness")
    if invalid:
        return {
            "status": "invalid_measurements",
            "validation_errors": invalid,
            "physical_validation_gate_passed": False,
        }
    patterns = tuple(dict.fromkeys(row["pattern"] for row in plan))
    groups = {}
    for pattern in patterns:
        specimens = [
            row["specimen_id"] for row in plan if row["pattern"] == pattern
        ]
        groups[pattern] = {
            field: _group_statistics(
                [parsed[specimen][field] for specimen in specimens]
            )
            for field in ("actual_mass_g", "bending_stiffness_N_per_mm")
        }
    toolpath = json.loads(
        Path(toolpath_manifest_path).read_text(encoding="utf-8")
    )
    if (
        toolpath.get("gates", {}).get(
            "candidate_set_admitted_for_physical_testing"
        )
        is not True
    ):
        raise ValueError("frozen toolpath replay gate is not admitted")
    comparisons = {}
    control_mean = groups["triangular"]["bending_stiffness_N_per_mm"][
        "mean"
    ]
    for pattern in patterns:
        measured_ratio = (
            groups[pattern]["bending_stiffness_N_per_mm"]["mean"]
            / control_mean
        )
        predicted_ratio = float(
            toolpath["fine_predictions"][pattern]["to_triangular_ratio"]
        )
        comparisons[pattern] = {
            "measured_to_triangular_ratio": measured_ratio,
            "predicted_to_triangular_ratio": predicted_ratio,
            "ratio_relative_error": abs(
                measured_ratio / predicted_ratio - 1.0
            ),
        }
    means = [
        groups[pattern]["actual_mass_g"]["mean"] for pattern in patterns
    ]
    mass_spread = max(means) / min(means) - 1.0
    cv_gate = all(
        groups[pattern]["bending_stiffness_N_per_mm"][
            "coefficient_of_variation"
        ]
        <= MAXIMUM_PHYSICAL_GROUP_CV
        for pattern in patterns
    )
    ratio_gate = all(
        comparisons[pattern]["ratio_relative_error"]
        <= MAXIMUM_PHYSICAL_RATIO_ERROR
        for pattern in patterns
    )
    candidate_gain = all(
        comparisons[pattern]["measured_to_triangular_ratio"] > 1.0
        for pattern in patterns
        if pattern != "triangular"
    )
    gate = bool(
        mass_spread <= 0.03 and cv_gate and ratio_gate and candidate_gain
    )
    return {
        "status": "validated" if gate else "physical_gate_failed",
        "groups": groups,
        "comparisons": comparisons,
        "mean_mass_spread": mass_spread,
        "group_cv_gate_passed": cv_gate,
        "prediction_agreement_gate_passed": ratio_gate,
        "all_candidates_beat_triangular": candidate_gain,
        "physical_validation_gate_passed": gate,
    }


def run_campaign(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
) -> Dict[str, object]:
    output_dir, runs_dir = Path(output_dir), Path(runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_perturbation_matrix()
    results = run_benchmark(
        output_dir / "beam_perturbations",
        runs_dir / "beam_perturbations",
        properties_path=PROPERTIES,
        patterns=list(matrix["geometries"]),
        load_cases=LOAD_CASES,
        domain_width_mm=WIDTH_MM,
        domain_height_mm=HEIGHT_MM,
        depth_mm=DEPTH_MM,
        relative_density=RELATIVE_DENSITY,
        tensile_strain=0.005,
        geometries=matrix["geometries"],
        strut_width_overrides=matrix["strut_widths"],
    )
    rows = [asdict(result) for result in results]
    robustness = aggregate_robustness(
        rows, matrix["metadata"], matrix["structural_breaks"]
    )
    _write_csv(output_dir / "robustness_ranking.csv", robustness["ranking"])
    detail_rows = []
    for summary in robustness["pattern_summaries"]:
        for load_case, detail in summary["load_details"].items():
            detail_rows.append(
                {
                    "pattern": summary["pattern"],
                    "load_case": load_case,
                    **detail,
                    "conservative_to_triangular": summary[
                        "conservative_to_triangular"
                    ][load_case],
                }
            )
    _write_csv(output_dir / "load_case_robustness.csv", detail_rows)
    selected = [
        "triangular",
        *[
            str(row["pattern"])
            for row in robustness["ranking"]
            if row["pattern"] != "triangular"
        ],
    ]
    physical = write_finalist_plate(
        output_dir / "physical_validation",
        selected,
        robustness["ranking"],
    )
    toolpath = run_finalist_toolpath_replay(
        output_dir / "toolpath_replay",
        runs_dir / "toolpath_replay",
        physical,
    )
    plan_path = output_dir / "physical_validation/physical_test_plan.csv"
    with plan_path.open(encoding="utf-8", newline="") as handle:
        plan_rows = list(csv.DictReader(handle))
    for row in plan_rows:
        row["toolpath_predicted_bending_ratio"] = toolpath[
            "fine_predictions"
        ][row["pattern"]]["to_triangular_ratio"]
    _write_csv(plan_path, plan_rows)
    evaluation = evaluate_finalist_measurements(
        plan_path,
        output_dir / "physical_validation/measurement_capture.csv",
        output_dir / "toolpath_replay/manifest.json",
    )
    evaluation_dir = output_dir / "physical_validation/evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "study_id": "bambu_pla_pattern_robustness_campaign_v1",
        "material": "Bambu PLA Basic",
        "patterns": list(PATTERNS),
        "load_cases": list(LOAD_CASES),
        "bead_width_scales": list(BEAD_WIDTH_SCALES),
        "solver_variants": len(matrix["geometries"]),
        "solver_cases": len(results),
        "structural_break_variants": len(matrix["structural_breaks"]),
        **robustness,
        "physical_validation": physical,
        "toolpath_replay": toolpath,
        "measurement_evaluation": evaluation,
        "gates": {
            "all_nominal_cases_solved": all(
                next(
                    row for row in rows
                    if row["pattern"]
                    == f"{SHORT_NAMES[base]}_o{angle}_nom"
                    and row["load_case"] == load_case
                )["status"]
                == "ok"
                for base in PATTERNS
                for angle in (0, 90)
                for load_case in LOAD_CASES
            ),
            "all_patterns_have_single_strut_envelope": all(
                any(
                    row["base_pattern"] == base
                    and row["scenario"]
                    == "single_missing_interior_strut"
                    for row in matrix["metadata"].values()
                )
                for base in PATTERNS
            ),
            "physical_manufacturing_gate_passed": physical[
                "manufacturing_gate_passed"
            ],
            "frozen_toolpath_candidate_set_gate_passed": toolpath["gates"][
                "candidate_set_admitted_for_physical_testing"
            ],
            "absolute_frozen_toolpath_convergence_gate_passed": toolpath[
                "gates"
            ][
                "absolute_toolpath_replay_gate_passed"
            ],
            "absolute_performance_validated": False,
        },
        "restrictions": [
            "B31 perturbations are screening models; the nominal finalists "
            "retain their existing C3D6 and frozen-toolpath evidence.",
            "A missing-strut case removes one connected centerline segment "
            "between graph junctions; it is not a stochastic printer model.",
            "Bead-width scales are structured sensitivity bounds, not a "
            "measured extrusion distribution.",
            "Absolute performance still requires the generated matched "
            "Bambu PLA Basic physical coupons.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Printable-pattern robustness tournament",
        "",
        (
            f"{len(matrix['geometries'])} geometry/process variants were "
            f"evaluated in {len(results)} CalculiX beam cases."
        ),
        "",
        "The conservative envelope takes the lower result across 0°/90° "
        "placement, 10% bead-width loss, and the 10th percentile of every "
        "single missing interior strut. It then compares each pattern with "
        "triangular across tension, shear, bending, compression, and buckling.",
        "",
        "| Rank | Pattern | Robust balanced | Worst load | "
        "Intended-orientation bending | Conservative bending |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(robustness["ranking"], start=1):
        report.append(
            f"| {rank} | {row['pattern']} | "
            f"{float(row['robust_balanced_ratio']):.3f} | "
            f"{float(row['robust_worst_load_ratio']):.3f} | "
            f"{float(row['nominal_orientation_0_bending_ratio']):.3f} | "
            f"{float(row['conservative_bending_ratio']):.3f} |"
        )
    report.extend(
        [
            "",
            "## Physical handoff",
            "",
            "A new single-plate showdown contains three matched copies of "
            + ", ".join(f"`{pattern}`" for pattern in selected)
            + ". This avoids comparing winners printed in separate slicer "
            "jobs.",
            "",
            f"Manufacturing gate: **{physical['manufacturing_gate_passed']}**; "
            f"predicted mass spread: "
            f"{physical['maximum_pattern_mean_mass_spread']:.2%}.",
            "",
            "The exact combined-plate G-code was replayed at "
            + ", ".join(
                f"{resolution}×{resolution}×20"
                for resolution in toolpath["resolutions_xy"]
            )
            + " voxel resolution. Intended-orientation fine ratios are:",
            "",
            "| Pattern | Frozen-toolpath bending / triangular |",
            "|---|---:|",
            *[
                f"| {pattern} | "
                f"{float(toolpath['fine_predictions'][pattern]['to_triangular_ratio']):.3f} |"
                for pattern in selected
            ],
            "",
            f"Candidate-set toolpath gate: **"
            f"{toolpath['gates']['candidate_set_admitted_for_physical_testing']}**. "
            f"Absolute-stiffness convergence gate: **"
            f"{toolpath['gates']['absolute_toolpath_replay_gate_passed']}**. "
            "The blank measurement sheet is correctly held at "
            f"`{evaluation['status']}`.",
            "",
            "This ranking selects robust virtual candidates; it does not "
            "replace the registered physical bending measurements.",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Challenge printable infill finalists against placement, "
            "bead-width, and missing-strut perturbations"
        )
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--capture")
    args = parser.parse_args()
    if args.evaluate_only:
        output = Path(args.output)
        capture = (
            Path(args.capture)
            if args.capture
            else output
            / "physical_validation/measurement_capture.csv"
        )
        evaluation = evaluate_finalist_measurements(
            output / "physical_validation/physical_test_plan.csv",
            capture,
            output / "toolpath_replay/manifest.json",
        )
        evaluation_dir = output / "physical_validation/evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        (evaluation_dir / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{evaluation['status']}; physical gate="
            f"{evaluation['physical_validation_gate_passed']}"
        )
        return
    payload = run_campaign(Path(args.output), Path(args.runs_dir))
    print(
        f"{payload['solver_cases']} solver cases; "
        f"balanced={payload['balanced_specialist']}; "
        f"bending={payload['bending_specialist']}; "
        f"plate gate="
        f"{payload['physical_validation']['manufacturing_gate_passed']}"
    )


if __name__ == "__main__":
    main()
