from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Mapping, Sequence
from xml.etree import ElementTree

import numpy as np

from .bambu_tds_profile import DEFAULT_TRANSVERSE_PROFILE
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
from .infill_benchmark import (
    LOAD_CASES,
    PatternGeometry,
    make_geometry_from_segments,
    make_pattern,
    run_benchmark,
)
from .solid_infill import (
    equal_solid_mass_strut_width,
    generate_prism_mesh,
)
from .solid_multiload import (
    _solid_preamble,
    run_solid_multiload,
    select_boundary_sets,
)
from .viscoelastic_pattern_modal_screen import (
    _run_modal,
    build_storage_profile,
    select_out_of_plane_mode,
)
from .volumetric_infill import inspect_binary_stl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/custom_infill/weird_kite_search"
)
DEFAULT_RUNS = PROJECT_ROOT / "runs/custom_infill/weird_kite_search"
DOMAIN_WIDTH_MM = 40.0
DOMAIN_HEIGHT_MM = 40.0
DEPTH_MM = 4.0
RELATIVE_DENSITY = 0.35
BEAM_APEX_VALUES_MM = (8.0, 12.0, 16.0)
BASELINE_APEXES_MM = (11.0, 15.0, 13.0, 16.0)
SOLID_MESH_SIZE_MM = 0.9
REFINED_MESH_SIZE_MM = 0.7
MODE_COUNT = 12
PROMOTE_COUNT = 6
PHYSICAL_CANDIDATE_COUNT = 2

STATIC_WEIGHTS = {
    "tension": 0.15,
    "shear": 0.15,
    "bending": 0.35,
    "compression": 0.15,
    "buckling": 0.20,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _point(x: float, y: float) -> tuple[float, float]:
    return (float(x), float(y))


def make_kite_geometry(
    apexes_mm: Sequence[float],
    name: str | None = None,
) -> PatternGeometry:
    if len(apexes_mm) != 4:
        raise ValueError("kite lattice requires four row apex positions")
    apexes = tuple(float(value) for value in apexes_mm)
    if any(not 5.0 <= value <= 17.0 for value in apexes):
        raise ValueError("kite apex positions must lie in [5, 17] mm")
    if name is None:
        name = "kite_" + "_".join(f"{int(value):02d}" for value in apexes)
    segments = []
    for x in (0.0, 20.0, 40.0):
        for row in range(4):
            segments.append(
                (
                    _point(x, row * 10.0),
                    _point(x, (row + 1) * 10.0),
                )
            )
    for y in (0.0, 40.0):
        segments.extend(
            [
                (_point(0.0, y), _point(20.0, y)),
                (_point(20.0, y), _point(40.0, y)),
            ]
        )
    for row, apex_x in enumerate(apexes):
        lower = row * 10.0
        upper = (row + 1) * 10.0
        middle = lower + 5.0
        left_apex = _point(apex_x, middle)
        right_apex = _point(40.0 - apex_x, middle)
        for corner in (
            _point(0.0, lower),
            _point(0.0, upper),
            _point(20.0, lower),
            _point(20.0, upper),
        ):
            segments.append((corner, left_apex))
        for corner in (
            _point(20.0, lower),
            _point(20.0, upper),
            _point(40.0, lower),
            _point(40.0, upper),
        ):
            segments.append((corner, right_apex))
    return make_geometry_from_segments(name, segments, DOMAIN_WIDTH_MM)


def build_candidate_geometries() -> Dict[str, PatternGeometry]:
    geometries: Dict[str, PatternGeometry] = {
        "triangular": make_pattern("triangular"),
    }
    for apexes in itertools.product(BEAM_APEX_VALUES_MM, repeat=4):
        geometry = make_kite_geometry(apexes)
        geometries[geometry.name] = geometry
    baseline = make_kite_geometry(
        BASELINE_APEXES_MM, name="graded_hourglass"
    )
    geometries[baseline.name] = baseline
    return geometries


def geometry_to_idea(
    geometry: PatternGeometry,
    apexes_mm: Sequence[float],
) -> Dict[str, object]:
    segments = [
        [
            list(geometry.nodes[first - 1]),
            list(geometry.nodes[second - 1]),
        ]
        for first, second in geometry.edges
    ]
    return {
        "schema_version": 1,
        "name": geometry.name,
        "description": (
            "Equal-mass mirrored four-row kite lattice with left-half apex "
            f"positions {list(apexes_mm)} mm."
        ),
        "units": "mm",
        "domain": {
            "width_mm": DOMAIN_WIDTH_MM,
            "height_mm": DOMAIN_HEIGHT_MM,
            "depth_mm": DEPTH_MM,
        },
        "relative_density": RELATIVE_DENSITY,
        "minimum_printable_strut_width_mm": 0.8,
        "minimum_segment_length_mm": 0.5,
        "design_parameters": {
            "left_half_row_apex_x_mm": list(apexes_mm),
            "right_half_mirror_rule": "x_right = 40 - x_left",
        },
        "segments": segments,
    }


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric(row: Mapping[str, str]) -> float | None:
    key = (
        "critical_buckling_load_N"
        if row["load_case"] == "buckling"
        else "stiffness_N_per_mm"
    )
    value = row.get(key, "")
    return None if value in {"", None} else float(value)


def _normalized_static_rows(
    summary_rows: Sequence[Mapping[str, str]],
    patterns: Sequence[str],
) -> list[Dict[str, object]]:
    lookup = {
        (row["pattern"], row["load_case"]): _metric(row)
        for row in summary_rows
        if row.get("status") == "ok"
    }
    maxima = {}
    for load_case in LOAD_CASES:
        values = [
            value
            for (pattern, case), value in lookup.items()
            if case == load_case
            and pattern in patterns
            and value is not None
            and value > 0.0
        ]
        if not values:
            raise ValueError(f"no valid {load_case} metrics")
        maxima[load_case] = max(values)
    rows = []
    for pattern in patterns:
        normalized = {
            load_case: (
                None
                if lookup.get((pattern, load_case)) is None
                else float(lookup[(pattern, load_case)])
                / maxima[load_case]
            )
            for load_case in LOAD_CASES
        }
        valid = all(value is not None and value > 0.0 for value in normalized.values())
        balanced = (
            None
            if not valid
            else math.exp(
                sum(math.log(float(value)) for value in normalized.values())
                / len(normalized)
            )
        )
        weighted = (
            None
            if not valid
            else math.exp(
                sum(
                    STATIC_WEIGHTS[case] * math.log(float(normalized[case]))
                    for case in LOAD_CASES
                )
            )
        )
        rows.append(
            {
                "pattern": pattern,
                **{
                    f"{case}_normalized": normalized[case]
                    for case in LOAD_CASES
                },
                "balanced_geometric_mean": balanced,
                "weighted_static_score": weighted,
                "worst_case_normalized": (
                    None if not valid else min(normalized.values())
                ),
            }
        )
    return rows


def _pareto_patterns(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> set[str]:
    front = set()
    for candidate in rows:
        values = [candidate.get(field) for field in fields]
        if any(value is None for value in values):
            continue
        dominated = False
        for other in rows:
            if other["pattern"] == candidate["pattern"]:
                continue
            other_values = [other.get(field) for field in fields]
            if any(value is None for value in other_values):
                continue
            if all(
                float(other_value) >= float(value)
                for other_value, value in zip(other_values, values)
            ) and any(
                float(other_value) > float(value)
                for other_value, value in zip(other_values, values)
            ):
                dominated = True
                break
        if not dominated:
            front.add(str(candidate["pattern"]))
    return front


def select_beam_promotions(
    rows: Sequence[Mapping[str, object]],
    count: int = PROMOTE_COUNT,
) -> list[str]:
    candidates = [
        row
        for row in rows
        if row["pattern"].startswith("kite_")
        or row["pattern"] == "graded_hourglass"
    ]
    if len(candidates) < count:
        raise ValueError("not enough valid kite candidates to promote")
    selections: list[str] = []

    def add(row: Mapping[str, object]) -> None:
        pattern = str(row["pattern"])
        if pattern not in selections:
            selections.append(pattern)

    orderings = (
        ("weighted_static_score",),
        ("balanced_geometric_mean",),
        ("worst_case_normalized",),
        ("bending_normalized", "weighted_static_score"),
        ("buckling_normalized", "bending_normalized"),
    )
    for fields in orderings:
        ranked = sorted(
            candidates,
            key=lambda row: tuple(
                float(row.get(field) or -1.0) for field in fields
            ),
            reverse=True,
        )
        for row in ranked:
            if (
                fields[0] != "bending_normalized"
                or float(row.get("worst_case_normalized") or 0.0) >= 0.35
            ):
                add(row)
                break
    baseline = next(
        row for row in candidates if row["pattern"] == "graded_hourglass"
    )
    add(baseline)
    for row in sorted(
        candidates,
        key=lambda row: float(row.get("weighted_static_score") or -1.0),
        reverse=True,
    ):
        add(row)
        if len(selections) >= count:
            break
    return selections[:count]


def render_solid_modal_deck(
    mesh,
    storage_profile: Mapping[str, object],
    mesh_size_mm: float,
    mode_count: int = MODE_COUNT,
) -> str:
    constants = storage_profile["screening_engineering_constants"]
    sets = select_boundary_sets(
        mesh,
        "bending",
        DOMAIN_WIDTH_MM,
        DOMAIN_HEIGHT_MM,
        DEPTH_MM,
        mesh_size_mm,
    )
    preamble = _solid_preamble(
        mesh,
        sets,
        "bending",
        float(constants["E1"]),
        float(constants["nu12"]),
        float(storage_profile["density_tonnes_mm3"]),
        DOMAIN_WIDTH_MM,
        DOMAIN_HEIGHT_MM,
        engineering_constants=constants,
    )
    return (
        preamble
        + "*STEP\n*FREQUENCY\n"
        + f"{mode_count}\n"
        + "*BOUNDARY\n"
        + "SUPPORTS,3,3,0.\n"
        + "ANCHOR,1,2,0.\n"
        + "ANCHOR_TWO,2,2,0.\n"
        + "*NODE FILE\nU\n*END STEP\n"
    )


def run_continuum_modal_batch(
    output_dir: Path,
    runs_dir: Path,
    geometries: Mapping[str, PatternGeometry],
    strut_widths: Mapping[str, float],
    mesh_size_mm: float,
) -> list[Dict[str, object]]:
    output_dir = Path(output_dir)
    for directory in ("meshes", "decks"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    profile = build_storage_profile()
    rows = []
    for pattern, geometry in geometries.items():
        mesh_path = output_dir / "meshes" / f"{pattern}.msh"
        mesh = generate_prism_mesh(
            geometry,
            float(strut_widths[pattern]),
            mesh_size_mm,
            mesh_path,
            DOMAIN_WIDTH_MM,
            DOMAIN_HEIGHT_MM,
            DEPTH_MM,
        )
        deck = output_dir / "decks" / f"{pattern}.inp"
        deck.write_text(
            render_solid_modal_deck(mesh, profile, mesh_size_mm),
            encoding="utf-8",
        )
        solver = _run_modal(
            deck,
            runs_dir,
            f"weird-modal-{pattern}-h{mesh_size_mm:g}",
        )
        selected = (
            None
            if solver["status"] != "ok"
            else select_out_of_plane_mode(solver["modes"])
        )
        rows.append(
            {
                "pattern": pattern,
                "mesh_size_mm": mesh_size_mm,
                "nodes": len(mesh.node_tags),
                "elements": len(mesh.element_tags),
                "actual_volume_mm3": mesh.actual_volume_mm3,
                "actual_mass_g": (
                    mesh.actual_volume_mm3
                    * float(profile["density_tonnes_mm3"])
                    * 1.0e6
                ),
                "status": solver["status"],
                "selected_mode": (
                    None if selected is None else int(selected["mode"])
                ),
                "out_of_plane_frequency_Hz": (
                    None if selected is None else selected["frequency_Hz"]
                ),
                "z_mass_fraction_within_extracted_modes": (
                    None
                    if selected is None
                    else selected[
                        "z_mass_fraction_within_extracted_modes"
                    ]
                ),
                "loss_factor": profile["loss_factor"],
                "run_directory": solver["run_directory"],
                "mesh_file": str(mesh_path.resolve()),
                "deck_file": str(deck.resolve()),
                "error": solver.get("error"),
            }
        )
    _write_csv(output_dir / "summary.csv", rows)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _strut_widths(summary_rows: Sequence[Mapping[str, str]]) -> Dict[str, float]:
    widths: Dict[str, set[float]] = {}
    for row in summary_rows:
        widths.setdefault(row["pattern"], set()).add(
            float(row["strut_width_mm"])
        )
    result = {}
    for pattern, values in widths.items():
        if len(values) != 1:
            raise ValueError(f"non-constant strut width for {pattern}")
        result[pattern] = values.pop()
    return result


def combine_solid_and_modal_scores(
    solid_summary: Sequence[Mapping[str, str]],
    modal_rows: Sequence[Mapping[str, object]],
    patterns: Sequence[str],
) -> list[Dict[str, object]]:
    static_rows = _normalized_static_rows(solid_summary, patterns)
    static_lookup = {row["pattern"]: row for row in static_rows}
    modal_lookup = {row["pattern"]: row for row in modal_rows}
    maximum_frequency = max(
        float(row["out_of_plane_frequency_Hz"])
        for row in modal_rows
        if row["status"] == "ok"
    )
    utilization = {}
    for pattern in patterns:
        values = [
            float(row["directional_normal_utilization_p99_mean"])
            for row in solid_summary
            if row["pattern"] == pattern
            and row["load_case"] != "buckling"
            and row.get("directional_normal_utilization_p99_mean")
            not in {"", None}
        ]
        utilization[pattern] = max(values) if values else float("inf")
    inverse_safety = {
        pattern: (
            0.0 if not math.isfinite(value) else 1.0 / (1.0 + value)
        )
        for pattern, value in utilization.items()
    }
    maximum_safety = max(inverse_safety.values())
    rows = []
    for pattern in patterns:
        static = static_lookup[pattern]
        modal = modal_lookup[pattern]
        frequency = (
            None
            if modal["status"] != "ok"
            else float(modal["out_of_plane_frequency_Hz"])
        )
        modal_normalized = (
            None if frequency is None else frequency / maximum_frequency
        )
        safety_normalized = (
            inverse_safety[pattern] / maximum_safety
            if maximum_safety > 0.0
            else None
        )
        static_score = static["weighted_static_score"]
        composite = (
            None
            if static_score is None
            or modal_normalized is None
            or safety_normalized is None
            or safety_normalized <= 0.0
            else float(static_score) ** 0.70
            * modal_normalized**0.20
            * safety_normalized**0.10
        )
        rows.append(
            {
                **static,
                "out_of_plane_frequency_Hz": frequency,
                "modal_frequency_normalized": modal_normalized,
                "maximum_p99_directional_utilization": utilization[pattern],
                "stress_safety_normalized": safety_normalized,
                "composite_score": composite,
            }
        )
    pareto = _pareto_patterns(
        rows,
        (
            "weighted_static_score",
            "bending_normalized",
            "modal_frequency_normalized",
            "stress_safety_normalized",
        ),
    )
    for row in rows:
        row["pareto_nondominated"] = row["pattern"] in pareto
    return sorted(
        rows,
        key=lambda row: float(row["composite_score"] or -1.0),
        reverse=True,
    )


def export_refined_stls(
    output_dir: Path,
    modal_rows: Sequence[Mapping[str, object]],
    selected_patterns: Sequence[str],
) -> list[Dict[str, object]]:
    try:
        import gmsh
    except ImportError as error:
        raise RuntimeError("Gmsh is required for STL export") from error
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = {str(row["pattern"]): row for row in modal_rows}
    rows = []
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Binary", 1)
        for pattern in selected_patterns:
            source = Path(str(lookup[pattern]["mesh_file"]))
            gmsh.clear()
            gmsh.open(str(source))
            target = output_dir / f"{pattern}.stl"
            gmsh.write(str(target))
            metrics = inspect_binary_stl(target)
            rows.append(
                {
                    "pattern": pattern,
                    "source_mesh": str(source.resolve()),
                    "source_mesh_sha256": _sha256(source),
                    "stl": str(target.resolve()),
                    "stl_sha256": _sha256(target),
                    **metrics,
                    "export_gate_passed": (
                        metrics["watertight"] is True
                        and int(metrics["nonmanifold_edge_count"]) == 0
                        and math.isclose(
                            float(metrics["volume_mm3"]),
                            RELATIVE_DENSITY
                            * DOMAIN_WIDTH_MM
                            * DOMAIN_HEIGHT_MM
                            * DEPTH_MM,
                            rel_tol=0.003,
                        )
                    ),
                }
            )
    finally:
        gmsh.finalize()
    _write_csv(output_dir / "manifest.csv", rows)
    if not all(row["export_gate_passed"] for row in rows):
        raise RuntimeError("one or more refined STL exports failed")
    return rows


def _three_mf_build_count(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("3D/3dmodel.model"))
    namespace = {
        "m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    }
    return len(root.findall(".//m:build/m:item", namespace))


def write_physical_plate(
    output_dir: Path,
    stl_rows: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, float]],
    replicates: int = 3,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orca = find_orca_slicer()
    stls = [Path(str(row["stl"])) for row in stl_rows]
    plate = output_dir / "bambu-pla-weird-kite-validation.3mf"
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
        raise RuntimeError("OrcaSlicer did not produce plate_1.gcode")
    extrusion = parse_object_extrusion(gcode)
    pattern_rows = {}
    for row in stl_rows:
        pattern = str(row["pattern"])
        matches = [
            values
            for name, values in extrusion.items()
            if name.startswith(f"{pattern}.stl")
        ]
        if len(matches) != replicates:
            raise ValueError(
                f"expected {replicates} sliced {pattern} objects, "
                f"found {len(matches)}"
            )
        pattern_rows[pattern] = matches
    means = {
        pattern: float(
            np.mean([row["predicted_deposited_mass_g"] for row in rows])
        )
        for pattern, rows in pattern_rows.items()
    }
    mass_spread = max(means.values()) / min(means.values()) - 1.0
    config = _config_from_gcode(gcode)
    build_count = _three_mf_build_count(plate)
    gate = (
        build_count == replicates * len(stls)
        and _expected_config_gate(config)
        and mass_spread <= 0.03
        and all(
            max(row["predicted_deposited_mass_g"] for row in rows)
            - min(row["predicted_deposited_mass_g"] for row in rows)
            <= 1.0e-9
            for rows in pattern_rows.values()
        )
    )
    plan_rows = []
    capture_rows = []
    order = 0
    for replicate in range(1, replicates + 1):
        for pattern in sorted(pattern_rows):
            order += 1
            predicted = predictions[pattern]
            mass = pattern_rows[pattern][replicate - 1][
                "predicted_deposited_mass_g"
            ]
            specimen = f"WK-{order:02d}-{pattern}-R{replicate}"
            plan_rows.append(
                {
                    "test_order": order,
                    "specimen_id": specimen,
                    "pattern": pattern,
                    "replicate": replicate,
                    "predicted_deposited_mass_g": mass,
                    "predicted_bending_stiffness_N_per_mm": predicted[
                        "bending_stiffness_N_per_mm"
                    ],
                    "predicted_out_of_plane_frequency_Hz": predicted[
                        "out_of_plane_frequency_Hz"
                    ],
                    "first_test": "supported_tap_or_swept_sine_modal_test",
                    "second_test": "three_point_bending_to_0p4mm",
                    "material": "Bambu PLA Basic",
                }
            )
            capture_rows.append(
                {
                    "specimen_id": specimen,
                    "actual_mass_g": "",
                    "ambient_temperature_C": "",
                    "relative_humidity_percent": "",
                    "measured_out_of_plane_frequency_Hz": "",
                    "measured_half_power_loss_factor": "",
                    "bending_stiffness_N_per_mm": "",
                    "failure_or_damage_observed": "",
                    "notes": "",
                }
            )
    _write_csv(output_dir / "physical_test_plan.csv", plan_rows)
    _write_csv(output_dir / "measurement_capture.csv", capture_rows)
    manifest = {
        "package_id": "bambu_pla_weird_kite_validation_v1",
        "material": "Bambu PLA Basic",
        "plate_3mf": str(plate.resolve()),
        "plate_3mf_sha256": _sha256(plate),
        "gcode": str(gcode.resolve()),
        "gcode_sha256": _sha256(gcode),
        "pattern_count": len(stls),
        "replicates_per_pattern": replicates,
        "build_item_count": build_count,
        "predicted_deposited_mass_g": means,
        "maximum_pattern_mean_mass_spread": mass_spread,
        "slicer_config_gate_passed": _expected_config_gate(config),
        "manufacturing_gate_passed": gate,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not gate:
        raise RuntimeError("weird-kite physical plate gate failed")
    return manifest


def run_search(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    properties_path: Path = DEFAULT_TRANSVERSE_PROFILE,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometries = build_candidate_geometries()
    patterns = list(geometries)
    run_benchmark(
        output_dir / "beam",
        runs_dir / "beam",
        properties_path=properties_path,
        patterns=patterns,
        load_cases=LOAD_CASES,
        geometries=geometries,
        tensile_strain=0.005,
    )
    beam_summary = _read_csv(output_dir / "beam/summary.csv")
    beam_scores = _normalized_static_rows(beam_summary, patterns)
    beam_pareto = _pareto_patterns(
        beam_scores,
        tuple(f"{case}_normalized" for case in LOAD_CASES),
    )
    for row in beam_scores:
        row["pareto_nondominated"] = row["pattern"] in beam_pareto
    beam_scores.sort(
        key=lambda row: float(row["weighted_static_score"] or -1.0),
        reverse=True,
    )
    _write_csv(output_dir / "beam_scores.csv", beam_scores)
    promoted = select_beam_promotions(beam_scores)
    solid_patterns = list(dict.fromkeys(["triangular", *promoted]))
    solid_geometries = {
        pattern: geometries[pattern] for pattern in solid_patterns
    }
    run_solid_multiload(
        output_dir / "solid",
        runs_dir / "solid",
        properties_path=properties_path,
        patterns=solid_patterns,
        load_cases=LOAD_CASES,
        mesh_size_mm=SOLID_MESH_SIZE_MM,
        relative_density=RELATIVE_DENSITY,
        deformation=0.005,
        geometries=solid_geometries,
    )
    solid_summary = _read_csv(output_dir / "solid/summary.csv")
    widths = _strut_widths(solid_summary)
    modal_rows = run_continuum_modal_batch(
        output_dir / "modal",
        runs_dir / "modal",
        solid_geometries,
        widths,
        SOLID_MESH_SIZE_MM,
    )
    combined = combine_solid_and_modal_scores(
        solid_summary, modal_rows, solid_patterns
    )
    _write_csv(output_dir / "combined_ranking.csv", combined)
    physical_candidates = [
        str(row["pattern"])
        for row in combined
        if row["pattern"] != "triangular"
    ][:PHYSICAL_CANDIDATE_COUNT]
    refined_patterns = ["triangular", *physical_candidates]
    refined_geometries = {
        pattern: geometries[pattern] for pattern in refined_patterns
    }
    run_solid_multiload(
        output_dir / "refined_bending",
        runs_dir / "refined_bending",
        properties_path=properties_path,
        patterns=refined_patterns,
        load_cases=("bending",),
        mesh_size_mm=REFINED_MESH_SIZE_MM,
        relative_density=RELATIVE_DENSITY,
        deformation=0.005,
        geometries=refined_geometries,
    )
    refined_static = _read_csv(
        output_dir / "refined_bending/summary.csv"
    )
    refined_widths = _strut_widths(refined_static)
    refined_modal = run_continuum_modal_batch(
        output_dir / "refined_modal",
        runs_dir / "refined_modal",
        refined_geometries,
        refined_widths,
        REFINED_MESH_SIZE_MM,
    )
    coarse_bending = {
        row["pattern"]: float(row["stiffness_N_per_mm"])
        for row in solid_summary
        if row["load_case"] == "bending" and row["status"] == "ok"
    }
    fine_bending = {
        row["pattern"]: float(row["stiffness_N_per_mm"])
        for row in refined_static
        if row["status"] == "ok"
    }
    coarse_modal = {
        str(row["pattern"]): float(row["out_of_plane_frequency_Hz"])
        for row in modal_rows
        if row["status"] == "ok"
    }
    fine_modal = {
        str(row["pattern"]): float(row["out_of_plane_frequency_Hz"])
        for row in refined_modal
        if row["status"] == "ok"
    }
    convergence = {
        pattern: {
            "bending_change_0p9_to_0p7": abs(
                coarse_bending[pattern] / fine_bending[pattern] - 1.0
            ),
            "modal_change_0p9_to_0p7": abs(
                coarse_modal[pattern] / fine_modal[pattern] - 1.0
            ),
        }
        for pattern in refined_patterns
    }
    idea_dir = output_dir / "promoted_ideas"
    idea_dir.mkdir(parents=True, exist_ok=True)
    for pattern in physical_candidates:
        apexes = tuple(float(value) for value in pattern.split("_")[1:])
        payload = geometry_to_idea(geometries[pattern], apexes)
        (idea_dir / f"{pattern}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    selected_for_print = ["triangular", *physical_candidates]
    stl_rows = export_refined_stls(
        output_dir / "physical_validation/stl",
        refined_modal,
        selected_for_print,
    )
    fine_static_lookup = {
        row["pattern"]: row for row in refined_static
    }
    fine_modal_lookup = {
        str(row["pattern"]): row for row in refined_modal
    }
    predictions = {
        pattern: {
            "bending_stiffness_N_per_mm": float(
                fine_static_lookup[pattern]["stiffness_N_per_mm"]
            ),
            "out_of_plane_frequency_Hz": float(
                fine_modal_lookup[pattern]["out_of_plane_frequency_Hz"]
            ),
        }
        for pattern in selected_for_print
    }
    physical = write_physical_plate(
        output_dir / "physical_validation/manufacturing",
        stl_rows,
        predictions,
    )
    gate = (
        all(
            value["bending_change_0p9_to_0p7"] <= 0.10
            and value["modal_change_0p9_to_0p7"] <= 0.10
            for value in convergence.values()
        )
        and physical["manufacturing_gate_passed"] is True
        and len(physical_candidates) == PHYSICAL_CANDIDATE_COUNT
    )
    payload: Dict[str, object] = {
        "study_id": "bambu_pla_weird_kite_multiobjective_search_v1",
        "material": "Bambu PLA Basic",
        "candidate_count": len(patterns) - 1,
        "beam_solver_case_count": len(patterns) * len(LOAD_CASES),
        "beam_pareto_count": len(beam_pareto),
        "promoted_patterns": promoted,
        "solid_pattern_count": len(solid_patterns),
        "physical_candidates": physical_candidates,
        "control_pattern": "triangular",
        "objective": {
            "static_weights": STATIC_WEIGHTS,
            "combined_weights": {
                "weighted_static_score": 0.70,
                "modal_frequency": 0.20,
                "stress_safety": 0.10,
            },
        },
        "refined_predictions": predictions,
        "mesh_convergence": convergence,
        "manufacturing_manifest": str(
            (
                output_dir
                / "physical_validation/manufacturing/manifest.json"
            ).resolve()
        ),
        "gates": {
            "all_refined_bending_and_modal_changes_at_most_10_percent": (
                all(
                    value["bending_change_0p9_to_0p7"] <= 0.10
                    and value["modal_change_0p9_to_0p7"] <= 0.10
                    for value in convergence.values()
                )
            ),
            "slicer_manufacturing_gate_passed": physical[
                "manufacturing_gate_passed"
            ],
            "candidate_search_admitted_for_physical_testing": gate,
            "absolute_performance_validated": False,
            "failure_or_lifetime_validated": False,
        },
        "restrictions": [
            "Beam screening only promotes candidates; decisions use unioned "
            "C3D6 solid and modal refinements.",
            "Continuum candidates are sliced after selection; their frozen "
            "toolpaths are not yet replayed as voxel FEA.",
            "The broadband card is a small-strain 18 C storage proxy.",
            "Physical tests are required for absolute bending, resonance, "
            "damage, fatigue, creep, and print-process bias.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    ranking_lookup = {str(row["pattern"]): row for row in combined}
    report = [
        "# Weird kite-lattice multiobjective search",
        "",
        (
            f"{len(patterns) - 1} mirrored four-row kite variants were "
            f"screened in {len(patterns) * len(LOAD_CASES)} beam cases. "
            f"{len(promoted)} candidates were promoted to equal-mass C3D6 "
            "static and exact-brand modal analysis."
        ),
        "",
        "## Physical-test promotion",
        "",
        "| Pattern | Composite | Fine bending (N/mm) | Fine mode (Hz) | "
        "Bending mesh change | Modal mesh change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pattern in selected_for_print:
        row = ranking_lookup[pattern]
        report.append(
            f"| {pattern} | {float(row['composite_score']):.4f} | "
            f"{predictions[pattern]['bending_stiffness_N_per_mm']:.3f} | "
            f"{predictions[pattern]['out_of_plane_frequency_Hz']:.3f} | "
            f"{convergence[pattern]['bending_change_0p9_to_0p7']:.2%} | "
            f"{convergence[pattern]['modal_change_0p9_to_0p7']:.2%} |"
        )
    report.extend(
        [
            "",
            f"Decision: **{'admitted for matched physical testing' if gate else 'not admitted'}**.",
            "",
            "The 3MF contains three copies of triangular and each promoted "
            "candidate. Test modal response first, then reuse each coupon for "
            "the small-displacement bending comparison. Absolute frequencies "
            "and stiffnesses remain predictions until those measurements are "
            "ingested.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search printable weird kite infills for Bambu PLA Basic"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()
    result = run_search(args.output, args.runs)
    print(
        f"weird infill search: "
        f"{result['gates']['candidate_search_admitted_for_physical_testing']}; "
        f"candidates={result['physical_candidates']}; "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
