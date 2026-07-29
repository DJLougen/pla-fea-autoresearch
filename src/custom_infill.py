from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .bambu_tds_profile import DEFAULT_TRANSVERSE_PROFILE
from .bambu_digital_twin import TwinRequest, select_model
from .infill_benchmark import (
    LOAD_CASES,
    PatternGeometry,
    PatternResult,
    make_geometry_from_segments,
    make_pattern,
    nominal_strut_width,
    run_benchmark,
)
from .multiaxial_failure_sensitivity import write_audit as write_failure_audit
from .solid_multiload import SolidLoadResult, run_solid_multiload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/custom_infill"
DEFAULT_RUNS = PROJECT_ROOT / "runs/custom_infill"
DEFAULT_TWIN_MANIFEST = (
    PROJECT_ROOT / "results/bambu_pla_digital_twin/manifest.json"
)
DEFAULT_TWIN_CARDS = (
    PROJECT_ROOT / "results/bambu_pla_digital_twin/canonical_cards.csv"
)


@dataclass(frozen=True)
class CustomPatternDefinition:
    name: str
    width_mm: float
    height_mm: float
    depth_mm: float
    relative_density: float
    minimum_printable_strut_width_mm: float
    minimum_segment_length_mm: float
    description: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def admit_pattern_study(
    pattern_name: str,
    properties_path: Path = DEFAULT_TRANSVERSE_PROFILE,
    *,
    temperature_C: float = 23.0,
    relative_humidity_percent: float = 20.0,
    conditioning: str = "as_printed_dry",
    twin_manifest_path: Path = DEFAULT_TWIN_MANIFEST,
    twin_cards_path: Path = DEFAULT_TWIN_CARDS,
) -> Dict[str, object]:
    """Bind a custom-pattern run to the canonical exact-brand twin."""
    properties_path = Path(properties_path)
    twin_manifest_path = Path(twin_manifest_path)
    twin_cards_path = Path(twin_cards_path)
    if not twin_manifest_path.exists() or not twin_cards_path.exists():
        raise ValueError(
            "canonical Bambu PLA digital twin is missing; run "
            "`python -m src.bambu_digital_twin` first"
        )
    manifest = json.loads(
        twin_manifest_path.read_text(encoding="utf-8")
    )
    if (
        manifest.get("twin_id")
        != "bambu_pla_basic_evidence_twin_v1"
        or manifest.get("selection_gate_passed") is not True
    ):
        raise ValueError("canonical Bambu PLA digital-twin gate is not valid")
    canonical_profile_sha = _sha256(DEFAULT_TRANSVERSE_PROFILE)
    supplied_profile_sha = _sha256(properties_path)
    if supplied_profile_sha != canonical_profile_sha:
        raise ValueError(
            "properties file does not match the canonical Bambu PLA "
            "directional screening profile"
        )
    request = TwinRequest(
        request_id=f"custom_{pattern_name}_static_elastic",
        analysis_kind="static_elastic",
        temperature_C=temperature_C,
        relative_humidity_percent=relative_humidity_percent,
        conditioning=conditioning,
        directionality="directional",
        material_direction="mixed",
    )
    selection = select_model(
        request,
        canonical_cards=manifest.get("canonical_cards", {}),
    )
    if selection.status not in {
        "admitted_with_restrictions",
        "sensitivity_only",
    }:
        raise ValueError(
            f"custom-pattern condition is {selection.status}: "
            f"{selection.reason}"
        )
    if selection.canonical_card is None:
        raise ValueError("admitted pattern study has no canonical material card")
    card_path = PROJECT_ROOT / selection.canonical_card
    if not card_path.exists():
        raise ValueError(
            f"canonical material card is missing: {selection.canonical_card}"
        )
    with twin_cards_path.open(encoding="utf-8", newline="") as handle:
        card_rows = {
            row["card_id"]: row for row in csv.DictReader(handle)
        }
    card_row = card_rows.get("room_tds_transverse")
    if (
        card_row is None
        or card_row.get("path") != selection.canonical_card
        or card_row.get("sha256") != _sha256(card_path)
    ):
        raise ValueError(
            "canonical Bambu PLA material-card checksum does not match"
        )
    return {
        "gate_passed": True,
        "twin_id": manifest["twin_id"],
        "twin_manifest": str(twin_manifest_path.resolve()),
        "twin_manifest_sha256": _sha256(twin_manifest_path),
        "request": asdict(request),
        "selection": asdict(selection),
        "properties_file": str(properties_path.resolve()),
        "properties_sha256": supplied_profile_sha,
        "canonical_profile_sha256": canonical_profile_sha,
        "canonical_card": str(card_path.resolve()),
        "canonical_card_sha256": _sha256(card_path),
    }


def _point(value: object, label: str) -> Tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a two-number [x, y] point")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain numeric coordinates") from error
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{label} coordinates must be finite")
    return x, y


def _connectivity(geometry: PatternGeometry) -> Tuple[bool, Dict[int, int]]:
    adjacency = {index: set() for index in range(1, len(geometry.nodes) + 1)}
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
    return len(visited) == len(adjacency), {
        node: len(neighbors) for node, neighbors in adjacency.items()
    }


def _minimum_joint_angle(
    geometry: PatternGeometry, degrees_by_node: Mapping[int, int]
) -> float | None:
    neighbors: Dict[int, List[int]] = {
        node: [] for node in degrees_by_node
    }
    for first, second in geometry.edges:
        neighbors[first].append(second)
        neighbors[second].append(first)
    minimum = None
    for node, adjacent in neighbors.items():
        origin = geometry.nodes[node - 1]
        vectors = []
        for other in adjacent:
            target = geometry.nodes[other - 1]
            length = math.hypot(
                target[0] - origin[0], target[1] - origin[1]
            )
            vectors.append(
                (
                    (target[0] - origin[0]) / length,
                    (target[1] - origin[1]) / length,
                )
            )
        for first_index, first in enumerate(vectors):
            for second in vectors[first_index + 1 :]:
                cosine = max(
                    -1.0,
                    min(1.0, first[0] * second[0] + first[1] * second[1]),
                )
                angle = math.degrees(math.acos(cosine))
                angle = min(angle, 180.0 - angle)
                if angle <= 1.0e-6:
                    continue
                minimum = angle if minimum is None else min(minimum, angle)
    return minimum


def load_custom_pattern(
    path: Path,
) -> Tuple[CustomPatternDefinition, PatternGeometry, Dict[str, object]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if payload.get("units") != "mm":
        raise ValueError("units must be 'mm'")
    domain = payload.get("domain")
    if not isinstance(domain, dict):
        raise ValueError("domain must be an object")
    width = float(domain.get("width_mm", 0.0))
    height = float(domain.get("height_mm", 0.0))
    depth = float(domain.get("depth_mm", 0.0))
    if not all(math.isfinite(value) and value > 0.0 for value in (
        width,
        height,
        depth,
    )):
        raise ValueError("domain dimensions must be finite and positive")
    relative_density = float(payload.get("relative_density", 0.35))
    if not 0.05 <= relative_density <= 0.8:
        raise ValueError("relative_density must lie in [0.05, 0.8]")
    minimum_width = float(
        payload.get("minimum_printable_strut_width_mm", 0.8)
    )
    minimum_segment = float(
        payload.get("minimum_segment_length_mm", 0.5)
    )
    if minimum_width <= 0.0 or minimum_segment <= 0.0:
        raise ValueError("minimum printable dimensions must be positive")

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("segments must be a non-empty list")
    segments = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError(
                f"segments[{index}] must be [[x1, y1], [x2, y2]]"
            )
        start = _point(segment[0], f"segments[{index}][0]")
        end = _point(segment[1], f"segments[{index}][1]")
        for point in (start, end):
            if not (
                0.0 <= point[0] <= width
                and 0.0 <= point[1] <= height
            ):
                raise ValueError(
                    f"segments[{index}] lies outside the coupon domain"
                )
        segments.append((start, end))

    definition = CustomPatternDefinition(
        name=str(payload.get("name", "")),
        width_mm=width,
        height_mm=height,
        depth_mm=depth,
        relative_density=relative_density,
        minimum_printable_strut_width_mm=minimum_width,
        minimum_segment_length_mm=minimum_segment,
        description=str(payload.get("description", "")),
    )
    geometry = make_geometry_from_segments(
        definition.name, segments, width, split_intersections=True
    )
    connected, degrees = _connectivity(geometry)
    if not connected:
        raise ValueError("pattern graph is disconnected")
    tolerance = max(width, height) * 1.0e-8
    boundary_counts = {
        "left": sum(abs(x) <= tolerance for x, _ in geometry.nodes),
        "right": sum(abs(x - width) <= tolerance for x, _ in geometry.nodes),
        "bottom": sum(abs(y) <= tolerance for _, y in geometry.nodes),
        "top": sum(abs(y - height) <= tolerance for _, y in geometry.nodes),
        "bending_load_line": sum(
            abs(x - width / 2.0) <= tolerance for x, _ in geometry.nodes
        ),
    }
    if any(boundary_counts[key] < 2 for key in (
        "left",
        "right",
        "bottom",
        "top",
    )):
        raise ValueError(
            "pattern needs at least two nodes on every coupon boundary"
        )
    if boundary_counts["bending_load_line"] < 1:
        raise ValueError("pattern does not cross the center bending-load line")
    internal_dangling = []
    for node, degree in degrees.items():
        x, y = geometry.nodes[node - 1]
        on_boundary = (
            abs(x) <= tolerance
            or abs(x - width) <= tolerance
            or abs(y) <= tolerance
            or abs(y - height) <= tolerance
        )
        if degree == 1 and not on_boundary:
            internal_dangling.append(node)
    if internal_dangling:
        raise ValueError(
            "pattern contains internal dangling struts at nodes "
            + ", ".join(str(node) for node in internal_dangling)
        )
    lengths = [
        math.dist(
            geometry.nodes[first - 1], geometry.nodes[second - 1]
        )
        for first, second in geometry.edges
    ]
    shortest = min(lengths)
    if shortest + tolerance < minimum_segment:
        raise ValueError(
            f"shortest connected segment is {shortest:.4g} mm, below the "
            f"{minimum_segment:.4g} mm declared minimum"
        )
    nominal_width = nominal_strut_width(
        geometry, width, height, relative_density
    )
    if nominal_width < minimum_width:
        raise ValueError(
            f"equal-mass nominal strut width is {nominal_width:.4g} mm, "
            f"below the {minimum_width:.4g} mm printability floor"
        )
    minimum_angle = _minimum_joint_angle(geometry, degrees)
    warnings = []
    if minimum_angle is not None and minimum_angle < 15.0:
        warnings.append(
            f"minimum non-collinear joint angle is {minimum_angle:.2f}°; "
            "inspect slicer fusion at acute joints"
        )
    validation = {
        "schema_version": 1,
        "source_file": str(path.resolve()),
        "source_sha256": _sha256(path),
        "status": "valid",
        "definition": asdict(definition),
        "normalized_nodes": len(geometry.nodes),
        "normalized_edges": len(geometry.edges),
        "total_centerline_length_mm": geometry.total_length_mm,
        "shortest_connected_segment_mm": shortest,
        "nominal_equal_mass_strut_width_mm": nominal_width,
        "minimum_joint_angle_deg": minimum_angle,
        "boundary_node_counts": boundary_counts,
        "degree_histogram": {
            str(degree): sum(value == degree for value in degrees.values())
            for degree in sorted(set(degrees.values()))
        },
        "crossings_are_connected": True,
        "graph_connected": True,
        "warnings": warnings,
    }
    return definition, geometry, validation


def _metric(
    row: PatternResult | SolidLoadResult,
) -> float | None:
    if row.load_case == "buckling":
        return row.critical_buckling_load_N
    return row.stiffness_N_per_mm


def _comparison_rows(
    fidelity: str,
    results: Sequence[PatternResult | SolidLoadResult],
    custom_name: str,
    control_name: str,
) -> List[Dict[str, object]]:
    lookup = {(row.pattern, row.load_case): row for row in results}
    rows = []
    for load_case in LOAD_CASES:
        custom = lookup[(custom_name, load_case)]
        control = lookup[(control_name, load_case)]
        custom_metric = _metric(custom)
        control_metric = _metric(control)
        ratio = (
            None
            if custom.status != "ok"
            or control.status != "ok"
            or custom_metric is None
            or control_metric is None
            or control_metric <= 0.0
            else custom_metric / control_metric
        )
        rows.append(
            {
                "fidelity": fidelity,
                "load_case": load_case,
                "custom_pattern": custom_name,
                "control_pattern": control_name,
                "custom_status": custom.status,
                "control_status": control.status,
                "metric": (
                    "critical_buckling_load_N"
                    if load_case == "buckling"
                    else "stiffness_N_per_mm"
                ),
                "custom_value": custom_metric,
                "control_value": control_metric,
                "custom_to_control_ratio": ratio,
            }
        )
    return rows


def run_custom_pattern_study(
    pattern_path: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    properties_path: Path = DEFAULT_TRANSVERSE_PROFILE,
    *,
    mesh_size_mm: float = 0.9,
    include_solid: bool = True,
    check_mesh_convergence: bool = True,
    control_name: str = "triangular",
    temperature_C: float = 23.0,
    relative_humidity_percent: float = 20.0,
    conditioning: str = "as_printed_dry",
    deformation: float = 0.01,
) -> Dict[str, object]:
    definition, geometry, validation = load_custom_pattern(pattern_path)
    if definition.name == control_name:
        raise ValueError("custom pattern name must differ from the control")
    if not 0.0 < deformation <= 0.02:
        raise ValueError(
            "deformation must lie in (0, 0.02] for the elastic pattern screen"
        )
    material_admission = admit_pattern_study(
        definition.name,
        properties_path,
        temperature_C=temperature_C,
        relative_humidity_percent=relative_humidity_percent,
        conditioning=conditioning,
    )
    output_dir = Path(output_dir) / definition.name
    runs_dir = Path(runs_dir) / definition.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    source_payload = json.loads(Path(pattern_path).read_text(encoding="utf-8"))
    (output_dir / "source_snapshot.json").write_text(
        json.dumps(source_payload, indent=2) + "\n", encoding="utf-8"
    )
    control = make_pattern(
        control_name, definition.width_mm, definition.height_mm
    )
    names = [definition.name, control_name]
    geometries = {definition.name: geometry, control_name: control}
    beam_results = run_benchmark(
        output_dir / "beam",
        runs_dir / "beam",
        properties_path=Path(properties_path),
        patterns=names,
        load_cases=LOAD_CASES,
        domain_width_mm=definition.width_mm,
        domain_height_mm=definition.height_mm,
        depth_mm=definition.depth_mm,
        relative_density=definition.relative_density,
        geometries=geometries,
        tensile_strain=deformation,
    )
    comparison = _comparison_rows(
        "beam_B31", beam_results, definition.name, control_name
    )
    solid_results: Sequence[SolidLoadResult] = []
    convergence_rows: List[Dict[str, object]] = []
    failure_sensitivity_static_fields = 0
    failure_sensitivity_path = None
    if include_solid:
        solid_results = run_solid_multiload(
            output_dir / "solid",
            runs_dir / "solid",
            properties_path=Path(properties_path),
            patterns=names,
            load_cases=LOAD_CASES,
            mesh_size_mm=mesh_size_mm,
            domain_width_mm=definition.width_mm,
            domain_height_mm=definition.height_mm,
            depth_mm=definition.depth_mm,
            relative_density=definition.relative_density,
            deformation=deformation,
            geometries=geometries,
        )
        comparison.extend(
            _comparison_rows(
                "solid_C3D6",
                solid_results,
                definition.name,
                control_name,
            )
        )
        for row in solid_results:
            if row.load_case == "bending":
                convergence_rows.append(
                    {
                        "mesh_size_mm": mesh_size_mm,
                        "pattern": row.pattern,
                        "status": row.status,
                        "stiffness_N_per_mm": row.stiffness_N_per_mm,
                        "nodes": row.nodes,
                        "elements": row.elements,
                    }
                )
        if check_mesh_convergence:
            for current_mesh_size in (1.2, 0.7):
                if math.isclose(
                    current_mesh_size, mesh_size_mm, rel_tol=0.0, abs_tol=1e-9
                ):
                    continue
                label = f"h{current_mesh_size:g}".replace(".", "p")
                current_results = run_solid_multiload(
                    output_dir / "solid_mesh_convergence" / label,
                    runs_dir / "solid_mesh_convergence" / label,
                    properties_path=Path(properties_path),
                    patterns=names,
                    load_cases=("bending",),
                    mesh_size_mm=current_mesh_size,
                    domain_width_mm=definition.width_mm,
                    domain_height_mm=definition.height_mm,
                    depth_mm=definition.depth_mm,
                    relative_density=definition.relative_density,
                    deformation=deformation,
                    geometries=geometries,
                )
                for row in current_results:
                    convergence_rows.append(
                        {
                            "mesh_size_mm": current_mesh_size,
                            "pattern": row.pattern,
                            "status": row.status,
                            "stiffness_N_per_mm": row.stiffness_N_per_mm,
                            "nodes": row.nodes,
                            "elements": row.elements,
                        }
                    )
        failure_output = output_dir / "failure_sensitivity"
        failure_rows = write_failure_audit(
            output_dir=failure_output,
            summaries=(
                (
                    f"custom_{definition.name}",
                    output_dir / "solid/summary.json",
                ),
            ),
            properties_path=Path(properties_path),
        )
        failure_sensitivity_static_fields = sum(
            row.get("screen_status") == "ok" for row in failure_rows
        )
        failure_sensitivity_path = str(
            (failure_output / "report.md").resolve()
        )
    convergence_rows.sort(
        key=lambda row: (
            str(row["pattern"]),
            -float(row["mesh_size_mm"]),
        )
    )
    convergence_change: Dict[str, float] = {}
    if convergence_rows:
        for pattern in names:
            stiffness_by_mesh = {
                float(row["mesh_size_mm"]): float(
                    row["stiffness_N_per_mm"]
                )
                for row in convergence_rows
                if row["pattern"] == pattern
                and row["status"] == "ok"
                and row["stiffness_N_per_mm"] is not None
            }
            if 0.9 in stiffness_by_mesh and 0.7 in stiffness_by_mesh:
                convergence_change[pattern] = abs(
                    stiffness_by_mesh[0.7] / stiffness_by_mesh[0.9] - 1.0
                )
    ratios = [
        float(row["custom_to_control_ratio"])
        for row in comparison
        if row["fidelity"] == (
            "solid_C3D6" if include_solid else "beam_B31"
        )
        and row["custom_to_control_ratio"] is not None
    ]
    balanced_ratio = (
        math.exp(sum(math.log(value) for value in ratios) / len(ratios))
        if len(ratios) == len(LOAD_CASES)
        else None
    )
    payload = {
        "pattern": definition.name,
        "control": control_name,
        "source_sha256": validation["source_sha256"],
        "material_admission": material_admission,
        "properties_file": str(Path(properties_path).resolve()),
        "properties_sha256": _sha256(Path(properties_path)),
        "temperature_C": temperature_C,
        "relative_humidity_percent": relative_humidity_percent,
        "conditioning": conditioning,
        "endpoint_deformation": deformation,
        "mesh_size_mm": mesh_size_mm if include_solid else None,
        "solid_run_included": include_solid,
        "successful_beam_cases": sum(
            row.status == "ok" for row in beam_results
        ),
        "successful_solid_cases": sum(
            row.status == "ok" for row in solid_results
        ),
        "failure_sensitivity_static_fields": (
            failure_sensitivity_static_fields
        ),
        "failure_sensitivity_report": failure_sensitivity_path,
        "balanced_custom_to_control_ratio": balanced_ratio,
        "bending_mesh_convergence": convergence_rows,
        "bending_medium_to_fine_change": convergence_change,
        "comparison": comparison,
    }
    lines = [
        f"# Custom infill study: {definition.name}",
        "",
        definition.description,
        "",
        "## Input gate",
        "",
        f"- Source SHA-256: `{validation['source_sha256']}`",
        f"- Connected graph: {validation['normalized_nodes']} nodes, "
        f"{validation['normalized_edges']} edges",
        f"- Shortest segment: "
        f"{validation['shortest_connected_segment_mm']:.3f} mm",
        f"- Nominal equal-mass width: "
        f"{validation['nominal_equal_mass_strut_width_mm']:.3f} mm",
        f"- Boundary node counts: {validation['boundary_node_counts']}",
        "",
    ]
    if validation["warnings"]:
        lines.extend(
            [
                "### Manufacturability warnings",
                "",
                *[
                    f"- {warning}" for warning in validation["warnings"]
                ],
                "",
            ]
        )
    lines.extend(
        [
        "## Equal-mass comparison",
        "",
        "| Fidelity | Load | Custom | Triangular | Ratio |",
        "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparison:
        custom_value = row["custom_value"]
        control_value = row["control_value"]
        ratio = row["custom_to_control_ratio"]
        lines.append(
            f"| {row['fidelity']} | {row['load_case']} | "
            f"{'n/a' if custom_value is None else f'{float(custom_value):.3f}'} | "
            f"{'n/a' if control_value is None else f'{float(control_value):.3f}'} | "
            f"{'n/a' if ratio is None else f'{float(ratio):.3f}'} |"
        )
    if convergence_rows:
        lines.extend(
            [
                "",
                "## Bending mesh convergence",
                "",
                "| Pattern | Mesh size (mm) | Nodes | Stiffness (N/mm) |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in convergence_rows:
            value = row["stiffness_N_per_mm"]
            lines.append(
                f"| {row['pattern']} | {float(row['mesh_size_mm']):.1f} | "
                f"{row['nodes']} | "
                f"{'n/a' if value is None else f'{float(value):.3f}'} |"
            )
        lines.extend(
            [
                "",
                "Medium-to-fine changes: "
                + ", ".join(
                    f"{pattern} {change:.2%}"
                    for pattern, change in convergence_change.items()
                )
                + ".",
            ]
        )
    highest_fidelity = "solid_C3D6" if include_solid else "beam_B31"
    final_rows = [
        row for row in comparison if row["fidelity"] == highest_fidelity
    ]
    specialist_wins = [
        str(row["load_case"])
        for row in final_rows
        if row["custom_to_control_ratio"] is not None
        and float(row["custom_to_control_ratio"]) > 1.05
    ]
    first_test_rows: List[Dict[str, object]] = []
    first_plan_path = output_dir / "first_physical_test_plan.csv"
    if include_solid and specialist_wins:
        full_plan_path = output_dir / "solid/physical_multiload_test_plan.csv"
        priority_path = (
            output_dir
            / "failure_sensitivity/physical_validation_priority.csv"
        )
        displacement_by_case: Dict[Tuple[str, str], str] = {}
        if priority_path.exists():
            with priority_path.open(
                encoding="utf-8", newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    displacement_by_case[
                        (row["pattern"], row["load_case"])
                    ] = row["recommended_first_displacement_mm"]
        with full_plan_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["load_case"] not in specialist_wins:
                    continue
                first_test_rows.append(
                    {
                        **row,
                        "recommended_first_displacement_mm": (
                            displacement_by_case.get(
                                (row["pattern"], row["load_case"]),
                                "",
                            )
                        ),
                    }
                )
        if first_test_rows:
            with first_plan_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(first_test_rows[0])
                )
                writer.writeheader()
                writer.writerows(first_test_rows)
    elif first_plan_path.exists():
        first_plan_path.unlink()
    converged_specialist = all(
        convergence_change.get(pattern, math.inf) <= 0.05
        for pattern in names
    )
    if specialist_wins and converged_specialist:
        next_action = (
            "inspect_slicer_then_print_specialist_validation"
            if validation["warnings"]
            else "print_specialist_validation"
        )
    elif specialist_wins:
        next_action = "refine_mesh_before_printing"
    else:
        next_action = "revise_pattern_before_printing"
    decision = {
        "pattern": definition.name,
        "control": control_name,
        "material": "Bambu PLA Basic",
        "material_gate_passed": material_admission["gate_passed"],
        "general_purpose_promotion": (
            balanced_ratio is not None
            and balanced_ratio >= 1.0
            and not validation["warnings"]
        ),
        "specialist_load_cases": specialist_wins,
        "mesh_convergence_gate_passed": converged_specialist,
        "first_test_specimen_count": len(first_test_rows),
        "next_action": next_action,
        "claims_not_supported": [
            "ultimate load or fracture",
            "post-collapse absorbed energy",
            "fatigue life",
            "creep or environmental lifetime",
            "behavior outside the admitted material condition",
        ],
    }
    payload["promotion_decision"] = decision
    payload["first_physical_test_plan"] = (
        str(first_plan_path.resolve())
        if first_test_rows
        else None
    )
    (output_dir / "study.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "promotion_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    lines.extend(["", "## Screening decision", ""])
    if balanced_ratio is not None and balanced_ratio >= 1.0:
        lines.append(
            "The custom concept passes the balanced screen against the "
            f"{control_name} control."
        )
    else:
        lines.append(
            "Do not promote this concept as a general-purpose replacement: "
            f"its balanced custom/control ratio is {balanced_ratio:.3f}."
            if balanced_ratio is not None
            else "Do not promote this concept until every load case succeeds."
        )
    if specialist_wins:
        lines.append(
            "It remains a specialist candidate for: "
            + ", ".join(specialist_wins)
            + "."
        )
    if validation["warnings"]:
        lines.append(
            "Resolve or explicitly inspect the manufacturability warnings in "
            "the slicer before physical testing."
        )
    if failure_sensitivity_path is not None:
        lines.append(
            f"The bounded tension-shear screen processed "
            f"{failure_sensitivity_static_fields}/8 static solid fields. "
            "Use its ranked three-replicate ramp before testing to the full "
            "simulated endpoint."
        )
    lines.extend(
        [
            "",
            "## Material admission",
            "",
            f"- Twin: `{material_admission['twin_id']}`",
            f"- Selection: "
            f"`{material_admission['selection']['model_id']}` "
            f"({material_admission['selection']['status']})",
            f"- Condition: {temperature_C:g} °C, "
            f"{relative_humidity_percent:g}% RH, `{conditioning}`",
            f"- Endpoint deformation: {deformation:.2%}",
            f"- Next action: `{next_action}`",
            f"- First physical set: {len(first_test_rows)} specimens",
            "",
        ]
    )
    lines.extend(
        [
            "",
            (
                "The five-load geometric-mean custom/control ratio at the "
                f"highest completed fidelity is {balanced_ratio:.3f}."
                if balanced_ratio is not None
                else "A balanced ratio was not computed because one or more "
                "load cases failed."
            ),
            "",
            "Ratios above one favor the custom concept. These are stiffness "
            "and elastic-buckling comparisons, not validated failure loads. "
            "Use the generated physical plan and exact-mass SCAD files before "
            "promoting a concept.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and simulate a custom infill centerline graph"
    )
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument(
        "--properties", default=str(DEFAULT_TRANSVERSE_PROFILE)
    )
    parser.add_argument("--mesh-size", type=float, default=0.9)
    parser.add_argument("--beam-only", action="store_true")
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--temperature-c", type=float, default=23.0)
    parser.add_argument(
        "--relative-humidity-percent", type=float, default=20.0
    )
    parser.add_argument(
        "--conditioning", default="as_printed_dry"
    )
    parser.add_argument("--deformation", type=float, default=0.01)
    args = parser.parse_args()
    payload = run_custom_pattern_study(
        Path(args.pattern),
        Path(args.output),
        Path(args.runs_dir),
        Path(args.properties),
        mesh_size_mm=args.mesh_size,
        include_solid=not args.beam_only,
        check_mesh_convergence=not args.skip_convergence,
        temperature_C=args.temperature_c,
        relative_humidity_percent=args.relative_humidity_percent,
        conditioning=args.conditioning,
        deformation=args.deformation,
    )
    solid_status = (
        f"{payload['successful_solid_cases']}/10"
        if payload["solid_run_included"]
        else "skipped"
    )
    report_path = (
        Path(args.output) / payload["pattern"] / "report.md"
    ).resolve()
    print(
        f"custom infill {payload['pattern']}: "
        f"beam={payload['successful_beam_cases']}/10 "
        f"solid={solid_status}; report: {report_path}"
    )


if __name__ == "__main__":
    main()
