from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .calculix import CalculixRunner
from .custom_pattern_toolpath import (
    DEFAULT_GCODE,
    DEFAULT_MATERIAL_PROFILE,
    DEFAULT_OUTPUT as DEFAULT_TOOLPATH_AUDIT,
    parse_deposited_segments,
)
from .materials import check_admissibility


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ROOT = PROJECT_ROOT / "results/custom_infill/graded_hourglass"
DEFAULT_OUTPUT = (
    CUSTOM_ROOT / "physical_validation/toolpath_resolved_bending"
)
DEFAULT_RUNS = PROJECT_ROOT / "runs/custom_infill/toolpath_resolved_bending"
DEFAULT_STUDY = CUSTOM_ROOT / "study.json"
DOMAIN_WIDTH_MM = 40.0
DOMAIN_HEIGHT_MM = 40.0
DEPTH_MM = 4.0
LAYER_HEIGHT_MM = 0.2
LAYER_COUNT = 20
FINE_RASTER_RESOLUTION = 240
DEFAULT_ANALYSIS_RESOLUTIONS = (40, 60, 80)
DISPLACEMENT_MM = 0.4
MAXIMUM_MASS_ERROR = 0.001
MAXIMUM_MEDIUM_TO_FINE_CHANGE = 0.05
MAXIMUM_RATIO_MEDIUM_TO_FINE_CHANGE = 0.05


@dataclass(frozen=True)
class FineToolpathGeometry:
    pattern: str
    object_name: str
    occupancy: np.ndarray
    raster_resolution: int
    raster_size_mm: float
    target_deposited_volume_mm3: float
    raster_volume_mm3: float
    mass_error_fraction: float
    x_offset_mm: float
    y_offset_mm: float
    layer_width_scales: Tuple[float, ...]
    component_count: int


@dataclass(frozen=True)
class ToolpathVoxelMesh:
    pattern: str
    resolution_xy: int
    layers_z: int
    cell_size_xy_mm: float
    layer_height_mm: float
    node_tags: np.ndarray
    coordinates: np.ndarray
    element_tags: np.ndarray
    connectivity: np.ndarray
    fill_fractions: np.ndarray
    supports: Tuple[int, ...]
    loads: Tuple[int, ...]
    anchor: int
    anchor_two: int
    component_count: int
    represented_volume_mm3: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pattern_for_object(name: str) -> str:
    match = re.fullmatch(
        r"([a-z][a-z0-9_]{0,63})\.stl_\d+",
        name,
    )
    if match is not None:
        return match.group(1)
    raise ValueError(f"unknown toolpath object {name!r}")


def _object_offsets(
    segments: Sequence[Mapping[str, object]],
) -> Tuple[float, float]:
    x_values = [
        float(segment[key])
        for segment in segments
        for key in ("x0_mm", "x1_mm")
    ]
    y_values = [
        float(segment[key])
        for segment in segments
        for key in ("y0_mm", "y1_mm")
    ]
    return (
        0.5 * (min(x_values) + max(x_values)) - DOMAIN_WIDTH_MM / 2.0,
        0.5 * (min(y_values) + max(y_values)) - DOMAIN_HEIGHT_MM / 2.0,
    )


def _segment_volume_mm3(segment: Mapping[str, object]) -> float:
    filament_radius = 1.75 / 2.0
    return (
        float(segment["filament_length_mm"])
        * math.pi
        * filament_radius**2
    )


def _normalized_score_field(
    segments: Sequence[Mapping[str, object]],
    resolution: int,
    x_offset_mm: float,
    y_offset_mm: float,
) -> np.ndarray:
    if resolution < 40:
        raise ValueError("fine toolpath raster must be at least 40 × 40")
    cell = DOMAIN_WIDTH_MM / resolution
    scores = np.full(
        (resolution, resolution, LAYER_COUNT), np.inf, dtype=np.float32
    )
    maximum_width_scale = 1.8
    for segment in segments:
        layer = int(
            round(float(segment["layer_z_mm"]) / LAYER_HEIGHT_MM)
        ) - 1
        if not 0 <= layer < LAYER_COUNT:
            raise ValueError(
                f"toolpath layer {segment['layer_z_mm']} lies outside coupon"
            )
        x0 = float(segment["x0_mm"]) - x_offset_mm
        y0 = float(segment["y0_mm"]) - y_offset_mm
        x1 = float(segment["x1_mm"]) - x_offset_mm
        y1 = float(segment["y1_mm"]) - y_offset_mm
        radius = 0.5 * float(segment["line_width_mm"] or 0.45)
        search_radius = maximum_width_scale * radius
        i0 = max(
            0, int(math.floor((min(x0, x1) - search_radius) / cell))
        )
        i1 = min(
            resolution - 1,
            int(math.floor((max(x0, x1) + search_radius) / cell)),
        )
        j0 = max(
            0, int(math.floor((min(y0, y1) - search_radius) / cell))
        )
        j1 = min(
            resolution - 1,
            int(math.floor((max(y0, y1) + search_radius) / cell)),
        )
        if i1 < i0 or j1 < j0:
            continue
        x_centers = (np.arange(i0, i1 + 1) + 0.5) * cell
        y_centers = (np.arange(j0, j1 + 1) + 0.5) * cell
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
        dx = x1 - x0
        dy = y1 - y0
        length_squared = dx * dx + dy * dy
        if length_squared > 0.0:
            parameter = np.clip(
                ((xx - x0) * dx + (yy - y0) * dy) / length_squared,
                0.0,
                1.0,
            )
            distance = np.hypot(
                xx - (x0 + parameter * dx),
                yy - (y0 + parameter * dy),
            )
        else:
            distance = np.hypot(xx - x0, yy - y0)
        normalized = distance / radius
        current = scores[i0 : i1 + 1, j0 : j1 + 1, layer]
        np.minimum(current, normalized.astype(np.float32), out=current)
    return scores


def build_mass_calibrated_fine_geometry(
    pattern: str,
    object_name: str,
    segments: Sequence[Mapping[str, object]],
    resolution: int = FINE_RASTER_RESOLUTION,
) -> FineToolpathGeometry:
    if _pattern_for_object(object_name) != pattern:
        raise ValueError("object name does not match requested pattern")
    x_offset, y_offset = _object_offsets(segments)
    scores = _normalized_score_field(
        segments, resolution, x_offset, y_offset
    )
    raster_size = DOMAIN_WIDTH_MM / resolution
    raster_cell_volume = raster_size**2 * LAYER_HEIGHT_MM
    occupancy = np.zeros_like(scores, dtype=bool)
    width_scales: List[float] = []
    target_total = 0.0
    for layer in range(LAYER_COUNT):
        layer_segments = [
            segment
            for segment in segments
            if int(
                round(float(segment["layer_z_mm"]) / LAYER_HEIGHT_MM)
            )
            - 1
            == layer
        ]
        target_volume = sum(
            _segment_volume_mm3(segment) for segment in layer_segments
        )
        target_total += target_volume
        target_cells = int(round(target_volume / raster_cell_volume))
        layer_scores = scores[:, :, layer].ravel()
        finite_count = int(np.isfinite(layer_scores).sum())
        if target_cells <= 0 or target_cells > finite_count:
            raise ValueError(
                f"layer {layer + 1} cannot represent deposited volume: "
                f"target={target_cells}, candidates={finite_count}"
            )
        selected = np.argpartition(layer_scores, target_cells - 1)[
            :target_cells
        ]
        layer_occupancy = np.zeros(
            (resolution, resolution), dtype=bool
        )
        layer_occupancy.flat[selected] = True
        occupancy[:, :, layer] = layer_occupancy
        width_scales.append(float(np.max(layer_scores[selected])))
    raster_volume = float(np.sum(occupancy) * raster_cell_volume)
    structure = ndimage.generate_binary_structure(3, 1)
    _, component_count = ndimage.label(occupancy, structure=structure)
    return FineToolpathGeometry(
        pattern=pattern,
        object_name=object_name,
        occupancy=occupancy,
        raster_resolution=resolution,
        raster_size_mm=raster_size,
        target_deposited_volume_mm3=target_total,
        raster_volume_mm3=raster_volume,
        mass_error_fraction=raster_volume / target_total - 1.0,
        x_offset_mm=x_offset,
        y_offset_mm=y_offset,
        layer_width_scales=tuple(width_scales),
        component_count=int(component_count),
    )


def _component_count(occupied: np.ndarray) -> int:
    structure = ndimage.generate_binary_structure(3, 1)
    _, count = ndimage.label(occupied, structure=structure)
    return int(count)


def coarsen_to_voxel_mesh(
    geometry: FineToolpathGeometry,
    resolution_xy: int,
) -> ToolpathVoxelMesh:
    fine = geometry.raster_resolution
    if fine % resolution_xy != 0:
        raise ValueError(
            "analysis resolution must divide the fine raster resolution"
        )
    factor = fine // resolution_xy
    fractions = geometry.occupancy.reshape(
        resolution_xy,
        factor,
        resolution_xy,
        factor,
        LAYER_COUNT,
    ).mean(axis=(1, 3))
    occupied = fractions > 0.0
    component_count = _component_count(occupied)
    cell_xy = DOMAIN_WIDTH_MM / resolution_xy
    node_lookup: Dict[tuple[int, int, int], int] = {}
    node_indices: List[tuple[int, int, int]] = []
    connectivity: List[List[int]] = []
    fill_fractions: List[float] = []
    corners = (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )
    for i, j, k in zip(*np.where(occupied)):
        element = []
        for di, dj, dk in corners:
            index = (int(i + di), int(j + dj), int(k + dk))
            if index not in node_lookup:
                node_lookup[index] = len(node_lookup) + 1
                node_indices.append(index)
            element.append(node_lookup[index])
        connectivity.append(element)
        fill_fractions.append(float(fractions[i, j, k]))
    node_tags = np.arange(1, len(node_indices) + 1, dtype=np.int64)
    coordinates = np.asarray(
        [
            (
                i * cell_xy,
                j * cell_xy,
                k * LAYER_HEIGHT_MM,
            )
            for i, j, k in node_indices
        ],
        dtype=float,
    )
    element_tags = np.arange(1, len(connectivity) + 1, dtype=np.int64)
    element_connectivity = np.asarray(connectivity, dtype=np.int64)
    fraction_array = np.asarray(fill_fractions, dtype=float)

    tolerance = 1e-10
    bottom = np.isclose(
        coordinates[:, 2], 0.0, atol=tolerance, rtol=0.0
    )
    minimum_x = float(np.min(coordinates[:, 0]))
    maximum_x = float(np.max(coordinates[:, 0]))
    edge_x = np.isclose(
        coordinates[:, 0], minimum_x, atol=tolerance, rtol=0.0
    ) | np.isclose(
        coordinates[:, 0], maximum_x, atol=tolerance, rtol=0.0
    )
    support_indices = np.flatnonzero(bottom & edge_x)
    top_indices = np.flatnonzero(
        np.isclose(
            coordinates[:, 2], DEPTH_MM, atol=tolerance, rtol=0.0
        )
    )
    center_distance = np.abs(
        coordinates[top_indices, 0] - DOMAIN_WIDTH_MM / 2.0
    )
    minimum_center = float(np.min(center_distance))
    strip_half_width = max(cell_xy * 0.6, minimum_center + tolerance)
    load_indices = top_indices[
        center_distance <= strip_half_width + tolerance
    ]
    if support_indices.size == 0 or load_indices.size == 0:
        raise RuntimeError("toolpath voxel mesh lacks bending boundaries")
    left = support_indices[
        np.isclose(
            coordinates[support_indices, 0],
            minimum_x,
            atol=tolerance,
            rtol=0.0,
        )
    ]
    right = support_indices[
        np.isclose(
            coordinates[support_indices, 0],
            maximum_x,
            atol=tolerance,
            rtol=0.0,
        )
    ]
    anchor_index = min(
        left,
        key=lambda index: (
            coordinates[index, 1],
            coordinates[index, 2],
        ),
    )
    anchor_two_index = min(
        right,
        key=lambda index: (
            coordinates[index, 1],
            coordinates[index, 2],
        ),
    )
    represented_volume = float(
        np.sum(fraction_array)
        * cell_xy**2
        * LAYER_HEIGHT_MM
    )
    return ToolpathVoxelMesh(
        pattern=geometry.pattern,
        resolution_xy=resolution_xy,
        layers_z=LAYER_COUNT,
        cell_size_xy_mm=cell_xy,
        layer_height_mm=LAYER_HEIGHT_MM,
        node_tags=node_tags,
        coordinates=coordinates,
        element_tags=element_tags,
        connectivity=element_connectivity,
        fill_fractions=fraction_array,
        supports=tuple(int(node_tags[index]) for index in support_indices),
        loads=tuple(int(node_tags[index]) for index in load_indices),
        anchor=int(node_tags[anchor_index]),
        anchor_two=int(node_tags[anchor_two_index]),
        component_count=component_count,
        represented_volume_mm3=represented_volume,
    )


def _id_set(keyword: str, name: str, identifiers: Iterable[int]) -> str:
    values = list(identifiers)
    lines = [f"*{keyword},{keyword}={name}"]
    for start in range(0, len(values), 16):
        lines.append(
            ",".join(str(value) for value in values[start : start + 16])
        )
    return "\n".join(lines)


def _engineering_constants(
    profile: Mapping[str, object],
) -> Dict[str, float]:
    source = profile["screening_engineering_constants"]
    if not isinstance(source, Mapping):
        raise ValueError("material profile lacks engineering constants")
    constants = {
        name: float(source[name])
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
        )
    }
    admissibility = check_admissibility(constants)
    if not admissibility.valid:
        raise ValueError(
            "inadmissible engineering constants: "
            + "; ".join(admissibility.reasons)
        )
    return constants


def render_toolpath_bending_deck(
    mesh: ToolpathVoxelMesh,
    profile: Mapping[str, object],
    displacement_mm: float = DISPLACEMENT_MM,
    include_field_output: bool = False,
) -> str:
    constants = _engineering_constants(profile)
    density = float(profile["density_tonnes_mm3"])
    nodes = "\n".join(
        f"{int(tag)}, {x:.12g}, {y:.12g}, {z:.12g}"
        for tag, (x, y, z) in zip(mesh.node_tags, mesh.coordinates)
    )
    elements = "\n".join(
        f"{int(tag)}, " + ",".join(str(int(node)) for node in element)
        for tag, element in zip(mesh.element_tags, mesh.connectivity)
    )
    unique_fractions = sorted(set(float(value) for value in mesh.fill_fractions))
    material_blocks = []
    for index, fraction in enumerate(unique_fractions, start=1):
        name = f"FILL_{index:03d}"
        element_ids = mesh.element_tags[
            np.isclose(
                mesh.fill_fractions,
                fraction,
                atol=1e-12,
                rtol=0.0,
            )
        ]
        scaled = {
            key: (
                value
                if key.startswith("nu")
                else value * fraction
            )
            for key, value in constants.items()
        }
        first_row = ",".join(
            f"{scaled[key]:.12g}"
            for key in (
                "E1",
                "E2",
                "E3",
                "nu12",
                "nu13",
                "nu23",
                "G12",
                "G13",
            )
        )
        material_blocks.append(
            f"""{_id_set("ELSET", name, element_ids)}
*MATERIAL,NAME={name}
*DENSITY
{density * fraction:.12g}
*ELASTIC,TYPE=ENGINEERING CONSTANTS
{first_row}
{scaled['G23']:.12g}
*SOLID SECTION,ELSET={name},MATERIAL={name},ORIENTATION=MAT_ORIENTATION
"""
        )
    field_output = (
        "*EL FILE,ELSET=ALL_TOOLPATH\nE\n"
        if include_field_output
        else ""
    )
    return f"""*HEADING
Mass-calibrated frozen-toolpath bending sensitivity, {mesh.pattern}
*NODE,NSET=NALL
{nodes}
*ELEMENT,TYPE=C3D8,ELSET=ALL_TOOLPATH
{elements}
{_id_set("NSET", "SUPPORTS", mesh.supports)}
{_id_set("NSET", "LOAD", mesh.loads)}
*NSET,NSET=ANCHOR
{mesh.anchor}
*NSET,NSET=ANCHOR_TWO
{mesh.anchor_two}
*ORIENTATION,NAME=MAT_ORIENTATION
1.,0.,0.,0.,1.,0.
{''.join(material_blocks)}
*STEP,NLGEOM=NO
*STATIC,DIRECT
1.,1.
*BOUNDARY
SUPPORTS,3,3,0.
ANCHOR,1,2,0.
ANCHOR_TWO,2,2,0.
LOAD,3,3,{-abs(displacement_mm):.12g}
*NODE PRINT,NSET=LOAD
U
*NODE PRINT,NSET=SUPPORTS
RF
*NODE FILE,NSET=LOAD
U
{field_output}\
*END STEP
"""


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_toolpath_bending_study(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    gcode_path: Path = DEFAULT_GCODE,
    material_profile_path: Path = DEFAULT_MATERIAL_PROFILE,
    toolpath_audit_manifest_path: Path = (
        DEFAULT_TOOLPATH_AUDIT / "manifest.json"
    ),
    study_path: Path = DEFAULT_STUDY,
    resolutions: Sequence[int] = DEFAULT_ANALYSIS_RESOLUTIONS,
    fine_raster_resolution: int = FINE_RASTER_RESOLUTION,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    gcode_path = Path(gcode_path)
    material_profile_path = Path(material_profile_path)
    toolpath_audit_manifest_path = Path(toolpath_audit_manifest_path)
    study_path = Path(study_path)
    if not resolutions:
        raise ValueError("at least one analysis resolution is required")
    if any(fine_raster_resolution % int(value) != 0 for value in resolutions):
        raise ValueError("all analysis resolutions must divide fine raster")
    audit = _json(toolpath_audit_manifest_path)
    if (
        audit.get("source_gcode_sha256") != _sha256(gcode_path)
        or audit.get("gates", {}).get(
            "relative_linear_stiffness_ranking_admitted"
        )
        is not True
    ):
        raise ValueError("frozen toolpath morphology audit is not admitted")
    profile = _json(material_profile_path)
    study = _json(study_path)
    custom_pattern = str(study.get("pattern", ""))
    control_pattern = str(study.get("control", ""))
    if (
        not custom_pattern
        or control_pattern != "triangular"
        or audit.get("custom_pattern", custom_pattern) != custom_pattern
    ):
        raise ValueError(
            "continuum study and toolpath audit do not identify the same "
            "custom pattern against triangular"
        )
    patterns = (control_pattern, custom_pattern)
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("fine_geometry", "voxel_meshes", "decks", "curves"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    parsed = parse_deposited_segments(gcode_path)
    selected_objects = {
        pattern: next(
            name
            for name in sorted(parsed)
            if _pattern_for_object(name) == pattern
        )
        for pattern in patterns
    }
    runner = CalculixRunner(
        runs_dir,
        executable="ccx",
        timeout_seconds=600.0,
        keep_successful=True,
    )
    rows: List[Dict[str, object]] = []
    geometry_rows: Dict[str, Dict[str, object]] = {}
    for pattern, object_name in selected_objects.items():
        geometry = build_mass_calibrated_fine_geometry(
            pattern,
            object_name,
            parsed[object_name],
            resolution=fine_raster_resolution,
        )
        geometry_path = (
            output_dir / "fine_geometry" / f"{pattern}-{fine_raster_resolution}.npz"
        )
        np.savez_compressed(
            geometry_path,
            occupancy=geometry.occupancy,
            raster_size_mm=np.asarray([geometry.raster_size_mm]),
        )
        geometry_rows[pattern] = {
            "object_name": object_name,
            "fine_raster_resolution": fine_raster_resolution,
            "fine_raster_size_mm": geometry.raster_size_mm,
            "target_deposited_volume_mm3": (
                geometry.target_deposited_volume_mm3
            ),
            "raster_volume_mm3": geometry.raster_volume_mm3,
            "mass_error_fraction": geometry.mass_error_fraction,
            "fine_geometry_component_count": geometry.component_count,
            "x_offset_mm": geometry.x_offset_mm,
            "y_offset_mm": geometry.y_offset_mm,
            "minimum_layer_width_scale": min(
                geometry.layer_width_scales
            ),
            "maximum_layer_width_scale": max(
                geometry.layer_width_scales
            ),
            "mean_layer_width_scale": float(
                np.mean(geometry.layer_width_scales)
            ),
            "fine_geometry_npz": str(geometry_path.resolve()),
            "fine_geometry_npz_sha256": _sha256(geometry_path),
        }
        for resolution in sorted(set(int(value) for value in resolutions)):
            mesh = coarsen_to_voxel_mesh(geometry, resolution)
            mesh_path = (
                output_dir / "voxel_meshes" / f"{pattern}-{resolution}.npz"
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
                run_id=f"toolpath-bending-{pattern}-{resolution}",
            )
            stiffness = None
            force = None
            endpoint = None
            if solver.ok:
                endpoint = float(abs(solver.curve["displacement"][-1]))
                force = float(abs(solver.curve["reaction_force"][-1]))
                stiffness = force / endpoint
                curve_path = (
                    output_dir / "curves" / f"{pattern}-{resolution}.csv"
                )
                with curve_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["displacement_mm", "force_N"],
                    )
                    writer.writeheader()
                    for current, current_force in zip(
                        np.abs(solver.curve["displacement"]),
                        np.abs(solver.curve["reaction_force"]),
                    ):
                        writer.writerow(
                            {
                                "displacement_mm": float(current),
                                "force_N": float(current_force),
                            }
                        )
            rows.append(
                {
                    "pattern": pattern,
                    "analysis_resolution_xy": resolution,
                    "cell_size_xy_mm": mesh.cell_size_xy_mm,
                    "layer_height_mm": mesh.layer_height_mm,
                    "nodes": len(mesh.node_tags),
                    "elements": len(mesh.element_tags),
                    "unique_fill_fractions": len(
                        set(float(value) for value in mesh.fill_fractions)
                    ),
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
                    "status": solver.status,
                    "endpoint_displacement_mm": endpoint,
                    "reaction_force_N": force,
                    "stiffness_N_per_mm": stiffness,
                    "run_directory": str(solver.run_dir.resolve()),
                    "voxel_mesh_npz": str(mesh_path.resolve()),
                    "voxel_mesh_npz_sha256": _sha256(mesh_path),
                    "deck": str(deck_path.resolve()),
                    "deck_sha256": _sha256(deck_path),
                    "error": solver.error,
                }
            )

    _write_csv(output_dir / "results.csv", rows)
    lookup = {
        (str(row["pattern"]), int(row["analysis_resolution_xy"])): row
        for row in rows
    }
    ordered_resolutions = sorted(set(int(value) for value in resolutions))
    fine_resolution = ordered_resolutions[-1]
    medium_resolution = (
        ordered_resolutions[-2]
        if len(ordered_resolutions) >= 2
        else fine_resolution
    )
    convergence = {}
    for pattern in patterns:
        medium = lookup[(pattern, medium_resolution)]["stiffness_N_per_mm"]
        fine = lookup[(pattern, fine_resolution)]["stiffness_N_per_mm"]
        convergence[pattern] = (
            None
            if medium is None or fine is None
            else abs(float(medium) / float(fine) - 1.0)
        )
    ratios = {}
    for resolution in ordered_resolutions:
        control = lookup[(control_pattern, resolution)]["stiffness_N_per_mm"]
        custom = lookup[
            (custom_pattern, resolution)
        ]["stiffness_N_per_mm"]
        ratios[str(resolution)] = (
            None
            if control is None or custom is None
            else float(custom) / float(control)
        )
    fine_ratio = ratios[str(fine_resolution)]
    medium_ratio = ratios[str(medium_resolution)]
    ratio_change = (
        None
        if fine_ratio is None or medium_ratio is None
        else abs(float(medium_ratio) / float(fine_ratio) - 1.0)
    )
    continuum_row = next(
        row
        for row in study["comparison"]
        if row["fidelity"] == "solid_C3D6"
        and row["load_case"] == "bending"
    )
    continuum_ratio = float(continuum_row["custom_to_control_ratio"])
    all_solver_cases_passed = all(row["status"] == "ok" for row in rows)
    all_geometry_connected = all(
        row["fine_geometry_component_count"] == 1
        for row in geometry_rows.values()
    ) and all(int(row["component_count"]) == 1 for row in rows)
    mass_gate = all(
        abs(float(row["mass_error_fraction"])) <= MAXIMUM_MASS_ERROR
        for row in rows
    )
    convergence_gate = all(
        value is not None and float(value) <= MAXIMUM_MEDIUM_TO_FINE_CHANGE
        for value in convergence.values()
    )
    ratio_convergence_gate = (
        ratio_change is not None
        and ratio_change <= MAXIMUM_RATIO_MEDIUM_TO_FINE_CHANGE
    )
    rank_survives = fine_ratio is not None and float(fine_ratio) > 1.0
    gate = all(
        (
            all_solver_cases_passed,
            all_geometry_connected,
            mass_gate,
            convergence_gate,
            ratio_convergence_gate,
            rank_survives,
        )
    )
    status = (
        "toolpath_aware_relative_bending_ranking_admitted"
        if gate
        else "toolpath_aware_ranking_not_admitted"
    )
    manifest: Dict[str, object] = {
        "study_id": f"{custom_pattern}_frozen_toolpath_bending_v1",
        "custom_pattern": custom_pattern,
        "control_pattern": control_pattern,
        "material": "Bambu PLA Basic",
        "status": status,
        "source_gcode": str(gcode_path.resolve()),
        "source_gcode_sha256": _sha256(gcode_path),
        "source_toolpath_audit": str(
            toolpath_audit_manifest_path.resolve()
        ),
        "source_toolpath_audit_sha256": _sha256(
            toolpath_audit_manifest_path
        ),
        "material_profile": str(material_profile_path.resolve()),
        "material_profile_sha256": _sha256(material_profile_path),
        "continuum_study": str(study_path.resolve()),
        "continuum_study_sha256": _sha256(study_path),
        "method": {
            "fine_raster_resolution_xy": fine_raster_resolution,
            "fine_raster_size_mm": DOMAIN_WIDTH_MM
            / fine_raster_resolution,
            "layer_height_mm": LAYER_HEIGHT_MM,
            "mass_calibration": (
                "each frozen G-code layer is rasterized from annotated bead "
                "centerlines and widths, then its occupied fine pixels are "
                "selected to match that layer's exact extruded filament volume"
            ),
            "coarse_element_law": (
                "C3D8 element elastic and shear moduli plus density are scaled "
                "linearly by fine-pixel occupancy fraction; Poisson ratios "
                "remain fixed (Voigt upper-bound voxel sensitivity)"
            ),
            "analysis_resolutions_xy": ordered_resolutions,
            "z_layers": LAYER_COUNT,
            "displacement_mm": DISPLACEMENT_MM,
        },
        "fine_geometries": geometry_rows,
        "solver_case_count": len(rows),
        "successful_solver_case_count": sum(
            row["status"] == "ok" for row in rows
        ),
        "toolpath_custom_to_control_ratio_by_resolution": ratios,
        "toolpath_fine_custom_to_control_ratio": fine_ratio,
        "continuum_custom_to_control_ratio": continuum_ratio,
        "toolpath_to_continuum_ratio_scale": (
            None
            if fine_ratio is None
            else float(fine_ratio) / continuum_ratio
        ),
        "medium_to_fine_stiffness_change": convergence,
        "medium_to_fine_ratio_change": ratio_change,
        "gates": {
            "all_solver_cases_passed": all_solver_cases_passed,
            "all_rasters_and_voxel_meshes_connected": (
                all_geometry_connected
            ),
            "mass_representation_error_at_most_0p1_percent": mass_gate,
            "each_medium_to_fine_stiffness_change_at_most_5_percent": (
                convergence_gate
            ),
            "medium_to_fine_ratio_change_at_most_5_percent": (
                ratio_convergence_gate
            ),
            "graded_hourglass_bending_advantage_survives": rank_survives,
            "custom_pattern_bending_advantage_survives": rank_survives,
            "relative_toolpath_aware_bending_ranking_admitted": gate,
            "absolute_stiffness_validated": False,
            "strength_or_failure_validated": False,
        },
        "restrictions": [
            "This is a mass-calibrated voxel sensitivity, not a resolved "
            "elliptical bead/contact mesh.",
            "Linear fill-fraction scaling is a Voigt upper-bound assumption.",
            "Inter-bead and interlayer interfaces are perfectly bonded.",
            "Physical coupons are still required for absolute stiffness and "
            "topology-bias validation.",
            "Strength, fracture, fatigue, creep, and post-collapse response "
            "are not admitted.",
        ],
        "results_csv": str((output_dir / "results.csv").resolve()),
        "results_csv_sha256": _sha256(output_dir / "results.csv"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    fine_control = lookup[
        (control_pattern, fine_resolution)
    ]["stiffness_N_per_mm"]
    fine_custom = lookup[
        (custom_pattern, fine_resolution)
    ]["stiffness_N_per_mm"]
    report = f"""# Frozen-toolpath bending sensitivity

The exact Orca G-code was converted into a layer-by-layer, mass-calibrated
voxel model and replayed with the admitted Bambu PLA Basic directional elastic
card.

| Model | Triangular stiffness (N/mm) | {custom_pattern} stiffness (N/mm) | Custom/control |
|---|---:|---:|---:|
| Continuum C3D6 | {float(continuum_row['control_value']):.3f} | {float(continuum_row['custom_value']):.3f} | {continuum_ratio:.4f} |
| Toolpath voxel, {fine_resolution} × {fine_resolution} × {LAYER_COUNT} | {float(fine_control) if fine_control is not None else float('nan'):.3f} | {float(fine_custom) if fine_custom is not None else float('nan'):.3f} | {float(fine_ratio) if fine_ratio is not None else float('nan'):.4f} |

The medium-to-fine stiffness changes are
{', '.join(f"{key} {float(value):.2%}" for key, value in convergence.items() if value is not None)};
the custom/control ratio changes by
{float(ratio_change) if ratio_change is not None else float('nan'):.2%}.

Decision: **{status}**.

This result tests whether the relative bending ranking survives the deposited
bead morphology. It does not validate absolute stiffness or any strength,
interface, fracture, fatigue, creep, or post-collapse claim.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen custom-pattern G-code as a voxel FEA model"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument("--gcode", default=str(DEFAULT_GCODE))
    parser.add_argument(
        "--material-profile", default=str(DEFAULT_MATERIAL_PROFILE)
    )
    parser.add_argument(
        "--toolpath-audit",
        default=str(DEFAULT_TOOLPATH_AUDIT / "manifest.json"),
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY))
    parser.add_argument(
        "--resolutions",
        default=",".join(str(value) for value in DEFAULT_ANALYSIS_RESOLUTIONS),
    )
    parser.add_argument(
        "--fine-raster-resolution", type=int, default=FINE_RASTER_RESOLUTION
    )
    args = parser.parse_args()
    manifest = run_toolpath_bending_study(
        Path(args.output),
        Path(args.runs),
        Path(args.gcode),
        Path(args.material_profile),
        Path(args.toolpath_audit),
        Path(args.study),
        tuple(int(value) for value in args.resolutions.split(",")),
        args.fine_raster_resolution,
    )
    print(
        f"toolpath bending: status={manifest['status']}; "
        f"solver cases={manifest['successful_solver_case_count']}/"
        f"{manifest['solver_case_count']}; "
        f"fine ratio={manifest['toolpath_fine_custom_to_control_ratio']}; "
        f"output: {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
