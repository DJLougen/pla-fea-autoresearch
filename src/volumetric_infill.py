from __future__ import annotations

import csv
import math
import struct
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


VOLUMETRIC_PATTERNS = ("honeycomb", "cubic", "grid", "gyroid")


@dataclass(frozen=True)
class VoxelMesh:
    pattern: str
    resolution: int
    side_length_mm: float
    occupancy: np.ndarray
    node_tags: np.ndarray
    coordinates: np.ndarray
    element_tags: np.ndarray
    connectivity: np.ndarray
    supports_by_axis: Mapping[str, Tuple[int, ...]]
    loads_by_axis: Mapping[str, Tuple[int, ...]]
    actual_relative_density: float
    voxel_size_mm: float
    actual_volume_mm3: float


def _distance_to_segments(
    points: np.ndarray,
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    distance = np.full(len(points), np.inf, dtype=float)
    for start, end in segments:
        direction = end - start
        length_squared = float(direction @ direction)
        parameter = np.clip(
            ((points - start) @ direction) / length_squared, 0.0, 1.0
        )
        projection = start + parameter[:, None] * direction
        distance = np.minimum(
            distance, np.linalg.norm(points - projection, axis=1)
        )
    return distance


def _bcc_segments(cells: int) -> List[tuple[np.ndarray, np.ndarray]]:
    segments = []
    for first in range(cells):
        for second in range(cells):
            for third in range(cells):
                lower = np.asarray([first, second, third], dtype=float) / cells
                upper = (
                    np.asarray([first + 1, second + 1, third + 1], dtype=float)
                    / cells
                )
                center = (lower + upper) / 2.0
                for dx in (0, 1):
                    for dy in (0, 1):
                        for dz in (0, 1):
                            corner = (
                                np.asarray(
                                    [first + dx, second + dy, third + dz],
                                    dtype=float,
                                )
                                / cells
                            )
                            segments.append((center, corner))
    return segments


def _pattern_score(
    pattern: str,
    resolution: int,
    cells: int = 2,
    grid_connectors: bool = False,
) -> np.ndarray:
    if pattern not in VOLUMETRIC_PATTERNS:
        raise ValueError(f"unknown volumetric pattern {pattern!r}")
    centers = (np.arange(resolution, dtype=float) + 0.5) / resolution
    x, y, z = np.meshgrid(centers, centers, centers, indexing="ij")
    angular = 2.0 * np.pi * cells
    if pattern == "grid":
        # Alternating orthogonal raster rods through successive z bands.
        dx = np.abs(np.sin(angular * x))
        dy = np.abs(np.sin(angular * y))
        dz_x = np.abs(np.sin(angular * z))
        dz_y = np.abs(np.cos(angular * z))
        alternating = np.minimum(
            np.sqrt(dy ** 2 + dz_x ** 2),
            np.sqrt(dx ** 2 + dz_y ** 2),
        )
        if grid_connectors:
            # Printed alternating rasters fuse at recurring crossings. A
            # sparse vertical connector family avoids a resolution-dependent
            # mechanism when outer walls are intentionally omitted.
            return np.minimum(
                alternating, np.sqrt(dx ** 2 + dy ** 2)
            )
        return alternating
    if pattern == "gyroid":
        level = (
            np.sin(angular * x) * np.cos(angular * y)
            + np.sin(angular * y) * np.cos(angular * z)
            + np.sin(angular * z) * np.cos(angular * x)
        )
        return np.abs(level)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    if pattern == "cubic":
        return _distance_to_segments(
            points, _bcc_segments(cells)
        ).reshape(x.shape)

    # Flat-top hexagonal walls repeated through thickness. Centers outside
    # the domain are included so clipping does not bias edge cells.
    radius = 0.28
    apothem = radius * np.sqrt(3.0) / 2.0
    score = np.full(len(points), np.inf, dtype=float)
    for column in range(-2, 8):
        center_x = column * 1.5 * radius
        for row in range(-2, 8):
            center_y = (
                row + 0.5 * (column % 2)
            ) * np.sqrt(3.0) * radius
            qx = np.abs(points[:, 0] - center_x)
            qy = np.abs(points[:, 1] - center_y)
            signed = np.maximum(
                qx * np.sqrt(3.0) / 2.0 + qy * 0.5,
                qy,
            ) - apothem
            score = np.minimum(score, np.abs(signed))
    return score.reshape(x.shape)


def pattern_occupancy(
    pattern: str,
    resolution: int,
    relative_density: float = 0.5,
    cells: int = 2,
    grid_connectors: bool = False,
) -> np.ndarray:
    if resolution < 6:
        raise ValueError("voxel resolution must be at least 6")
    if not 0.1 <= relative_density <= 0.9:
        raise ValueError("relative density must lie in [0.1, 0.9]")
    score = _pattern_score(
        pattern, resolution, cells, grid_connectors
    )
    count = int(round(relative_density * score.size))
    order = np.argsort(score.ravel(), kind="stable")
    occupancy = np.zeros(score.size, dtype=bool)
    occupancy[order[:count]] = True
    occupancy = occupancy.reshape(score.shape)
    components = _component_masks(occupancy)
    if len(components) > 1:
        occupancy = max(components, key=lambda item: int(np.sum(item)))
        while int(np.sum(occupancy)) < count:
            adjacent = np.zeros_like(occupancy)
            adjacent[1:, :, :] |= occupancy[:-1, :, :]
            adjacent[:-1, :, :] |= occupancy[1:, :, :]
            adjacent[:, 1:, :] |= occupancy[:, :-1, :]
            adjacent[:, :-1, :] |= occupancy[:, 1:, :]
            adjacent[:, :, 1:] |= occupancy[:, :, :-1]
            adjacent[:, :, :-1] |= occupancy[:, :, 1:]
            candidates = np.flatnonzero((adjacent & ~occupancy).ravel())
            if len(candidates) == 0:
                raise RuntimeError(
                    f"could not repair {pattern} voxel connectivity"
                )
            missing = count - int(np.sum(occupancy))
            candidate_scores = score.ravel()[candidates]
            selected = candidates[
                np.argsort(candidate_scores, kind="stable")[:missing]
            ]
            occupancy.ravel()[selected] = True
    return occupancy


def _component_masks(occupancy: np.ndarray) -> List[np.ndarray]:
    occupied = np.asarray(occupancy, dtype=bool)
    visited = np.zeros_like(occupied)
    components: List[np.ndarray] = []
    neighbors = (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    )
    shape = occupied.shape
    for seed in zip(*np.where(occupied & ~visited)):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        component = np.zeros_like(occupied)
        while stack:
            current = stack.pop()
            component[current] = True
            for offset in neighbors:
                candidate = tuple(
                    current[index] + offset[index] for index in range(3)
                )
                if any(
                    candidate[index] < 0
                    or candidate[index] >= shape[index]
                    for index in range(3)
                ):
                    continue
                if occupied[candidate] and not visited[candidate]:
                    visited[candidate] = True
                    stack.append(candidate)
        components.append(component)
    return components


def connectivity_components(occupancy: np.ndarray) -> List[int]:
    return sorted(
        (int(np.sum(component)) for component in _component_masks(occupancy)),
        reverse=True,
    )


def make_voxel_mesh(
    pattern: str,
    resolution: int = 16,
    side_length_mm: float = 20.0,
    relative_density: float = 0.5,
    cells: int = 2,
    required_axes: Sequence[str] = ("x", "y", "z"),
    grid_connectors: bool = False,
) -> VoxelMesh:
    if not required_axes or any(
        axis not in {"x", "y", "z"} for axis in required_axes
    ):
        raise ValueError(
            "required_axes must contain one or more of x, y, and z"
        )
    occupancy = pattern_occupancy(
        pattern,
        resolution,
        relative_density,
        cells,
        grid_connectors,
    )
    components = connectivity_components(occupancy)
    if len(components) != 1:
        raise RuntimeError(
            f"{pattern} voxel morphology has {len(components)} components"
        )
    voxel_size = side_length_mm / resolution
    node_lookup: Dict[tuple[int, int, int], int] = {}
    node_indices: List[tuple[int, int, int]] = []
    connectivity = []
    corner_offsets = (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )
    for i, j, k in zip(*np.where(occupancy)):
        element = []
        for di, dj, dk in corner_offsets:
            index = (int(i + di), int(j + dj), int(k + dk))
            if index not in node_lookup:
                node_lookup[index] = len(node_lookup) + 1
                node_indices.append(index)
            element.append(node_lookup[index])
        connectivity.append(element)
    node_tags = np.arange(1, len(node_indices) + 1, dtype=np.int64)
    coordinates = (
        np.asarray(node_indices, dtype=float) * voxel_size
    )
    element_tags = np.arange(
        1, len(connectivity) + 1, dtype=np.int64
    )
    element_connectivity = np.asarray(connectivity, dtype=np.int64)
    supports: Dict[str, Tuple[int, ...]] = {}
    loads: Dict[str, Tuple[int, ...]] = {}
    for index, axis in enumerate(("x", "y", "z")):
        if axis not in required_axes:
            continue
        supports[axis] = tuple(
            int(tag)
            for tag, coordinate in zip(node_tags, coordinates[:, index])
            if math.isclose(coordinate, 0.0, abs_tol=1.0e-10)
        )
        loads[axis] = tuple(
            int(tag)
            for tag, coordinate in zip(node_tags, coordinates[:, index])
            if math.isclose(
                coordinate, side_length_mm, abs_tol=1.0e-10
            )
        )
        if not supports[axis] or not loads[axis]:
            raise RuntimeError(
                f"{pattern} does not span the {axis} loading direction"
            )
    density = float(np.mean(occupancy))
    volume = float(np.sum(occupancy) * voxel_size ** 3)
    return VoxelMesh(
        pattern,
        resolution,
        side_length_mm,
        occupancy,
        node_tags,
        coordinates,
        element_tags,
        element_connectivity,
        supports,
        loads,
        density,
        voxel_size,
        volume,
    )


def _id_set(name: str, identifiers: Iterable[int]) -> str:
    values = list(identifiers)
    lines = [f"*NSET,NSET={name}"]
    for start in range(0, len(values), 16):
        lines.append(
            ",".join(str(value) for value in values[start:start + 16])
        )
    return "\n".join(lines)


def render_voxel_compression_deck(
    mesh: VoxelMesh,
    axis: str,
    deformation: float,
    youngs_modulus_MPa: float = 2600.0,
    poisson_ratio: float = 0.33,
    density_tonnes_mm3: float = 1.25e-9,
    plastic_points: Sequence[Mapping[str, float]] | None = None,
) -> str:
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be x, y, or z")
    if not 0.0 < deformation <= 0.10:
        raise ValueError("deformation must lie in (0, 0.10]")
    axis_dof = {"x": 1, "y": 2, "z": 3}[axis]
    transverse = [value for value in (1, 2, 3) if value != axis_dof]
    support_tags = mesh.supports_by_axis[axis]
    support_indices = np.asarray(
        [int(tag) - 1 for tag in support_tags], dtype=int
    )
    transverse_coordinates = mesh.coordinates[
        support_indices
    ][:, [value - 1 for value in transverse]]
    anchor_position = int(
        np.lexsort(
            (
                transverse_coordinates[:, 1],
                transverse_coordinates[:, 0],
            )
        )[0]
    )
    anchor_two_position = int(
        np.lexsort(
            (
                -transverse_coordinates[:, 0],
                transverse_coordinates[:, 1],
            )
        )[0]
    )
    anchor = support_tags[anchor_position]
    anchor_two = support_tags[anchor_two_position]
    nodes = "\n".join(
        f"{int(tag)}, {x:.12g}, {y:.12g}, {z:.12g}"
        for tag, (x, y, z) in zip(mesh.node_tags, mesh.coordinates)
    )
    elements = "\n".join(
        f"{int(tag)}, " + ",".join(str(int(node)) for node in element)
        for tag, element in zip(mesh.element_tags, mesh.connectivity)
    )
    plastic = ""
    if plastic_points:
        rows = "\n".join(
            f"{float(point['stress_MPa']):.12g},"
            f"{float(point['equivalent_plastic_strain']):.12g}"
            for point in plastic_points
        )
        plastic = f"*PLASTIC\n{rows}\n"
        step = "*STEP,NLGEOM=YES\n*STATIC\n0.025,1.,1.e-06,0.05"
    else:
        step = "*STEP,NLGEOM=NO\n*STATIC,DIRECT\n1.,1."
    displacement = -mesh.side_length_mm * deformation
    return f"""*HEADING
Explicit voxel {mesh.pattern} compression along {axis}
*NODE,NSET=NALL
{nodes}
*ELEMENT,TYPE=C3D8,ELSET=SOLID
{elements}
{_id_set("SUPPORTS", support_tags)}
{_id_set("LOAD", mesh.loads_by_axis[axis])}
*NSET,NSET=ANCHOR
{anchor}
*NSET,NSET=ANCHOR_TWO
{anchor_two}
*MATERIAL,NAME=BAMBU_PLA
*DENSITY
{density_tonnes_mm3:.12g}
*ELASTIC
{youngs_modulus_MPa:.12g},{poisson_ratio:.12g}
{plastic}*SOLID SECTION,ELSET=SOLID,MATERIAL=BAMBU_PLA
{step}
*BOUNDARY
SUPPORTS,{axis_dof},{axis_dof},0.
ANCHOR,{transverse[0]},{transverse[1]},0.
ANCHOR_TWO,{transverse[1]},{transverse[1]},0.
LOAD,{axis_dof},{axis_dof},{displacement:.12g}
*NODE PRINT,NSET=LOAD
U
*NODE PRINT,NSET=SUPPORTS
RF
*NODE FILE,NSET=LOAD
U
*END STEP
"""


def morphology_metrics(mesh: VoxelMesh, axis: str = "x") -> Dict[str, float]:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    line_fraction = np.mean(mesh.occupancy, axis=axis_index)
    transitions = np.diff(
        mesh.occupancy.astype(np.int8), axis=axis_index
    )
    interface_count = np.sum(np.abs(transitions), axis=axis_index)
    occupied = mesh.occupancy
    exposed_faces = 0
    for current_axis in range(3):
        exposed_faces += int(
            np.sum(
                np.abs(
                    np.diff(
                        occupied.astype(np.int8),
                        axis=current_axis,
                    )
                )
            )
        )
        low = np.take(occupied, 0, axis=current_axis)
        high = np.take(occupied, -1, axis=current_axis)
        exposed_faces += int(np.sum(low) + np.sum(high))
    return {
        "line_of_sight_mean_solid_fraction": float(np.mean(line_fraction)),
        "line_of_sight_minimum_solid_fraction": float(np.min(line_fraction)),
        "line_of_sight_standard_deviation": float(np.std(line_fraction)),
        "fully_solid_line_fraction": float(np.mean(line_fraction == 1.0)),
        "empty_line_fraction": float(np.mean(line_fraction == 0.0)),
        "mean_material_void_intersections": float(np.mean(interface_count)),
        "surface_area_proxy_mm2": float(
            exposed_faces * mesh.voxel_size_mm ** 2
        ),
        "surface_area_proxy_per_solid_volume_mm_inv": float(
            exposed_faces * mesh.voxel_size_mm ** 2
            / mesh.actual_volume_mm3
        ),
    }


def write_occupancy_csv(mesh: VoxelMesh, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("i", "j", "k", "occupied"))
        for i in range(mesh.resolution):
            for j in range(mesh.resolution):
                for k in range(mesh.resolution):
                    writer.writerow(
                        (i, j, k, int(mesh.occupancy[i, j, k]))
                    )


def write_voxel_stl(mesh: VoxelMesh, path: Path) -> int:
    """Write a manifold, volume-matched surface around the voxel solid."""
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise RuntimeError(
            "printable voxel STL export requires scikit-image"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    volume = np.pad(
        mesh.occupancy.astype(np.float32),
        1,
        mode="constant",
        constant_values=0.0,
    )

    def surface(level: float) -> tuple[np.ndarray, np.ndarray, float]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"skimage\.measure\._marching_cubes_lewiner",
            )
            vertices, faces, _, _ = marching_cubes(
                volume,
                level=level,
                spacing=(mesh.voxel_size_mm,) * 3,
                gradient_direction="descent",
                allow_degenerate=False,
                method="lewiner",
            )
        vertices -= 0.5 * mesh.voxel_size_mm
        triangles = vertices[faces]
        signed = np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        )
        return vertices, faces, float(abs(np.sum(signed)) / 6.0)

    vertices, faces, surface_volume = surface(0.5)
    target_volume = mesh.actual_volume_mm3
    scale = (target_volume / surface_volume) ** (1.0 / 3.0)
    center = 0.5 * (
        np.min(vertices, axis=0) + np.max(vertices, axis=0)
    )
    vertices = center + scale * (vertices - center)
    vertices -= np.min(vertices, axis=0)
    triangle_count = len(faces)
    header = (
        f"Bambu PLA {mesh.pattern} n{mesh.resolution} voxel boundary"
        .encode("ascii")[:80]
        .ljust(80, b"\0")
    )
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", triangle_count))
        for face in faces:
            triangle = vertices[face]
            normal = np.cross(
                triangle[1] - triangle[0],
                triangle[2] - triangle[0],
            )
            norm = float(np.linalg.norm(normal))
            if norm:
                normal /= norm
            values = list(normal)
            for vertex in triangle:
                values.extend(float(value) for value in vertex)
            handle.write(struct.pack("<12fH", *values, 0))
    return triangle_count


def inspect_binary_stl(path: Path) -> Dict[str, object]:
    """Return topology, volume, and bounds checks for a binary STL."""
    payload = Path(path).read_bytes()
    if len(payload) < 84:
        raise ValueError("binary STL is shorter than its header")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    if len(payload) != 84 + 50 * triangle_count:
        raise ValueError("binary STL byte count does not match its header")
    edges: Counter[tuple[tuple[float, ...], tuple[float, ...]]] = Counter()
    triangles = []
    for index in range(triangle_count):
        values = struct.unpack_from("<12fH", payload, 84 + 50 * index)
        vertices = np.asarray(
            [values[3:6], values[6:9], values[9:12]],
            dtype=float,
        )
        triangles.append(vertices)
        rounded = [
            tuple(round(float(value), 6) for value in vertex)
            for vertex in vertices
        ]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edges[tuple(sorted((rounded[first], rounded[second])))] += 1
    triangle_array = np.asarray(triangles)
    signed = np.einsum(
        "ij,ij->i",
        triangle_array[:, 0],
        np.cross(triangle_array[:, 1], triangle_array[:, 2]),
    )
    all_vertices = triangle_array.reshape(-1, 3)
    nonmanifold = sum(count != 2 for count in edges.values())
    return {
        "triangle_count": triangle_count,
        "watertight": nonmanifold == 0,
        "nonmanifold_edge_count": nonmanifold,
        "volume_mm3": float(abs(np.sum(signed)) / 6.0),
        "minimum_x_mm": float(np.min(all_vertices[:, 0])),
        "minimum_y_mm": float(np.min(all_vertices[:, 1])),
        "minimum_z_mm": float(np.min(all_vertices[:, 2])),
        "maximum_x_mm": float(np.max(all_vertices[:, 0])),
        "maximum_y_mm": float(np.max(all_vertices[:, 1])),
        "maximum_z_mm": float(np.max(all_vertices[:, 2])),
    }
