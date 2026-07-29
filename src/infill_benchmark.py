from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .calculix import CalculixRunner, find_ccx
from .materials import PRIORS, material_card


Point = Tuple[float, float]
Segment = Tuple[Point, Point]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROPERTIES = PROJECT_ROOT / "data/reported/bambu_pla_screening.json"
PATTERN_NAMES = (
    "grid",
    "triangular",
    "x_braced",
    "selective_x",
    "staggered_diamond",
    "bowtie",
    "reentrant",
    "concentric_diamond",
    "sinusoidal",
    "zigzag_90",
    "zigzag_45",
    "hilbert",
)
LOAD_CASES = ("tension", "shear", "bending", "compression", "buckling")


@dataclass(frozen=True)
class PatternGeometry:
    name: str
    nodes: Tuple[Point, ...]
    edges: Tuple[Tuple[int, int], ...]
    total_length_mm: float


@dataclass
class PatternResult:
    pattern: str
    load_case: str
    status: str
    nodes: int
    beams: int
    total_centerline_length_mm: float
    strut_width_mm: float
    nominal_relative_density: float
    nominal_mass_g: float
    applied_displacement_mm: float
    reaction_force_N: float | None
    stiffness_N_per_mm: float | None
    effective_modulus_MPa: float | None
    specific_stiffness_N_per_mm_g: float | None
    maximum_von_mises_MPa: float | None
    tensile_reference_utilization: float | None
    critical_buckling_load_N: float | None
    critical_buckling_load_N_per_g: float | None
    run_directory: str
    error: str | None = None


def _point(x: float, y: float) -> Point:
    return (round(float(x), 9), round(float(y), 9))


def _canonical_segment(a: Point, b: Point) -> Segment:
    if a == b:
        raise ValueError("zero-length pattern segment")
    return (a, b) if a < b else (b, a)


def _geometry(name: str, segments: Iterable[Segment]) -> PatternGeometry:
    unique = sorted(
        {
            _canonical_segment(_point(*start), _point(*end))
            for start, end in segments
        }
    )
    points = sorted({point for segment in unique for point in segment})
    indices = {point: index + 1 for index, point in enumerate(points)}
    edges = tuple((indices[start], indices[end]) for start, end in unique)
    total_length = sum(
        math.hypot(end[0] - start[0], end[1] - start[1])
        for start, end in unique
    )
    return PatternGeometry(name, tuple(points), edges, total_length)


def _split_at_vertical_line(
    segments: Iterable[Segment], x_coordinate: float
) -> List[Segment]:
    """Insert connected nodes where beams cross a vertical loading line."""
    result: List[Segment] = []
    tolerance = 1.0e-9
    for start, end in segments:
        delta_x = end[0] - start[0]
        if abs(delta_x) <= tolerance:
            result.append((start, end))
            continue
        parameter = (x_coordinate - start[0]) / delta_x
        if tolerance < parameter < 1.0 - tolerance:
            y_coordinate = start[1] + parameter * (end[1] - start[1])
            split = _point(x_coordinate, y_coordinate)
            result.extend([(start, split), (split, end)])
        else:
            result.append((start, end))
    return result


def _finalize_pattern(
    name: str, segments: Iterable[Segment], domain_width_mm: float
) -> PatternGeometry:
    return _geometry(
        name, _split_at_vertical_line(segments, domain_width_mm / 2.0)
    )


def make_geometry_from_segments(
    name: str,
    segments: Iterable[Segment],
    domain_width_mm: float,
    *,
    split_intersections: bool = True,
) -> PatternGeometry:
    """Create a beam/solid geometry and connect every crossing strut.

    Custom concepts often describe two crossing lines without explicitly
    inserting a node. The printable capsule union connects them physically, so
    the beam model must do the same to preserve fidelity between screening
    levels.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
        raise ValueError(
            "pattern name must match [a-z][a-z0-9_]{0,63}"
        )
    original = sorted(
        {
            _canonical_segment(_point(*start), _point(*end))
            for start, end in segments
        }
    )
    if not original:
        raise ValueError("pattern must contain at least one segment")
    if not split_intersections:
        return _finalize_pattern(name, original, domain_width_mm)

    tolerance = 1.0e-9

    def cross(first: Point, second: Point) -> float:
        return first[0] * second[1] - first[1] * second[0]

    parameters: List[set[float]] = [
        {0.0, 1.0} for _ in original
    ]
    for first_index, (p, p2) in enumerate(original):
        r = (p2[0] - p[0], p2[1] - p[1])
        rr = r[0] ** 2 + r[1] ** 2
        for second_index in range(first_index + 1, len(original)):
            q, q2 = original[second_index]
            s = (q2[0] - q[0], q2[1] - q[1])
            q_minus_p = (q[0] - p[0], q[1] - p[1])
            denominator = cross(r, s)
            if abs(denominator) <= tolerance:
                if abs(cross(q_minus_p, r)) > tolerance:
                    continue
                t0 = (
                    q_minus_p[0] * r[0] + q_minus_p[1] * r[1]
                ) / rr
                q2_minus_p = (q2[0] - p[0], q2[1] - p[1])
                t1 = (
                    q2_minus_p[0] * r[0] + q2_minus_p[1] * r[1]
                ) / rr
                overlap = min(1.0, max(t0, t1)) - max(
                    0.0, min(t0, t1)
                )
                if overlap > tolerance:
                    raise ValueError(
                        "custom segments contain a collinear overlap; split "
                        "the intended centerline into non-overlapping pieces"
                    )
                continue
            t = cross(q_minus_p, s) / denominator
            u = cross(q_minus_p, r) / denominator
            if (
                -tolerance <= t <= 1.0 + tolerance
                and -tolerance <= u <= 1.0 + tolerance
            ):
                parameters[first_index].add(min(1.0, max(0.0, t)))
                parameters[second_index].add(min(1.0, max(0.0, u)))

    split: List[Segment] = []
    for (start, end), values in zip(original, parameters):
        ordered = sorted(values)
        for lower, upper in zip(ordered[:-1], ordered[1:]):
            if upper - lower <= tolerance:
                continue
            first = _point(
                start[0] + lower * (end[0] - start[0]),
                start[1] + lower * (end[1] - start[1]),
            )
            second = _point(
                start[0] + upper * (end[0] - start[0]),
                start[1] + upper * (end[1] - start[1]),
            )
            split.append((first, second))
    return _finalize_pattern(name, split, domain_width_mm)


def _grid_segments(width: float, height: float, cells: int) -> List[Segment]:
    dx, dy = width / cells, height / cells
    segments: List[Segment] = []
    for ix in range(cells + 1):
        x = ix * dx
        for iy in range(cells):
            segments.append((_point(x, iy * dy), _point(x, (iy + 1) * dy)))
    for iy in range(cells + 1):
        y = iy * dy
        for ix in range(cells):
            segments.append((_point(ix * dx, y), _point((ix + 1) * dx, y)))
    return segments


def _hilbert_xy(order: int, index: int) -> Tuple[int, int]:
    """Map a Hilbert-curve distance to integer grid coordinates."""
    side = 1 << order
    x = y = 0
    distance = index
    scale = 1
    while scale < side:
        rx = 1 & (distance // 2)
        ry = 1 & (distance ^ rx)
        if ry == 0:
            if rx == 1:
                x = scale - 1 - x
                y = scale - 1 - y
            x, y = y, x
        x += scale * rx
        y += scale * ry
        distance //= 4
        scale *= 2
    return x, y


def make_hilbert_pattern(
    name: str = "hilbert",
    width: float = 40.0,
    height: float = 40.0,
    order: int = 3,
) -> PatternGeometry:
    """Return a printable Hilbert centerline joined to a specimen wall."""
    if order < 1 or order > 5:
        raise ValueError("Hilbert order must lie in [1, 5]")
    side = 1 << order
    margin_x = width / (2.0 * side)
    margin_y = height / (2.0 * side)
    points = [
        _point(
            margin_x + (width - 2.0 * margin_x) * x / (side - 1),
            margin_y + (height - 2.0 * margin_y) * y / (side - 1),
        )
        for x, y in (
            _hilbert_xy(order, index) for index in range(side * side)
        )
    ]
    segments = list(zip(points[:-1], points[1:]))
    segments.extend(
        [
            (_point(0.0, 0.0), _point(width, 0.0)),
            (_point(width, 0.0), _point(width, height)),
            (_point(width, height), _point(0.0, height)),
            (_point(0.0, height), _point(0.0, 0.0)),
            (points[0], _point(points[0][0], 0.0)),
            (points[-1], _point(points[-1][0], 0.0)),
        ]
    )
    return make_geometry_from_segments(name, segments, width)


def make_zigzag_pattern(
    angle_degrees: int,
    name: str | None = None,
    width: float = 40.0,
    height: float = 40.0,
    lines: int = 8,
) -> PatternGeometry:
    """Return a raster-like 90° or 45° centerline comparison pattern."""
    if angle_degrees not in {45, 90}:
        raise ValueError("zigzag angle must be 45 or 90 degrees")
    if lines < 3:
        raise ValueError("zigzag lines must be at least 3")
    name = name or f"zigzag_{angle_degrees}"
    if angle_degrees == 90:
        x_values = [
            width * (index + 1) / (lines + 1) for index in range(lines)
        ]
        segments: List[Segment] = [
            (_point(0.0, 0.0), _point(width, 0.0)),
            (_point(width, 0.0), _point(width, height)),
            (_point(width, height), _point(0.0, height)),
            (_point(0.0, height), _point(0.0, 0.0)),
        ]
        segments.extend(
            (_point(x, 0.0), _point(x, height)) for x in x_values
        )
        return make_geometry_from_segments(name, segments, width)

    # Parallel +45° rasters clipped to the square. The perimeter represents
    # the specimen wall and joins every raster endpoint without inventing
    # interior cross-bracing.
    segments = [
        ((_point(0.0, 0.0)), (_point(width, 0.0))),
        ((_point(width, 0.0)), (_point(width, height))),
        ((_point(width, height)), (_point(0.0, height))),
        ((_point(0.0, height)), (_point(0.0, 0.0))),
    ]
    offsets = np.linspace(-width, height, lines + 2)[1:-1]
    for offset in offsets:
        candidates: List[Point] = []
        for x, y in (
            (0.0, offset),
            (width, width + offset),
            (-offset, 0.0),
            (height - offset, height),
        ):
            if -1.0e-9 <= x <= width + 1.0e-9 and -1.0e-9 <= y <= height + 1.0e-9:
                point = _point(
                    min(width, max(0.0, x)),
                    min(height, max(0.0, y)),
                )
                if point not in candidates:
                    candidates.append(point)
        if len(candidates) == 2:
            segments.append((candidates[0], candidates[1]))
    return make_geometry_from_segments(name, segments, width)


def make_staggered_diamond_variant(
    name: str,
    width: float = 40.0,
    height: float = 40.0,
    cells: int = 3,
    offset_fraction: float = 0.5,
) -> PatternGeometry:
    if cells < 2:
        raise ValueError("staggered-diamond cells must be at least 2")
    if not 0.1 <= offset_fraction <= 0.9:
        raise ValueError("offset_fraction must lie in [0.1, 0.9]")
    dx, dy = width / cells, height / cells
    segments: List[Segment] = []
    rows: List[List[Point]] = []
    for iy in range(cells + 1):
        if iy % 2 == 0:
            x_values = [ix * dx for ix in range(cells + 1)]
        else:
            x_values = [0.0]
            x_values.extend(
                (ix + offset_fraction) * dx for ix in range(cells)
            )
            x_values.append(width)
        rows.append([_point(x, iy * dy) for x in x_values])
    for lower, upper in zip(rows[:-1], rows[1:]):
        for start in lower:
            nearest = sorted(
                upper, key=lambda end: (abs(end[0] - start[0]), end[0])
            )[:2]
            for end in nearest:
                segments.append((start, end))
        for end in upper:
            nearest = min(lower, key=lambda start: abs(start[0] - end[0]))
            segments.append((nearest, end))
    for row in (rows[0], rows[-1]):
        segments.extend(zip(row[:-1], row[1:]))
    for x in (0.0, width):
        for iy in range(cells):
            segments.append(
                (_point(x, iy * dy), _point(x, (iy + 1) * dy))
            )
    return _finalize_pattern(name, segments, width)


def make_pattern(
    name: str, width: float = 40.0, height: float = 40.0, cells: int = 3
) -> PatternGeometry:
    if name not in PATTERN_NAMES:
        raise ValueError(f"unknown pattern {name!r}; choose from {PATTERN_NAMES}")
    dx, dy = width / cells, height / cells

    if name in {"grid", "triangular", "x_braced", "selective_x"}:
        segments = _grid_segments(width, height, cells)
        for ix in range(cells):
            for iy in range(cells):
                x0, x1 = ix * dx, (ix + 1) * dx
                y0, y1 = iy * dy, (iy + 1) * dy
                if name == "triangular":
                    if (ix + iy) % 2:
                        segments.append((_point(x0, y1), _point(x1, y0)))
                    else:
                        segments.append((_point(x0, y0), _point(x1, y1)))
                elif name == "x_braced" or (
                    name == "selective_x" and (ix + iy) % 2 == 0
                ):
                    center = _point((x0 + x1) / 2.0, (y0 + y1) / 2.0)
                    for corner in (
                        _point(x0, y0),
                        _point(x1, y0),
                        _point(x0, y1),
                        _point(x1, y1),
                    ):
                        segments.append((corner, center))
        return _finalize_pattern(name, segments, width)

    if name == "staggered_diamond":
        return make_staggered_diamond_variant(
            name, width, height, cells, 0.5
        )

    if name == "hilbert":
        return make_hilbert_pattern(name, width, height, order=3)

    if name in {"zigzag_90", "zigzag_45"}:
        return make_zigzag_pattern(
            int(name.rsplit("_", 1)[1]),
            name,
            width,
            height,
            lines=8,
        )

    if name == "bowtie":
        segments = []
        # A vertical center spine is tied to both side rails by alternating
        # hourglass cells. The topology is intentionally sparse and unusual,
        # but every load path remains continuous and printable.
        for x in (0.0, width / 2.0, width):
            for iy in range(cells):
                segments.append(
                    (
                        _point(x, iy * dy),
                        _point(x, (iy + 1) * dy),
                    )
                )
        for y in (0.0, height):
            segments.extend(
                [
                    (_point(0.0, y), _point(width / 2.0, y)),
                    (_point(width / 2.0, y), _point(width, y)),
                ]
            )
        for iy in range(cells):
            y0, y1 = iy * dy, (iy + 1) * dy
            center = _point(width / 2.0, (y0 + y1) / 2.0)
            for corner in (
                _point(0.0, y0),
                _point(0.0, y1),
                _point(width, y0),
                _point(width, y1),
            ):
                segments.append((corner, center))
        return _finalize_pattern(name, segments, width)

    if name == "reentrant":
        segments = []
        # Horizontal ribs make adjacent cells share printable material.
        for iy in range(cells + 1):
            for ix in range(cells):
                segments.append(
                    (
                        _point(ix * dx, iy * dy),
                        _point((ix + 1) * dx, iy * dy),
                    )
                )
        inset = 0.35
        for ix in range(cells):
            for iy in range(cells):
                x0, x1 = ix * dx, (ix + 1) * dx
                y0, y1 = iy * dy, (iy + 1) * dy
                middle_y = (y0 + y1) / 2.0
                left = _point(x0 + inset * dx, middle_y)
                right = _point(x1 - inset * dx, middle_y)
                segments.extend(
                    [
                        (_point(x0, y0), left),
                        (left, _point(x0, y1)),
                        (_point(x1, y0), right),
                        (right, _point(x1, y1)),
                        (left, right),
                    ]
                )
        return _finalize_pattern(name, segments, width)

    if name == "concentric_diamond":
        segments = []
        # Printable loading rails and side rails.
        perimeter = [
            _point(0, 0),
            _point(width, 0),
            _point(width, height),
            _point(0, height),
        ]
        for index, start in enumerate(perimeter):
            segments.append((start, perimeter[(index + 1) % len(perimeter)]))
        center = _point(width / 2.0, height / 2.0)
        outer = [
            _point(width / 2.0, 0),
            _point(width, height / 2.0),
            _point(width / 2.0, height),
            _point(0, height / 2.0),
        ]
        inner = [
            _point(width / 2.0, height * 0.25),
            _point(width * 0.75, height / 2.0),
            _point(width / 2.0, height * 0.75),
            _point(width * 0.25, height / 2.0),
        ]
        for ring in (outer, inner):
            for index, start in enumerate(ring):
                segments.append((start, ring[(index + 1) % len(ring)]))
        for outer_point, inner_point in zip(outer, inner):
            segments.append((outer_point, inner_point))
            segments.append((inner_point, center))
        return _finalize_pattern(name, segments, width)

    # A toolpath-like wavy-column pattern with cross ties at each cell height.
    segments = []
    samples_per_cell = 4
    amplitude = 0.18 * dx
    column_points: List[List[Point]] = []
    for ix in range(cells + 1):
        points = []
        phase = 0.0 if ix % 2 == 0 else math.pi
        for sample in range(cells * samples_per_cell + 1):
            y = height * sample / (cells * samples_per_cell)
            x = ix * dx + amplitude * math.sin(math.pi * y / dy + phase)
            if ix in (0, cells) or sample in (0, cells * samples_per_cell):
                x = ix * dx
            points.append(_point(x, y))
        column_points.append(points)
        segments.extend(zip(points[:-1], points[1:]))
    for iy in range(cells + 1):
        sample = iy * samples_per_cell
        for ix in range(cells):
            segments.append(
                (column_points[ix][sample], column_points[ix + 1][sample])
            )
    return _finalize_pattern(name, segments, width)


def nominal_strut_width(
    geometry: PatternGeometry,
    domain_width_mm: float,
    domain_height_mm: float,
    relative_density: float,
) -> float:
    if not 0.0 < relative_density < 1.0:
        raise ValueError("relative density must lie between zero and one")
    return (
        relative_density
        * domain_width_mm
        * domain_height_mm
        / geometry.total_length_mm
    )


def _nset(name: str, node_ids: Sequence[int]) -> str:
    lines = [f"*NSET,NSET={name}"]
    for start in range(0, len(node_ids), 16):
        lines.append(",".join(str(value) for value in node_ids[start : start + 16]))
    return "\n".join(lines)


def _sets_for_load_case(
    geometry: PatternGeometry,
    load_case: str,
    domain_width_mm: float,
    domain_height_mm: float,
) -> Tuple[List[int], List[int], int, int]:
    tolerance = max(domain_height_mm, 1.0) * 1.0e-9
    if load_case == "bending":
        supports = [
            index
            for index, (x, _) in enumerate(geometry.nodes, start=1)
            if abs(x) <= tolerance or abs(x - domain_width_mm) <= tolerance
        ]
        loads = [
            index
            for index, (x, _) in enumerate(geometry.nodes, start=1)
            if abs(x - domain_width_mm / 2.0) <= tolerance
        ]
        left = [
            index
            for index in supports
            if abs(geometry.nodes[index - 1][0]) <= tolerance
        ]
        right = [
            index
            for index in supports
            if abs(geometry.nodes[index - 1][0] - domain_width_mm) <= tolerance
        ]
        anchor = min(left, key=lambda index: geometry.nodes[index - 1][1])
        anchor_two = min(right, key=lambda index: geometry.nodes[index - 1][1])
    else:
        supports = [
            index
            for index, (_, y) in enumerate(geometry.nodes, start=1)
            if abs(y) <= tolerance
        ]
        loads = [
            index
            for index, (_, y) in enumerate(geometry.nodes, start=1)
            if abs(y - domain_height_mm) <= tolerance
        ]
        anchor = min(supports, key=lambda index: geometry.nodes[index - 1][0])
        anchor_two = anchor
    if not supports or not loads:
        raise ValueError(
            f"{geometry.name} does not connect {load_case} loading boundaries"
        )
    return supports, loads, anchor, anchor_two


def _node_lines(
    geometry: PatternGeometry,
    load_case: str,
    domain_width_mm: float,
    domain_height_mm: float,
) -> str:
    lines = []
    for index, (x, y) in enumerate(geometry.nodes, start=1):
        if (
            load_case == "compression"
            and 0.0 < x < domain_width_mm
            and 0.0 < y < domain_height_mm
        ):
            # Small deterministic as-printed imperfection to activate P-delta
            # response without inventing a random defect field.
            x += (
                0.0005
                * domain_width_mm
                * math.sin(math.pi * y / domain_height_mm)
                * math.sin(2.0 * math.pi * x / domain_width_mm)
            )
        lines.append(f"{index}, {x:.9g}, {y:.9g}, 0.")
    return "\n".join(lines)


def _deck_preamble(
    geometry: PatternGeometry,
    load_case: str,
    strut_width_mm: float,
    depth_mm: float,
    youngs_modulus_MPa: float,
    poisson_ratio: float,
    domain_width_mm: float,
    domain_height_mm: float,
    density_tonnes_mm3: float,
) -> Tuple[str, List[int], List[int], int, int]:
    nodes = _node_lines(
        geometry, load_case, domain_width_mm, domain_height_mm
    )
    elements = "\n".join(
        f"{index}, {first}, {second}"
        for index, (first, second) in enumerate(geometry.edges, start=1)
    )
    supports, loads, anchor, anchor_two = _sets_for_load_case(
        geometry, load_case, domain_width_mm, domain_height_mm
    )
    preamble = f"""** Explicit printable-centerline infill screening model.
*HEADING
{geometry.name} {load_case} equal-material infill benchmark
*NODE,NSET=NALL
{nodes}
*ELEMENT,TYPE=B31,ELSET=INFILL
{elements}
{_nset("SUPPORTS", supports)}
{_nset("LOAD", loads)}
*NSET,NSET=ANCHOR
{anchor}
*NSET,NSET=ANCHOR_TWO
{anchor_two}
*MATERIAL,NAME=PLA_REPORTED
*DENSITY
{density_tonnes_mm3:.9g}
*ELASTIC
{youngs_modulus_MPa:.9g}, {poisson_ratio:.9g}
*BEAM SECTION,ELSET=INFILL,MATERIAL=PLA_REPORTED,SECTION=RECT
{depth_mm:.9g}, {strut_width_mm:.9g}
0., 0., 1.
"""
    return preamble, supports, loads, anchor, anchor_two


def render_load_case_deck(
    geometry: PatternGeometry,
    load_case: str,
    strut_width_mm: float,
    depth_mm: float,
    youngs_modulus_MPa: float,
    poisson_ratio: float,
    displacement_mm: float,
    domain_width_mm: float,
    domain_height_mm: float,
    density_tonnes_mm3: float,
) -> str:
    if load_case not in {"tension", "shear", "bending", "compression"}:
        raise ValueError(f"unsupported displacement load case: {load_case}")
    preamble, _, _, _, _ = _deck_preamble(
        geometry,
        load_case,
        strut_width_mm,
        depth_mm,
        youngs_modulus_MPa,
        poisson_ratio,
        domain_width_mm,
        domain_height_mm,
        density_tonnes_mm3,
    )
    if load_case == "tension":
        boundaries = f"""NALL,3,5,0.
SUPPORTS,2,2,0.
ANCHOR,1,1,0.
LOAD,2,2,{displacement_mm:.9g}"""
        step = "*STEP,NLGEOM=NO\n*STATIC,DIRECT\n1., 1."
    elif load_case == "compression":
        boundaries = f"""NALL,3,5,0.
SUPPORTS,2,2,0.
ANCHOR,1,1,0.
LOAD,2,2,{-abs(displacement_mm):.9g}"""
        step = "*STEP,NLGEOM=YES\n*STATIC\n0.05, 1., 1.e-06, 0.1"
    elif load_case == "shear":
        boundaries = f"""NALL,3,5,0.
SUPPORTS,1,2,0.
LOAD,2,2,0.
LOAD,1,1,{displacement_mm:.9g}"""
        step = "*STEP,NLGEOM=NO\n*STATIC,DIRECT\n1., 1."
    else:
        boundaries = f"""SUPPORTS,3,3,0.
ANCHOR,1,2,0.
ANCHOR_TWO,2,2,0.
LOAD,3,3,{-abs(displacement_mm):.9g}"""
        step = "*STEP,NLGEOM=NO\n*STATIC,DIRECT\n1., 1."
    return f"""{preamble}{step}
*BOUNDARY
{boundaries}
*NODE PRINT,NSET=LOAD
U
*NODE PRINT,NSET=SUPPORTS
RF
*EL PRINT,ELSET=INFILL
S
*NODE FILE,NSET=LOAD
U
*EL FILE
S
*END STEP
"""


def render_beam_deck(
    geometry: PatternGeometry,
    strut_width_mm: float,
    depth_mm: float,
    youngs_modulus_MPa: float,
    poisson_ratio: float,
    displacement_mm: float,
    domain_height_mm: float,
    density_tonnes_mm3: float,
) -> str:
    """Backward-compatible tensile deck used by the original benchmark API."""
    domain_width_mm = max(x for x, _ in geometry.nodes)
    return render_load_case_deck(
        geometry,
        "tension",
        strut_width_mm,
        depth_mm,
        youngs_modulus_MPa,
        poisson_ratio,
        displacement_mm,
        domain_width_mm,
        domain_height_mm,
        density_tonnes_mm3,
    )


def render_buckling_deck(
    geometry: PatternGeometry,
    strut_width_mm: float,
    depth_mm: float,
    youngs_modulus_MPa: float,
    poisson_ratio: float,
    domain_width_mm: float,
    domain_height_mm: float,
    density_tonnes_mm3: float,
) -> str:
    preamble, _, loads, _, _ = _deck_preamble(
        geometry,
        "buckling",
        strut_width_mm,
        depth_mm,
        youngs_modulus_MPa,
        poisson_ratio,
        domain_width_mm,
        domain_height_mm,
        density_tonnes_mm3,
    )
    load_each = -1.0 / len(loads)
    cloads = "\n".join(f"{node},2,{load_each:.12g}" for node in loads)
    return f"""{preamble}*BOUNDARY
NALL,3,5,0.
SUPPORTS,2,2,0.
ANCHOR,1,1,0.
*STEP
*BUCKLE
1, 0.01
*CLOAD
{cloads}
*NODE FILE
U
*END STEP
"""


def _maximum_von_mises(curve: Mapping[str, np.ndarray]) -> float | None:
    stress = curve.get("frd_stress")
    if stress is None or stress.ndim != 2 or stress.shape[1] < 6:
        return None
    values = stress[:, :6]
    values = values[np.isfinite(values).all(axis=1)]
    if values.size == 0:
        return None
    sx, sy, sz, sxy, syz, szx = values.T
    von_mises = np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
    )
    return float(np.max(von_mises))


def _parse_buckling_factor(path: Path) -> float:
    if not path.exists():
        raise ValueError("buckling analysis did not create a .dat file")
    text = path.read_text(encoding="utf-8", errors="replace")
    # CalculiX 2.23 renders the title with spaces between every letter and
    # places "MODE NO", "BUCKLING", and "FACTOR" on separate lines.
    # Anchor the parse on MODE NO so unrelated numbered output cannot match.
    heading = re.search(r"^\s*MODE\s+NO\b", text, flags=re.IGNORECASE | re.MULTILINE)
    if not heading or "BUCKLING" not in text[heading.start() :].upper():
        raise ValueError("no buckling-factor table found in CalculiX output")
    table = text[heading.end() : heading.end() + 1000]
    rows = re.findall(
        r"^\s*\d+\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)",
        table,
        flags=re.MULTILINE,
    )
    if not rows:
        raise ValueError("buckling-factor table contained no eigenvalue")
    return float(rows[0].replace("D", "E").replace("d", "e"))


def _run_buckling_deck(
    deck_path: Path,
    runs_dir: Path,
    run_id: str,
    timeout_seconds: float = 120.0,
) -> Tuple[str, float | None, Path, str | None]:
    executable = find_ccx("ccx")
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists():
        run_dir = Path(runs_dir) / f"{run_id}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    if executable is None:
        return "executable_missing", None, run_dir, "CalculiX was not found"
    (run_dir / "model.inp").write_text(
        deck_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    start = time.monotonic()
    try:
        process = subprocess.run(
            [str(executable), "model"],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return (
            "timeout",
            None,
            run_dir,
            f"buckling solver exceeded {timeout_seconds:g} seconds: {error}",
        )
    runtime = time.monotonic() - start
    (run_dir / "stdout.txt").write_text(process.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        return (
            "solver_failed",
            None,
            run_dir,
            f"CalculiX returned {process.returncode} after {runtime:.3f} s",
        )
    try:
        factor = _parse_buckling_factor(run_dir / "model.dat")
    except ValueError as error:
        return "parse_failed", None, run_dir, str(error)
    return "ok", factor, run_dir, None


def render_svg(
    geometry: PatternGeometry,
    path: Path,
    strut_width_mm: float,
    domain_width_mm: float,
    domain_height_mm: float,
) -> None:
    scale = 10.0
    margin = 24.0
    width = domain_width_mm * scale + 2.0 * margin
    height = domain_height_mm * scale + 2.0 * margin
    lines = []
    for first, second in geometry.edges:
        x1, y1 = geometry.nodes[first - 1]
        x2, y2 = geometry.nodes[second - 1]
        lines.append(
            f'<line x1="{margin + x1 * scale:.3f}" '
            f'y1="{margin + (domain_height_mm - y1) * scale:.3f}" '
            f'x2="{margin + x2 * scale:.3f}" '
            f'y2="{margin + (domain_height_mm - y2) * scale:.3f}"/>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.3f} {height:.3f}">
  <rect width="100%" height="100%" fill="#f7f5ef"/>
  <g stroke="#e36b2c" stroke-width="{strut_width_mm * scale:.3f}"
     stroke-linecap="round" stroke-linejoin="round">
    {''.join(lines)}
  </g>
  <rect x="{margin}" y="{margin}" width="{domain_width_mm * scale:.3f}"
        height="{domain_height_mm * scale:.3f}" fill="none" stroke="#333"
        stroke-width="1"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def render_scad(
    geometry: PatternGeometry,
    path: Path,
    strut_width_mm: float,
    depth_mm: float,
    domain_width_mm: float | None = None,
    domain_height_mm: float | None = None,
) -> None:
    pairs = []
    for first, second in geometry.edges:
        start, end = geometry.nodes[first - 1], geometry.nodes[second - 1]
        pairs.append(f"  [[{start[0]:.9g},{start[1]:.9g}],[{end[0]:.9g},{end[1]:.9g}]]")
    segments = ",\n".join(pairs)
    domain_width_mm = (
        max(x for x, _ in geometry.nodes)
        if domain_width_mm is None
        else domain_width_mm
    )
    domain_height_mm = (
        max(y for _, y in geometry.nodes)
        if domain_height_mm is None
        else domain_height_mm
    )
    text = f"""// Generated centerline geometry for a printable comparison coupon.
// Nominal beam-model width: {strut_width_mm:.6g} mm
$fn = 32;
strut_width = {strut_width_mm:.9g};
coupon_depth = {depth_mm:.9g};
segments = [
{segments}
];

linear_extrude(height=coupon_depth)
intersection() {{
  square([{domain_width_mm:.9g}, {domain_height_mm:.9g}]);
  union() {{
    for (segment = segments)
      hull() {{
        translate(segment[0]) circle(d=strut_width);
        translate(segment[1]) circle(d=strut_width);
      }}
  }}
}}
"""
    path.write_text(text, encoding="utf-8")


def run_benchmark(
    output_dir: Path,
    runs_dir: Path,
    properties_path: Path = DEFAULT_PROPERTIES,
    patterns: Sequence[str] = PATTERN_NAMES,
    load_cases: Sequence[str] = LOAD_CASES,
    domain_width_mm: float = 40.0,
    domain_height_mm: float = 40.0,
    depth_mm: float = 4.0,
    relative_density: float = 0.35,
    tensile_strain: float = 0.01,
    geometries: Mapping[str, PatternGeometry] | None = None,
    strut_width_overrides: Mapping[str, float] | None = None,
) -> List[PatternResult]:
    properties = json.loads(Path(properties_path).read_text(encoding="utf-8"))
    elastic = properties["screening_isotropic"]
    strength = properties["reported_strength_reference"]
    youngs_modulus = float(elastic["E_MPa"])
    poisson = float(elastic["nu"])
    density = float(properties["density_tonnes_mm3"])
    tensile_reference = float(strength["overall_tensile_UTS_MPa"])
    displacement = domain_height_mm * tensile_strain

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "patterns").mkdir(exist_ok=True)
    (output_dir / "curves").mkdir(exist_ok=True)
    (output_dir / "decks").mkdir(exist_ok=True)
    unknown_cases = set(load_cases) - set(LOAD_CASES)
    if unknown_cases:
        raise ValueError(f"unknown load cases: {sorted(unknown_cases)}")
    runner = CalculixRunner(
        runs_dir,
        executable="ccx",
        timeout_seconds=120.0,
        keep_successful=True,
    )
    results: List[PatternResult] = []

    for name in patterns:
        geometry = (
            geometries[name]
            if geometries is not None and name in geometries
            else make_pattern(name, domain_width_mm, domain_height_mm)
        )
        if strut_width_overrides is not None and name in strut_width_overrides:
            strut_width = float(strut_width_overrides[name])
            if not math.isfinite(strut_width) or strut_width <= 0.0:
                raise ValueError(
                    f"strut-width override for {name} must be positive"
                )
        else:
            strut_width = nominal_strut_width(
                geometry,
                domain_width_mm,
                domain_height_mm,
                relative_density,
            )
        volume = geometry.total_length_mm * strut_width * depth_mm
        mass_g = volume * density * 1.0e6
        actual_relative_density = (
            geometry.total_length_mm
            * strut_width
            / (domain_width_mm * domain_height_mm)
        )
        render_svg(
            geometry,
            output_dir / "patterns" / f"{name}.svg",
            strut_width,
            domain_width_mm,
            domain_height_mm,
        )
        render_scad(
            geometry,
            output_dir / "patterns" / f"{name}.scad",
            strut_width,
            depth_mm,
            domain_width_mm,
            domain_height_mm,
        )
        for load_case in load_cases:
            if load_case == "buckling":
                deck = render_buckling_deck(
                    geometry,
                    strut_width,
                    depth_mm,
                    youngs_modulus,
                    poisson,
                    domain_width_mm,
                    domain_height_mm,
                    density,
                )
                deck_path = output_dir / "decks" / f"{name}-{load_case}.inp"
                deck_path.write_text(deck, encoding="utf-8")
                status, factor, run_dir, error = _run_buckling_deck(
                    deck_path, runs_dir, f"infill-{name}-{load_case}"
                )
                critical_load = None if factor is None else abs(float(factor))
                results.append(
                    PatternResult(
                        name,
                        load_case,
                        status,
                        len(geometry.nodes),
                        len(geometry.edges),
                        geometry.total_length_mm,
                        strut_width,
                        actual_relative_density,
                        mass_g,
                        0.0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        critical_load,
                        (
                            None
                            if critical_load is None
                            else critical_load / mass_g
                        ),
                        str(run_dir.resolve()),
                        error,
                    )
                )
                continue

            deck = render_load_case_deck(
                geometry,
                load_case,
                strut_width,
                depth_mm,
                youngs_modulus,
                poisson,
                displacement,
                domain_width_mm,
                domain_height_mm,
                density,
            )
            deck_path = output_dir / "decks" / f"{name}-{load_case}.inp"
            deck_path.write_text(deck, encoding="utf-8")
            solver = runner.run(
                deck_path, {}, run_id=f"infill-{name}-{load_case}"
            )
            if not solver.ok:
                results.append(
                    PatternResult(
                        name,
                        load_case,
                        solver.status,
                        len(geometry.nodes),
                        len(geometry.edges),
                        geometry.total_length_mm,
                        strut_width,
                        actual_relative_density,
                        mass_g,
                        displacement,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        str(solver.run_dir.resolve()),
                        solver.error,
                    )
                )
                continue

            endpoint_displacement = float(abs(solver.curve["displacement"][-1]))
            endpoint_force = float(abs(solver.curve["reaction_force"][-1]))
            stiffness = endpoint_force / endpoint_displacement
            if load_case == "bending":
                section_inertia = domain_height_mm * depth_mm**3 / 12.0
                effective_modulus = (
                    stiffness * domain_width_mm**3 / (48.0 * section_inertia)
                )
                reference_length = domain_width_mm
            else:
                effective_modulus = (
                    endpoint_force
                    * domain_height_mm
                    / (
                        domain_width_mm
                        * depth_mm
                        * endpoint_displacement
                    )
                )
                reference_length = domain_height_mm
            maximum_stress = _maximum_von_mises(solver.curve)
            utilization = (
                None
                if maximum_stress is None
                else maximum_stress / tensile_reference
            )
            displacements = np.abs(solver.curve["displacement"])
            forces = np.abs(solver.curve["reaction_force"])
            if len(displacements) == 1:
                displacements = np.linspace(0.0, endpoint_displacement, 21)
                forces = endpoint_force * displacements / endpoint_displacement
            curve_path = output_dir / "curves" / f"{name}-{load_case}.csv"
            with curve_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "pattern",
                        "load_case",
                        "applied_displacement_mm",
                        "reaction_force_N",
                        "normalized_displacement",
                        "nominal_force_per_area_MPa",
                    ],
                )
                writer.writeheader()
                for current, force in zip(displacements, forces):
                    writer.writerow(
                        {
                            "pattern": name,
                            "load_case": load_case,
                            "applied_displacement_mm": current,
                            "reaction_force_N": force,
                            "normalized_displacement": current
                            / reference_length,
                            "nominal_force_per_area_MPa": force
                            / (domain_width_mm * depth_mm),
                        }
                    )
            results.append(
                PatternResult(
                    name,
                    load_case,
                    "ok",
                    len(geometry.nodes),
                    len(geometry.edges),
                    geometry.total_length_mm,
                    strut_width,
                    actual_relative_density,
                    mass_g,
                    endpoint_displacement,
                    endpoint_force,
                    stiffness,
                    effective_modulus,
                    stiffness / mass_g,
                    maximum_stress,
                    utilization,
                    None,
                    None,
                    str(solver.run_dir.resolve()),
                )
            )

    fields = list(asdict(results[0]))
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    (output_dir / "summary.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    _write_multi_load_ranking(output_dir, results, patterns, load_cases)
    _write_physical_test_plan(
        output_dir,
        results,
        properties,
        domain_width_mm,
        domain_height_mm,
        depth_mm,
        replicates=3,
        top_patterns=3,
    )
    (output_dir / "screening_isotropic_material.inp").write_text(
        (
            "*MATERIAL, NAME=PLA_REPORTED_SCREENING\n"
            f"*DENSITY\n{density:.9g}\n"
            f"*ELASTIC\n{youngs_modulus:.9g}, {poisson:.9g}\n"
        ),
        encoding="utf-8",
    )
    (output_dir / "tds_orthotropic_initialization_prior.inp").write_text(
        material_card(PRIORS, name="PLA_TDS_PRIOR"),
        encoding="utf-8",
    )
    _write_report(
        output_dir,
        results,
        properties,
        domain_width_mm,
        domain_height_mm,
        depth_mm,
        relative_density,
        tensile_strain,
    )
    return results


def _write_multi_load_ranking(
    output_dir: Path,
    results: Sequence[PatternResult],
    patterns: Sequence[str],
    load_cases: Sequence[str],
) -> None:
    metric_by_case: Dict[str, Dict[str, float]] = {}
    for load_case in load_cases:
        values: Dict[str, float] = {}
        for result in results:
            if result.pattern not in patterns or result.load_case != load_case:
                continue
            if load_case == "buckling":
                metric = result.critical_buckling_load_N_per_g
            else:
                metric = result.specific_stiffness_N_per_mm_g
            if result.status == "ok" and metric is not None and metric > 0.0:
                values[result.pattern] = float(metric)
        if values:
            maximum = max(values.values())
            metric_by_case[load_case] = {
                pattern: value / maximum for pattern, value in values.items()
            }
    rows = []
    for pattern in patterns:
        normalized = {
            load_case: metric_by_case.get(load_case, {}).get(pattern)
            for load_case in load_cases
        }
        available = [value for value in normalized.values() if value is not None]
        score = (
            None
            if len(available) != len(load_cases)
            else float(np.prod(available) ** (1.0 / len(available)))
        )
        rows.append(
            {
                "pattern": pattern,
                "balanced_geometric_mean": score,
                **{
                    f"{load_case}_normalized": normalized[load_case]
                    for load_case in load_cases
                },
            }
        )
    rows.sort(
        key=lambda row: (
            row["balanced_geometric_mean"] is not None,
            row["balanced_geometric_mean"] or -1.0,
        ),
        reverse=True,
    )
    with (output_dir / "multi_load_ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_physical_test_plan(
    output_dir: Path,
    results: Sequence[PatternResult],
    properties: Mapping[str, object],
    width: float,
    height: float,
    depth: float,
    replicates: int,
    top_patterns: int,
) -> None:
    """Create a matched-coupon plan that directly mirrors the FEA matrix."""
    with (output_dir / "multi_load_ranking.csv").open(encoding="utf-8") as handle:
        ranking = [
            row
            for row in csv.DictReader(handle)
            if row.get("balanced_geometric_mean")
        ]
    selected = [row["pattern"] for row in ranking[:top_patterns]]
    lookup = {(result.pattern, result.load_case): result for result in results}
    fields = [
        "specimen_id",
        "pattern",
        "load_case",
        "replicate",
        "source_scad",
        "material",
        "build_orientation",
        "coupon_width_mm",
        "coupon_height_mm",
        "coupon_depth_mm",
        "strut_width_mm",
        "nominal_relative_density",
        "nominal_mass_g",
        "measure",
        "simulation_metric",
        "simulation_value",
        "simulation_run_directory",
    ]
    rows = []
    for pattern in selected:
        cases = [
            result
            for result in results
            if result.pattern == pattern and result.status == "ok"
        ]
        for result in cases:
            if result.load_case == "buckling":
                metric = "critical_buckling_load_N"
                value = result.critical_buckling_load_N
                measure = "force_N, crosshead_displacement_mm, failure_mode"
            else:
                metric = "stiffness_N_per_mm"
                value = result.stiffness_N_per_mm
                measure = "force_N, crosshead_displacement_mm, failure_mode"
            for replicate in range(1, replicates + 1):
                rows.append(
                    {
                        "specimen_id": (
                            f"{pattern}-{result.load_case}-r{replicate:02d}"
                        ),
                        "pattern": pattern,
                        "load_case": result.load_case,
                        "replicate": replicate,
                        "source_scad": (
                            f"patterns/{pattern}.scad"
                        ),
                        "material": properties["material"],
                        "build_orientation": (
                            "flat; keep slicer profile and bead orientation "
                            "identical across every specimen"
                        ),
                        "coupon_width_mm": width,
                        "coupon_height_mm": height,
                        "coupon_depth_mm": depth,
                        "strut_width_mm": result.strut_width_mm,
                        "nominal_relative_density": (
                            result.nominal_relative_density
                        ),
                        "nominal_mass_g": result.nominal_mass_g,
                        "measure": measure,
                        "simulation_metric": metric,
                        "simulation_value": value,
                        "simulation_run_directory": result.run_directory,
                    }
                )
    with (output_dir / "physical_test_plan.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    output_dir: Path,
    results: Sequence[PatternResult],
    properties: Mapping[str, object],
    width: float,
    height: float,
    depth: float,
    relative_density: float,
    strain: float,
) -> None:
    lines = [
        "# Reported Bambu PLA infill screening",
        "",
        (
            f"All patterns use a {width:g} × {height:g} mm centerline envelope "
            f"extruded {depth:g} mm, "
            f"{relative_density:.0%} nominal line-volume fraction, and "
            f"{strain:.1%} prescribed deformation for displacement-controlled cases."
        ),
        (
            f"The beam material uses E = "
            f"{properties['screening_isotropic']['E_MPa']:g} MPa and "
            f"ν = {properties['screening_isotropic']['nu']:g}."
        ),
    ]
    load_cases = list(dict.fromkeys(result.load_case for result in results))
    for load_case in load_cases:
        successful = [
            result
            for result in results
            if result.load_case == load_case and result.status == "ok"
        ]
        if load_case == "buckling":
            ranking = sorted(
                (
                    result
                    for result in successful
                    if result.critical_buckling_load_N_per_g is not None
                ),
                key=lambda result: float(result.critical_buckling_load_N_per_g),
                reverse=True,
            )
            lines.extend(
                [
                    "",
                    "## Linear buckling",
                    "",
                    "| Rank | Pattern | Critical load (N) | Critical load (N/g) |",
                    "|---:|---|---:|---:|",
                ]
            )
            for rank, result in enumerate(ranking, start=1):
                lines.append(
                    f"| {rank} | {result.pattern} | "
                    f"{result.critical_buckling_load_N:.3f} | "
                    f"{result.critical_buckling_load_N_per_g:.3f} |"
                )
            continue
        ranking = sorted(
            (
                result
                for result in successful
                if result.specific_stiffness_N_per_mm_g is not None
            ),
            key=lambda result: float(result.specific_stiffness_N_per_mm_g),
            reverse=True,
        )
        lines.extend(
            [
                "",
                f"## {load_case.replace('_', ' ').title()}",
                "",
                "| Rank | Pattern | Stiffness (N/mm) | Apparent modulus (MPa) | "
                "Specific stiffness (N/mm/g) | Max von Mises (MPa) | "
                "UTS-reference utilization |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, result in enumerate(ranking, start=1):
            stress = (
                "n/a"
                if result.maximum_von_mises_MPa is None
                else f"{result.maximum_von_mises_MPa:.3f}"
            )
            utilization = (
                "n/a"
                if result.tensile_reference_utilization is None
                else f"{result.tensile_reference_utilization:.3f}"
            )
            lines.append(
                f"| {rank} | {result.pattern} | "
                f"{result.stiffness_N_per_mm:.3f} | "
                f"{result.effective_modulus_MPa:.3f} | "
                f"{result.specific_stiffness_N_per_mm_g:.3f} | {stress} | "
                f"{utilization} |"
            )
    ranking_path = output_dir / "multi_load_ranking.csv"
    if ranking_path.exists():
        with ranking_path.open(encoding="utf-8") as handle:
            balanced = list(csv.DictReader(handle))
        lines.extend(
            [
                "",
                "## Balanced multi-load index",
                "",
                "The index is the geometric mean of each pattern's normalized "
                "specific result in every completed load case. Higher is better.",
                "",
                "| Rank | Pattern | Balanced index |",
                "|---:|---|---:|",
            ]
        )
        for rank, row in enumerate(balanced, start=1):
            value = row["balanced_geometric_mean"]
            lines.append(
                f"| {rank} | {row['pattern']} | "
                f"{'n/a' if not value else f'{float(value):.3f}'} |"
            )
    failures = [result for result in results if result.status != "ok"]
    if failures:
        lines.extend(["", "## Solver failures", ""])
        for result in failures:
            lines.append(
                f"- `{result.pattern}` / `{result.load_case}`: "
                f"{result.status}: {result.error}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Rankings compare geometry at equal nominal beam volume and mass.",
            "- Tension, shear, and bending are linear static. Compression uses "
            "geometric nonlinearity plus a 0.02 mm deterministic imperfection. "
            "Buckling is a planar linear-eigenvalue screen under a unit load.",
            "- Maximum stress is compared only with the reported mean tensile UTS as "
            "a screening reference; it is not a predicted failure load.",
            "- Contact, plasticity, fracture, bead separation, creep, fatigue, "
            "temperature dependence, and print defects are not yet represented.",
            "- OpenSCAD files under `patterns/` use the same centerlines and nominal "
            "strut widths, providing a direct route to printable comparison coupons.",
            "- `physical_test_plan.csv` schedules three independent specimens per "
            "load case for the top three balanced patterns; record the actual "
            "printed mass and all slicer settings alongside those measurements.",
            "- Before relying on a winner, remesh it with 3D solids, add contact and "
            "measured nonlinear material data, then print at least three matched "
            "coupons per pattern.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare explicit infill geometries using reported Bambu PLA stiffness"
    )
    parser.add_argument("--output", default="results/infill_benchmark")
    parser.add_argument("--runs-dir", default="runs/infill_benchmark")
    parser.add_argument("--properties", default=str(DEFAULT_PROPERTIES))
    parser.add_argument("--relative-density", type=float, default=0.35)
    parser.add_argument("--strain", type=float, default=0.01)
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=PATTERN_NAMES,
        default=list(PATTERN_NAMES),
    )
    parser.add_argument(
        "--load-cases",
        nargs="+",
        choices=LOAD_CASES,
        default=list(LOAD_CASES),
    )
    args = parser.parse_args()
    results = run_benchmark(
        Path(args.output),
        Path(args.runs_dir),
        properties_path=Path(args.properties),
        patterns=args.patterns,
        load_cases=args.load_cases,
        relative_density=args.relative_density,
        tensile_strain=args.strain,
    )
    successful = sum(result.status == "ok" for result in results)
    print(
        f"completed {successful}/{len(results)} infill simulations; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
