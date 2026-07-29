from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .calculix import CalculixRunner
from .infill_benchmark import (
    DEFAULT_PROPERTIES,
    PatternGeometry,
    make_pattern,
    nominal_strut_width,
    render_load_case_deck,
    render_scad,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEAM_RESULTS = PROJECT_ROOT / "results/infill_benchmark/summary.csv"
DEFAULT_PATTERNS = ("x_braced", "triangular", "grid")


@dataclass(frozen=True)
class SolidMesh:
    node_tags: np.ndarray
    coordinates: np.ndarray
    element_tags: np.ndarray
    connectivity: np.ndarray
    supports: Tuple[int, ...]
    loads: Tuple[int, ...]
    anchor: int
    anchor_two: int
    actual_volume_mm3: float
    through_thickness_layers: int
    mesh_path: Path


@dataclass
class SolidResult:
    pattern: str
    fidelity: str
    mesh_size_mm: float
    through_thickness_layers: int
    nodes: int
    elements: int
    strut_width_mm: float
    actual_relative_density: float
    actual_mass_g: float
    status: str
    reaction_force_N: float | None
    stiffness_N_per_mm: float | None
    specific_stiffness_N_per_mm_g: float | None
    apparent_modulus_MPa: float | None
    beam_stiffness_N_per_mm: float | None
    solid_to_beam_stiffness_ratio: float | None
    lower_E_stiffness_N_per_mm: float | None
    upper_E_stiffness_N_per_mm: float | None
    run_directory: str
    mesh_file: str
    error: str | None = None


def _require_gmsh():
    try:
        import gmsh
    except ImportError as error:
        raise RuntimeError(
            "3D solid validation requires the optional 'solid' dependencies; "
            "run `uv sync --extra solid`"
        ) from error
    return gmsh


def _capsule_union(
    gmsh,
    geometry: PatternGeometry,
    strut_width_mm: float,
    domain_width_mm: float,
    domain_height_mm: float,
):
    occ = gmsh.model.occ
    surfaces = []
    for first, second in geometry.edges:
        x1, y1 = geometry.nodes[first - 1]
        x2, y2 = geometry.nodes[second - 1]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.atan2(y2 - y1, x2 - x1)
        tag = occ.addRectangle(
            0.0, -strut_width_mm / 2.0, 0.0, length, strut_width_mm
        )
        occ.rotate([(2, tag)], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, angle)
        occ.translate([(2, tag)], x1, y1, 0.0)
        surfaces.append((2, tag))
    for x, y in geometry.nodes:
        surfaces.append(
            (
                2,
                occ.addDisk(
                    x,
                    y,
                    0.0,
                    strut_width_mm / 2.0,
                    strut_width_mm / 2.0,
                ),
            )
        )
    union, _ = occ.fuse(
        [surfaces[0]], surfaces[1:], removeObject=True, removeTool=True
    )
    clip = occ.addRectangle(
        0.0, 0.0, 0.0, domain_width_mm, domain_height_mm
    )
    clipped, _ = occ.intersect(
        union, [(2, clip)], removeObject=True, removeTool=True
    )
    occ.synchronize()
    if not clipped:
        raise RuntimeError("Gmsh produced an empty clipped pattern")
    return clipped


def solid_union_area(
    geometry: PatternGeometry,
    strut_width_mm: float,
    domain_width_mm: float = 40.0,
    domain_height_mm: float = 40.0,
) -> float:
    """Return exact printable capsule-union area after clipping to the coupon."""
    gmsh = _require_gmsh()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("area")
        surfaces = _capsule_union(
            gmsh,
            geometry,
            strut_width_mm,
            domain_width_mm,
            domain_height_mm,
        )
        return float(
            sum(gmsh.model.occ.getMass(2, tag) for _, tag in surfaces)
        )
    finally:
        gmsh.finalize()


def equal_solid_mass_strut_width(
    geometry: PatternGeometry,
    relative_density: float,
    domain_width_mm: float = 40.0,
    domain_height_mm: float = 40.0,
    relative_tolerance: float = 2.0e-4,
) -> float:
    """Solve for width using the unioned printable solid, including joints."""
    target_area = relative_density * domain_width_mm * domain_height_mm
    initial = nominal_strut_width(
        geometry, domain_width_mm, domain_height_mm, relative_density
    )
    lower = initial * 0.5
    upper = initial * 1.75
    lower_area = solid_union_area(
        geometry, lower, domain_width_mm, domain_height_mm
    )
    upper_area = solid_union_area(
        geometry, upper, domain_width_mm, domain_height_mm
    )
    if not lower_area < target_area < upper_area:
        raise RuntimeError("could not bracket the equal-solid-mass strut width")
    for _ in range(18):
        middle = (lower + upper) / 2.0
        area = solid_union_area(
            geometry, middle, domain_width_mm, domain_height_mm
        )
        if abs(area - target_area) / target_area <= relative_tolerance:
            return middle
        if area < target_area:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def generate_prism_mesh(
    geometry: PatternGeometry,
    strut_width_mm: float,
    mesh_size_mm: float,
    mesh_path: Path,
    domain_width_mm: float = 40.0,
    domain_height_mm: float = 40.0,
    depth_mm: float = 4.0,
) -> SolidMesh:
    """Mesh the clipped printable geometry with layered C3D6-compatible prisms."""
    gmsh = _require_gmsh()
    mesh_path = Path(mesh_path)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    layers = max(2, int(math.ceil(depth_mm / mesh_size_mm)))
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(f"{geometry.name}-solid")
        surfaces = _capsule_union(
            gmsh,
            geometry,
            strut_width_mm,
            domain_width_mm,
            domain_height_mm,
        )
        area = sum(
            gmsh.model.occ.getMass(2, tag) for _, tag in surfaces
        )
        gmsh.model.occ.extrude(
            surfaces,
            0.0,
            0.0,
            depth_mm,
            numElements=[layers],
            recombine=True,
        )
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_mm)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(mesh_path))

        node_tags, flat_coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(flat_coordinates, dtype=float).reshape(-1, 3)
        element_types, element_tag_blocks, connectivity_blocks = (
            gmsh.model.mesh.getElements(3)
        )
        prism_blocks = []
        prism_tag_blocks = []
        for element_type, tags, connectivity in zip(
            element_types, element_tag_blocks, connectivity_blocks
        ):
            name, _, _, nodes_per_element, _, _ = (
                gmsh.model.mesh.getElementProperties(element_type)
            )
            if name == "Prism 6":
                prism_tag_blocks.append(np.asarray(tags, dtype=np.int64))
                prism_blocks.append(
                    np.asarray(connectivity, dtype=np.int64).reshape(
                        -1, nodes_per_element
                    )
                )
        if not prism_blocks:
            raise RuntimeError("Gmsh did not produce six-node prism elements")
        element_tags = np.concatenate(prism_tag_blocks)
        connectivity = np.vstack(prism_blocks)
        tolerance = 1.0e-7
        support_mask = np.isclose(
            coordinates[:, 1], 0.0, atol=tolerance, rtol=0.0
        )
        load_mask = np.isclose(
            coordinates[:, 1],
            domain_height_mm,
            atol=tolerance,
            rtol=0.0,
        )
        supports = tuple(int(tag) for tag in node_tags[support_mask])
        loads = tuple(int(tag) for tag in node_tags[load_mask])
        if not supports or not loads:
            raise RuntimeError("solid mesh is missing a flat loading boundary")
        support_indices = np.flatnonzero(support_mask)
        anchor_index = min(
            support_indices,
            key=lambda index: (
                coordinates[index, 0],
                coordinates[index, 2],
            ),
        )
        anchor_two_index = max(
            support_indices,
            key=lambda index: (
                coordinates[index, 0],
                -coordinates[index, 2],
            ),
        )
        return SolidMesh(
            np.asarray(node_tags, dtype=np.int64),
            coordinates,
            element_tags,
            connectivity,
            supports,
            loads,
            int(node_tags[anchor_index]),
            int(node_tags[anchor_two_index]),
            float(area * depth_mm),
            layers,
            mesh_path.resolve(),
        )
    finally:
        gmsh.finalize()


def _id_set(keyword: str, name: str, identifiers: Iterable[int]) -> str:
    identifiers = list(identifiers)
    lines = [f"*{keyword},{keyword}={name}"]
    for start in range(0, len(identifiers), 16):
        lines.append(
            ",".join(str(value) for value in identifiers[start : start + 16])
        )
    return "\n".join(lines)


def render_solid_tension_deck(
    mesh: SolidMesh,
    youngs_modulus_MPa: float,
    poisson_ratio: float,
    density_tonnes_mm3: float,
    displacement_mm: float,
) -> str:
    nodes = "\n".join(
        f"{int(tag)}, {x:.12g}, {y:.12g}, {z:.12g}"
        for tag, (x, y, z) in zip(mesh.node_tags, mesh.coordinates)
    )
    elements = "\n".join(
        f"{int(tag)}, " + ",".join(str(int(node)) for node in element)
        for tag, element in zip(mesh.element_tags, mesh.connectivity)
    )
    return f"""*HEADING
3D unioned infill tension fidelity model
*NODE,NSET=NALL
{nodes}
*ELEMENT,TYPE=C3D6,ELSET=SOLID
{elements}
{_id_set("NSET", "SUPPORTS", mesh.supports)}
{_id_set("NSET", "LOAD", mesh.loads)}
*NSET,NSET=ANCHOR
{mesh.anchor}
*NSET,NSET=ANCHOR_TWO
{mesh.anchor_two}
*ORIENTATION,NAME=MAT_ORIENTATION
1.,0.,0.,0.,1.,0.
*MATERIAL,NAME=PLA_REPORTED
*DENSITY
{density_tonnes_mm3:.12g}
*ELASTIC
{youngs_modulus_MPa:.12g},{poisson_ratio:.12g}
*SOLID SECTION,ELSET=SOLID,MATERIAL=PLA_REPORTED,ORIENTATION=MAT_ORIENTATION
*STEP,NLGEOM=NO
*STATIC,DIRECT
1.,1.
*BOUNDARY
SUPPORTS,2,2,0.
ANCHOR,1,1,0.
ANCHOR,3,3,0.
ANCHOR_TWO,3,3,0.
LOAD,2,2,{displacement_mm:.12g}
*NODE PRINT,NSET=LOAD
U
*NODE PRINT,NSET=SUPPORTS
RF
*NODE FILE,NSET=LOAD
U
*END STEP
"""


def _matching_beam_stiffness(
    runner: CalculixRunner,
    geometry: PatternGeometry,
    strut_width_mm: float,
    depth_mm: float,
    youngs_modulus_MPa: float,
    poisson_ratio: float,
    density_tonnes_mm3: float,
    displacement_mm: float,
    domain_width_mm: float,
    domain_height_mm: float,
    deck_path: Path,
) -> Tuple[float | None, str]:
    deck = render_load_case_deck(
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
    deck_path.write_text(deck, encoding="utf-8")
    result = runner.run(
        deck_path, {}, run_id=f"solid-reference-{geometry.name}-beam"
    )
    if not result.ok:
        return None, str(result.run_dir.resolve())
    displacement = float(abs(result.curve["displacement"][-1]))
    force = float(abs(result.curve["reaction_force"][-1]))
    return force / displacement, str(result.run_dir.resolve())


def run_solid_validation(
    output_dir: Path,
    runs_dir: Path,
    properties_path: Path = DEFAULT_PROPERTIES,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
    mesh_sizes_mm: Sequence[float] = (1.2, 0.9, 0.7),
    domain_width_mm: float = 40.0,
    domain_height_mm: float = 40.0,
    depth_mm: float = 4.0,
    relative_density: float = 0.35,
    tensile_strain: float = 0.01,
) -> List[SolidResult]:
    properties = json.loads(Path(properties_path).read_text(encoding="utf-8"))
    elastic = properties["screening_isotropic"]
    youngs_modulus = float(elastic["E_MPa"])
    modulus_sd = float(elastic["E_standard_deviation_MPa"])
    poisson = float(elastic["nu"])
    density = float(properties["density_tonnes_mm3"])
    displacement = domain_height_mm * tensile_strain
    output_dir = Path(output_dir)
    (output_dir / "meshes").mkdir(parents=True, exist_ok=True)
    (output_dir / "decks").mkdir(exist_ok=True)
    (output_dir / "patterns").mkdir(exist_ok=True)
    runner = CalculixRunner(
        runs_dir,
        executable="ccx",
        timeout_seconds=300.0,
        keep_successful=True,
    )
    results: List[SolidResult] = []
    for pattern in patterns:
        geometry = make_pattern(pattern, domain_width_mm, domain_height_mm)
        strut_width = equal_solid_mass_strut_width(
            geometry,
            relative_density,
            domain_width_mm,
            domain_height_mm,
        )
        render_scad(
            geometry,
            output_dir / "patterns" / f"{pattern}.scad",
            strut_width,
            depth_mm,
            domain_width_mm,
            domain_height_mm,
        )
        beam_stiffness, _ = _matching_beam_stiffness(
            runner,
            geometry,
            strut_width,
            depth_mm,
            youngs_modulus,
            poisson,
            density,
            displacement,
            domain_width_mm,
            domain_height_mm,
            output_dir / "decks" / f"{pattern}-beam.inp",
        )
        for mesh_size in mesh_sizes_mm:
            label = f"h{mesh_size:g}".replace(".", "p")
            mesh = generate_prism_mesh(
                geometry,
                strut_width,
                mesh_size,
                output_dir / "meshes" / f"{pattern}-{label}.msh",
                domain_width_mm,
                domain_height_mm,
                depth_mm,
            )
            deck_path = output_dir / "decks" / f"{pattern}-{label}.inp"
            deck_path.write_text(
                render_solid_tension_deck(
                    mesh,
                    youngs_modulus,
                    poisson,
                    density,
                    displacement,
                ),
                encoding="utf-8",
            )
            solver = runner.run(
                deck_path, {}, run_id=f"solid-{pattern}-{label}"
            )
            actual_relative_density = mesh.actual_volume_mm3 / (
                domain_width_mm * domain_height_mm * depth_mm
            )
            actual_mass = (
                mesh.actual_volume_mm3 * density * 1.0e6
            )
            if not solver.ok:
                results.append(
                    SolidResult(
                        pattern,
                        "solid_C3D6",
                        mesh_size,
                        mesh.through_thickness_layers,
                        len(mesh.node_tags),
                        len(mesh.element_tags),
                        strut_width,
                        actual_relative_density,
                        actual_mass,
                        solver.status,
                        None,
                        None,
                        None,
                        None,
                        beam_stiffness,
                        None,
                        None,
                        None,
                        str(solver.run_dir.resolve()),
                        str(mesh.mesh_path),
                        solver.error,
                    )
                )
                continue
            endpoint_displacement = float(
                abs(solver.curve["displacement"][-1])
            )
            endpoint_force = float(
                abs(solver.curve["reaction_force"][-1])
            )
            stiffness = endpoint_force / endpoint_displacement
            apparent_modulus = (
                endpoint_force
                * domain_height_mm
                / (
                    domain_width_mm
                    * depth_mm
                    * endpoint_displacement
                )
            )
            results.append(
                SolidResult(
                    pattern,
                    "solid_C3D6",
                    mesh_size,
                    mesh.through_thickness_layers,
                    len(mesh.node_tags),
                    len(mesh.element_tags),
                    strut_width,
                    actual_relative_density,
                    actual_mass,
                    "ok",
                    endpoint_force,
                    stiffness,
                    stiffness / actual_mass,
                    apparent_modulus,
                    beam_stiffness,
                    (
                        None
                        if beam_stiffness is None
                        else stiffness / beam_stiffness
                    ),
                    stiffness
                    * (youngs_modulus - modulus_sd)
                    / youngs_modulus,
                    stiffness
                    * (youngs_modulus + modulus_sd)
                    / youngs_modulus,
                    str(solver.run_dir.resolve()),
                    str(mesh.mesh_path),
                )
            )
    _write_outputs(
        output_dir,
        results,
        properties,
        domain_width_mm,
        domain_height_mm,
        depth_mm,
        relative_density,
    )
    return results


def _write_outputs(
    output_dir: Path,
    results: Sequence[SolidResult],
    properties: Mapping[str, object],
    width: float,
    height: float,
    depth: float,
    relative_density: float,
) -> None:
    fields = list(asdict(results[0]))
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    (output_dir / "summary.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    finest: Dict[str, SolidResult] = {}
    by_pattern: Dict[str, List[SolidResult]] = {}
    for result in results:
        if result.status == "ok":
            by_pattern.setdefault(result.pattern, []).append(result)
    for pattern, rows in by_pattern.items():
        finest[pattern] = min(rows, key=lambda row: row.mesh_size_mm)
    ranking = sorted(
        finest.values(),
        key=lambda row: float(row.specific_stiffness_N_per_mm_g or 0.0),
        reverse=True,
    )
    test_fields = [
        "specimen_id",
        "pattern",
        "replicate",
        "source_scad",
        "material",
        "target_actual_mass_g",
        "strut_width_mm",
        "predicted_stiffness_N_per_mm",
        "reported_E_lower_stiffness_N_per_mm",
        "reported_E_upper_stiffness_N_per_mm",
        "required_measurements",
        "print_control",
    ]
    test_rows = []
    for result in ranking:
        for replicate in range(1, 4):
            test_rows.append(
                {
                    "specimen_id": (
                        f"{result.pattern}-solid-tension-r{replicate:02d}"
                    ),
                    "pattern": result.pattern,
                    "replicate": replicate,
                    "source_scad": f"patterns/{result.pattern}.scad",
                    "material": properties["material"],
                    "target_actual_mass_g": result.actual_mass_g,
                    "strut_width_mm": result.strut_width_mm,
                    "predicted_stiffness_N_per_mm": (
                        result.stiffness_N_per_mm
                    ),
                    "reported_E_lower_stiffness_N_per_mm": (
                        result.lower_E_stiffness_N_per_mm
                    ),
                    "reported_E_upper_stiffness_N_per_mm": (
                        result.upper_E_stiffness_N_per_mm
                    ),
                    "required_measurements": (
                        "actual_mass_g, force_N, crosshead_displacement_mm, "
                        "failure_mode"
                    ),
                    "print_control": (
                        "flat; identical spool, drying, slicer profile, bead "
                        "orientation, and machine for all specimens"
                    ),
                }
            )
    with (output_dir / "physical_tension_validation_plan.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=test_fields)
        writer.writeheader()
        writer.writerows(test_rows)
    lines = [
        "# 3D solid-mesh infill validation",
        "",
        (
            f"Patterns are clipped to an exact {width:g} × {height:g} × "
            f"{depth:g} mm coupon and their printable capsule-union widths are "
            f"solved to {relative_density:.0%} actual solid volume."
        ),
        (
            "The material is Bambu PLA Basic with reported screening "
            f"E = {properties['screening_isotropic']['E_MPa']:g} ± "
            f"{properties['screening_isotropic']['E_standard_deviation_MPa']:g} "
            "MPa and the project-prior Poisson ratio."
        ),
        "",
        "## Finest-mesh ranking",
        "",
        "| Rank | Pattern | Mesh (mm) | Nodes | Stiffness (N/mm) | "
        "Specific stiffness (N/mm/g) | Solid/beam |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, result in enumerate(ranking, start=1):
        lines.append(
            f"| {rank} | {result.pattern} | {result.mesh_size_mm:g} | "
            f"{result.nodes} | {result.stiffness_N_per_mm:.3f} | "
            f"{result.specific_stiffness_N_per_mm_g:.3f} | "
            f"{result.solid_to_beam_stiffness_ratio:.3f} |"
        )
    lines.extend(["", "## Mesh convergence", ""])
    for pattern, rows in sorted(by_pattern.items()):
        ordered = sorted(rows, key=lambda row: row.mesh_size_mm, reverse=True)
        lines.extend(
            [
                f"### {pattern}",
                "",
                "| Mesh (mm) | Layers | Elements | Stiffness (N/mm) | "
                "Reported-E interval (N/mm) |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for result in ordered:
            lines.append(
                f"| {result.mesh_size_mm:g} | "
                f"{result.through_thickness_layers} | {result.elements} | "
                f"{result.stiffness_N_per_mm:.3f} | "
                f"{result.lower_E_stiffness_N_per_mm:.3f}–"
                f"{result.upper_E_stiffness_N_per_mm:.3f} |"
            )
        if len(ordered) >= 2:
            fine = ordered[-1]
            previous = ordered[-2]
            change = abs(
                fine.stiffness_N_per_mm - previous.stiffness_N_per_mm
            ) / fine.stiffness_N_per_mm
            lines.extend(
                [
                    "",
                    f"Finest-step stiffness change: {change:.2%}.",
                    "",
                ]
            )
    failures = [result for result in results if result.status != "ok"]
    if failures:
        lines.extend(["## Failures", ""])
        for result in failures:
            lines.append(
                f"- {result.pattern} at {result.mesh_size_mm:g} mm: "
                f"{result.status}: {result.error}"
            )
        lines.append("")
    lines.extend(
        [
            "## Scope",
            "",
            "- These C3D6 continuum models include unioned joints, finite strut "
            "width, through-thickness deformation, and exact printable mass.",
            "- The reported modulus uncertainty scales absolute elastic force but "
            "does not by itself change the geometry ranking because every pattern "
            "uses the same linear material law.",
            "- `physical_tension_validation_plan.csv` defines three matched "
            "replicates of each corrected-mass geometry and the force interval "
            "that would be consistent with the reported modulus spread.",
            "- Failure, bead interfaces, void morphology, residual stress, and "
            "nonlinear material behavior still require matched Bambu PLA Basic "
            "measurements.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate leading infill patterns with 3D solid CalculiX models"
    )
    parser.add_argument("--output", default="results/solid_infill_validation")
    parser.add_argument("--runs-dir", default="runs/solid_infill_validation")
    parser.add_argument("--properties", default=str(DEFAULT_PROPERTIES))
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=list(DEFAULT_PATTERNS),
    )
    parser.add_argument(
        "--mesh-sizes",
        nargs="+",
        type=float,
        default=[1.2, 0.9, 0.7],
    )
    parser.add_argument("--relative-density", type=float, default=0.35)
    args = parser.parse_args()
    results = run_solid_validation(
        Path(args.output),
        Path(args.runs_dir),
        properties_path=Path(args.properties),
        patterns=args.patterns,
        mesh_sizes_mm=args.mesh_sizes,
        relative_density=args.relative_density,
    )
    successful = sum(result.status == "ok" for result in results)
    print(
        f"completed {successful}/{len(results)} solid simulations; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
