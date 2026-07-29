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
    LOAD_CASES,
    _maximum_von_mises,
    _run_buckling_deck,
    make_pattern,
    PatternGeometry,
    render_buckling_deck,
    render_load_case_deck,
    render_scad,
)
from .materials import check_admissibility
from .strength_screening import screen_curve_directional_normal_strength
from .solid_infill import (
    DEFAULT_PATTERNS,
    SolidMesh,
    equal_solid_mass_strut_width,
    generate_prism_mesh,
)


@dataclass(frozen=True)
class BoundarySets:
    supports: Tuple[int, ...]
    loads: Tuple[int, ...]
    anchor: int
    anchor_two: int


@dataclass
class SolidLoadResult:
    pattern: str
    load_case: str
    status: str
    mesh_size_mm: float
    through_thickness_layers: int
    nodes: int
    elements: int
    strut_width_mm: float
    actual_relative_density: float
    actual_mass_g: float
    applied_displacement_mm: float
    reaction_force_N: float | None
    stiffness_N_per_mm: float | None
    specific_stiffness_N_per_mm_g: float | None
    apparent_modulus_MPa: float | None
    maximum_von_mises_MPa: float | None
    tensile_reference_utilization: float | None
    critical_buckling_load_N: float | None
    critical_buckling_load_N_per_g: float | None
    matching_beam_metric: float | None
    solid_to_beam_ratio: float | None
    lower_E_metric: float | None
    upper_E_metric: float | None
    run_directory: str
    mesh_file: str
    error: str | None = None
    maximum_tensile_sigma1_MPa: float | None = None
    maximum_tensile_sigma2_MPa: float | None = None
    maximum_tensile_sigma3_MPa: float | None = None
    directional_normal_utilization_mean: float | None = None
    directional_normal_utilization_lower_1sd: float | None = None
    directional_normal_utilization_p99_mean: float | None = None
    directional_normal_governing_direction: str | None = None


def _nearest_tag(
    mesh: SolidMesh,
    candidates: np.ndarray,
    key,
) -> int:
    index = min(candidates, key=key)
    return int(mesh.node_tags[index])


def select_boundary_sets(
    mesh: SolidMesh,
    load_case: str,
    domain_width_mm: float,
    domain_height_mm: float,
    depth_mm: float,
    mesh_size_mm: float,
) -> BoundarySets:
    if load_case not in LOAD_CASES:
        raise ValueError(f"unsupported solid load case {load_case!r}")
    xyz = mesh.coordinates
    tags = mesh.node_tags
    tolerance = 1.0e-7
    if load_case == "bending":
        bottom = np.isclose(xyz[:, 2], 0.0, atol=tolerance, rtol=0.0)
        edge_x = np.isclose(
            xyz[:, 0], 0.0, atol=tolerance, rtol=0.0
        ) | np.isclose(
            xyz[:, 0], domain_width_mm, atol=tolerance, rtol=0.0
        )
        support_mask = bottom & edge_x
        top_indices = np.flatnonzero(
            np.isclose(
                xyz[:, 2], depth_mm, atol=tolerance, rtol=0.0
            )
        )
        if top_indices.size == 0:
            raise RuntimeError("solid mesh has no top surface for bending")
        center_distance = np.abs(
            xyz[top_indices, 0] - domain_width_mm / 2.0
        )
        minimum = float(np.min(center_distance))
        strip_half_width = max(mesh_size_mm * 0.6, minimum + tolerance)
        load_indices = top_indices[
            center_distance <= strip_half_width + tolerance
        ]
        support_indices = np.flatnonzero(support_mask)
        if support_indices.size == 0 or load_indices.size == 0:
            raise RuntimeError("solid bending fixture sets are empty")
        left = support_indices[
            np.isclose(
                xyz[support_indices, 0], 0.0, atol=tolerance, rtol=0.0
            )
        ]
        right = support_indices[
            np.isclose(
                xyz[support_indices, 0],
                domain_width_mm,
                atol=tolerance,
                rtol=0.0,
            )
        ]
        anchor = _nearest_tag(
            mesh,
            left,
            lambda index: (xyz[index, 1], xyz[index, 2]),
        )
        anchor_two = _nearest_tag(
            mesh,
            right,
            lambda index: (xyz[index, 1], xyz[index, 2]),
        )
    else:
        support_indices = np.flatnonzero(
            np.isclose(
                xyz[:, 1], 0.0, atol=tolerance, rtol=0.0
            )
        )
        load_indices = np.flatnonzero(
            np.isclose(
                xyz[:, 1],
                domain_height_mm,
                atol=tolerance,
                rtol=0.0,
            )
        )
        if support_indices.size == 0 or load_indices.size == 0:
            raise RuntimeError("solid loading fixture sets are empty")
        anchor = _nearest_tag(
            mesh,
            support_indices,
            lambda index: (xyz[index, 0], xyz[index, 2]),
        )
        anchor_two = _nearest_tag(
            mesh,
            support_indices,
            lambda index: (-xyz[index, 0], xyz[index, 2]),
        )
    return BoundarySets(
        tuple(int(value) for value in tags[support_indices]),
        tuple(int(value) for value in tags[load_indices]),
        anchor,
        anchor_two,
    )


def _id_set(name: str, identifiers: Iterable[int]) -> str:
    identifiers = list(identifiers)
    lines = [f"*NSET,NSET={name}"]
    for start in range(0, len(identifiers), 16):
        lines.append(
            ",".join(str(value) for value in identifiers[start : start + 16])
        )
    return "\n".join(lines)


def _mesh_text(
    mesh: SolidMesh,
    load_case: str,
    domain_width_mm: float,
    domain_height_mm: float,
) -> Tuple[str, str]:
    coordinates = mesh.coordinates.copy()
    if load_case == "compression":
        interior = (
            (coordinates[:, 0] > 0.0)
            & (coordinates[:, 0] < domain_width_mm)
            & (coordinates[:, 1] > 0.0)
            & (coordinates[:, 1] < domain_height_mm)
        )
        x = coordinates[interior, 0]
        y = coordinates[interior, 1]
        coordinates[interior, 0] += (
            0.0005
            * domain_width_mm
            * np.sin(np.pi * y / domain_height_mm)
            * np.sin(2.0 * np.pi * x / domain_width_mm)
        )
    nodes = "\n".join(
        f"{int(tag)}, {x:.12g}, {y:.12g}, {z:.12g}"
        for tag, (x, y, z) in zip(mesh.node_tags, coordinates)
    )
    elements = "\n".join(
        f"{int(tag)}, " + ",".join(str(int(node)) for node in element)
        for tag, element in zip(mesh.element_tags, mesh.connectivity)
    )
    return nodes, elements


def _solid_preamble(
    mesh: SolidMesh,
    sets: BoundarySets,
    load_case: str,
    youngs_modulus_MPa: float,
    poisson_ratio: float,
    density_tonnes_mm3: float,
    domain_width_mm: float,
    domain_height_mm: float,
    plastic_points: Sequence[Mapping[str, float]] | None = None,
    engineering_constants: Mapping[str, float] | None = None,
) -> str:
    nodes, elements = _mesh_text(
        mesh, load_case, domain_width_mm, domain_height_mm
    )
    plastic_card = ""
    if plastic_points:
        rows = "\n".join(
            f"{point['true_stress_MPa']:.12g},"
            f"{point['equivalent_plastic_strain']:.12g}"
            for point in plastic_points
        )
        plastic_card = f"*PLASTIC\n{rows}\n"
    if engineering_constants is None:
        elastic_card = (
            "*ELASTIC\n"
            f"{youngs_modulus_MPa:.12g},{poisson_ratio:.12g}\n"
        )
    else:
        constants = {
            name: float(engineering_constants[name])
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
        first_row = ",".join(
            f"{constants[name]:.12g}"
            for name in (
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
        elastic_card = (
            "*ELASTIC,TYPE=ENGINEERING CONSTANTS\n"
            f"{first_row}\n{constants['G23']:.12g}\n"
        )
    return f"""*HEADING
3D unioned infill {load_case} model
*NODE,NSET=NALL
{nodes}
*ELEMENT,TYPE=C3D6,ELSET=SOLID
{elements}
{_id_set("SUPPORTS", sets.supports)}
{_id_set("LOAD", sets.loads)}
*NSET,NSET=ANCHOR
{sets.anchor}
*NSET,NSET=ANCHOR_TWO
{sets.anchor_two}
*ORIENTATION,NAME=MAT_ORIENTATION
1.,0.,0.,0.,1.,0.
*MATERIAL,NAME=PLA_REPORTED
*DENSITY
{density_tonnes_mm3:.12g}
{elastic_card}\
{plastic_card}\
*SOLID SECTION,ELSET=SOLID,MATERIAL=PLA_REPORTED,ORIENTATION=MAT_ORIENTATION
"""


def render_solid_static_deck(
    mesh: SolidMesh,
    load_case: str,
    youngs_modulus_MPa: float,
    poisson_ratio: float,
    density_tonnes_mm3: float,
    displacement_mm: float,
    domain_width_mm: float = 40.0,
    domain_height_mm: float = 40.0,
    depth_mm: float = 4.0,
    mesh_size_mm: float = 0.7,
    plastic_points: Sequence[Mapping[str, float]] | None = None,
    incremental_nlgeom: bool = False,
    output_element_fields: bool = True,
    engineering_constants: Mapping[str, float] | None = None,
) -> str:
    if load_case not in {"tension", "shear", "bending", "compression"}:
        raise ValueError(f"unsupported static load case {load_case!r}")
    sets = select_boundary_sets(
        mesh,
        load_case,
        domain_width_mm,
        domain_height_mm,
        depth_mm,
        mesh_size_mm,
    )
    preamble = _solid_preamble(
        mesh,
        sets,
        load_case,
        youngs_modulus_MPa,
        poisson_ratio,
        density_tonnes_mm3,
        domain_width_mm,
        domain_height_mm,
        plastic_points,
        engineering_constants,
    )
    if load_case == "tension":
        step = (
            "*STEP,NLGEOM=YES\n*STATIC\n0.025,1.,1.e-06,0.05"
            if plastic_points or incremental_nlgeom
            else "*STEP,NLGEOM=NO\n*STATIC,DIRECT\n1.,1."
        )
        boundaries = f"""SUPPORTS,2,2,0.
ANCHOR,1,1,0.
ANCHOR,3,3,0.
ANCHOR_TWO,3,3,0.
LOAD,2,2,{abs(displacement_mm):.12g}"""
    elif load_case == "compression":
        step = "*STEP,NLGEOM=YES\n*STATIC\n0.05,1.,1.e-06,0.1"
        boundaries = f"""SUPPORTS,2,2,0.
ANCHOR,1,1,0.
ANCHOR,3,3,0.
ANCHOR_TWO,3,3,0.
LOAD,2,2,{-abs(displacement_mm):.12g}"""
    elif load_case == "shear":
        step = "*STEP,NLGEOM=NO\n*STATIC,DIRECT\n1.,1."
        boundaries = f"""SUPPORTS,1,3,0.
LOAD,2,2,0.
LOAD,1,1,{abs(displacement_mm):.12g}"""
    else:
        step = "*STEP,NLGEOM=NO\n*STATIC,DIRECT\n1.,1."
        boundaries = f"""SUPPORTS,3,3,0.
ANCHOR,1,2,0.
ANCHOR_TWO,2,2,0.
LOAD,3,3,{-abs(displacement_mm):.12g}"""
    element_output = (
        "*EL FILE,ELSET=SOLID\nS,PEEQ\n"
        if output_element_fields
        else ""
    )
    return f"""{preamble}{step}
*BOUNDARY
{boundaries}
*NODE PRINT,NSET=LOAD
U
*NODE PRINT,NSET=SUPPORTS
RF
*NODE FILE,NSET=LOAD
U
{element_output}\
*END STEP
"""


def render_solid_buckling_deck(
    mesh: SolidMesh,
    youngs_modulus_MPa: float,
    poisson_ratio: float,
    density_tonnes_mm3: float,
    domain_width_mm: float = 40.0,
    domain_height_mm: float = 40.0,
    depth_mm: float = 4.0,
    mesh_size_mm: float = 0.7,
    engineering_constants: Mapping[str, float] | None = None,
) -> str:
    sets = select_boundary_sets(
        mesh,
        "buckling",
        domain_width_mm,
        domain_height_mm,
        depth_mm,
        mesh_size_mm,
    )
    preamble = _solid_preamble(
        mesh,
        sets,
        "buckling",
        youngs_modulus_MPa,
        poisson_ratio,
        density_tonnes_mm3,
        domain_width_mm,
        domain_height_mm,
        engineering_constants=engineering_constants,
    )
    load_each = -1.0 / len(sets.loads)
    cloads = "\n".join(
        f"{node},2,{load_each:.12g}" for node in sets.loads
    )
    return f"""{preamble}*BOUNDARY
SUPPORTS,2,2,0.
ANCHOR,1,1,0.
ANCHOR,3,3,0.
ANCHOR_TWO,3,3,0.
*STEP
*BUCKLE
1,0.01
*CLOAD
{cloads}
*NODE FILE
U
*END STEP
"""


def _run_matching_beam(
    runner: CalculixRunner,
    geometry,
    load_case: str,
    strut_width_mm: float,
    depth_mm: float,
    youngs_modulus_MPa: float,
    poisson_ratio: float,
    density_tonnes_mm3: float,
    displacement_mm: float,
    domain_width_mm: float,
    domain_height_mm: float,
    deck_path: Path,
    runs_dir: Path,
) -> float | None:
    if load_case == "buckling":
        deck = render_buckling_deck(
            geometry,
            strut_width_mm,
            depth_mm,
            youngs_modulus_MPa,
            poisson_ratio,
            domain_width_mm,
            domain_height_mm,
            density_tonnes_mm3,
        )
        deck_path.write_text(deck, encoding="utf-8")
        status, factor, _, _ = _run_buckling_deck(
            deck_path,
            runs_dir,
            f"solid-multiload-beam-{geometry.name}-buckling",
        )
        return None if status != "ok" else factor
    deck = render_load_case_deck(
        geometry,
        load_case,
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
    solver = runner.run(
        deck_path,
        {},
        run_id=f"solid-multiload-beam-{geometry.name}-{load_case}",
    )
    if not solver.ok:
        return None
    endpoint_displacement = float(abs(solver.curve["displacement"][-1]))
    endpoint_force = float(abs(solver.curve["reaction_force"][-1]))
    return endpoint_force / endpoint_displacement


def run_solid_multiload(
    output_dir: Path,
    runs_dir: Path,
    properties_path: Path = DEFAULT_PROPERTIES,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
    load_cases: Sequence[str] = LOAD_CASES,
    mesh_size_mm: float = 0.7,
    domain_width_mm: float = 40.0,
    domain_height_mm: float = 40.0,
    depth_mm: float = 4.0,
    relative_density: float = 0.35,
    deformation: float = 0.01,
    geometries: Mapping[str, PatternGeometry] | None = None,
) -> List[SolidLoadResult]:
    unknown = set(load_cases) - set(LOAD_CASES)
    if unknown:
        raise ValueError(f"unknown load cases: {sorted(unknown)}")
    properties = json.loads(Path(properties_path).read_text(encoding="utf-8"))
    elastic = properties["screening_isotropic"]
    engineering_constants = properties.get(
        "screening_engineering_constants"
    )
    if engineering_constants is not None:
        engineering_constants = {
            name: float(engineering_constants[name])
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
        admissibility = check_admissibility(engineering_constants)
        if not admissibility.valid:
            raise ValueError(
                "profile engineering constants are inadmissible: "
                + "; ".join(admissibility.reasons)
            )
    strength = properties["reported_strength_reference"]
    youngs_modulus = float(elastic["E_MPa"])
    modulus_sd = float(elastic["E_standard_deviation_MPa"])
    poisson = float(elastic["nu"])
    density = float(properties["density_tonnes_mm3"])
    tensile_reference = float(strength["overall_tensile_UTS_MPa"])
    displacement = domain_height_mm * deformation
    output_dir = Path(output_dir)
    for directory in ("meshes", "decks", "patterns", "curves"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    runner = CalculixRunner(
        runs_dir,
        executable="ccx",
        timeout_seconds=600.0,
        keep_successful=True,
    )
    results: List[SolidLoadResult] = []
    for pattern in patterns:
        geometry = (
            geometries[pattern]
            if geometries is not None and pattern in geometries
            else make_pattern(pattern, domain_width_mm, domain_height_mm)
        )
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
        mesh = generate_prism_mesh(
            geometry,
            strut_width,
            mesh_size_mm,
            output_dir / "meshes" / f"{pattern}.msh",
            domain_width_mm,
            domain_height_mm,
            depth_mm,
        )
        actual_density = mesh.actual_volume_mm3 / (
            domain_width_mm * domain_height_mm * depth_mm
        )
        actual_mass = mesh.actual_volume_mm3 * density * 1.0e6
        for load_case in load_cases:
            beam_metric = _run_matching_beam(
                runner,
                geometry,
                load_case,
                strut_width,
                depth_mm,
                youngs_modulus,
                poisson,
                density,
                displacement,
                domain_width_mm,
                domain_height_mm,
                output_dir / "decks" / f"{pattern}-{load_case}-beam.inp",
                runs_dir,
            )
            deck_path = (
                output_dir / "decks" / f"{pattern}-{load_case}-solid.inp"
            )
            if load_case == "buckling":
                deck_path.write_text(
                    render_solid_buckling_deck(
                        mesh,
                        youngs_modulus,
                        poisson,
                        density,
                        domain_width_mm,
                        domain_height_mm,
                        depth_mm,
                        mesh_size_mm,
                        engineering_constants=engineering_constants,
                    ),
                    encoding="utf-8",
                )
                status, factor, run_dir, error = _run_buckling_deck(
                    deck_path,
                    runs_dir,
                    f"solid-multiload-{pattern}-buckling",
                    timeout_seconds=600.0,
                )
                critical = None if factor is None else abs(float(factor))
                ratio = (
                    None
                    if critical is None or not beam_metric
                    else critical / beam_metric
                )
                results.append(
                    SolidLoadResult(
                        pattern,
                        load_case,
                        status,
                        mesh_size_mm,
                        mesh.through_thickness_layers,
                        len(mesh.node_tags),
                        len(mesh.element_tags),
                        strut_width,
                        actual_density,
                        actual_mass,
                        0.0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        critical,
                        None if critical is None else critical / actual_mass,
                        beam_metric,
                        ratio,
                        (
                            None
                            if critical is None
                            else critical
                            * (youngs_modulus - modulus_sd)
                            / youngs_modulus
                        ),
                        (
                            None
                            if critical is None
                            else critical
                            * (youngs_modulus + modulus_sd)
                            / youngs_modulus
                        ),
                        str(run_dir.resolve()),
                        str(mesh.mesh_path),
                        error,
                    )
                )
                continue
            deck_path.write_text(
                render_solid_static_deck(
                    mesh,
                    load_case,
                    youngs_modulus,
                    poisson,
                    density,
                    displacement,
                    domain_width_mm,
                    domain_height_mm,
                    depth_mm,
                    mesh_size_mm,
                    engineering_constants=engineering_constants,
                ),
                encoding="utf-8",
            )
            solver = runner.run(
                deck_path,
                {},
                run_id=f"solid-multiload-{pattern}-{load_case}",
            )
            if not solver.ok:
                results.append(
                    SolidLoadResult(
                        pattern,
                        load_case,
                        solver.status,
                        mesh_size_mm,
                        mesh.through_thickness_layers,
                        len(mesh.node_tags),
                        len(mesh.element_tags),
                        strut_width,
                        actual_density,
                        actual_mass,
                        displacement,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        beam_metric,
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
            if load_case == "bending":
                inertia = domain_height_mm * depth_mm**3 / 12.0
                apparent_modulus = (
                    stiffness * domain_width_mm**3 / (48.0 * inertia)
                )
                reference_length = domain_width_mm
            else:
                apparent_modulus = (
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
            strength_screen = screen_curve_directional_normal_strength(
                solver.curve,
                properties,
            )
            displacements = np.abs(solver.curve["displacement"])
            forces = np.abs(solver.curve["reaction_force"])
            curve_path = (
                output_dir / "curves" / f"{pattern}-{load_case}.csv"
            )
            with curve_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "pattern",
                        "load_case",
                        "displacement_mm",
                        "force_N",
                        "normalized_displacement",
                    ],
                )
                writer.writeheader()
                for current, force in zip(displacements, forces):
                    writer.writerow(
                        {
                            "pattern": pattern,
                            "load_case": load_case,
                            "displacement_mm": current,
                            "force_N": force,
                            "normalized_displacement": current
                            / reference_length,
                        }
                    )
            ratio = (
                None
                if not beam_metric
                else stiffness / beam_metric
            )
            results.append(
                SolidLoadResult(
                    pattern,
                    load_case,
                    "ok",
                    mesh_size_mm,
                    mesh.through_thickness_layers,
                    len(mesh.node_tags),
                    len(mesh.element_tags),
                    strut_width,
                    actual_density,
                    actual_mass,
                    endpoint_displacement,
                    endpoint_force,
                    stiffness,
                    stiffness / actual_mass,
                    apparent_modulus,
                    maximum_stress,
                    (
                        None
                        if maximum_stress is None
                        else maximum_stress / tensile_reference
                    ),
                    None,
                    None,
                    beam_metric,
                    ratio,
                    stiffness
                    * (youngs_modulus - modulus_sd)
                    / youngs_modulus,
                    stiffness
                    * (youngs_modulus + modulus_sd)
                    / youngs_modulus,
                    str(solver.run_dir.resolve()),
                    str(mesh.mesh_path),
                    maximum_tensile_sigma1_MPa=(
                        None
                        if strength_screen is None
                        else strength_screen.maximum_tensile_sigma1_MPa
                    ),
                    maximum_tensile_sigma2_MPa=(
                        None
                        if strength_screen is None
                        else strength_screen.maximum_tensile_sigma2_MPa
                    ),
                    maximum_tensile_sigma3_MPa=(
                        None
                        if strength_screen is None
                        else strength_screen.maximum_tensile_sigma3_MPa
                    ),
                    directional_normal_utilization_mean=(
                        None
                        if strength_screen is None
                        else strength_screen.maximum_mean_strength_utilization
                    ),
                    directional_normal_utilization_lower_1sd=(
                        None
                        if strength_screen is None
                        else strength_screen.maximum_lower_1sd_strength_utilization
                    ),
                    directional_normal_utilization_p99_mean=(
                        None
                        if strength_screen is None
                        else strength_screen.p99_mean_strength_utilization
                    ),
                    directional_normal_governing_direction=(
                        None
                        if strength_screen is None
                        else strength_screen.governing_direction
                    ),
                )
            )
    _write_outputs(
        output_dir,
        results,
        properties,
        patterns,
        load_cases,
    )
    return results


def _metric(result: SolidLoadResult) -> float | None:
    if result.load_case == "buckling":
        return result.critical_buckling_load_N_per_g
    return result.specific_stiffness_N_per_mm_g


def _balanced_rows(
    results: Sequence[SolidLoadResult],
    patterns: Sequence[str],
    load_cases: Sequence[str],
) -> List[Dict[str, object]]:
    normalized: Dict[str, Dict[str, float]] = {}
    for load_case in load_cases:
        values = {
            row.pattern: float(_metric(row))
            for row in results
            if row.load_case == load_case
            and row.status == "ok"
            and _metric(row) is not None
            and _metric(row) > 0.0
        }
        if values:
            maximum = max(values.values())
            normalized[load_case] = {
                pattern: value / maximum
                for pattern, value in values.items()
            }
    rows: List[Dict[str, object]] = []
    for pattern in patterns:
        values = [
            normalized.get(load_case, {}).get(pattern)
            for load_case in load_cases
        ]
        score = (
            None
            if any(value is None for value in values)
            else float(
                np.prod([float(value) for value in values])
                ** (1.0 / len(values))
            )
        )
        rows.append(
            {
                "pattern": pattern,
                "balanced_geometric_mean": score,
                **{
                    f"{load_case}_normalized": normalized.get(
                        load_case, {}
                    ).get(pattern)
                    for load_case in load_cases
                },
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["balanced_geometric_mean"] is not None,
            row["balanced_geometric_mean"] or -1.0,
        ),
        reverse=True,
    )


def _write_outputs(
    output_dir: Path,
    results: Sequence[SolidLoadResult],
    properties: Mapping[str, object],
    patterns: Sequence[str],
    load_cases: Sequence[str],
) -> None:
    (output_dir / "material_profile_used.json").write_text(
        json.dumps(properties, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(asdict(results[0]))
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)
    (output_dir / "summary.json").write_text(
        json.dumps([asdict(row) for row in results], indent=2) + "\n",
        encoding="utf-8",
    )
    balanced = _balanced_rows(results, patterns, load_cases)
    with (output_dir / "multi_load_ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(balanced[0]))
        writer.writeheader()
        writer.writerows(balanced)
    lookup = {(row.pattern, row.load_case): row for row in results}
    plan_fields = [
        "specimen_id",
        "pattern",
        "load_case",
        "replicate",
        "source_scad",
        "material",
        "target_actual_mass_g",
        "strut_width_mm",
        "predicted_metric",
        "predicted_value",
        "reported_E_lower_value",
        "reported_E_upper_value",
        "required_measurements",
        "print_control",
    ]
    plan_rows = []
    for ranking_row in balanced:
        pattern = str(ranking_row["pattern"])
        for load_case in load_cases:
            result = lookup.get((pattern, load_case))
            if result is None or result.status != "ok":
                continue
            predicted_metric = (
                "critical_buckling_load_N"
                if load_case == "buckling"
                else "stiffness_N_per_mm"
            )
            predicted_value = (
                result.critical_buckling_load_N
                if load_case == "buckling"
                else result.stiffness_N_per_mm
            )
            for replicate in range(1, 4):
                plan_rows.append(
                    {
                        "specimen_id": (
                            f"{pattern}-{load_case}-r{replicate:02d}"
                        ),
                        "pattern": pattern,
                        "load_case": load_case,
                        "replicate": replicate,
                        "source_scad": f"patterns/{pattern}.scad",
                        "material": properties["material"],
                        "target_actual_mass_g": result.actual_mass_g,
                        "strut_width_mm": result.strut_width_mm,
                        "predicted_metric": predicted_metric,
                        "predicted_value": predicted_value,
                        "reported_E_lower_value": result.lower_E_metric,
                        "reported_E_upper_value": result.upper_E_metric,
                        "required_measurements": (
                            "actual_mass_g, force_N, "
                            "crosshead_displacement_mm, failure_mode"
                        ),
                        "print_control": (
                            "flat; identical spool, drying, slicer profile, "
                            "bead orientation, and machine"
                        ),
                    }
                )
    with (output_dir / "physical_multiload_test_plan.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=plan_fields)
        writer.writeheader()
        writer.writerows(plan_rows)
    engineering = properties.get("screening_engineering_constants")
    if engineering:
        material_description = (
            f"engineering constants E1/E2/E3 = "
            f"{float(engineering['E1']):.1f}/"
            f"{float(engineering['E2']):.1f}/"
            f"{float(engineering['E3']):.1f} MPa, with local 1/2/3 mapped "
            "to global X/Y/Z for these flat coupons"
        )
    else:
        material_description = (
            f"isotropic E = "
            f"{properties['screening_isotropic']['E_MPa']:.3f} MPa and "
            f"ν = {properties['screening_isotropic']['nu']:.3f}"
        )
    lines = [
        "# Refined 3D solid multi-load infill screen",
        "",
        (
            "All patterns use the same Bambu PLA Basic elastic profile, the "
            "same converged C3D6 mesh size, and equal actual printable mass."
        ),
        (
            f"Profile `{properties.get('profile_id', 'unspecified')}` uses "
            f"{material_description}."
        ),
    ]
    for load_case in load_cases:
        candidates = [
            row
            for row in results
            if row.load_case == load_case and row.status == "ok"
        ]
        candidates.sort(
            key=lambda row: float(_metric(row) or 0.0), reverse=True
        )
        if load_case == "buckling":
            lines.extend(
                [
                    "",
                    "## Buckling",
                    "",
                    "| Rank | Pattern | Critical load (N) | N/g | "
                    "Solid/beam |",
                    "|---:|---|---:|---:|---:|",
                ]
            )
            for rank, row in enumerate(candidates, start=1):
                lines.append(
                    f"| {rank} | {row.pattern} | "
                    f"{row.critical_buckling_load_N:.3f} | "
                    f"{row.critical_buckling_load_N_per_g:.3f} | "
                    f"{row.solid_to_beam_ratio:.3f} |"
                )
        else:
            lines.extend(
                [
                    "",
                    f"## {load_case.title()}",
                    "",
                    "| Rank | Pattern | Stiffness (N/mm) | N/mm/g | "
                    "Max von Mises (MPa) | Directional normal utilization | "
                    "Governing axis | Solid/beam |",
                    "|---:|---|---:|---:|---:|---:|---|---:|",
                ]
            )
            for rank, row in enumerate(candidates, start=1):
                stress = (
                    "n/a"
                    if row.maximum_von_mises_MPa is None
                    else f"{row.maximum_von_mises_MPa:.3f}"
                )
                ratio = (
                    "n/a"
                    if row.solid_to_beam_ratio is None
                    else f"{row.solid_to_beam_ratio:.3f}"
                )
                directional = (
                    "n/a"
                    if row.directional_normal_utilization_mean is None
                    else f"{row.directional_normal_utilization_mean:.3f}"
                )
                governing = (
                    "n/a"
                    if row.directional_normal_governing_direction is None
                    else row.directional_normal_governing_direction
                )
                lines.append(
                    f"| {rank} | {row.pattern} | "
                    f"{row.stiffness_N_per_mm:.3f} | "
                    f"{row.specific_stiffness_N_per_mm_g:.3f} | "
                    f"{stress} | {directional} | {governing} | {ratio} |"
                )
    lines.extend(
        [
            "",
            "## Balanced continuum index",
            "",
            "| Rank | Pattern | Score |",
            "|---:|---|---:|",
        ]
    )
    for rank, row in enumerate(balanced, start=1):
        value = row["balanced_geometric_mean"]
        lines.append(
            f"| {rank} | {row['pattern']} | "
            f"{'n/a' if value is None else f'{float(value):.3f}'} |"
        )
    failures = [row for row in results if row.status != "ok"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            lines.append(
                f"- {row.pattern}/{row.load_case}: "
                f"{row.status}: {row.error}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Static rankings use stiffness per measured target mass; buckling "
            "uses critical load per target mass.",
            "- Compression includes geometric nonlinearity and a deterministic "
            "0.02 mm lateral imperfection.",
            "- The bending fixture uses a narrow top-surface displacement strip "
            "and supported bottom edges; physical fixtures should mirror it.",
            "- The reported elastic-modulus spread is propagated to every force "
            "metric as a proportional XY-modulus sensitivity; for engineering-"
            "constants profiles this is not a joint directional uncertainty "
            "interval. Failure remains a reference check rather than a "
            "constitutive prediction.",
            "- Directional normal utilization uses the TDS tensile means "
            "(local 1/2 = XY 35 MPa; local 3 = Z 31 MPa). It ignores "
            "compression and shear because matching strengths are unavailable, "
            "and it is a hotspot screen rather than a damage/fracture law.",
            "- `physical_multiload_test_plan.csv` schedules three independent "
            "replicates for every pattern/load combination.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the refined 3D solid multi-load infill screen"
    )
    parser.add_argument("--output", default="results/solid_multiload")
    parser.add_argument("--runs-dir", default="runs/solid_multiload")
    parser.add_argument("--properties", default=str(DEFAULT_PROPERTIES))
    parser.add_argument(
        "--patterns", nargs="+", default=list(DEFAULT_PATTERNS)
    )
    parser.add_argument(
        "--load-cases",
        nargs="+",
        choices=LOAD_CASES,
        default=list(LOAD_CASES),
    )
    parser.add_argument("--mesh-size", type=float, default=0.7)
    parser.add_argument("--relative-density", type=float, default=0.35)
    args = parser.parse_args()
    results = run_solid_multiload(
        Path(args.output),
        Path(args.runs_dir),
        properties_path=Path(args.properties),
        patterns=args.patterns,
        load_cases=args.load_cases,
        mesh_size_mm=args.mesh_size,
        relative_density=args.relative_density,
    )
    successful = sum(row.status == "ok" for row in results)
    print(
        f"completed {successful}/{len(results)} refined solid simulations; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
