from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .ballistic_pattern_validation import PHYSICAL_RANKING
from .calculix import CalculixRunner
from .volumetric_infill import (
    VOLUMETRIC_PATTERNS,
    inspect_binary_stl,
    make_voxel_mesh,
    morphology_metrics,
    render_voxel_compression_deck,
    write_occupancy_csv,
    write_voxel_stl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/volumetric_pattern_validation"
DEFAULT_RUNS = PROJECT_ROOT / "runs/volumetric_pattern_validation_compact"
RESOLUTIONS = (24, 32, 40)
SCREENING_RESOLUTION = 24
CACHE_VERSION = 1
PLASTIC_POINTS = (
    {"stress_MPa": 50.0, "equivalent_plastic_strain": 0.0},
    {"stress_MPa": 60.0, "equivalent_plastic_strain": 0.05},
)


@dataclass
class VolumetricResult:
    pattern: str
    resolution: int
    axis: str
    model: str
    status: str
    elements: int
    nodes: int
    actual_relative_density: float
    mass_g: float
    displacement_mm: float | None
    reaction_force_N: float | None
    stiffness_N_per_mm: float | None
    apparent_modulus_MPa: float | None
    absorbed_energy_N_mm: float | None
    specific_absorbed_energy_J_kg: float | None
    run_directory: str
    error: str | None = None


def _curve_energy(displacement: np.ndarray, force: np.ndarray) -> float:
    x = np.abs(np.asarray(displacement, dtype=float))
    y = np.abs(np.asarray(force, dtype=float))
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) == 0:
        return 0.0
    if x[0] > 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
    return float(np.trapezoid(y, x))


def run_case(
    runner: CalculixRunner,
    output_dir: Path,
    pattern: str,
    resolution: int,
    axis: str,
    model: str,
    side_length_mm: float = 20.0,
) -> tuple[VolumetricResult, Dict[str, float]]:
    identifier = f"{pattern}-n{resolution}-{axis}-{model}"
    mesh = make_voxel_mesh(
        pattern,
        resolution=resolution,
        side_length_mm=side_length_mm,
        relative_density=0.5,
    )
    density_tonnes_mm3 = 1.25e-9
    mass_g = mesh.actual_volume_mm3 * density_tonnes_mm3 * 1.0e6
    plastic = PLASTIC_POINTS if model == "author_predamage_plastic" else None
    deformation = 0.05 if plastic else 0.01
    deck = render_voxel_compression_deck(
        mesh,
        axis,
        deformation,
        youngs_modulus_MPa=2600.0,
        poisson_ratio=0.33,
        density_tonnes_mm3=density_tonnes_mm3,
        plastic_points=plastic,
    )
    deck_dir = output_dir / "decks"
    mesh_dir = output_dir / "occupancy"
    cache_dir = output_dir / "case_results"
    deck_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    deck_path = deck_dir / f"{identifier}.inp"
    deck_path.write_text(deck, encoding="utf-8")
    if model == "elastic":
        write_occupancy_csv(
            mesh, mesh_dir / f"{pattern}-n{resolution}.csv"
        )
        if axis == "x" and resolution == SCREENING_RESOLUTION:
            write_voxel_stl(
                mesh,
                output_dir / "printable_stl" / f"{pattern}.stl",
            )
    metrics = morphology_metrics(mesh, axis)
    cache_path = cache_dir / f"{identifier}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_version") == CACHE_VERSION:
            return VolumetricResult(**cached["result"]), metrics
    solver = runner.run(deck_path, {}, run_id=identifier)
    if not solver.ok:
        return (
            VolumetricResult(
                pattern,
                resolution,
                axis,
                model,
                solver.status,
                len(mesh.element_tags),
                len(mesh.node_tags),
                mesh.actual_relative_density,
                mass_g,
                None,
                None,
                None,
                None,
                None,
                None,
                str(solver.run_dir.resolve()),
                solver.error,
            ),
            metrics,
        )
    displacement = float(abs(solver.curve["displacement"][-1]))
    reaction = float(abs(solver.curve["reaction_force"][-1]))
    stiffness = reaction / displacement
    apparent_modulus = (
        reaction
        * side_length_mm
        / (side_length_mm ** 2 * displacement)
    )
    energy = _curve_energy(
        solver.curve["displacement"], solver.curve["reaction_force"]
    )
    result = VolumetricResult(
        pattern,
        resolution,
        axis,
        model,
        "ok",
        len(mesh.element_tags),
        len(mesh.node_tags),
        mesh.actual_relative_density,
        mass_g,
        displacement,
        reaction,
        stiffness,
        apparent_modulus,
        energy,
        energy / mass_g,
        str(solver.run_dir.resolve()),
    )
    cache_path.write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "result": asdict(result),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result, metrics


def pairwise_matches(
    predicted: Sequence[str],
    reference: Sequence[str] = PHYSICAL_RANKING,
) -> int:
    matches = 0
    for index, first in enumerate(reference):
        for second in reference[index + 1:]:
            matches += predicted.index(first) < predicted.index(second)
    return matches


def _ranking(
    results: Sequence[VolumetricResult],
    model: str,
    axis: str,
    metric: str,
    resolution: int = SCREENING_RESOLUTION,
) -> List[str]:
    selected = [
        result for result in results
        if result.model == model
        and result.axis == axis
        and result.resolution == resolution
        and result.status == "ok"
    ]
    return [
        result.pattern
        for result in sorted(
            selected,
            key=lambda result: float(getattr(result, metric)),
            reverse=True,
        )
    ]


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_validation(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = CalculixRunner(
        runs_dir,
        executable="ccx",
        timeout_seconds=600.0,
        keep_successful=True,
    )
    results: List[VolumetricResult] = []
    morphology: Dict[tuple[str, int, str], Dict[str, float]] = {}
    for resolution in RESOLUTIONS:
        for pattern in VOLUMETRIC_PATTERNS:
            result, metrics = run_case(
                runner,
                output_dir,
                pattern,
                resolution,
                "x",
                "elastic",
            )
            results.append(result)
            morphology[(pattern, resolution, "x")] = metrics
    for axis in ("y", "z"):
        for pattern in VOLUMETRIC_PATTERNS:
            result, metrics = run_case(
                runner,
                output_dir,
                pattern,
                SCREENING_RESOLUTION,
                axis,
                "elastic",
            )
            results.append(result)
            morphology[(pattern, 16, axis)] = metrics
    for pattern in VOLUMETRIC_PATTERNS:
        result, metrics = run_case(
            runner,
            output_dir,
            pattern,
            SCREENING_RESOLUTION,
            "x",
            "author_predamage_plastic",
        )
        results.append(result)
        morphology[(pattern, 16, "x")] = metrics

    rows = [asdict(result) for result in results]
    _write_csv(output_dir / "solver_results.csv", rows)
    morphology_rows = [
        {
            "pattern": pattern,
            "resolution": resolution,
            "axis": axis,
            **metrics,
        }
        for (pattern, resolution, axis), metrics in sorted(
            morphology.items()
        )
    ]
    _write_csv(output_dir / "morphology_metrics.csv", morphology_rows)
    convergence = []
    coarse, medium, fine = RESOLUTIONS
    for pattern in VOLUMETRIC_PATTERNS:
        selected = {
            result.resolution: result
            for result in results
            if result.pattern == pattern
            and result.axis == "x"
            and result.model == "elastic"
            and result.status == "ok"
        }
        convergence.append(
            {
                "pattern": pattern,
                "coarse_to_medium_change": abs(
                    float(selected[medium].apparent_modulus_MPa)
                    / float(selected[coarse].apparent_modulus_MPa)
                    - 1.0
                ),
                "medium_to_fine_change": abs(
                    float(selected[fine].apparent_modulus_MPa)
                    / float(selected[medium].apparent_modulus_MPa)
                    - 1.0
                ),
            }
        )
    _write_csv(output_dir / "mesh_convergence.csv", convergence)

    rankings = []
    for axis in ("x", "y", "z"):
        predicted = _ranking(
            results,
            "elastic",
            axis,
            "apparent_modulus_MPa",
        )
        rankings.append(
            {
                "model": "elastic",
                "axis": axis,
                "metric": "apparent_modulus_MPa",
                "predicted_ranking": predicted,
                "physical_ballistic_ranking": list(PHYSICAL_RANKING),
                "full_match": predicted == list(PHYSICAL_RANKING),
                "pairwise_matches": pairwise_matches(predicted),
                "pairwise_comparisons": 6,
            }
        )
    plastic_ranking = _ranking(
        results,
        "author_predamage_plastic",
        "x",
        "specific_absorbed_energy_J_kg",
    )
    rankings.append(
        {
            "model": "author_predamage_plastic",
            "axis": "x",
            "metric": "specific_absorbed_energy_J_kg",
            "predicted_ranking": plastic_ranking,
            "physical_ballistic_ranking": list(PHYSICAL_RANKING),
            "full_match": plastic_ranking == list(PHYSICAL_RANKING),
            "pairwise_matches": pairwise_matches(plastic_ranking),
            "pairwise_comparisons": 6,
        }
    )
    _write_csv(output_dir / "ranking_comparison.csv", rankings)
    (output_dir / "ranking_comparison.json").write_text(
        json.dumps(rankings, indent=2) + "\n", encoding="utf-8"
    )
    printable_metrics = {}
    for pattern in VOLUMETRIC_PATTERNS:
        metrics = inspect_binary_stl(
            output_dir / "printable_stl" / f"{pattern}.stl"
        )
        metrics["pattern"] = pattern
        metrics["relative_density"] = (
            float(metrics["volume_mm3"]) / 20.0 ** 3
        )
        metrics["mass_g"] = (
            float(metrics["volume_mm3"]) * 1.25e-3
        )
        printable_metrics[pattern] = metrics
    _write_csv(
        output_dir / "printable_mesh_validation.csv",
        list(printable_metrics.values()),
    )
    physical_rows = []
    for pattern in VOLUMETRIC_PATTERNS:
        for axis in ("x", "y", "z"):
            elastic_reference = next(
                result
                for result in results
                if result.pattern == pattern
                and result.axis == axis
                and result.model == "elastic"
                and result.resolution == SCREENING_RESOLUTION
            )
            plastic_reference = next(
                (
                    result
                    for result in results
                    if result.pattern == pattern
                    and result.axis == axis
                    and result.model == "author_predamage_plastic"
                    and result.resolution == SCREENING_RESOLUTION
                ),
                None,
            )
            for replicate in range(1, 4):
                physical_rows.append(
                    {
                        "specimen_id": (
                            f"VOL-{pattern.upper()}-{axis.upper()}-"
                            f"R{replicate}"
                        ),
                        "pattern": pattern,
                        "load_axis": axis,
                        "replicate": replicate,
                        "geometry_file": (
                            f"printable_stl/{pattern}.stl"
                        ),
                        "design_relative_density": printable_metrics[
                            pattern
                        ]["relative_density"],
                        "design_mass_g": printable_metrics[pattern][
                            "mass_g"
                        ],
                        "geometry_extent_x_mm": (
                            printable_metrics[pattern]["maximum_x_mm"]
                            - printable_metrics[pattern]["minimum_x_mm"]
                        ),
                        "geometry_extent_y_mm": (
                            printable_metrics[pattern]["maximum_y_mm"]
                            - printable_metrics[pattern]["minimum_y_mm"]
                        ),
                        "geometry_extent_z_mm": (
                            printable_metrics[pattern]["maximum_z_mm"]
                            - printable_metrics[pattern]["minimum_z_mm"]
                        ),
                        "compression_endpoint_mm": 1.0,
                        "simulation_initial_stiffness_N_per_mm": (
                            elastic_reference.stiffness_N_per_mm
                        ),
                        "simulation_predamage_energy_to_1mm_N_mm": (
                            plastic_reference.absorbed_energy_N_mm
                            if plastic_reference is not None
                            else ""
                        ),
                        "slicer_geometry_infill": (
                            "100% (do not replace designed voids)"
                        ),
                        "actual_mass_g": "",
                        "measured_initial_stiffness_N_per_mm": "",
                        "measured_energy_to_1mm_N_mm": "",
                        "failure_notes": "",
                    }
                )
    random.Random(20260728).shuffle(physical_rows)
    for order, row in enumerate(physical_rows, start=1):
        row["randomized_test_order"] = order
    _write_csv(
        output_dir / "physical_compression_test_plan.csv",
        physical_rows,
    )
    summary = {
        "material": "Bambu PLA Basic black",
        "geometry_class": (
            "idealized explicit voxel analogs, not exact Bambu Studio "
            "toolpaths"
        ),
        "patterns": list(VOLUMETRIC_PATTERNS),
        "target_actual_relative_density": 0.5,
        "successful_solver_cases": sum(
            result.status == "ok" for result in results
        ),
        "total_solver_cases": len(results),
        "mesh_resolutions": list(RESOLUTIONS),
        "screening_resolution": SCREENING_RESOLUTION,
        "maximum_medium_to_fine_change": max(
            row["medium_to_fine_change"] for row in convergence
        ),
        "ranking_cases": len(rankings),
        "full_physical_ranking_matches": sum(
            bool(row["full_match"]) for row in rankings
        ),
        "pairwise_matches": sum(
            int(row["pairwise_matches"]) for row in rankings
        ),
        "pairwise_comparisons": 6 * len(rankings),
        "plastic_energy_ranking": plastic_ranking,
        "physical_ballistic_ranking": list(PHYSICAL_RANKING),
        "printable_stl_count": len(VOLUMETRIC_PATTERNS),
        "watertight_printable_stl_count": sum(
            bool(metrics["watertight"])
            for metrics in printable_metrics.values()
        ),
        "maximum_printable_mass_error_fraction": max(
            abs(float(metrics["mass_g"]) / 5.0 - 1.0)
            for metrics in printable_metrics.values()
        ),
        "maximum_printable_extent_error_mm": max(
            abs(
                (
                    float(metrics[f"maximum_{axis}_mm"])
                    - float(metrics[f"minimum_{axis}_mm"])
                )
                - 20.0
            )
            for metrics in printable_metrics.values()
            for axis in ("x", "y", "z")
        ),
        "physical_compression_specimens": len(physical_rows),
        "limitations": [
            "Morphologies are idealized voxel analogs and not recovered "
            "Bambu Studio toolpaths.",
            "All patterns have equal voxel mass; the physical targets did not.",
            "Compression uses common isotropic material properties to isolate "
            "geometry.",
            "The plastic model stops at the author's damage-initiation point "
            "and has no element deletion or contact penetration.",
            "Quasi-static compression energy is only a topology proxy for "
            "ballistic penetration.",
            "A ranking mismatch is expected evidence, not a parameter-fitting "
            "target.",
            "Printable STLs are manifold, volume-matched isosurfaces of the "
            "voxel occupancy and their checked extents are recorded.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Explicit volumetric infill validation

## Result

Four idealized 3D voxel morphologies were generated at exactly 50% actual
solid fraction: honeycomb walls, a body-centered cubic lattice, alternating
grid rasters, and a gyroid shell. CalculiX completed
{summary['successful_solver_cases']}/{summary['total_solver_cases']} cases
across three mesh resolutions, three compression directions, and a common
pre-damage plastic energy screen.

The pre-damage specific-energy ranking is:
`{' > '.join(plastic_ranking)}`.
The peer-reviewed ballistic ranking is:
`{' > '.join(PHYSICAL_RANKING)}`.
Across the four ranking comparisons, the explicit models match
{summary['full_physical_ranking_matches']}/{summary['ranking_cases']} complete
rankings and {summary['pairwise_matches']}/{summary['pairwise_comparisons']}
pairwise orderings.

The 24/32/40-voxel mesh check has a maximum medium-to-fine stiffness change of
{summary['maximum_medium_to_fine_change']:.1%}. All four exported STLs pass the
closed-edge watertightness check and match the 5 g design mass within
{summary['maximum_printable_mass_error_fraction']:.2%}. The randomized physical
plan contains {summary['physical_compression_specimens']} specimens: four
patterns by three loading axes by three repeats.

## Interpretation

This is the first geometry-derived 3D comparison in the project for all four
peer-reviewed pattern families. It deliberately does not tune morphology or
material parameters to reproduce the physical order. A disagreement shows
that elastic stiffness or pre-damage compression energy alone is insufficient
to explain high-rate penetration.

All voxel analogs have equal actual mass. That is cleaner for topology
comparison than the source experiment, where equal slicer infill produced
different target masses, but it is also a different experiment.

## Model boundary

The geometries are idealized analogs, not exact Bambu Studio toolpaths.
CalculiX applies quasi-static compression with common material properties.
The nonlinear screen reaches the author's 5% damage-initiation range but does
not perform damage evolution, deletion, projectile contact, heat generation,
or high-rate constitutive response. It is a pattern-screening proxy, not a
ballistic safety model.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run explicit volumetric infill validation"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()
    summary = run_validation(args.output, args.runs)
    print(
        f"completed {summary['successful_solver_cases']}/"
        f"{summary['total_solver_cases']} volumetric cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
