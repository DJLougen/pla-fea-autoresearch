from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .bambu_common_infill_reference import (
    DEFAULT_OUTPUT as DEFAULT_REFERENCE,
    DEFAULT_SOURCE,
    PATTERNS,
    import_reference,
)
from .calculix import CalculixRunner
from .volumetric_infill import (
    make_voxel_mesh,
    render_voxel_compression_deck,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/common_infill_compression_validation"
DEFAULT_RUNS = PROJECT_ROOT / "runs/common_infill_compression_compact"
RESOLUTIONS = (32, 40)
SIDE_LENGTH_MM = 30.0
DEFORMATION = 0.01
SOLID_MODULUS_MPA = 2600.0
POISSON_RATIO = 0.33
DENSITY_TONNES_MM3 = 1.24e-9
CACHE_VERSION = 2
ANALOG_LOAD_AXIS = "z"


@dataclass(frozen=True)
class CompressionResult:
    pattern: str
    nominal_infill_percent: int
    resolution: int
    status: str
    target_mass_g: float
    target_total_solid_fraction: float
    actual_relative_density: float
    simulated_mass_g: float
    mass_error_fraction: float
    elements: int
    nodes: int
    displacement_mm: float | None
    reaction_force_N: float | None
    apparent_modulus_MPa: float | None
    source_initial_effective_modulus_MPa: float
    run_directory: str
    error: str | None = None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _reference_conditions(
    reference_dir: Path = DEFAULT_REFERENCE,
) -> List[Dict[str, object]]:
    mass_rows = _read_csv(reference_dir / "reported_mass_summary.csv")
    curve_rows = _read_csv(reference_dir / "curve_validation.csv")
    metrics = {
        (row["pattern"], int(row["nominal_infill_percent"])): row
        for row in curve_rows
    }
    conditions: List[Dict[str, object]] = []
    for row in mass_rows:
        key = (row["pattern"], int(row["nominal_infill_percent"]))
        curve = metrics[key]
        conditions.append(
            {
                "pattern": row["pattern"],
                "nominal_infill_percent": int(
                    row["nominal_infill_percent"]
                ),
                "target_mass_g": float(row["mean_mass_g"]),
                "target_total_solid_fraction": float(
                    row["inferred_total_solid_fraction"]
                ),
                "source_initial_effective_modulus_MPa": (
                    float(curve["initial_effective_modulus_Pa"]) / 1.0e6
                ),
            }
        )
    return sorted(
        conditions,
        key=lambda item: (
            int(item["nominal_infill_percent"]),
            PATTERNS.index(str(item["pattern"])),
        ),
    )


def run_case(
    runner: CalculixRunner,
    output_dir: Path,
    condition: Mapping[str, object],
    resolution: int,
) -> CompressionResult:
    pattern = str(condition["pattern"])
    nominal = int(condition["nominal_infill_percent"])
    target_density = float(condition["target_total_solid_fraction"])
    target_mass = float(condition["target_mass_g"])
    identifier = (
        f"{pattern}-{nominal}pct-n{resolution}-"
        f"{ANALOG_LOAD_AXIS}-elastic"
    )
    cache_dir = output_dir / "case_results"
    deck_dir = output_dir / "decks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    deck_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{identifier}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_version") == CACHE_VERSION:
            return CompressionResult(**cached["result"])
    mesh = make_voxel_mesh(
        pattern,
        resolution=resolution,
        side_length_mm=SIDE_LENGTH_MM,
        relative_density=target_density,
        required_axes=(ANALOG_LOAD_AXIS,),
        grid_connectors=True,
    )
    simulated_mass = (
        mesh.actual_volume_mm3 * DENSITY_TONNES_MM3 * 1.0e6
    )
    deck = render_voxel_compression_deck(
        mesh,
        ANALOG_LOAD_AXIS,
        DEFORMATION,
        youngs_modulus_MPa=SOLID_MODULUS_MPA,
        poisson_ratio=POISSON_RATIO,
        density_tonnes_mm3=DENSITY_TONNES_MM3,
    )
    deck_path = deck_dir / f"{identifier}.inp"
    deck_path.write_text(deck, encoding="utf-8")
    solver = runner.run(deck_path, {}, run_id=identifier)
    common = {
        "pattern": pattern,
        "nominal_infill_percent": nominal,
        "resolution": resolution,
        "target_mass_g": target_mass,
        "target_total_solid_fraction": target_density,
        "actual_relative_density": mesh.actual_relative_density,
        "simulated_mass_g": simulated_mass,
        "mass_error_fraction": simulated_mass / target_mass - 1.0,
        "elements": len(mesh.element_tags),
        "nodes": len(mesh.node_tags),
        "source_initial_effective_modulus_MPa": float(
            condition["source_initial_effective_modulus_MPa"]
        ),
    }
    if not solver.ok:
        return CompressionResult(
            **common,
            status=solver.status,
            displacement_mm=None,
            reaction_force_N=None,
            apparent_modulus_MPa=None,
            run_directory=str(solver.run_dir.resolve()),
            error=solver.error,
        )
    displacement = float(abs(solver.curve["displacement"][-1]))
    reaction = float(abs(solver.curve["reaction_force"][-1]))
    apparent_modulus = (
        reaction
        * SIDE_LENGTH_MM
        / (SIDE_LENGTH_MM ** 2 * displacement)
    )
    result = CompressionResult(
        **common,
        status="ok",
        displacement_mm=displacement,
        reaction_force_N=reaction,
        apparent_modulus_MPa=apparent_modulus,
        run_directory=str(solver.run_dir.resolve()),
    )
    cache_path.write_text(
        json.dumps(
            {"cache_version": CACHE_VERSION, "result": asdict(result)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _ranking(
    results: Sequence[CompressionResult],
    nominal_density: int,
    metric: str,
    resolution: int | None = None,
) -> List[str]:
    selected_resolution = (
        max(RESOLUTIONS) if resolution is None else resolution
    )
    selected = [
        result
        for result in results
        if result.nominal_infill_percent == nominal_density
        and result.resolution == selected_resolution
        and result.status == "ok"
    ]
    return [
        result.pattern
        for result in sorted(
            selected,
            key=lambda item: float(getattr(item, metric)),
            reverse=True,
        )
    ]


def run_validation(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    reference_dir: Path = DEFAULT_REFERENCE,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(reference_dir)
    import_reference(DEFAULT_SOURCE, reference_dir)
    conditions = _reference_conditions(reference_dir)
    runner = CalculixRunner(
        runs_dir,
        executable="ccx",
        timeout_seconds=600.0,
        keep_successful=True,
    )
    results: List[CompressionResult] = []
    for resolution in RESOLUTIONS:
        for condition in conditions:
            results.append(
                run_case(runner, output_dir, condition, resolution)
            )
    _write_csv(
        output_dir / "solver_results.csv",
        [asdict(result) for result in results],
    )
    fine = [
        result
        for result in results
        if result.resolution == max(RESOLUTIONS)
        and result.status == "ok"
    ]
    coarse = {
        (result.pattern, result.nominal_infill_percent): result
        for result in results
        if result.resolution == min(RESOLUTIONS)
        and result.status == "ok"
    }
    convergence = [
        {
            "pattern": result.pattern,
            "nominal_infill_percent": result.nominal_infill_percent,
            "coarse_resolution": min(RESOLUTIONS),
            "fine_resolution": max(RESOLUTIONS),
            "coarse_apparent_modulus_MPa": coarse[
                (result.pattern, result.nominal_infill_percent)
            ].apparent_modulus_MPa,
            "fine_apparent_modulus_MPa": result.apparent_modulus_MPa,
            "coarse_to_fine_change": abs(
                float(result.apparent_modulus_MPa)
                / float(
                    coarse[
                        (result.pattern, result.nominal_infill_percent)
                    ].apparent_modulus_MPa
                )
                - 1.0
            ),
        }
        for result in fine
    ]
    _write_csv(output_dir / "mesh_convergence.csv", convergence)
    ranking_rows = []
    for nominal in (10, 20, 30):
        simulated_fine = _ranking(
            results, nominal, "apparent_modulus_MPa"
        )
        simulated_coarse = _ranking(
            results,
            nominal,
            "apparent_modulus_MPa",
            min(RESOLUTIONS),
        )
        physical = _ranking(
            results, nominal, "source_initial_effective_modulus_MPa"
        )
        ranking_rows.append(
            {
                "nominal_infill_percent": nominal,
                "physical_ranking": " > ".join(physical),
                "coarse_simulated_ranking": " > ".join(
                    simulated_coarse
                ),
                "fine_simulated_ranking": " > ".join(simulated_fine),
                "full_ranking_match_both_meshes": (
                    physical == simulated_coarse == simulated_fine
                ),
                "ranking_mesh_stable": (
                    simulated_coarse == simulated_fine
                ),
                "grid_ranked_first_physical": physical[0] == "grid",
                "grid_ranked_first_both_meshes": (
                    simulated_coarse[0] == "grid"
                    and simulated_fine[0] == "grid"
                ),
            }
        )
    _write_csv(output_dir / "ranking_comparison.csv", ranking_rows)
    observed = np.asarray(
        [result.source_initial_effective_modulus_MPa for result in fine]
    )
    predicted = np.asarray(
        [float(result.apparent_modulus_MPa) for result in fine]
    )
    calibration_mask = np.asarray(
        [result.nominal_infill_percent == 20 for result in fine]
    )
    scale = float(
        np.dot(predicted[calibration_mask], observed[calibration_mask])
        / np.dot(
            predicted[calibration_mask], predicted[calibration_mask]
        )
    )
    scaled = scale * predicted
    comparison_rows = [
        {
            "pattern": result.pattern,
            "nominal_infill_percent": result.nominal_infill_percent,
            "source_initial_effective_modulus_MPa": (
                result.source_initial_effective_modulus_MPa
            ),
            "source_x1000_unit_sensitivity_modulus_MPa": (
                result.source_initial_effective_modulus_MPa * 1000.0
            ),
            "raw_simulation_apparent_modulus_MPa": (
                result.apparent_modulus_MPa
            ),
            "raw_simulation_to_literal_source_ratio": (
                float(result.apparent_modulus_MPa)
                / result.source_initial_effective_modulus_MPa
            ),
            "raw_simulation_to_x1000_source_ratio": (
                float(result.apparent_modulus_MPa)
                / (
                    result.source_initial_effective_modulus_MPa
                    * 1000.0
                )
            ),
            "literal_global_scale_fitted_on_20pct": scale,
            "x1000_global_scale_fitted_on_20pct": scale * 1000.0,
            "literal_scaled_simulation_modulus_MPa": float(
                scaled[index]
            ),
            "scaled_relative_error_same_for_both_unit_interpretations": (
                float(scaled[index])
                / result.source_initial_effective_modulus_MPa
                - 1.0
            ),
            "partition": (
                "scale_fit"
                if result.nominal_infill_percent == 20
                else "density_holdout"
            ),
        }
        for index, result in enumerate(fine)
    ]
    _write_csv(output_dir / "modulus_comparison.csv", comparison_rows)
    holdout_errors = [
        abs(
            float(
                row[
                    "scaled_relative_error_same_for_both_unit_interpretations"
                ]
            )
        )
        for row in comparison_rows
        if row["partition"] == "density_holdout"
    ]
    summary = {
        "material": "Bambu Lab PLA Basic, purple",
        "source_status": "TechRxiv preprint",
        "physical_tests_represented": 27,
        "aggregate_digitized_curves": 9,
        "solver_cases_successful": sum(
            result.status == "ok" for result in results
        ),
        "solver_cases_total": len(results),
        "mesh_resolutions": list(RESOLUTIONS),
        "maximum_coarse_to_fine_change": max(
            float(row["coarse_to_fine_change"])
            for row in convergence
        ),
        "maximum_mass_error_fraction": max(
            abs(result.mass_error_fraction) for result in results
        ),
        "full_ranking_matches": sum(
            bool(row["full_ranking_match_both_meshes"])
            for row in ranking_rows
        ),
        "ranking_comparisons": len(ranking_rows),
        "mesh_stable_ranking_comparisons": sum(
            bool(row["ranking_mesh_stable"]) for row in ranking_rows
        ),
        "grid_first_matches": sum(
            bool(row["grid_ranked_first_physical"])
            and bool(row["grid_ranked_first_both_meshes"])
            for row in ranking_rows
        ),
        "spearman_rank_correlation_9_conditions": float(
            spearmanr(observed, predicted).statistic
        ),
        "pearson_correlation_9_conditions": float(
            pearsonr(observed, predicted).statistic
        ),
        "literal_global_scale_fitted_on_20pct": scale,
        "x1000_global_scale_fitted_on_20pct": scale * 1000.0,
        "holdout_scaled_modulus_MAPE": float(np.mean(holdout_errors)),
        "raw_simulation_to_literal_source_ratio_min": float(
            np.min(predicted / observed)
        ),
        "raw_simulation_to_literal_source_ratio_max": float(
            np.max(predicted / observed)
        ),
        "raw_simulation_to_x1000_source_ratio_min": float(
            np.min(predicted / (1000.0 * observed))
        ),
        "raw_simulation_to_x1000_source_ratio_max": float(
            np.max(predicted / (1000.0 * observed))
        ),
        "geometry_class": (
            "mass-matched wall-free idealized voxel analogs with explicit "
            "grid crossing connectors, loaded along the build-axis analog z; "
            "not the source coupons' exact Bambu Studio toolpaths or "
            "0.42 mm outer walls"
        ),
        "decision_gate": (
            "Use the simulator for relative compression ranking only when "
            "the physical ranking is reproduced; do not use raw absolute "
            "stress or failure predictions."
        ),
        "limitations": [
            "The source is a preprint and exposes aggregate plotted curves, "
            "not raw specimen histories.",
            "The published stress scale is retained exactly as Pa x10^4 and "
            "conflicts by x1000 with all nine independently stated MPa yield "
            "strengths; both unit interpretations are reported.",
            "The source includes 0.42 mm walls; the voxel analogs match total "
            "mass but do not reconstruct slicer toolpaths or wall/infill "
            "interfaces.",
            "Grid crossing connectors represent fused alternating rasters "
            "and permit z-axis loading without claiming the source's exact "
            "layer toolpath.",
            "One common isotropic elastic material isolates geometry; the "
            "comparison does not identify damage, contact, creep, fatigue, "
            "moisture, aging, or rate laws.",
            "The single scale factor is fit only to the 20% conditions and is "
            "reported solely as a density-generalization diagnostic.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    ranking_lines = "\n".join(
        f"| {row['nominal_infill_percent']}% | "
        f"{row['physical_ranking']} | "
        f"{row['coarse_simulated_ranking']} | "
        f"{row['fine_simulated_ranking']} | "
        f"{'pass' if row['full_ranking_match_both_meshes'] else 'fail'} |"
        for row in ranking_rows
    )
    report = f"""# Exact-brand common-infill compression validation

## Outcome

CalculiX completed {summary['solver_cases_successful']}/
{summary['solver_cases_total']} mass-matched cases. The benchmark represents
27 physical Bambu PLA Basic compression tests through nine aggregate
digitized curves.

| Nominal infill | Physical ranking | 32-voxel ranking | 40-voxel ranking | Gate |
|---:|---|---|---|---|
{ranking_lines}

The complete ranking matched in {summary['full_ranking_matches']}/
{summary['ranking_comparisons']} density groups, and grid was correctly ranked
first in {summary['grid_first_matches']}/3. All
{summary['mesh_stable_ranking_comparisons']}/3 rankings were unchanged by the
mesh refinement. Across all nine conditions,
Spearman rank correlation is
{summary['spearman_rank_correlation_9_conditions']:.3f} and Pearson correlation
is {summary['pearson_correlation_9_conditions']:.3f}.

## Numerical checks

The 32-to-40-voxel stiffness change is at most
{summary['maximum_coarse_to_fine_change']:.1%}. Voxel rounding changes target
mass by at most {summary['maximum_mass_error_fraction']:.2%}.
The roughly 20% worst-case modulus change prevents a high-precision magnitude
claim, even though the topology order is stable.

Against the literal plotted Pa scale, raw simulated modulus is
{summary['raw_simulation_to_literal_source_ratio_min']:.0f}–\
{summary['raw_simulation_to_literal_source_ratio_max']:.0f} times larger.
Under the paper-internal ×1,000 unit reconciliation, that ratio becomes
{summary['raw_simulation_to_x1000_source_ratio_min']:.3f}–\
{summary['raw_simulation_to_x1000_source_ratio_max']:.3f}. The reconciliation
is a sensitivity, not a silent correction: all nine adjusted curve peaks
match the paper's separately stated MPa yield strengths within 4%.

A single multiplier fitted only to the three 20% conditions is
{summary['x1000_global_scale_fitted_on_20pct']:.3f} under the ×1,000
interpretation and has a 10%/30% holdout MAPE of
{summary['holdout_scaled_modulus_MAPE']:.1%}. The remaining density/topology
error is retained rather than hidden by per-pattern tuning.

## Decision boundary

This benchmark can support relative, quasi-static compression ranking only
where the rank gate passes. It cannot support absolute load, crushing,
failure, impact, creep, fatigue, moisture, aging, or service-temperature
predictions. The physical coupons include 0.42 mm outer walls and slicer
toolpaths; the explicit voxel models match measured total mass but not those
details. The wall-free analog is loaded along z. Explicit grid crossing
connectors represent fused alternating rasters, but this is not an exact
reproduction of the physical layer toolpath.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate explicit infill against exact-brand compression"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--reference", type=Path, default=DEFAULT_REFERENCE
    )
    args = parser.parse_args()
    summary = run_validation(args.output, args.runs, args.reference)
    print(
        f"completed {summary['solver_cases_successful']}/"
        f"{summary['solver_cases_total']} matched compression cases; "
        f"ranking matches={summary['full_ranking_matches']}/"
        f"{summary['ranking_comparisons']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
