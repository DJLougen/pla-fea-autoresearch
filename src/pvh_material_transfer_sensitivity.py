from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .bambu_pvh_reference import (
    DEFAULT_OUTPUT as DEFAULT_REFERENCE,
    DEFAULT_SOURCE,
    import_reference,
)
from .calculix import CalculixRunner
from .volumetric_infill import (
    VOLUMETRIC_PATTERNS,
    make_voxel_mesh,
    render_voxel_compression_deck,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/pvh_material_transfer_sensitivity"
DEFAULT_RUNS = PROJECT_ROOT / "runs/pvh_material_transfer_compact"
PROJECT_CARD = (
    PROJECT_ROOT / "data/reported/bambu_pla_raw_curve_prepeak_plastic.json"
)
RESOLUTION = 24
SIDE_LENGTH_MM = 20.0
RELATIVE_DENSITY = 0.5
DEFORMATION = 0.05
LOAD_AXIS = "x"
DENSITY_TONNES_MM3 = 1.24e-9
CACHE_VERSION = 1


@dataclass(frozen=True)
class TransferResult:
    pattern: str
    material_card: str
    status: str
    resolution: int
    elements: int
    nodes: int
    actual_relative_density: float
    mass_g: float
    youngs_modulus_MPa: float
    poisson_ratio: float
    yield_stress_MPa: float
    endpoint_displacement_mm: float | None
    endpoint_reaction_force_N: float | None
    peak_reaction_force_N: float | None
    initial_stiffness_N_per_mm: float | None
    absorbed_energy_N_mm: float | None
    specific_absorbed_energy_J_kg: float | None
    run_directory: str
    error: str | None = None


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def material_cards(
    project_card_path: Path = PROJECT_CARD,
    reference_dir: Path = DEFAULT_REFERENCE,
) -> Dict[str, Dict[str, object]]:
    project = json.loads(
        Path(project_card_path).read_text(encoding="utf-8")
    )
    author = json.loads(
        (Path(reference_dir) / "author_model.json").read_text(
            encoding="utf-8"
        )
    )["material_card"]
    project_points = [
        {
            "stress_MPa": point["true_stress_MPa"],
            "equivalent_plastic_strain": point[
                "equivalent_plastic_strain"
            ],
        }
        for point in project["plastic_isotropic"]["points"]
    ]
    return {
        "project_raw_tensile": {
            "youngs_modulus_MPa": project["screening_isotropic"]["E_MPa"],
            "poisson_ratio": project["screening_isotropic"]["nu"],
            "yield_stress_MPa": project["plastic_isotropic"][
                "yield_true_stress_MPa"
            ],
            "plastic_points": project_points,
            "provenance": (
                "19 non-Z pre-peak Bambu PLA Basic tensile curves; "
                "compression symmetry unverified"
            ),
        },
        "independent_pvh_author": {
            "youngs_modulus_MPa": author["youngs_modulus_MPa"],
            "poisson_ratio": author["poisson_ratio"],
            "yield_stress_MPa": author["yield_stress_MPa"],
            "plastic_points": [
                {
                    "stress_MPa": author["yield_stress_MPa"],
                    "equivalent_plastic_strain": 0.0,
                }
            ],
            "provenance": (
                "independent peer-reviewed Bambu PLA Basic PVH study; "
                "elastic-perfectly-plastic completion because the author's "
                "hardening table is unpublished"
            ),
        },
    }


def _curve_energy(
    displacement: np.ndarray, reaction: np.ndarray
) -> float:
    x = np.abs(np.asarray(displacement, dtype=float))
    y = np.abs(np.asarray(reaction, dtype=float))
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x[0] > 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
    return float(np.trapezoid(y, x))


def _initial_stiffness(
    displacement: np.ndarray, reaction: np.ndarray
) -> float:
    x = np.abs(np.asarray(displacement, dtype=float))
    y = np.abs(np.asarray(reaction, dtype=float))
    selected = (x > 0.0) & (x <= SIDE_LENGTH_MM * 0.01)
    if int(np.sum(selected)) < 2:
        selected = np.arange(len(x)) < min(3, len(x))
    return float(
        np.dot(x[selected], y[selected])
        / np.dot(x[selected], x[selected])
    )


def run_case(
    runner: CalculixRunner,
    output_dir: Path,
    pattern: str,
    card_name: str,
    card: Mapping[str, object],
    resolution: int = RESOLUTION,
) -> TransferResult:
    identifier = f"{pattern}-n{resolution}-{LOAD_AXIS}-{card_name}"
    cache_dir = output_dir / "case_results"
    deck_dir = output_dir / "decks"
    curve_dir = output_dir / "curves"
    for directory in (cache_dir, deck_dir, curve_dir):
        directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{identifier}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_version") == CACHE_VERSION:
            return TransferResult(**cached["result"])

    mesh = make_voxel_mesh(
        pattern,
        resolution=resolution,
        side_length_mm=SIDE_LENGTH_MM,
        relative_density=RELATIVE_DENSITY,
        required_axes=(LOAD_AXIS,),
    )
    mass_g = (
        mesh.actual_volume_mm3 * DENSITY_TONNES_MM3 * 1.0e6
    )
    deck = render_voxel_compression_deck(
        mesh,
        LOAD_AXIS,
        DEFORMATION,
        youngs_modulus_MPa=float(card["youngs_modulus_MPa"]),
        poisson_ratio=float(card["poisson_ratio"]),
        density_tonnes_mm3=DENSITY_TONNES_MM3,
        plastic_points=card["plastic_points"],
    )
    deck_path = deck_dir / f"{identifier}.inp"
    deck_path.write_text(deck, encoding="utf-8")
    solver = runner.run(deck_path, {}, run_id=identifier)
    common = {
        "pattern": pattern,
        "material_card": card_name,
        "resolution": resolution,
        "elements": len(mesh.element_tags),
        "nodes": len(mesh.node_tags),
        "actual_relative_density": mesh.actual_relative_density,
        "mass_g": mass_g,
        "youngs_modulus_MPa": float(card["youngs_modulus_MPa"]),
        "poisson_ratio": float(card["poisson_ratio"]),
        "yield_stress_MPa": float(card["yield_stress_MPa"]),
    }
    if not solver.ok:
        return TransferResult(
            **common,
            status=solver.status,
            endpoint_displacement_mm=None,
            endpoint_reaction_force_N=None,
            peak_reaction_force_N=None,
            initial_stiffness_N_per_mm=None,
            absorbed_energy_N_mm=None,
            specific_absorbed_energy_J_kg=None,
            run_directory=str(solver.run_dir.resolve()),
            error=solver.error,
        )
    displacement = np.asarray(
        solver.curve["displacement"], dtype=float
    )
    reaction = np.asarray(
        solver.curve["reaction_force"], dtype=float
    )
    energy = _curve_energy(displacement, reaction)
    curve_rows = [
        {
            "displacement_mm": float(current_displacement),
            "reaction_force_N": float(current_reaction),
        }
        for current_displacement, current_reaction in zip(
            displacement, reaction
        )
    ]
    _write_csv(curve_dir / f"{identifier}.csv", curve_rows)
    result = TransferResult(
        **common,
        status="ok",
        endpoint_displacement_mm=float(abs(displacement[-1])),
        endpoint_reaction_force_N=float(abs(reaction[-1])),
        peak_reaction_force_N=float(np.max(np.abs(reaction))),
        initial_stiffness_N_per_mm=_initial_stiffness(
            displacement, reaction
        ),
        absorbed_energy_N_mm=energy,
        specific_absorbed_energy_J_kg=energy / mass_g,
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
    results: Sequence[TransferResult],
    card_name: str,
    metric: str,
) -> List[str]:
    selected = [
        row
        for row in results
        if row.material_card == card_name and row.status == "ok"
    ]
    return [
        row.pattern
        for row in sorted(
            selected,
            key=lambda row: float(getattr(row, metric)),
            reverse=True,
        )
    ]


def run_sensitivity(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    reference_dir: Path = DEFAULT_REFERENCE,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(reference_dir)
    import_reference(DEFAULT_SOURCE, reference_dir)
    cards = material_cards(reference_dir=reference_dir)
    (output_dir / "material_cards.json").write_text(
        json.dumps(cards, indent=2) + "\n", encoding="utf-8"
    )
    runner = CalculixRunner(
        Path(runs_dir),
        executable="ccx",
        timeout_seconds=600.0,
        keep_successful=True,
    )
    results: List[TransferResult] = []
    for card_name, card in cards.items():
        for pattern in VOLUMETRIC_PATTERNS:
            results.append(
                run_case(
                    runner,
                    output_dir,
                    pattern,
                    card_name,
                    card,
                )
            )
    _write_csv(
        output_dir / "solver_results.csv",
        [asdict(result) for result in results],
    )
    card_names = list(cards)
    rankings = {
        name: _ranking(
            results, name, "specific_absorbed_energy_J_kg"
        )
        for name in card_names
    }
    lookup = {
        (row.pattern, row.material_card): row
        for row in results
        if row.status == "ok"
    }
    comparison = []
    for pattern in VOLUMETRIC_PATTERNS:
        project = lookup[(pattern, "project_raw_tensile")]
        independent = lookup[(pattern, "independent_pvh_author")]
        comparison.append(
            {
                "pattern": pattern,
                "project_specific_energy_J_kg": (
                    project.specific_absorbed_energy_J_kg
                ),
                "independent_specific_energy_J_kg": (
                    independent.specific_absorbed_energy_J_kg
                ),
                "independent_to_project_energy_ratio": (
                    float(independent.specific_absorbed_energy_J_kg)
                    / float(project.specific_absorbed_energy_J_kg)
                ),
                "project_rank": (
                    rankings["project_raw_tensile"].index(pattern) + 1
                ),
                "independent_rank": (
                    rankings["independent_pvh_author"].index(pattern) + 1
                ),
                "rank_stable": (
                    rankings["project_raw_tensile"].index(pattern)
                    == rankings["independent_pvh_author"].index(pattern)
                ),
                "physical_next_step": (
                    "print three repeats in each load axis using the "
                    "existing compression plan"
                ),
            }
        )
    _write_csv(output_dir / "candidate_priority.csv", comparison)
    project_values = np.asarray(
        [
            lookup[(pattern, "project_raw_tensile")].specific_absorbed_energy_J_kg
            for pattern in VOLUMETRIC_PATTERNS
        ],
        dtype=float,
    )
    independent_values = np.asarray(
        [
            lookup[(pattern, "independent_pvh_author")].specific_absorbed_energy_J_kg
            for pattern in VOLUMETRIC_PATTERNS
        ],
        dtype=float,
    )
    ratios = independent_values / project_values
    successful = sum(row.status == "ok" for row in results)
    rank_correlation = float(
        spearmanr(project_values, independent_values).statistic
    )
    summary = {
        "material": "Bambu PLA Basic",
        "purpose": (
            "transfer independently measured exact-brand PLA properties "
            "onto reproducible explicit infill geometries"
        ),
        "solver_cases_successful": successful,
        "solver_cases_total": len(results),
        "patterns": list(VOLUMETRIC_PATTERNS),
        "material_cards": card_names,
        "specific_energy_rankings": rankings,
        "full_ranking_stable": (
            rankings["project_raw_tensile"]
            == rankings["independent_pvh_author"]
        ),
        "rank_stability_spearman": rank_correlation,
        "independent_to_project_energy_ratio_min": float(np.min(ratios)),
        "independent_to_project_energy_ratio_max": float(np.max(ratios)),
        "exact_published_pvh_replay_attempted": False,
        "exact_published_pvh_replay_exclusion": (
            "unpublished perturbation realization, CAD/model files, and "
            "post-yield hardening table"
        ),
        "interpretation": (
            "A stable ranking is robustness to two exact-brand material "
            "cards, not physical validation of the weird geometry."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Independent Bambu PLA material-card transfer

All **{successful}/{len(results)}** CalculiX cases solved. The same four
equal-mass explicit geometries were compressed to 5% strain with:

1. the project's 19-curve pre-peak Bambu PLA Basic card; and
2. the independent PVH paper's E = 2140 MPa, yield = 36 MPa card, completed
   as elastic-perfectly-plastic because its hardening curve is unpublished.

Project-card energy ranking:
`{' > '.join(rankings['project_raw_tensile'])}`.

Independent-card energy ranking:
`{' > '.join(rankings['independent_pvh_author'])}`.

Ranking Spearman correlation is **{rank_correlation:.3f}**. The independent
card changes predicted energy by **{float(np.min(ratios)):.3f}x to
{float(np.max(ratios)):.3f}x** depending on geometry.

This is a material-transfer robustness test. It does not claim an exact replay
of the paper's random PVH specimen and does not replace the 36-coupon physical
compression plan already exported by the volumetric benchmark.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer the independent PVH Bambu PLA card onto explicit "
            "weird infills"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()
    result = run_sensitivity(args.output, args.runs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
