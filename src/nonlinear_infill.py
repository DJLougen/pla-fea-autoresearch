from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .calculix import CalculixRunner
from .infill_benchmark import (
    PatternGeometry,
    _maximum_von_mises,
    make_pattern,
    render_scad,
)
from .parsing import parse_frd_scalar_max_history
from .solid_infill import (
    DEFAULT_PATTERNS,
    equal_solid_mass_strut_width,
    generate_prism_mesh,
)
from .solid_multiload import render_solid_static_deck


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_raw_curve_prepeak_plastic.json"
)
DEFAULT_ELASTIC_RESULTS = (
    PROJECT_ROOT
    / "results/solid_multiload_raw_calibrated/summary.json"
)


@dataclass
class NonlinearResult:
    pattern: str
    status: str
    mesh_size_mm: float
    nodes: int
    elements: int
    actual_mass_g: float
    target_global_strain: float
    endpoint_displacement_mm: float | None
    endpoint_force_N: float | None
    endpoint_secant_stiffness_N_per_mm: float | None
    elastic_reference_stiffness_N_per_mm: float | None
    endpoint_force_retention: float | None
    detected_yield_global_strain: float | None
    detected_yield_force_N: float | None
    maximum_von_mises_MPa: float | None
    maximum_equivalent_plastic_strain: float | None
    hardening_range_utilization: float | None
    within_calibrated_hardening_range: bool | None
    run_directory: str
    mesh_file: str
    error: str | None = None


def _maximum_peeq(curve: Dict[str, np.ndarray]) -> float | None:
    for key in ("frd_peeq", "frd_pe"):
        values = curve.get(key)
        if values is None:
            continue
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            return float(np.max(finite))
    return None


def _elastic_reference(path: Path) -> Dict[str, float]:
    if not Path(path).exists():
        return {}
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        row["pattern"]: float(row["stiffness_N_per_mm"])
        for row in rows
        if row["load_case"] == "tension"
        and row["status"] == "ok"
        and row["stiffness_N_per_mm"] is not None
    }


def _yield_from_force_curve(
    displacement: np.ndarray,
    force: np.ndarray,
    elastic_stiffness: float,
    reference_length_mm: float,
    retention_threshold: float = 0.995,
) -> tuple[float | None, float | None]:
    nonzero = displacement > max(1.0e-9, 0.002 * reference_length_mm)
    retention = np.divide(
        force,
        elastic_stiffness * displacement,
        out=np.ones_like(force),
        where=displacement > 0.0,
    )
    candidates = np.flatnonzero(nonzero & (retention < retention_threshold))
    if candidates.size == 0:
        return None, None
    index = int(candidates[0])
    return (
        float(displacement[index] / reference_length_mm),
        float(force[index]),
    )


def _yield_from_reference_curve(
    displacement: np.ndarray,
    force: np.ndarray,
    reference_displacement: np.ndarray,
    reference_force: np.ndarray,
    reference_length_mm: float,
    retention_threshold: float = 0.995,
) -> tuple[float | None, float | None, np.ndarray]:
    matched_reference = np.interp(
        displacement, reference_displacement, reference_force
    )
    retention = np.divide(
        force,
        matched_reference,
        out=np.ones_like(force),
        where=matched_reference > 0.0,
    )
    candidates = np.flatnonzero(
        (displacement > 0.002 * reference_length_mm)
        & (retention < retention_threshold)
    )
    if candidates.size == 0:
        return None, None, matched_reference
    index = int(candidates[0])
    return (
        float(displacement[index] / reference_length_mm),
        float(force[index]),
        matched_reference,
    )


def _yield_from_peeq_history(
    time: np.ndarray,
    displacement: np.ndarray,
    force: np.ndarray,
    peeq_history: Dict[float, float],
    reference_length_mm: float,
    peeq_threshold: float = 1.0e-10,
) -> tuple[float | None, float | None]:
    """Return the first saved state with non-negligible plastic strain."""
    candidates = [
        current_time
        for current_time, maximum_peeq in sorted(peeq_history.items())
        if maximum_peeq > peeq_threshold
    ]
    if not candidates:
        return None, None
    yield_time = candidates[0]
    order = np.argsort(time)
    ordered_time = np.asarray(time, dtype=float)[order]
    ordered_displacement = np.asarray(displacement, dtype=float)[order]
    ordered_force = np.asarray(force, dtype=float)[order]
    yield_displacement = float(
        np.interp(yield_time, ordered_time, ordered_displacement)
    )
    yield_force = float(np.interp(yield_time, ordered_time, ordered_force))
    return yield_displacement / reference_length_mm, yield_force


def run_nonlinear_tension(
    output_dir: Path,
    runs_dir: Path,
    profile_path: Path = DEFAULT_PROFILE,
    elastic_results_path: Path = DEFAULT_ELASTIC_RESULTS,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
    mesh_size_mm: float = 0.7,
    domain_width_mm: float = 40.0,
    domain_height_mm: float = 40.0,
    depth_mm: float = 4.0,
    relative_density: float = 0.35,
    target_global_strain: float = 0.015,
    geometries: Mapping[str, PatternGeometry] | None = None,
) -> List[NonlinearResult]:
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    elastic = profile["screening_isotropic"]
    plastic = profile["plastic_isotropic"]
    points = plastic["points"]
    modulus = float(elastic["E_MPa"])
    poisson = float(elastic["nu"])
    density = float(profile["density_tonnes_mm3"])
    maximum_supported_plastic_strain = float(
        plastic["maximum_supported_equivalent_plastic_strain"]
    )
    elastic_reference = _elastic_reference(elastic_results_path)
    target_displacement = domain_height_mm * target_global_strain
    output_dir = Path(output_dir)
    for directory in ("meshes", "decks", "patterns", "curves"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    (output_dir / "material_profile_used.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    runner = CalculixRunner(
        runs_dir,
        executable="ccx",
        timeout_seconds=900.0,
        keep_successful=True,
    )
    results: List[NonlinearResult] = []
    sampled_rows: List[Dict[str, object]] = []
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
        mass = mesh.actual_volume_mm3 * density * 1.0e6
        reference_deck_path = (
            output_dir / "decks" / f"{pattern}-elastic-reference.inp"
        )
        reference_deck_path.write_text(
            render_solid_static_deck(
                mesh,
                "tension",
                modulus,
                poisson,
                density,
                target_displacement,
                domain_width_mm,
                domain_height_mm,
                depth_mm,
                mesh_size_mm,
                incremental_nlgeom=True,
                output_element_fields=False,
            ),
            encoding="utf-8",
        )
        reference_solver = runner.run(
            reference_deck_path,
            {},
            run_id=f"nonlinear-reference-{pattern}",
        )
        if not reference_solver.ok:
            results.append(
                NonlinearResult(
                    pattern,
                    "elastic_reference_failed",
                    mesh_size_mm,
                    len(mesh.node_tags),
                    len(mesh.element_tags),
                    mass,
                    target_global_strain,
                    None,
                    None,
                    None,
                    elastic_reference.get(pattern),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    str(reference_solver.run_dir.resolve()),
                    str(mesh.mesh_path),
                    reference_solver.error,
                )
            )
            continue
        reference_displacement = np.abs(
            np.asarray(
                reference_solver.curve["displacement"], dtype=float
            )
        )
        reference_force = np.abs(
            np.asarray(
                reference_solver.curve["reaction_force"], dtype=float
            )
        )
        reference_order = np.argsort(reference_displacement)
        reference_displacement = reference_displacement[reference_order]
        reference_force = reference_force[reference_order]
        reference_stiffness = float(
            reference_force[-1] / reference_displacement[-1]
        )
        deck_path = output_dir / "decks" / f"{pattern}-tension.inp"
        deck_path.write_text(
            render_solid_static_deck(
                mesh,
                "tension",
                modulus,
                poisson,
                density,
                target_displacement,
                domain_width_mm,
                domain_height_mm,
                depth_mm,
                mesh_size_mm,
                plastic_points=points,
            ),
            encoding="utf-8",
        )
        solver = runner.run(
            deck_path,
            {},
            run_id=f"nonlinear-tension-{pattern}",
        )
        reference = reference_stiffness
        if not solver.ok:
            results.append(
                NonlinearResult(
                    pattern,
                    solver.status,
                    mesh_size_mm,
                    len(mesh.node_tags),
                    len(mesh.element_tags),
                    mass,
                    target_global_strain,
                    None,
                    None,
                    None,
                    reference,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    str(solver.run_dir.resolve()),
                    str(mesh.mesh_path),
                    solver.error,
                )
            )
            continue
        raw_displacement = np.abs(
            np.asarray(solver.curve["displacement"], dtype=float)
        )
        raw_force = np.abs(
            np.asarray(solver.curve["reaction_force"], dtype=float)
        )
        solver_time = np.asarray(solver.curve["time"], dtype=float)
        displacement = raw_displacement.copy()
        force = raw_force.copy()
        order = np.argsort(displacement)
        displacement = displacement[order]
        force = force[order]
        endpoint_displacement = float(displacement[-1])
        endpoint_force = float(force[-1])
        secant = endpoint_force / endpoint_displacement
        matched_reference = np.interp(
            displacement, reference_displacement, reference_force
        )
        peeq_history = parse_frd_scalar_max_history(
            solver.run_dir / "model.frd", "pe"
        )
        yield_strain, yield_force = _yield_from_peeq_history(
            solver_time,
            raw_displacement,
            raw_force,
            peeq_history,
            domain_height_mm,
        )
        retention = endpoint_force / matched_reference[-1]
        maximum_stress = _maximum_von_mises(solver.curve)
        maximum_peeq = _maximum_peeq(solver.curve)
        utilization = (
            None
            if maximum_peeq is None
            else maximum_peeq / maximum_supported_plastic_strain
        )
        curve_path = output_dir / "curves" / f"{pattern}.csv"
        with curve_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "pattern",
                    "global_strain",
                    "displacement_mm",
                    "force_N",
                    "elastic_reference_force_N",
                    "force_retention",
                ],
            )
            writer.writeheader()
            for current_displacement, current_force in zip(
                displacement, force
            ):
                elastic_force = float(
                    np.interp(
                        current_displacement,
                        reference_displacement,
                        reference_force,
                    )
                )
                writer.writerow(
                    {
                        "pattern": pattern,
                        "global_strain": (
                            current_displacement / domain_height_mm
                        ),
                        "displacement_mm": current_displacement,
                        "force_N": current_force,
                        "elastic_reference_force_N": elastic_force,
                        "force_retention": current_force / elastic_force,
                    }
                )
        for strain_level in dict.fromkeys(
            (0.01, 0.0125, 0.015, 0.0175, target_global_strain)
        ):
            if strain_level > target_global_strain + 1.0e-12:
                continue
            current_displacement = strain_level * domain_height_mm
            current_force = float(
                np.interp(current_displacement, displacement, force)
            )
            elastic_force = float(
                np.interp(
                    current_displacement,
                    reference_displacement,
                    reference_force,
                )
            )
            sampled_rows.append(
                {
                    "pattern": pattern,
                    "global_strain": strain_level,
                    "force_N": current_force,
                    "elastic_reference_force_N": elastic_force,
                    "force_retention": current_force / elastic_force,
                }
            )
        results.append(
            NonlinearResult(
                pattern,
                "ok",
                mesh_size_mm,
                len(mesh.node_tags),
                len(mesh.element_tags),
                mass,
                target_global_strain,
                endpoint_displacement,
                endpoint_force,
                secant,
                reference,
                retention,
                yield_strain,
                yield_force,
                maximum_stress,
                maximum_peeq,
                utilization,
                None if utilization is None else utilization <= 1.0,
                str(solver.run_dir.resolve()),
                str(mesh.mesh_path),
            )
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
    with (output_dir / "force_at_strain.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(sampled_rows[0])
        )
        writer.writeheader()
        writer.writerows(sampled_rows)
    _write_report(output_dir, results, profile)
    return results


def _write_report(
    output_dir: Path,
    results: Sequence[NonlinearResult],
    profile: Dict[str, object],
) -> None:
    successful = [row for row in results if row.status == "ok"]
    ranking = sorted(
        successful,
        key=lambda row: float(row.endpoint_force_N or 0.0),
        reverse=True,
    )
    lines = [
        "# Nonlinear pre-peak infill tension",
        "",
        (
            f"Profile `{profile['profile_id']}` applies the tensile-derived "
            "piecewise J2 hardening card to the refined equal-mass solid coupons."
        ),
        "",
        "| Rank | Pattern | Force (N) | Secant stiffness (N/mm) | "
        "Elastic-force retention | Detected yield strain | Max PEEQ | "
        "Hardening-range utilization |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranking, start=1):
        yield_text = (
            "n/a"
            if row.detected_yield_global_strain is None
            else f"{row.detected_yield_global_strain:.3%}"
        )
        lines.append(
            f"| {rank} | {row.pattern} | {row.endpoint_force_N:.3f} | "
            f"{row.endpoint_secant_stiffness_N_per_mm:.3f} | "
            f"{row.endpoint_force_retention:.3f} | "
            f"{yield_text} | "
            f"{'n/a' if row.maximum_equivalent_plastic_strain is None else f'{row.maximum_equivalent_plastic_strain:.6f}'} | "
            f"{'n/a' if row.hardening_range_utilization is None else f'{row.hardening_range_utilization:.3f}'} |"
        )
    failures = [row for row in results if row.status != "ok"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            lines.append(
                f"- {row.pattern}: {row.status}: {row.error}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Yield detection is the first saved state whose maximum equivalent plastic strain (PEEQ) exceeds 1e-10.",
            "- PEEQ is checked against the maximum plastic strain directly supported by the aggregate raw curve.",
            "- A result outside that range is retained for diagnosis but is not a calibrated prediction.",
            "- This J2 card is a monotonic in-layer tensile sensitivity model only; it does not predict crack initiation or final failure.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nonlinear pre-peak tension on refined infill solids"
    )
    parser.add_argument("--output", default="results/nonlinear_infill")
    parser.add_argument("--runs-dir", default="runs/nonlinear_infill")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument(
        "--elastic-results", default=str(DEFAULT_ELASTIC_RESULTS)
    )
    parser.add_argument(
        "--patterns", nargs="+", default=list(DEFAULT_PATTERNS)
    )
    parser.add_argument("--mesh-size", type=float, default=0.7)
    parser.add_argument("--target-strain", type=float, default=0.015)
    args = parser.parse_args()
    results = run_nonlinear_tension(
        Path(args.output),
        Path(args.runs_dir),
        profile_path=Path(args.profile),
        elastic_results_path=Path(args.elastic_results),
        patterns=args.patterns,
        mesh_size_mm=args.mesh_size,
        target_global_strain=args.target_strain,
    )
    successful = sum(row.status == "ok" for row in results)
    print(
        f"completed {successful}/{len(results)} nonlinear solid runs; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
