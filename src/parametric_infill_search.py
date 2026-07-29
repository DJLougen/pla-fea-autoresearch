from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .bambu_tds_profile import DEFAULT_TRANSVERSE_PROFILE
from .infill_benchmark import (
    LOAD_CASES,
    PatternGeometry,
    make_pattern,
    make_staggered_diamond_variant,
    run_benchmark,
)
from .nonlinear_infill import run_nonlinear_tension
from .solid_multiload import _balanced_rows, run_solid_multiload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/parametric_infill_search"
DEFAULT_RUNS = PROJECT_ROOT / "runs/parametric_infill_search"
CELL_COUNTS = (2, 3, 4, 5)
OFFSETS = (0.2, 0.35, 0.5, 0.65, 0.8)
CONTROLS = ("triangular", "x_braced", "grid")


def build_search_geometries(
    cell_counts: Sequence[int] = CELL_COUNTS,
    offsets: Sequence[float] = OFFSETS,
) -> Dict[str, PatternGeometry]:
    geometries = {
        name: make_pattern(name) for name in CONTROLS
    }
    for cells in cell_counts:
        for offset in offsets:
            name = f"sd_c{cells}_o{int(round(100 * offset)):02d}"
            geometries[name] = make_staggered_diamond_variant(
                name,
                cells=int(cells),
                offset_fraction=float(offset),
            )
    return geometries


def _read_ranking(path: Path) -> List[Dict[str, object]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key == "pattern":
                continue
            row[key] = None if value == "" else float(value)
    return rows


def _write_search_report(
    output_dir: Path,
    beam_ranking: Sequence[Mapping[str, object]],
    solid_ranking: Sequence[Mapping[str, object]],
    promoted: Sequence[str],
) -> None:
    lines = [
        "# Parametric staggered-diamond infill search",
        "",
        "All candidates use the same 40 × 40 × 4 mm envelope and 35% "
        "material fraction. The beam stage screens cell count and alternating "
        "row offset; the promoted stage uses equal-actual-mass 0.7 mm C3D6 "
        "solids with the Bambu TDS transverse-isotropic sensitivity profile.",
        "",
        f"Promoted variants: {', '.join(promoted)}.",
        "",
        "## Beam-screen leaders",
        "",
        "| Rank | Pattern | Balanced score |",
        "|---:|---|---:|",
    ]
    for rank, row in enumerate(beam_ranking[:12], start=1):
        lines.append(
            f"| {rank} | {row['pattern']} | "
            f"{float(row['balanced_geometric_mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Refined 3D leaders",
            "",
            "| Rank | Pattern | Balanced score |",
            "|---:|---|---:|",
        ]
    )
    for rank, row in enumerate(solid_ranking, start=1):
        score = row["balanced_geometric_mean"]
        lines.append(
            f"| {rank} | {row['pattern']} | "
            f"{'n/a' if score is None else f'{float(score):.3f}'} |"
        )
    winner = solid_ranking[0]["pattern"] if solid_ranking else "n/a"
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"The refined equal-mass winner in this search is `{winner}`. "
            "Beam rank alone is not treated as decisive because earlier planar "
            "buckling results overpredicted X-braced performance by orders of "
            "magnitude.",
            "",
            "See `solid/report.md` for per-load forces, stresses, buckling "
            "loads, exact strut widths, and the generated physical test plan.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_search(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    properties_path: Path = DEFAULT_TRANSVERSE_PROFILE,
    promote_count: int = 3,
    mesh_size_mm: float = 0.7,
) -> List[Mapping[str, object]]:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometries = build_search_geometries()
    names = list(geometries)
    run_benchmark(
        output_dir / "beam",
        runs_dir / "beam",
        properties_path=properties_path,
        patterns=names,
        geometries=geometries,
    )
    beam_ranking = _read_ranking(
        output_dir / "beam/multi_load_ranking.csv"
    )
    variants = [
        row for row in beam_ranking if str(row["pattern"]).startswith("sd_")
    ]
    promoted = [
        str(row["pattern"]) for row in variants[:promote_count]
    ]
    current = "sd_c3_o50"
    solid_names = list(
        dict.fromkeys([*CONTROLS, current, *promoted])
    )
    solid_geometries = {
        name: geometries[name] for name in solid_names
    }
    solid_results = run_solid_multiload(
        output_dir / "solid",
        runs_dir / "solid",
        properties_path=properties_path,
        patterns=solid_names,
        load_cases=LOAD_CASES,
        mesh_size_mm=mesh_size_mm,
        geometries=solid_geometries,
    )
    solid_ranking = _balanced_rows(
        solid_results, solid_names, LOAD_CASES
    )
    (output_dir / "search_config.json").write_text(
        json.dumps(
            {
                "cell_counts": list(CELL_COUNTS),
                "offsets": list(OFFSETS),
                "controls": list(CONTROLS),
                "promoted": promoted,
                "solid_patterns": solid_names,
                "properties": str(Path(properties_path).resolve()),
                "mesh_size_mm": mesh_size_mm,
                "relative_density": 0.35,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_search_report(
        output_dir, beam_ranking, solid_ranking, promoted
    )
    return solid_ranking


def run_promoted_nonlinear_check(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    challenger: str = "sd_c2_o50",
    target_strain: float = 0.008,
):
    geometries = build_search_geometries()
    if challenger not in geometries:
        raise ValueError(f"unknown parametric challenger {challenger!r}")
    selected = {
        "triangular": geometries["triangular"],
        challenger: geometries[challenger],
    }
    results = run_nonlinear_tension(
        Path(output_dir) / "nonlinear",
        Path(runs_dir) / "nonlinear",
        patterns=list(selected),
        target_global_strain=target_strain,
        geometries=selected,
    )
    _write_nonlinear_decision(Path(output_dir), results, challenger)
    return results


def _write_nonlinear_decision(output_dir, results, challenger: str) -> None:
    lookup = {row.pattern: row for row in results}
    challenger_row = lookup[challenger]
    control = lookup["triangular"]
    yield_text = (
        "not detected"
        if challenger_row.detected_yield_global_strain is None
        else f"{challenger_row.detected_yield_global_strain:.3%}"
    )
    section = [
        "## Nonlinear promotion gate",
        "",
        (
            f"At {challenger_row.target_global_strain:.2%} global tension, "
            f"`{challenger}` carries {challenger_row.endpoint_force_N:.3f} N "
            f"versus triangular's {control.endpoint_force_N:.3f} N."
        ),
        (
            f"The challenger first records local plasticity at {yield_text}; "
            f"its maximum PEEQ is "
            f"{challenger_row.maximum_equivalent_plastic_strain:.6f}, "
            f"using {challenger_row.hardening_range_utilization:.1%} of the "
            "raw-curve-supported hardening range."
        ),
        "",
        (
            f"`{challenger}` remains inside the raw-curve-supported plastic-"
            f"strain range at this {challenger_row.target_global_strain:.2%} "
            "endpoint. This is not a strength-safety gate: directional "
            "normal-stress screening and physical fracture testing remain "
            "separate requirements."
        ),
        "",
    ]
    decision_path = output_dir / "nonlinear_decision.md"
    decision_path.write_text("\n".join(section), encoding="utf-8")
    report_path = output_dir / "report.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        marker = "\n## Nonlinear promotion gate\n"
        if marker in text:
            text = text.split(marker, 1)[0].rstrip() + "\n"
        report_path.write_text(
            text.rstrip() + "\n\n" + "\n".join(section),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search staggered-diamond cell count and row offset"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument(
        "--properties", default=str(DEFAULT_TRANSVERSE_PROFILE)
    )
    parser.add_argument("--promote-count", type=int, default=3)
    parser.add_argument("--mesh-size", type=float, default=0.7)
    parser.add_argument(
        "--nonlinear-only",
        action="store_true",
        help="Run the promoted challenger and triangular at 0.8% strain",
    )
    parser.add_argument("--challenger", default="sd_c2_o50")
    parser.add_argument("--target-strain", type=float, default=0.008)
    args = parser.parse_args()
    if args.nonlinear_only:
        results = run_promoted_nonlinear_check(
            Path(args.output),
            Path(args.runs_dir),
            challenger=args.challenger,
            target_strain=args.target_strain,
        )
        successful = sum(row.status == "ok" for row in results)
        print(
            f"nonlinear follow-up completed {successful}/{len(results)}; "
            f"report: {(Path(args.output) / 'nonlinear/report.md').resolve()}"
        )
        return
    ranking = run_search(
        Path(args.output),
        Path(args.runs_dir),
        Path(args.properties),
        promote_count=args.promote_count,
        mesh_size_mm=args.mesh_size,
    )
    print(
        f"parametric search completed; winner={ranking[0]['pattern']}; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
