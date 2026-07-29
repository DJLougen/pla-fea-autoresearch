from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .bambu_tds_profile import (
    DEFAULT_ISOTROPIC_PROFILE,
    DEFAULT_TRANSVERSE_PROFILE,
    write_tds_profiles,
)
from .infill_benchmark import LOAD_CASES
from .solid_multiload import (
    SolidLoadResult,
    _balanced_rows,
    _metric,
    run_solid_multiload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/tds_anisotropy_sensitivity"
DEFAULT_RUNS = PROJECT_ROOT / "runs/tds_anisotropy_sensitivity"
DEFAULT_PATTERNS = (
    "triangular",
    "x_braced",
    "staggered_diamond",
    "grid",
)


def compare_results(
    isotropic: Sequence[SolidLoadResult],
    transverse: Sequence[SolidLoadResult],
) -> List[Dict[str, object]]:
    iso_lookup = {(row.pattern, row.load_case): row for row in isotropic}
    rows: List[Dict[str, object]] = []
    for transverse_row in transverse:
        key = (transverse_row.pattern, transverse_row.load_case)
        isotropic_row = iso_lookup.get(key)
        iso_metric = None if isotropic_row is None else _metric(isotropic_row)
        transverse_metric = _metric(transverse_row)
        ratio = (
            None
            if iso_metric is None
            or transverse_metric is None
            or iso_metric <= 0.0
            else float(transverse_metric / iso_metric)
        )
        rows.append(
            {
                "pattern": transverse_row.pattern,
                "load_case": transverse_row.load_case,
                "isotropic_metric": iso_metric,
                "transverse_metric": transverse_metric,
                "transverse_to_isotropic_ratio": ratio,
                "percent_change": (
                    None if ratio is None else 100.0 * (ratio - 1.0)
                ),
                "isotropic_status": (
                    None if isotropic_row is None else isotropic_row.status
                ),
                "transverse_status": transverse_row.status,
            }
        )
    return rows


def _write_comparison(
    output_dir: Path,
    comparison: Sequence[Mapping[str, object]],
    isotropic: Sequence[SolidLoadResult],
    transverse: Sequence[SolidLoadResult],
    patterns: Sequence[str],
    load_cases: Sequence[str],
) -> None:
    with (output_dir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(comparison[0])
        )
        writer.writeheader()
        writer.writerows(comparison)
    (output_dir / "comparison.json").write_text(
        json.dumps(list(comparison), indent=2) + "\n", encoding="utf-8"
    )
    isotropic_balanced = {
        row["pattern"]: row["balanced_geometric_mean"]
        for row in _balanced_rows(isotropic, patterns, load_cases)
    }
    transverse_balanced = {
        row["pattern"]: row["balanced_geometric_mean"]
        for row in _balanced_rows(transverse, patterns, load_cases)
    }
    lines = [
        "# Bambu PLA Basic TDS anisotropy sensitivity",
        "",
        "Both paired models use E1 = E2 = 2580 MPa. The isotropic model also "
        "uses E3 = 2580 MPa and isotropic shear. The transverse sensitivity "
        "uses reported E3 = 2060 MPa, prior G13 = G23 = 780 MPa, and local "
        "1/2/3 aligned with global X/Y/Z for the flat coupons.",
        "",
        "| Pattern | Load | Isotropic metric | Transverse metric | Change |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison:
        iso = row["isotropic_metric"]
        transverse_value = row["transverse_metric"]
        change = row["percent_change"]
        lines.append(
            f"| {row['pattern']} | {row['load_case']} | "
            f"{'n/a' if iso is None else f'{float(iso):.3f}'} | "
            f"{'n/a' if transverse_value is None else f'{float(transverse_value):.3f}'} | "
            f"{'n/a' if change is None else f'{float(change):+.2f}%'} |"
        )
    lines.extend(
        [
            "",
            "## Balanced-score sensitivity",
            "",
            "| Pattern | Isotropic | Transverse | Change |",
            "|---|---:|---:|---:|",
        ]
    )
    for pattern in patterns:
        iso = isotropic_balanced.get(pattern)
        transverse_value = transverse_balanced.get(pattern)
        change = (
            None
            if iso is None or transverse_value is None or iso <= 0.0
            else 100.0 * (float(transverse_value) / float(iso) - 1.0)
        )
        lines.append(
            f"| {pattern} | "
            f"{'n/a' if iso is None else f'{float(iso):.3f}'} | "
            f"{'n/a' if transverse_value is None else f'{float(transverse_value):.3f}'} | "
            f"{'n/a' if change is None else f'{change:+.2f}%'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This paired run isolates the TDS engineering-constants "
            "hypothesis; it is not an inverse fit of shear or Poisson values.",
            "- The TDS supplies summary properties rather than raw directional "
            "curves, so no fixed held-out error can be computed for this run.",
            "- Strength, damage, fracture, temperature dependence, creep, "
            "fatigue, and print defects remain outside this elastic comparison.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_anisotropy_sensitivity(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
    load_cases: Sequence[str] = LOAD_CASES,
    mesh_size_mm: float = 0.7,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tds_profiles(output_dir / "profiles")
    isotropic = run_solid_multiload(
        output_dir / "isotropic",
        Path(runs_dir) / "isotropic",
        properties_path=DEFAULT_ISOTROPIC_PROFILE,
        patterns=patterns,
        load_cases=load_cases,
        mesh_size_mm=mesh_size_mm,
    )
    transverse = run_solid_multiload(
        output_dir / "transverse",
        Path(runs_dir) / "transverse",
        properties_path=DEFAULT_TRANSVERSE_PROFILE,
        patterns=patterns,
        load_cases=load_cases,
        mesh_size_mm=mesh_size_mm,
    )
    comparison = compare_results(isotropic, transverse)
    _write_comparison(
        output_dir,
        comparison,
        isotropic,
        transverse,
        patterns,
        load_cases,
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired isotropic/transverse Bambu TDS solid models"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
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
    args = parser.parse_args()
    comparison = run_anisotropy_sensitivity(
        Path(args.output),
        Path(args.runs_dir),
        patterns=args.patterns,
        load_cases=args.load_cases,
        mesh_size_mm=args.mesh_size,
    )
    successful = sum(
        row["isotropic_status"] == "ok"
        and row["transverse_status"] == "ok"
        for row in comparison
    )
    print(
        f"completed {successful}/{len(comparison)} paired comparisons; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
