from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

from .solid_infill import DEFAULT_PATTERNS
from .infill_benchmark import DEFAULT_PROPERTIES
from .solid_multiload import SolidLoadResult, run_solid_multiload


def _mesh_label(mesh_size_mm: float) -> str:
    return f"h{mesh_size_mm:g}".replace(".", "p")


def run_buckling_convergence(
    output_dir: Path,
    runs_dir: Path,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
    mesh_sizes_mm: Sequence[float] = (1.2, 0.9, 0.7),
    properties_path: Path = DEFAULT_PROPERTIES,
) -> List[SolidLoadResult]:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    results: List[SolidLoadResult] = []
    for mesh_size in mesh_sizes_mm:
        label = _mesh_label(mesh_size)
        results.extend(
            run_solid_multiload(
                output_dir / label,
                runs_dir / label,
                properties_path=properties_path,
                patterns=patterns,
                load_cases=["buckling"],
                mesh_size_mm=mesh_size,
            )
        )
    fields = [
        "pattern",
        "mesh_size_mm",
        "nodes",
        "elements",
        "critical_buckling_load_N",
        "critical_buckling_load_N_per_g",
        "status",
        "run_directory",
    ]
    with (output_dir / "buckling_mesh_convergence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: getattr(row, field) for field in fields})
    grouped: Dict[str, List[SolidLoadResult]] = {}
    for row in results:
        grouped.setdefault(row.pattern, []).append(row)
    lines = [
        "# 3D buckling mesh convergence",
        "",
        (
            "Critical loads use the first unconstrained 3D eigenmode of the "
            "equal-actual-mass Bambu PLA Basic coupon."
        ),
    ]
    for pattern in patterns:
        rows = sorted(
            grouped.get(pattern, []),
            key=lambda row: row.mesh_size_mm,
            reverse=True,
        )
        lines.extend(
            [
                "",
                f"## {pattern}",
                "",
                "| Mesh (mm) | Nodes | Elements | Critical load (N) |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            value = (
                "n/a"
                if row.critical_buckling_load_N is None
                else f"{row.critical_buckling_load_N:.3f}"
            )
            lines.append(
                f"| {row.mesh_size_mm:g} | {row.nodes} | "
                f"{row.elements} | {value} |"
            )
        successful = [
            row
            for row in rows
            if row.status == "ok"
            and row.critical_buckling_load_N is not None
        ]
        if len(successful) >= 2:
            previous, fine = successful[-2], successful[-1]
            change = abs(
                fine.critical_buckling_load_N
                - previous.critical_buckling_load_N
            ) / fine.critical_buckling_load_N
            lines.extend(
                [
                    "",
                    f"Finest-step critical-load change: {change:.2%}.",
                ]
            )
    lines.extend(
        [
            "",
            "The refined 0.7 mm values are used by the solid multi-load ranking.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run three-level convergence of 3D solid buckling loads"
    )
    parser.add_argument(
        "--output", default="results/solid_buckling_convergence"
    )
    parser.add_argument(
        "--runs-dir", default="runs/solid_buckling_convergence"
    )
    parser.add_argument("--properties", default=str(DEFAULT_PROPERTIES))
    parser.add_argument(
        "--patterns", nargs="+", default=list(DEFAULT_PATTERNS)
    )
    parser.add_argument(
        "--mesh-sizes",
        nargs="+",
        type=float,
        default=[1.2, 0.9, 0.7],
    )
    args = parser.parse_args()
    rows = run_buckling_convergence(
        Path(args.output),
        Path(args.runs_dir),
        patterns=args.patterns,
        mesh_sizes_mm=args.mesh_sizes,
        properties_path=Path(args.properties),
    )
    successful = sum(row.status == "ok" for row in rows)
    print(
        f"completed {successful}/{len(rows)} buckling convergence runs; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
