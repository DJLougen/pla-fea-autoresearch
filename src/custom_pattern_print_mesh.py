from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping

from .volumetric_infill import inspect_binary_stl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ROOT = (
    PROJECT_ROOT / "results/custom_infill/graded_hourglass"
)
DEFAULT_PLAN = CUSTOM_ROOT / "first_physical_test_plan.csv"
DEFAULT_OUTPUT = CUSTOM_ROOT / "physical_validation/printable_stl"
DENSITY_G_PER_MM3 = 1.24e-3
MAXIMUM_MASS_RELATIVE_ERROR = 0.001
BOUND_TOLERANCE_MM = 1e-6


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _target_masses(plan_path: Path) -> Dict[str, float]:
    with Path(plan_path).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    targets: Dict[str, set[float]] = {}
    for row in rows:
        targets.setdefault(row["pattern"], set()).add(
            float(row["target_actual_mass_g"])
        )
    if "triangular" not in targets or len(targets) != 2:
        raise ValueError(
            "print mesh plan must contain triangular and exactly one custom "
            "pattern"
        )
    result = {}
    for pattern, values in targets.items():
        if len(values) != 1:
            raise ValueError(
                f"target mass is not constant for {pattern}"
            )
        result[pattern] = values.pop()
    return result


def _bounds_pass(metrics: Mapping[str, object]) -> bool:
    expected = {
        "minimum_x_mm": 0.0,
        "minimum_y_mm": 0.0,
        "minimum_z_mm": 0.0,
        "maximum_x_mm": 40.0,
        "maximum_y_mm": 40.0,
        "maximum_z_mm": 4.0,
    }
    return all(
        abs(float(metrics[key]) - value) <= BOUND_TOLERANCE_MM
        for key, value in expected.items()
    )


def write_print_meshes(
    output_dir: Path = DEFAULT_OUTPUT,
    plan_path: Path = DEFAULT_PLAN,
    mesh_dir: Path = CUSTOM_ROOT / "solid/meshes",
) -> Dict[str, object]:
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(
            "Gmsh is required to export the exact simulated surfaces"
        ) from exc
    output_dir = Path(output_dir)
    mesh_dir = Path(mesh_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = _target_masses(plan_path)
    patterns = ["triangular"] + sorted(
        pattern for pattern in targets if pattern != "triangular"
    )
    rows = []
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Binary", 1)
        for pattern in patterns:
            source = mesh_dir / f"{pattern}.msh"
            if not source.exists():
                raise FileNotFoundError(
                    f"simulated volume mesh is missing: {source}"
                )
            gmsh.clear()
            gmsh.open(str(source))
            destination = output_dir / f"{pattern}.stl"
            gmsh.write(str(destination))
            metrics = inspect_binary_stl(destination)
            volume = float(metrics["volume_mm3"])
            mass = volume * DENSITY_G_PER_MM3
            target = targets[pattern]
            mass_error = mass / target - 1.0
            bounds_passed = _bounds_pass(metrics)
            gate_passed = (
                metrics["watertight"] is True
                and int(metrics["nonmanifold_edge_count"]) == 0
                and int(metrics["triangle_count"]) > 0
                and bounds_passed
                and abs(mass_error) <= MAXIMUM_MASS_RELATIVE_ERROR
            )
            rows.append(
                {
                    "pattern": pattern,
                    "source_volume_mesh": str(source.resolve()),
                    "source_volume_mesh_sha256": _sha256(source),
                    "printable_stl": str(destination.resolve()),
                    "printable_stl_sha256": _sha256(destination),
                    **metrics,
                    "density_g_per_mm3": DENSITY_G_PER_MM3,
                    "calculated_mass_g": mass,
                    "target_mass_g": target,
                    "mass_relative_error": mass_error,
                    "bounds_gate_passed": bounds_passed,
                    "print_mesh_gate_passed": gate_passed,
                }
            )
    finally:
        gmsh.finalize()
    gate = len(rows) == 2 and all(
        row["print_mesh_gate_passed"] for row in rows
    )
    manifest = {
        "export_id": (
            f"{patterns[1]}_exact_simulated_meshes_v1"
        ),
        "patterns": patterns,
        "method": (
            "binary STL exterior surfaces exported directly from the exact "
            "C3D6 volume meshes used in the CalculiX comparison"
        ),
        "material": "Bambu PLA Basic",
        "density_g_per_mm3": DENSITY_G_PER_MM3,
        "mesh_count": len(rows),
        "watertight_mesh_count": sum(
            bool(row["watertight"]) for row in rows
        ),
        "maximum_allowed_mass_relative_error": (
            MAXIMUM_MASS_RELATIVE_ERROR
        ),
        "all_print_mesh_gates_passed": gate,
        "meshes": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Slicer-ready custom-pattern meshes",
        "",
        "These binary STLs are exterior surfaces of the exact volume meshes "
        "used by CalculiX. This avoids an independent CAD tessellation "
        "between simulation and printing.",
        "",
        "| Pattern | Triangles | Watertight | Bounds (mm) | Volume (mm³) | "
        "Mass (g) | Target (g) | Error |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['pattern']} | {row['triangle_count']} | "
            f"{'yes' if row['watertight'] else 'no'} | 40 × 40 × 4 | "
            f"{row['volume_mm3']:.3f} | "
            f"{row['calculated_mass_g']:.6f} | "
            f"{row['target_mass_g']:.6f} | "
            f"{row['mass_relative_error']:.3%} |"
        )
    report.extend(
        [
            "",
            f"Print-mesh gate: **{'passed' if gate else 'failed'}**.",
            "",
            "Import in millimetres without rescaling. Print three copies of "
            "each STL using the frozen common slicer settings.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    if not gate:
        raise RuntimeError("one or more printable STL gates failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export exact simulated custom-pattern meshes to STL"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument(
        "--mesh-dir", default=str(CUSTOM_ROOT / "solid/meshes")
    )
    args = parser.parse_args()
    manifest = write_print_meshes(
        Path(args.output), Path(args.plan), Path(args.mesh_dir)
    )
    print(
        f"print meshes: {manifest['watertight_mesh_count']}/"
        f"{manifest['mesh_count']} watertight; "
        f"gate={manifest['all_print_mesh_gates_passed']}; "
        f"output: {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
