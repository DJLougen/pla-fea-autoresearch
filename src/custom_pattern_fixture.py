from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from .volumetric_infill import inspect_binary_stl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ROOT = (
    PROJECT_ROOT / "results/custom_infill/graded_hourglass"
)
DEFAULT_OUTPUT = CUSTOM_ROOT / "physical_validation/fixture"
SUPPORT_SPAN_MM = 40.0
ROLLER_DIAMETER_MM = 4.0
GROOVE_DIAMETER_MM = 4.1
COUPON_WIDTH_MM = 40.0
COUPON_HEIGHT_MM = 40.0
COUPON_THICKNESS_MM = 4.0
ENDPOINT_DISPLACEMENT_MM = 0.4
CROSSHEAD_RATE_MM_MIN = 1.2
PRELOAD_N = 0.5
MAXIMUM_FIXTURE_COMPLIANCE_MM_N = 0.0006


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_nodes_and_set(
    deck_path: Path, set_name: str
) -> Sequence[tuple[float, float, float]]:
    text = Path(deck_path).read_text(encoding="utf-8")
    node_match = re.search(
        r"\*NODE,NSET=NALL\n(.*?)(?=\n\*)", text, re.S
    )
    set_match = re.search(
        rf"\*NSET,NSET={set_name}\n(.*?)(?=\n\*)", text, re.S
    )
    if node_match is None or set_match is None:
        raise ValueError(
            f"deck does not contain nodes and {set_name}: {deck_path}"
        )
    nodes = {}
    for line in node_match.group(1).splitlines():
        values = [value.strip() for value in line.split(",")]
        nodes[int(values[0])] = tuple(float(value) for value in values[1:4])
    identifiers = [
        int(value)
        for value in re.split(r"[,\s]+", set_match.group(1).strip())
        if value
    ]
    return [nodes[identifier] for identifier in identifiers]


def _coordinate_summary(
    coordinates: Sequence[tuple[float, float, float]]
) -> Dict[str, object]:
    return {
        "node_count": len(coordinates),
        "minimum_x_mm": min(point[0] for point in coordinates),
        "maximum_x_mm": max(point[0] for point in coordinates),
        "minimum_y_mm": min(point[1] for point in coordinates),
        "maximum_y_mm": max(point[1] for point in coordinates),
        "minimum_z_mm": min(point[2] for point in coordinates),
        "maximum_z_mm": max(point[2] for point in coordinates),
    }


def trace_solver_fixture(
    deck_dir: Path = CUSTOM_ROOT / "solid/decks",
    summary_path: Path = CUSTOM_ROOT / "solid/summary.csv",
) -> Dict[str, object]:
    deck_dir = Path(deck_dir)
    traces = {}
    for pattern in ("triangular", "graded_hourglass"):
        deck = deck_dir / f"{pattern}-bending-solid.inp"
        supports = _coordinate_summary(
            _parse_nodes_and_set(deck, "SUPPORTS")
        )
        load = _coordinate_summary(
            _parse_nodes_and_set(deck, "LOAD")
        )
        traces[pattern] = {
            "deck": str(deck.resolve()),
            "deck_sha256": _sha256(deck),
            "supports": supports,
            "load": load,
            "load_strip_width_mm": (
                float(load["maximum_x_mm"])
                - float(load["minimum_x_mm"])
            ),
        }
    with Path(summary_path).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["load_case"] == "bending"
        ]
    predictions = {
        row["pattern"]: {
            "endpoint_displacement_mm": float(
                row["applied_displacement_mm"]
            ),
            "reaction_force_N": float(row["reaction_force_N"]),
            "stiffness_N_per_mm": float(row["stiffness_N_per_mm"]),
        }
        for row in rows
    }
    geometry_gate = True
    for trace in traces.values():
        supports = trace["supports"]
        load = trace["load"]
        geometry_gate = geometry_gate and (
            math.isclose(
                float(supports["minimum_x_mm"]), 0.0, abs_tol=1e-8
            )
            and math.isclose(
                float(supports["maximum_x_mm"]), 40.0, abs_tol=1e-8
            )
            and math.isclose(
                float(supports["minimum_y_mm"]), 0.0, abs_tol=1e-8
            )
            and math.isclose(
                float(supports["maximum_y_mm"]), 40.0, abs_tol=1e-8
            )
            and math.isclose(
                float(supports["minimum_z_mm"]), 0.0, abs_tol=1e-8
            )
            and math.isclose(
                float(supports["maximum_z_mm"]), 0.0, abs_tol=1e-8
            )
            and float(trace["load_strip_width_mm"]) <= 1.1
            and math.isclose(
                (
                    float(load["minimum_x_mm"])
                    + float(load["maximum_x_mm"])
                )
                / 2.0,
                20.0,
                abs_tol=0.05,
            )
            and math.isclose(
                float(load["minimum_y_mm"]), 0.0, abs_tol=1e-8
            )
            and math.isclose(
                float(load["maximum_y_mm"]), 40.0, abs_tol=1e-8
            )
            and math.isclose(
                float(load["minimum_z_mm"]), 4.0, abs_tol=1e-8
            )
            and math.isclose(
                float(load["maximum_z_mm"]), 4.0, abs_tol=1e-8
            )
        )
    prediction_gate = (
        set(predictions) == {"triangular", "graded_hourglass"}
        and all(
            math.isclose(
                row["endpoint_displacement_mm"],
                ENDPOINT_DISPLACEMENT_MM,
                abs_tol=1e-8,
            )
            for row in predictions.values()
        )
    )
    return {
        "fixture_trace_id": "graded_hourglass_bending_boundary_v1",
        "traces": traces,
        "predictions": predictions,
        "solver_boundary_geometry_gate_passed": geometry_gate,
        "solver_endpoint_gate_passed": prediction_gate,
    }


def _component_mesh(
    name: str, output_path: Path
) -> Dict[str, object]:
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError("Gmsh is required for fixture export") from exc
    gmsh.clear()
    gmsh.model.add(name)
    if name == "support_base":
        box = gmsh.model.occ.addBox(0, 0, 0, 60, 50, 8)
        left = gmsh.model.occ.addCylinder(
            10, 0, 8, 0, 50, 0, GROOVE_DIAMETER_MM / 2.0
        )
        right = gmsh.model.occ.addCylinder(
            50, 0, 8, 0, 50, 0, GROOVE_DIAMETER_MM / 2.0
        )
        gmsh.model.occ.cut(
            [(3, box)],
            [(3, left), (3, right)],
            removeObject=True,
            removeTool=True,
        )
        expected_bounds = (60.0, 50.0, 8.0)
    elif name == "loading_nose":
        box = gmsh.model.occ.addBox(0, 0, 0, 10, 50, 8)
        groove = gmsh.model.occ.addCylinder(
            5, 0, 0, 0, 50, 0, GROOVE_DIAMETER_MM / 2.0
        )
        gmsh.model.occ.cut(
            [(3, box)],
            [(3, groove)],
            removeObject=True,
            removeTool=True,
        )
        expected_bounds = (10.0, 50.0, 8.0)
    else:
        raise ValueError(f"unknown fixture component {name}")
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.4)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.6)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(output_path))
    metrics = inspect_binary_stl(output_path)
    bounds_gate = (
        abs(float(metrics["minimum_x_mm"])) <= 1e-6
        and abs(float(metrics["minimum_y_mm"])) <= 1e-6
        and abs(float(metrics["minimum_z_mm"])) <= 1e-6
        and math.isclose(
            float(metrics["maximum_x_mm"]),
            expected_bounds[0],
            abs_tol=1e-6,
        )
        and math.isclose(
            float(metrics["maximum_y_mm"]),
            expected_bounds[1],
            abs_tol=1e-6,
        )
        and math.isclose(
            float(metrics["maximum_z_mm"]),
            expected_bounds[2],
            abs_tol=1e-6,
        )
    )
    return {
        "component": name,
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        **metrics,
        "bounds_gate_passed": bounds_gate,
        "component_gate_passed": (
            metrics["watertight"] is True
            and int(metrics["nonmanifold_edge_count"]) == 0
            and bounds_gate
        ),
    }


def write_fixture_package(
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError("Gmsh is required for fixture export") from exc
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = trace_solver_fixture()
    components = []
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Binary", 1)
        for name in ("support_base", "loading_nose"):
            components.append(
                _component_mesh(name, output_dir / f"{name}.stl")
            )
    finally:
        gmsh.finalize()
    gate = (
        trace["solver_boundary_geometry_gate_passed"] is True
        and trace["solver_endpoint_gate_passed"] is True
        and len(components) == 2
        and all(
            component["component_gate_passed"]
            for component in components
        )
    )
    protocol = {
        "fixture_id": "bambu-pattern-bend-v1",
        "test_mode": "three_point_bending_stiffness",
        "coupon_orientation": (
            "as-printed flat; 4 mm print thickness vertical; original "
            "40 mm x-axis along the support span"
        ),
        "support_span_mm": SUPPORT_SPAN_MM,
        "support_span_tolerance_mm": 0.05,
        "support_roller_count": 2,
        "support_roller_diameter_mm": ROLLER_DIAMETER_MM,
        "loading_roller_diameter_mm": ROLLER_DIAMETER_MM,
        "steel_rod_length_mm": 50.0,
        "groove_diameter_mm": GROOVE_DIAMETER_MM,
        "centerline_alignment_tolerance_mm": 0.05,
        "preload_N": PRELOAD_N,
        "crosshead_rate_mm_min": CROSSHEAD_RATE_MM_MIN,
        "endpoint_displacement_mm": ENDPOINT_DISPLACEMENT_MM,
        "minimum_acquisition_rate_Hz": 10.0,
        "minimum_curve_points": 20,
        "maximum_fixture_compliance_mm_per_N": (
            MAXIMUM_FIXTURE_COMPLIANCE_MM_N
        ),
        "fixture_compliance_requirement": (
            "measure with a stiff calibration bridge or use local coupon "
            "deflection; correct crosshead displacement before ingestion"
        ),
        "stop_conditions": [
            "0.400 mm corrected midpoint displacement",
            "visible crack or unstable force drop",
            "coupon leaves either support",
        ],
        "required_hardware": [
            "two 4.00 mm diameter hardened-steel support rods, 50 mm long",
            "one 4.00 mm diameter hardened-steel loading rod, 50 mm long",
            "compression test frame or load cell with at least 50 N capacity",
            "displacement measurement resolution 0.005 mm or better",
            "balance resolution 0.01 g or better",
        ],
    }
    manifest = {
        "fixture_id": protocol["fixture_id"],
        "material_scope": "Bambu PLA Basic coupon comparison",
        "solver_trace": trace,
        "protocol": protocol,
        "components": components,
        "steel_rods_are_not_printed": True,
        "all_fixture_gates_passed": gate,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    triangular = trace["predictions"]["triangular"]
    custom = trace["predictions"]["graded_hourglass"]
    report = f"""# Three-point bending fixture and protocol

The fixture reproduces the solver's 40.00 mm support span and centered line
load. The printed parts locate three 4.00 mm hardened-steel rods; the rods,
not printed plastic edges, contact the coupon.

| Parameter | Registered value |
|---|---:|
| Support span | 40.00 ± 0.05 mm |
| Support/loading rods | 4.00 mm diameter × 50 mm steel |
| Coupon | 40 × 40 × 4 mm, printed thickness vertical |
| Preload | 0.5 N |
| Crosshead rate | 1.2 mm/min |
| Corrected endpoint displacement | 0.400 mm |
| Minimum acquisition | 10 Hz |
| Maximum fixture compliance | {MAXIMUM_FIXTURE_COMPLIANCE_MM_N:.4f} mm/N |

The CalculiX endpoint predicts {triangular['reaction_force_N']:.3f} N for the
triangular control and {custom['reaction_force_N']:.3f} N for the
graded-hourglass coupon. Stop at the registered displacement; this is a
stiffness test, not a fracture test.

Before coupon testing, measure fixture/machine compliance with a stiff bridge
or use a local midpoint displacement sensor. Supply compliance-corrected
displacement to the capture CSV. Do not accept a run if compliance exceeds
the registered limit.

Fixture gate: **{'passed' if gate else 'failed'}**.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    if not gate:
        raise RuntimeError("one or more fixture gates failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the matched custom-pattern bending fixture"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    manifest = write_fixture_package(Path(args.output))
    print(
        f"bending fixture: gate={manifest['all_fixture_gates_passed']}; "
        f"components={len(manifest['components'])}; "
        f"output: {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
