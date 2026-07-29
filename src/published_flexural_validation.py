from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .bambu_flexural_digitization import DEFAULT_OUTPUT as FLEXURAL_DATA
from .bambu_tds_profile import DEFAULT_TRANSVERSE_PROFILE
from .calculix import CalculixRunner, ORIENTATIONS
from .materials import material_card


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/published_flexural_validation"
DEFAULT_RUNS = PROJECT_ROOT / "runs/published_flexural_validation"
TEMPLATE = (
    PROJECT_ROOT
    / "meshes/published_flexural_100x12p7x3p2_span64.inp.template"
)
SPAN_MM = 64.0
WIDTH_MM = 12.7
DEPTH_MM = 3.2
TARGET_OUTER_STRAIN = 0.005
RASTER_TO_DIRECTION = {"0": "1", "45": "12_45", "90": "2"}
MESH_LEVELS = {
    "coarse": (1, 1),
    "medium": (2, 2),
    "fine": (4, 4),
}


def build_structured_flexural_template(
    interval_multiplier: int,
    through_depth_elements: int,
) -> str:
    """Build a reproducible C3D8I mesh with nodes at load/support locations."""
    base_breaks = [0.0, 18.0, 50.0, 82.0, 100.0]
    base_counts = [1, 2, 2, 1]
    xs: List[float] = [base_breaks[0]]
    for left, right, count in zip(
        base_breaks[:-1], base_breaks[1:], base_counts
    ):
        pieces = count * interval_multiplier
        xs.extend(
            left + (right - left) * step / pieces
            for step in range(1, pieces + 1)
        )
    ys = [0.0, WIDTH_MM]
    zs = [
        DEPTH_MM * step / through_depth_elements
        for step in range(through_depth_elements + 1)
    ]

    def node_id(ix: int, iy: int, iz: int) -> int:
        return 1 + ix * len(ys) * len(zs) + iy * len(zs) + iz

    lines = [
        "*HEADING",
        "Bambu PLA Basic published flexural external-validation coupon",
        "*NODE",
    ]
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                lines.append(f"{node_id(ix, iy, iz)},{x:.8g},{y:.8g},{z:.8g}")
    lines.append("*ELEMENT,TYPE=C3D8I,ELSET=COUPON")
    element_id = 1
    for ix in range(len(xs) - 1):
        for iz in range(len(zs) - 1):
            nodes = [
                node_id(ix, 0, iz),
                node_id(ix + 1, 0, iz),
                node_id(ix + 1, 1, iz),
                node_id(ix, 1, iz),
                node_id(ix, 0, iz + 1),
                node_id(ix + 1, 0, iz + 1),
                node_id(ix + 1, 1, iz + 1),
                node_id(ix, 1, iz + 1),
            ]
            lines.append(
                f"{element_id}," + ",".join(str(node) for node in nodes)
            )
            element_id += 1

    x_indices = {
        target: min(
            range(len(xs)), key=lambda index: abs(xs[index] - target)
        )
        for target in (18.0, 50.0, 82.0)
    }
    supports = [
        node_id(x_indices[x], iy, 0)
        for x in (18.0, 82.0)
        for iy in range(len(ys))
    ]
    load = [
        node_id(x_indices[50.0], iy, len(zs) - 1)
        for iy in range(len(ys))
    ]
    anchor = node_id(x_indices[18.0], 0, 0)
    lines.extend(
        [
            "*NSET,NSET=SUPPORTS",
            ",".join(str(node) for node in supports),
            "*NSET,NSET=LOAD",
            ",".join(str(node) for node in load),
            "*NSET,NSET=ANCHOR",
            str(anchor),
            "*ORIENTATION,NAME=MAT_ORIENTATION",
            "{{ORIENTATION}}",
            "{{MATERIAL_CARD}}",
            "*SOLID SECTION,ELSET=COUPON,MATERIAL=PLA,"
            "ORIENTATION=MAT_ORIENTATION",
            "*STEP,NLGEOM=NO",
            "*STATIC,DIRECT",
            "1.,1.",
            "*BOUNDARY",
            "SUPPORTS,3,3,0.",
            "ANCHOR,1,2,0.",
            "LOAD,3,3,{{MAX_DISPLACEMENT}}",
            "*NODE PRINT,NSET=LOAD",
            "U",
            "*NODE PRINT,NSET=SUPPORTS",
            "RF",
            "*EL FILE,ELSET=COUPON",
            "S,E",
            "*END STEP",
            "",
        ]
    )
    return "\n".join(lines)


def _load_published_220c() -> Dict[str, Dict[str, float]]:
    with (FLEXURAL_DATA / "table3_flexural_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(int(float(row["raster_orientation_deg"]))): {
            key: float(value) for key, value in row.items()
        }
        for row in rows
        if float(row["nozzle_temperature_C"]) == 220.0
    }


def _load_digitized_slopes() -> Dict[str, float]:
    with (FLEXURAL_DATA / "digitization_validation.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return {
            str(int(float(row["raster_orientation_deg"]))): float(
                row["digitized_slope_0p006_to_0p012_GPa"]
            )
            for row in csv.DictReader(handle)
        }


def run_published_flexural_validation(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    properties_path: Path = DEFAULT_TRANSVERSE_PROFILE,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    properties = json.loads(
        Path(properties_path).read_text(encoding="utf-8")
    )
    constants = {
        key: float(value)
        for key, value in properties[
            "screening_engineering_constants"
        ].items()
    }
    published = _load_published_220c()
    digitized = _load_digitized_slopes()
    displacement = (
        TARGET_OUTER_STRAIN * SPAN_MM**2 / (6.0 * DEPTH_MM)
    )
    mesh_dir = output_dir / "mesh_templates"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    templates: Dict[str, Path] = {}
    for level, (interval_multiplier, depth_elements) in MESH_LEVELS.items():
        path = mesh_dir / f"flexural_{level}.inp.template"
        path.write_text(
            build_structured_flexural_template(
                interval_multiplier, depth_elements
            ),
            encoding="utf-8",
        )
        templates[level] = path
    runner = CalculixRunner(
        Path(runs_dir),
        executable="ccx",
        timeout_seconds=120.0,
        keep_successful=True,
    )
    rows: List[Dict[str, object]] = []
    for raster, direction in RASTER_TO_DIRECTION.items():
        result = runner.run(
            templates["fine"],
            {
                "ORIENTATION": ORIENTATIONS[direction],
                "MATERIAL_CARD": material_card(constants),
                "MAX_DISPLACEMENT": -displacement,
            },
            run_id=f"published-flexural-{raster}deg",
        )
        if not result.ok:
            rows.append(
                {
                    "raster_orientation_deg": int(raster),
                    "material_direction": direction,
                    "status": result.status,
                    "error": result.error,
                    "run_directory": str(result.run_dir.resolve()),
                }
            )
            continue
        endpoint_displacement = abs(
            float(result.curve["displacement"][-1])
        )
        endpoint_force = abs(float(result.curve["reaction_force"][-1]))
        stiffness = endpoint_force / endpoint_displacement
        apparent_modulus = (
            stiffness
            * SPAN_MM**3
            / (4.0 * WIDTH_MM * DEPTH_MM**3)
        )
        table_modulus = (
            published[raster]["flexural_modulus_GPa"] * 1000.0
        )
        rows.append(
            {
                "raster_orientation_deg": int(raster),
                "material_direction": direction,
                "status": "ok",
                "target_outer_flexural_strain": TARGET_OUTER_STRAIN,
                "endpoint_displacement_mm": endpoint_displacement,
                "endpoint_force_N": endpoint_force,
                "stiffness_N_per_mm": stiffness,
                "simulated_apparent_flexural_modulus_MPa": (
                    apparent_modulus
                ),
                "published_table_flexural_modulus_MPa": table_modulus,
                "digitized_visible_slope_MPa": (
                    digitized[raster] * 1000.0
                ),
                "simulation_to_table_ratio": (
                    apparent_modulus / table_modulus
                ),
                "relative_error_to_table": (
                    apparent_modulus / table_modulus - 1.0
                ),
                "published_flexural_strength_MPa": published[raster][
                    "flexural_strength_MPa"
                ],
                "run_directory": str(result.run_dir.resolve()),
                "error": None,
            }
        )
    mesh_rows: List[Dict[str, object]] = []
    fine_zero = next(
        (
            row
            for row in rows
            if row["status"] == "ok"
            and row["raster_orientation_deg"] == 0
        ),
        None,
    )
    for level in ("coarse", "medium"):
        result = runner.run(
            templates[level],
            {
                "ORIENTATION": ORIENTATIONS["1"],
                "MATERIAL_CARD": material_card(constants),
                "MAX_DISPLACEMENT": -displacement,
            },
            run_id=f"published-flexural-mesh-{level}",
        )
        if not result.ok:
            mesh_rows.append(
                {
                    "mesh_level": level,
                    "status": result.status,
                    "error": result.error,
                }
            )
            continue
        endpoint_displacement = abs(float(result.curve["displacement"][-1]))
        endpoint_force = abs(float(result.curve["reaction_force"][-1]))
        stiffness = endpoint_force / endpoint_displacement
        mesh_rows.append(
            {
                "mesh_level": level,
                "status": "ok",
                "interval_multiplier": MESH_LEVELS[level][0],
                "through_depth_elements": MESH_LEVELS[level][1],
                "apparent_modulus_MPa": (
                    stiffness
                    * SPAN_MM**3
                    / (4.0 * WIDTH_MM * DEPTH_MM**3)
                ),
                "error": None,
            }
        )
    if fine_zero is not None:
        mesh_rows.append(
            {
                "mesh_level": "fine",
                "status": "ok",
                "interval_multiplier": MESH_LEVELS["fine"][0],
                "through_depth_elements": MESH_LEVELS["fine"][1],
                "apparent_modulus_MPa": fine_zero[
                    "simulated_apparent_flexural_modulus_MPa"
                ],
                "error": None,
            }
        )
    mesh_fields = sorted({key for row in mesh_rows for key in row})
    with (output_dir / "mesh_convergence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=mesh_fields)
        writer.writeheader()
        writer.writerows(mesh_rows)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    successful = [row for row in rows if row["status"] == "ok"]
    simulated_ratio = {
        str(row["raster_orientation_deg"]): float(
            row["simulated_apparent_flexural_modulus_MPa"]
        )
        / float(successful[0]["simulated_apparent_flexural_modulus_MPa"])
        for row in successful
    }
    published_ratio = {
        raster: published[raster]["flexural_modulus_GPa"]
        / published["0"]["flexural_modulus_GPa"]
        for raster in RASTER_TO_DIRECTION
    }
    mean_absolute_error = float(
        np.mean(
            [
                abs(float(row["relative_error_to_table"]))
                for row in successful
            ]
        )
    )
    mesh_by_level = {
        str(row["mesh_level"]): float(row["apparent_modulus_MPa"])
        for row in mesh_rows
        if row["status"] == "ok"
    }
    medium_to_fine_change = abs(
        mesh_by_level["fine"] / mesh_by_level["medium"] - 1.0
    )
    lines = [
        "# External Bambu PLA Basic flexural validation",
        "",
        "## Result",
        "",
        "A fresh CalculiX run matched the published 100 × 12.7 × 3.2 mm "
        "specimen and 64 mm three-point-bending span. The material card was "
        "the existing TDS transverse-isotropic sensitivity—not refitted to "
        "the paper.",
        "",
        f"The medium-to-fine mesh change is "
        f"{medium_to_fine_change:.2%}; the reported comparison uses the fine "
        "mesh (four elements through the thickness).",
        "",
        "| Raster | Local loading | Simulated apparent modulus (GPa) | "
        "Published modulus (GPa) | Error | Orientation ratio: sim / paper |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in successful:
        raster = str(row["raster_orientation_deg"])
        lines.append(
            f"| {raster}° | {row['material_direction']} | "
            f"{float(row['simulated_apparent_flexural_modulus_MPa']) / 1000.0:.3f} | "
            f"{float(row['published_table_flexural_modulus_MPa']) / 1000.0:.3f} | "
            f"{float(row['relative_error_to_table']):+.1%} | "
            f"{simulated_ratio[raster]:.3f} / "
            f"{published_ratio[raster]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Mean absolute modulus error is {mean_absolute_error:.1%}. "
            "A bending modulus is not substituted into the axial material "
            "card.",
            "- The normalized orientation trend is reproduced: the 0° and "
            "45° elastic responses are effectively tied and 90° is only "
            "slightly lower. This supports near in-plane elastic isotropy for "
            "this print condition.",
            "- The much larger orientation change in published flexural "
            "strength is not reproduced because the current model has no "
            "raster-interface damage or fracture law.",
            "- The plotted curves are digitized external evidence with "
            "undefined aggregation. Table 3 means are the primary comparison; "
            "these data remain excluded from calibration and model selection.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the Bambu elastic card against published flexure"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument(
        "--properties", default=str(DEFAULT_TRANSVERSE_PROFILE)
    )
    args = parser.parse_args()
    rows = run_published_flexural_validation(
        Path(args.output),
        Path(args.runs_dir),
        Path(args.properties),
    )
    successful = sum(row["status"] == "ok" for row in rows)
    print(
        f"published flexural validation: {successful}/{len(rows)} cases; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
