from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .calculix import CalculixRunner
from .custom_pattern_toolpath import (
    DEFAULT_MATERIAL_PROFILE,
    parse_deposited_segments,
)
from .custom_pattern_toolpath_fea import (
    FINE_RASTER_RESOLUTION,
    build_mass_calibrated_fine_geometry,
    coarsen_to_voxel_mesh,
    render_toolpath_bending_deck,
)
from .viscoelastic_pattern_modal_screen import (
    _run_modal,
    build_storage_profile,
    render_modal_deck,
    select_out_of_plane_mode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOT = (
    PROJECT_ROOT / "results/custom_infill/weird_kite_search"
)
DEFAULT_GCODE = (
    SEARCH_ROOT
    / "physical_validation/manufacturing/sliced/plate_1.gcode"
)
DEFAULT_OUTPUT = SEARCH_ROOT / "toolpath_validation"
DEFAULT_RUNS = (
    PROJECT_ROOT / "runs/custom_infill/weird_kite_search/toolpath_validation"
)
DEFAULT_RESOLUTIONS = (60, 80)
PATTERNS = (
    "triangular",
    "kite_08_08_08_08",
    "kite_08_08_12_08",
)
MAXIMUM_CONVERGENCE_CHANGE = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_representative_objects(
    parsed: Mapping[str, Sequence[Mapping[str, object]]],
    patterns: Sequence[str] = PATTERNS,
) -> Dict[str, str]:
    selected = {}
    for pattern in patterns:
        matches = sorted(
            name for name in parsed if name.startswith(f"{pattern}.stl_")
        )
        if len(matches) != 3:
            raise ValueError(
                f"expected three sliced {pattern} objects, found {len(matches)}"
            )
        selected[pattern] = matches[0]
    return selected


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_toolpath_gate(
    rows: Sequence[Mapping[str, object]],
    patterns: Sequence[str] = PATTERNS,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
) -> Dict[str, object]:
    lookup = {
        (
            str(row["pattern"]),
            int(row["resolution_xy"]),
            str(row["analysis"]),
        ): row
        for row in rows
    }
    fine = max(resolutions)
    previous = sorted(resolutions)[-2] if len(resolutions) > 1 else fine
    convergence: Dict[str, Dict[str, float | None]] = {}
    for pattern in patterns:
        convergence[pattern] = {}
        for analysis, field in (
            ("static_bending", "bending_stiffness_N_per_mm"),
            ("storage_modal", "out_of_plane_frequency_Hz"),
        ):
            coarse = lookup[(pattern, previous, analysis)].get(field)
            fine_value = lookup[(pattern, fine, analysis)].get(field)
            convergence[pattern][analysis] = (
                None
                if coarse is None or fine_value is None
                else abs(float(coarse) / float(fine_value) - 1.0)
            )
    control_static = float(
        lookup[
            ("triangular", fine, "static_bending")
        ]["bending_stiffness_N_per_mm"]
    )
    control_modal = float(
        lookup[
            ("triangular", fine, "storage_modal")
        ]["out_of_plane_frequency_Hz"]
    )
    ratios = {}
    for pattern in patterns:
        ratios[pattern] = {
            "bending_to_triangular": float(
                lookup[
                    (pattern, fine, "static_bending")
                ]["bending_stiffness_N_per_mm"]
            )
            / control_static,
            "modal_to_triangular": float(
                lookup[
                    (pattern, fine, "storage_modal")
                ]["out_of_plane_frequency_Hz"]
            )
            / control_modal,
        }
    all_ok = all(row["status"] == "ok" for row in rows)
    converged = all(
        value is not None and float(value) <= MAXIMUM_CONVERGENCE_CHANGE
        for by_analysis in convergence.values()
        for value in by_analysis.values()
    )
    candidates_survive = all(
        ratios[pattern]["bending_to_triangular"] > 1.0
        and ratios[pattern]["modal_to_triangular"] > 1.0
        for pattern in patterns
        if pattern != "triangular"
    )
    candidate_order_static = sorted(
        (pattern for pattern in patterns if pattern != "triangular"),
        key=lambda pattern: ratios[pattern]["bending_to_triangular"],
        reverse=True,
    )
    candidate_order_modal = sorted(
        (pattern for pattern in patterns if pattern != "triangular"),
        key=lambda pattern: ratios[pattern]["modal_to_triangular"],
        reverse=True,
    )
    order_agrees = candidate_order_static == candidate_order_modal
    return {
        "fine_resolution_xy": fine,
        "convergence": convergence,
        "fine_ratios": ratios,
        "all_solver_cases_passed": all_ok,
        "all_static_and_modal_changes_at_most_5_percent": converged,
        "both_candidates_beat_triangular_in_bending_and_modal": (
            candidates_survive
        ),
        "candidate_static_and_modal_order_agrees": order_agrees,
        "toolpath_candidate_gate_passed": (
            all_ok and converged and candidates_survive and order_agrees
        ),
    }


def run_toolpath_validation(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    gcode_path: Path = DEFAULT_GCODE,
    static_profile_path: Path = DEFAULT_MATERIAL_PROFILE,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    gcode_path = Path(gcode_path)
    resolutions = tuple(sorted(set(int(value) for value in resolutions)))
    if len(resolutions) < 2:
        raise ValueError("at least two resolutions are required")
    if any(FINE_RASTER_RESOLUTION % value != 0 for value in resolutions):
        raise ValueError("all resolutions must divide the fine raster")
    search = _json(SEARCH_ROOT / "summary.json")
    if (
        search.get("gates", {}).get(
            "candidate_search_admitted_for_physical_testing"
        )
        is not True
    ):
        raise ValueError("weird-kite continuum search is not admitted")
    physical = _json(
        SEARCH_ROOT / "physical_validation/manufacturing/manifest.json"
    )
    if (
        physical.get("manufacturing_gate_passed") is not True
        or physical.get("gcode_sha256") != _sha256(gcode_path)
    ):
        raise ValueError("frozen physical G-code gate is not valid")
    static_profile = _json(static_profile_path)
    modal_profile = build_storage_profile()
    parsed = parse_deposited_segments(gcode_path)
    selected = select_representative_objects(parsed)
    for directory in (
        "fine_geometry",
        "voxel_meshes",
        "static_decks",
        "modal_decks",
        "static_curves",
    ):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    static_runner = CalculixRunner(
        runs_dir / "static",
        executable="ccx",
        timeout_seconds=600.0,
        keep_successful=True,
    )
    rows = []
    geometry_rows = {}
    for pattern, object_name in selected.items():
        geometry = build_mass_calibrated_fine_geometry(
            pattern,
            object_name,
            parsed[object_name],
            resolution=FINE_RASTER_RESOLUTION,
        )
        fine_path = (
            output_dir
            / "fine_geometry"
            / f"{pattern}-{FINE_RASTER_RESOLUTION}.npz"
        )
        np.savez_compressed(
            fine_path,
            occupancy=geometry.occupancy,
            raster_size_mm=np.asarray([geometry.raster_size_mm]),
        )
        geometry_rows[pattern] = {
            "object_name": object_name,
            "segment_count": len(parsed[object_name]),
            "target_deposited_volume_mm3": (
                geometry.target_deposited_volume_mm3
            ),
            "fine_raster_volume_mm3": geometry.raster_volume_mm3,
            "fine_mass_error_fraction": geometry.mass_error_fraction,
            "component_count": geometry.component_count,
            "fine_geometry": str(fine_path.resolve()),
            "fine_geometry_sha256": _sha256(fine_path),
        }
        for resolution in resolutions:
            mesh = coarsen_to_voxel_mesh(geometry, resolution)
            mesh_path = (
                output_dir
                / "voxel_meshes"
                / f"{pattern}-{resolution}.npz"
            )
            np.savez_compressed(
                mesh_path,
                node_tags=mesh.node_tags,
                coordinates=mesh.coordinates,
                element_tags=mesh.element_tags,
                connectivity=mesh.connectivity,
                fill_fractions=mesh.fill_fractions,
            )
            static_deck = (
                output_dir
                / "static_decks"
                / f"{pattern}-{resolution}.inp"
            )
            static_deck.write_text(
                render_toolpath_bending_deck(mesh, static_profile),
                encoding="utf-8",
            )
            static_solver = static_runner.run(
                static_deck,
                {},
                run_id=f"weird-toolpath-static-{pattern}-{resolution}",
            )
            stiffness = None
            if static_solver.ok:
                displacement = float(
                    abs(static_solver.curve["displacement"][-1])
                )
                force = float(
                    abs(static_solver.curve["reaction_force"][-1])
                )
                stiffness = force / displacement
                curve_path = (
                    output_dir
                    / "static_curves"
                    / f"{pattern}-{resolution}.csv"
                )
                _write_csv(
                    curve_path,
                    [
                        {
                            "displacement_mm": float(current),
                            "force_N": float(current_force),
                        }
                        for current, current_force in zip(
                            np.abs(static_solver.curve["displacement"]),
                            np.abs(static_solver.curve["reaction_force"]),
                        )
                    ],
                )
            rows.append(
                {
                    "pattern": pattern,
                    "resolution_xy": resolution,
                    "analysis": "static_bending",
                    "status": static_solver.status,
                    "nodes": len(mesh.node_tags),
                    "elements": len(mesh.element_tags),
                    "represented_volume_mm3": mesh.represented_volume_mm3,
                    "bending_stiffness_N_per_mm": stiffness,
                    "out_of_plane_frequency_Hz": None,
                    "selected_mode": None,
                    "z_mass_fraction_within_extracted_modes": None,
                    "run_directory": str(static_solver.run_dir.resolve()),
                    "deck": str(static_deck.resolve()),
                    "mesh": str(mesh_path.resolve()),
                    "error": static_solver.error,
                }
            )
            modal_deck = (
                output_dir
                / "modal_decks"
                / f"{pattern}-{resolution}.inp"
            )
            modal_deck.write_text(
                render_modal_deck(mesh, modal_profile),
                encoding="utf-8",
            )
            modal_solver = _run_modal(
                modal_deck,
                runs_dir / "modal",
                f"weird-toolpath-modal-{pattern}-{resolution}",
            )
            selected_mode = (
                None
                if modal_solver["status"] != "ok"
                else select_out_of_plane_mode(modal_solver["modes"])
            )
            rows.append(
                {
                    "pattern": pattern,
                    "resolution_xy": resolution,
                    "analysis": "storage_modal",
                    "status": modal_solver["status"],
                    "nodes": len(mesh.node_tags),
                    "elements": len(mesh.element_tags),
                    "represented_volume_mm3": mesh.represented_volume_mm3,
                    "bending_stiffness_N_per_mm": None,
                    "out_of_plane_frequency_Hz": (
                        None
                        if selected_mode is None
                        else selected_mode["frequency_Hz"]
                    ),
                    "selected_mode": (
                        None
                        if selected_mode is None
                        else int(selected_mode["mode"])
                    ),
                    "z_mass_fraction_within_extracted_modes": (
                        None
                        if selected_mode is None
                        else selected_mode[
                            "z_mass_fraction_within_extracted_modes"
                        ]
                    ),
                    "run_directory": modal_solver["run_directory"],
                    "deck": str(modal_deck.resolve()),
                    "mesh": str(mesh_path.resolve()),
                    "error": modal_solver.get("error"),
                }
            )
    _write_csv(output_dir / "results.csv", rows)
    gate = evaluate_toolpath_gate(rows, PATTERNS, resolutions)
    fine = int(gate["fine_resolution_xy"])
    continuum = search["refined_predictions"]
    toolpath_values = {}
    for pattern in PATTERNS:
        toolpath_values[pattern] = {
            "bending_stiffness_N_per_mm": next(
                float(row["bending_stiffness_N_per_mm"])
                for row in rows
                if row["pattern"] == pattern
                and row["resolution_xy"] == fine
                and row["analysis"] == "static_bending"
            ),
            "out_of_plane_frequency_Hz": next(
                float(row["out_of_plane_frequency_Hz"])
                for row in rows
                if row["pattern"] == pattern
                and row["resolution_xy"] == fine
                and row["analysis"] == "storage_modal"
            ),
        }
    payload: Dict[str, object] = {
        "study_id": "bambu_pla_weird_kite_frozen_toolpath_validation_v1",
        "material": "Bambu PLA Basic",
        "source_search": str((SEARCH_ROOT / "summary.json").resolve()),
        "source_search_sha256": _sha256(SEARCH_ROOT / "summary.json"),
        "source_gcode": str(gcode_path.resolve()),
        "source_gcode_sha256": _sha256(gcode_path),
        "patterns": list(PATTERNS),
        "representative_objects": selected,
        "resolutions_xy": list(resolutions),
        "z_layers": 20,
        "fine_geometries": geometry_rows,
        "continuum_predictions": continuum,
        "fine_toolpath_predictions": toolpath_values,
        "gate": gate,
        "status": (
            "frozen_toolpath_candidates_admitted_for_physical_testing"
            if gate["toolpath_candidate_gate_passed"]
            else "frozen_toolpath_candidate_gate_failed"
        ),
        "restrictions": [
            "Mass-calibrated voxel replay uses linear fill-fraction stiffness "
            "and density scaling.",
            "Beads and layer contacts are perfectly bonded.",
            "Absolute bending and resonance require the generated physical "
            "coupon measurements.",
            "No strength, damage, fatigue, creep, or environmental lifetime "
            "claim is admitted.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Frozen-toolpath validation of weird kite candidates",
        "",
        "The exact nine-coupon Orca G-code was rasterized layer by layer and "
        "replayed with matched static and 18 C storage-modal material cards.",
        "",
        "| Pattern | Toolpath bend (N/mm) | vs triangular | Toolpath mode "
        "(Hz) | vs triangular |",
        "|---|---:|---:|---:|---:|",
    ]
    ratios = gate["fine_ratios"]
    for pattern in PATTERNS:
        report.append(
            f"| {pattern} | "
            f"{toolpath_values[pattern]['bending_stiffness_N_per_mm']:.3f} | "
            f"{ratios[pattern]['bending_to_triangular']:.4f} | "
            f"{toolpath_values[pattern]['out_of_plane_frequency_Hz']:.3f} | "
            f"{ratios[pattern]['modal_to_triangular']:.4f} |"
        )
    report.extend(
        [
            "",
            (
                "Decision: **admitted for matched physical testing**."
                if gate["toolpath_candidate_gate_passed"]
                else "Decision: **toolpath gate failed**."
            ),
            "",
            "The gate requires both candidates to beat triangular in static "
            "bending and dominant out-of-plane mode after slicing, all "
            "60→80 changes to remain within 5%, and the two candidate "
            "rankings to agree. It does not validate absolute response.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the weird-kite physical plate as voxel FEA"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--gcode", type=Path, default=DEFAULT_GCODE)
    args = parser.parse_args()
    result = run_toolpath_validation(
        args.output, args.runs, args.gcode
    )
    print(
        f"weird-kite toolpath gate: "
        f"{result['gate']['toolpath_candidate_gate_passed']}; "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
