from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .bambu_short_time_relaxation import (
    DEFAULT_OUTPUT as RELAXATION_REFERENCE,
    generalized_maxwell_complex,
)
from .bambu_viscoelastic_wave_reference import (
    POISSON_RATIO,
    REPORTED_DENSITY_KG_M3,
)
from .custom_pattern_toolpath import parse_deposited_segments
from .custom_pattern_toolpath_fea import (
    FINE_RASTER_RESOLUTION,
    build_mass_calibrated_fine_geometry,
    coarsen_to_voxel_mesh,
)
from .viscoelastic_pattern_modal_screen import (
    _run_modal,
    render_modal_deck,
    select_out_of_plane_mode,
)
from .weird_infill_toolpath_validation import (
    DEFAULT_GCODE,
    PATTERNS,
    SEARCH_ROOT,
    select_representative_objects,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = SEARCH_ROOT / "frequency_dependent_modal_screen"
DEFAULT_RUNS = (
    PROJECT_ROOT
    / "runs/custom_infill/weird_kite_search/frequency_dependent_modal_screen"
)
DEFAULT_RESOLUTIONS = (60, 80)
MAXIMUM_ITERATIONS = 8
MAXIMUM_SELF_CONSISTENCY_CHANGE = 1.0e-3
MAXIMUM_MESH_CHANGE = 0.05
MODE_COUNT = 12


def _json(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_relaxation_spectrum(
    root: Path = RELAXATION_REFERENCE,
) -> Dict[str, object]:
    root = Path(root)
    summary = _json(root / "summary.json")
    if (
        summary.get("overall_status")
        != "storage_and_short_time_relaxation_sensitivity_only"
        or summary.get("storage_fit_gate_passed") is not True
        or summary.get("positive_spectrum_gate_passed") is not True
    ):
        raise ValueError("short-time storage spectrum is not admitted")
    rows = _read_csv(root / "maxwell_spectrum.csv")
    if len(rows) != int(summary["selected_term_count"]):
        raise ValueError("Maxwell spectrum row count does not match summary")
    times = np.asarray(
        [float(row["relaxation_time_s"]) for row in rows], dtype=float
    )
    branches = np.asarray(
        [float(row["branch_shear_modulus_MPa"]) for row in rows],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(times))
        or np.any(~np.isfinite(branches))
        or np.any(times <= 0.0)
        or np.any(branches <= 0.0)
    ):
        raise ValueError("Maxwell spectrum contains inadmissible values")
    return {
        "summary": summary,
        "relaxation_time_s": times,
        "branch_shear_modulus_MPa": branches,
        "equilibrium_shear_modulus_MPa": float(
            summary["mean_equilibrium_shear_modulus_MPa"]
        ),
        "minimum_frequency_Hz": float(
            summary["minimum_supported_frequency_Hz"]
        ),
        "maximum_frequency_Hz": float(
            summary["maximum_supported_frequency_Hz"]
        ),
        "summary_path": root / "summary.json",
        "spectrum_path": root / "maxwell_spectrum.csv",
    }


def build_frequency_profile(
    frequency_hz: float,
    spectrum: Mapping[str, object],
) -> Dict[str, object]:
    frequency = float(frequency_hz)
    minimum = float(spectrum["minimum_frequency_Hz"])
    maximum = float(spectrum["maximum_frequency_Hz"])
    if not minimum <= frequency <= maximum:
        raise ValueError(
            f"frequency {frequency:g} Hz lies outside [{minimum:g}, "
            f"{maximum:g}] Hz"
        )
    storage, loss = generalized_maxwell_complex(
        np.asarray([frequency]),
        float(spectrum["equilibrium_shear_modulus_MPa"]),
        np.asarray(spectrum["branch_shear_modulus_MPa"], dtype=float),
        np.asarray(spectrum["relaxation_time_s"], dtype=float),
    )
    shear_storage = float(storage[0])
    shear_loss = float(loss[0])
    equivalent_e = 2.0 * shear_storage * (1.0 + POISSON_RATIO)
    return {
        "profile_id": "bambu_pla_18C_frequency_specific_storage_proxy_v1",
        "material": "Bambu PLA Basic",
        "density_tonnes_mm3": REPORTED_DENSITY_KG_M3 * 1.0e-12,
        "screening_engineering_constants": {
            "E1": equivalent_e,
            "E2": equivalent_e,
            "E3": equivalent_e,
            "nu12": POISSON_RATIO,
            "nu13": POISSON_RATIO,
            "nu23": POISSON_RATIO,
            "G12": shear_storage,
            "G13": shear_storage,
            "G23": shear_storage,
        },
        "evaluation_frequency_Hz": frequency,
        "shear_storage_modulus_MPa": shear_storage,
        "shear_loss_modulus_MPa": shear_loss,
        "loss_factor": shear_loss / shear_storage,
        "reference_temperature_C": 18.0,
        "minimum_reduced_frequency_Hz": minimum,
        "maximum_reduced_frequency_Hz": maximum,
        "restriction": (
            "Frequency-specific in-phase elastic proxy inside a "
            "self-consistent modal iteration; fitted loss is not admitted "
            "for precision damping."
        ),
    }


def evaluate_frequency_dependent_gate(
    rows: Sequence[Mapping[str, object]],
    patterns: Sequence[str] = PATTERNS,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
) -> Dict[str, object]:
    terminal = [
        row for row in rows if bool(row.get("terminal_iteration"))
    ]
    lookup = {
        (str(row["pattern"]), int(row["resolution_xy"])): row
        for row in terminal
    }
    expected = {
        (pattern, int(resolution))
        for pattern in patterns
        for resolution in resolutions
    }
    if set(lookup) != expected:
        raise ValueError("terminal modal rows do not cover the requested matrix")
    all_ok = all(str(row["status"]) == "ok" for row in rows)
    all_converged = all(
        bool(row["self_consistency_converged"]) for row in terminal
    )
    mode_consistent = all(
        bool(row["selected_mode_consistent"]) for row in terminal
    )
    fine = max(int(value) for value in resolutions)
    previous = (
        sorted(int(value) for value in resolutions)[-2]
        if len(resolutions) > 1
        else fine
    )
    mesh_change = {}
    for pattern in patterns:
        fine_value = float(
            lookup[(pattern, fine)]["out_of_plane_frequency_Hz"]
        )
        previous_value = float(
            lookup[(pattern, previous)]["out_of_plane_frequency_Hz"]
        )
        mesh_change[pattern] = abs(previous_value / fine_value - 1.0)
    mesh_converged = all(
        change <= MAXIMUM_MESH_CHANGE for change in mesh_change.values()
    )
    control = float(
        lookup[("triangular", fine)]["out_of_plane_frequency_Hz"]
    )
    ratios = {
        pattern: (
            float(
                lookup[(pattern, fine)]["out_of_plane_frequency_Hz"]
            )
            / control
        )
        for pattern in patterns
    }
    candidates = [
        pattern for pattern in patterns if pattern != "triangular"
    ]
    candidates_beat_control = all(ratios[pattern] > 1.0 for pattern in candidates)
    frequency_order = sorted(
        candidates, key=lambda pattern: ratios[pattern], reverse=True
    )
    expected_order = [
        "kite_08_08_08_08",
        "kite_08_08_12_08",
    ]
    candidate_order_preserved = frequency_order == expected_order
    admitted = (
        all_ok
        and all_converged
        and mode_consistent
        and mesh_converged
        and candidates_beat_control
        and candidate_order_preserved
    )
    return {
        "fine_resolution_xy": fine,
        "mesh_frequency_change": mesh_change,
        "fine_frequency_ratios_to_triangular": ratios,
        "all_calculix_iterations_passed": all_ok,
        "all_material_frequency_iterations_converged": all_converged,
        "selected_mode_consistent_within_each_iteration": mode_consistent,
        "all_60_to_80_frequency_changes_at_most_5_percent": mesh_converged,
        "both_candidates_beat_triangular": candidates_beat_control,
        "candidate_order_preserved": candidate_order_preserved,
        "frequency_dependent_modal_gate_passed": admitted,
    }


def run_frequency_dependent_modal_screen(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    gcode_path: Path = DEFAULT_GCODE,
    spectrum_root: Path = RELAXATION_REFERENCE,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    maximum_iterations: int = MAXIMUM_ITERATIONS,
    convergence_tolerance: float = MAXIMUM_SELF_CONSISTENCY_CHANGE,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    gcode_path = Path(gcode_path)
    resolutions = tuple(sorted(set(int(value) for value in resolutions)))
    if len(resolutions) < 2:
        raise ValueError("at least two mesh resolutions are required")
    if any(FINE_RASTER_RESOLUTION % value != 0 for value in resolutions):
        raise ValueError("mesh resolutions must divide the fine raster")
    if maximum_iterations < 2:
        raise ValueError("at least two material iterations are required")
    if not 0.0 < convergence_tolerance <= 0.05:
        raise ValueError("invalid self-consistency tolerance")

    frozen = _json(SEARCH_ROOT / "toolpath_validation/summary.json")
    if (
        frozen.get("gate", {}).get("toolpath_candidate_gate_passed")
        is not True
        or frozen.get("source_gcode_sha256") != _sha256(gcode_path)
    ):
        raise ValueError("frozen weird-infill toolpath gate is not admitted")
    spectrum = load_relaxation_spectrum(spectrum_root)
    initial_rows = _read_csv(
        SEARCH_ROOT / "toolpath_validation/results.csv"
    )
    initial_frequency = {
        (row["pattern"], int(row["resolution_xy"])): float(
            row["out_of_plane_frequency_Hz"]
        )
        for row in initial_rows
        if row["analysis"] == "storage_modal" and row["status"] == "ok"
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decks").mkdir(parents=True, exist_ok=True)
    parsed = parse_deposited_segments(gcode_path)
    selected = select_representative_objects(parsed)
    rows: list[Dict[str, object]] = []
    terminal_rows: list[Dict[str, object]] = []
    for pattern, object_name in selected.items():
        geometry = build_mass_calibrated_fine_geometry(
            pattern,
            object_name,
            parsed[object_name],
            resolution=FINE_RASTER_RESOLUTION,
        )
        for resolution in resolutions:
            mesh = coarsen_to_voxel_mesh(geometry, resolution)
            current_frequency = initial_frequency[(pattern, resolution)]
            selected_modes: list[int] = []
            converged = False
            last_row: Dict[str, object] | None = None
            for iteration in range(1, maximum_iterations + 1):
                profile = build_frequency_profile(current_frequency, spectrum)
                deck = (
                    output_dir
                    / "decks"
                    / f"{pattern}-{resolution}-iteration-{iteration}.inp"
                )
                deck.write_text(
                    render_modal_deck(mesh, profile, MODE_COUNT),
                    encoding="utf-8",
                )
                solver = _run_modal(
                    deck,
                    runs_dir,
                    (
                        f"frequency-modal-{pattern}-{resolution}-"
                        f"iteration-{iteration}"
                    ),
                )
                selected_mode = (
                    None
                    if solver["status"] != "ok"
                    else select_out_of_plane_mode(solver["modes"])
                )
                solved_frequency = (
                    None
                    if selected_mode is None
                    else float(selected_mode["frequency_Hz"])
                )
                change = (
                    None
                    if solved_frequency is None
                    else abs(solved_frequency / current_frequency - 1.0)
                )
                if selected_mode is not None:
                    selected_modes.append(int(selected_mode["mode"]))
                last_row = {
                    "pattern": pattern,
                    "resolution_xy": resolution,
                    "iteration": iteration,
                    "input_frequency_Hz": current_frequency,
                    "shear_storage_modulus_MPa": profile[
                        "shear_storage_modulus_MPa"
                    ],
                    "shear_loss_modulus_MPa": profile[
                        "shear_loss_modulus_MPa"
                    ],
                    "loss_factor_sensitivity": profile["loss_factor"],
                    "equivalent_youngs_modulus_MPa": profile[
                        "screening_engineering_constants"
                    ]["E1"],
                    "status": solver["status"],
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
                    "out_of_plane_frequency_Hz": solved_frequency,
                    "self_consistency_change": change,
                    "self_consistency_converged": False,
                    "selected_mode_consistent": False,
                    "terminal_iteration": False,
                    "run_directory": solver["run_directory"],
                    "deck": str(deck.resolve()),
                    "deck_sha256": _sha256(deck),
                    "error": solver.get("error"),
                }
                rows.append(last_row)
                if solver["status"] != "ok":
                    break
                if change is not None and change <= convergence_tolerance:
                    converged = True
                    break
                current_frequency = float(solved_frequency)
            if last_row is None:
                raise RuntimeError("modal iteration produced no rows")
            mode_consistent = (
                bool(selected_modes)
                and len(set(selected_modes)) == 1
            )
            last_row["self_consistency_converged"] = converged
            last_row["selected_mode_consistent"] = mode_consistent
            last_row["terminal_iteration"] = True
            terminal_rows.append(last_row)

    _write_csv(output_dir / "iterations.csv", rows)
    _write_csv(output_dir / "terminal_results.csv", terminal_rows)
    gate = evaluate_frequency_dependent_gate(
        rows, PATTERNS, resolutions
    )
    fine = int(gate["fine_resolution_xy"])
    terminal_lookup = {
        (row["pattern"], int(row["resolution_xy"])): row
        for row in terminal_rows
    }
    fine_results = {
        pattern: {
            "out_of_plane_frequency_Hz": float(
                terminal_lookup[(pattern, fine)][
                    "out_of_plane_frequency_Hz"
                ]
            ),
            "shear_storage_modulus_MPa": float(
                terminal_lookup[(pattern, fine)][
                    "shear_storage_modulus_MPa"
                ]
            ),
            "material_iterations": int(
                terminal_lookup[(pattern, fine)]["iteration"]
            ),
            "self_consistency_change": float(
                terminal_lookup[(pattern, fine)][
                    "self_consistency_change"
                ]
            ),
        }
        for pattern in PATTERNS
    }
    constant_fine = frozen["fine_toolpath_predictions"]
    for pattern in PATTERNS:
        previous = float(
            constant_fine[pattern]["out_of_plane_frequency_Hz"]
        )
        current = fine_results[pattern]["out_of_plane_frequency_Hz"]
        fine_results[pattern][
            "change_from_reported_constant_storage_proxy"
        ] = current / previous - 1.0

    payload: Dict[str, object] = {
        "study_id": (
            "bambu_pla_weird_kite_frequency_dependent_modal_screen_v1"
        ),
        "material": "Bambu PLA Basic",
        "status": (
            "frequency_dependent_relative_modal_screen_admitted"
            if gate["frequency_dependent_modal_gate_passed"]
            else "frequency_dependent_modal_screen_not_admitted"
        ),
        "source_gcode": str(gcode_path.resolve()),
        "source_gcode_sha256": _sha256(gcode_path),
        "source_spectrum_summary": str(
            Path(spectrum["summary_path"]).resolve()
        ),
        "source_spectrum_summary_sha256": _sha256(
            spectrum["summary_path"]
        ),
        "source_spectrum": str(Path(spectrum["spectrum_path"]).resolve()),
        "source_spectrum_sha256": _sha256(spectrum["spectrum_path"]),
        "patterns": list(PATTERNS),
        "resolutions_xy": list(resolutions),
        "maximum_material_iterations": maximum_iterations,
        "self_consistency_tolerance": convergence_tolerance,
        "selected_mode_rule": (
            "largest effective Z modal mass among 12 extracted modes"
        ),
        "fine_results": fine_results,
        "gate": gate,
        "solver_iteration_count": len(rows),
        "restrictions": [
            "The positive storage spectrum is inferred from frequency-domain "
            "DMA and is valid only at 18 C / 40% RH.",
            "Each CalculiX solve uses an in-phase elastic proxy evaluated at "
            "the preceding modal frequency.",
            "The loss-modulus precision gate failed, so fitted loss factors "
            "are sensitivity metadata rather than validated damping.",
            "Frozen sliced paths are mass-calibrated voxel solids with "
            "perfect bead and layer bonding.",
            "Absolute resonance, support compliance, harmonic response, "
            "fatigue, failure, and lifetime still require physical tests.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Frequency-dependent modal screen for promoted weird infills",
        "",
        "Each frozen-toolpath CalculiX eigenfrequency was iterated against "
        "the exact-brand positive storage spectrum until the input material "
        "frequency and solved out-of-plane mode agreed.",
        "",
        "| Pattern | Fine frequency (Hz) | G' (MPa) | Iterations | "
        "vs triangular |",
        "|---|---:|---:|---:|---:|",
    ]
    ratios = gate["fine_frequency_ratios_to_triangular"]
    for pattern in PATTERNS:
        result = fine_results[pattern]
        report.append(
            f"| {pattern} | "
            f"{result['out_of_plane_frequency_Hz']:.3f} | "
            f"{result['shear_storage_modulus_MPa']:.3f} | "
            f"{result['material_iterations']} | "
            f"{ratios[pattern]:.4f} |"
        )
    report.extend(
        [
            "",
            f"Decision: **{payload['status']}**.",
            "",
            "This closes the internal stiffness-frequency loop for the "
            "comparative modal screen. It does not validate absolute "
            "resonance or damping.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate frozen-toolpath CalculiX modes against the exact-brand "
            "frequency-dependent storage spectrum"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--gcode", type=Path, default=DEFAULT_GCODE)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
    )
    args = parser.parse_args()
    result = run_frequency_dependent_modal_screen(
        output_dir=args.output,
        runs_dir=args.runs,
        gcode_path=args.gcode,
        resolutions=args.resolutions,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "solver_iteration_count": result[
                    "solver_iteration_count"
                ],
                "fine_frequency_ratios_to_triangular": result["gate"][
                    "fine_frequency_ratios_to_triangular"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
