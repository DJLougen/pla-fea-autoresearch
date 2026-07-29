from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .bambu_viscoelastic_wave_reference import DEFAULT_OUTPUT as WAVE_REFERENCE
from .calculix import find_ccx
from .custom_pattern_toolpath import DEFAULT_GCODE, parse_deposited_segments
from .custom_pattern_toolpath_fea import (
    DEFAULT_OUTPUT as TOOLPATH_BENDING,
    FINE_RASTER_RESOLUTION,
    _pattern_for_object,
    build_mass_calibrated_fine_geometry,
    coarsen_to_voxel_mesh,
    render_toolpath_bending_deck,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ROOT = PROJECT_ROOT / "results/custom_infill/graded_hourglass"
DEFAULT_OUTPUT = CUSTOM_ROOT / "viscoelastic_modal_screen"
DEFAULT_RUNS = PROJECT_ROOT / "runs/custom_infill/viscoelastic_modal_screen"
DEFAULT_RESOLUTIONS = (60, 80)
MODE_COUNT = 12
MAXIMUM_FREQUENCY_CONVERGENCE_CHANGE = 0.10


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_storage_profile(
    reference_path: Path = WAVE_REFERENCE / "provenance.json",
) -> Dict[str, object]:
    reference = _json(reference_path)
    if reference.get("quality_gate_passed") is not True:
        raise ValueError("viscoelastic reference quality gate is not passed")
    static = reference["reported_static_measurements"]
    dynamic = reference["reported_dynamic_measurements"]
    nu = float(static["poisson_ratio"])
    g_storage = float(dynamic["shear_storage_modulus_MPa"])
    equivalent_e = float(dynamic["storage_equivalent_youngs_modulus_MPa"])
    return {
        "profile_id": "bambu_pla_18C_broadband_storage_proxy_v1",
        "material": "Bambu PLA Basic",
        "density_tonnes_mm3": (
            float(static["printed_density_kg_m3"]) * 1.0e-12
        ),
        "screening_engineering_constants": {
            "E1": equivalent_e,
            "E2": equivalent_e,
            "E3": equivalent_e,
            "nu12": nu,
            "nu13": nu,
            "nu23": nu,
            "G12": g_storage,
            "G13": g_storage,
            "G23": g_storage,
        },
        "loss_factor": float(dynamic["loss_factor"]),
        "reference_temperature_C": float(
            dynamic["reference_temperature_C"]
        ),
        "minimum_reduced_frequency_Hz": float(
            dynamic["minimum_reduced_frequency_Hz"]
        ),
        "maximum_reduced_frequency_Hz": float(
            dynamic["maximum_reduced_frequency_Hz"]
        ),
        "source_reference": str(Path(reference_path).resolve()),
        "restriction": (
            "Isotropic in-phase storage proxy for small-strain modal "
            "sensitivity; not a static, failure, or lifetime material card."
        ),
    }


def render_modal_deck(
    mesh,
    profile: Mapping[str, object],
    mode_count: int = MODE_COUNT,
) -> str:
    if mode_count < 3:
        raise ValueError("at least three modes are required")
    preamble = render_toolpath_bending_deck(mesh, profile).split(
        "*STEP", 1
    )[0]
    return (
        preamble
        + "*STEP\n"
        + "*FREQUENCY\n"
        + f"{mode_count}\n"
        + "*BOUNDARY\n"
        + "SUPPORTS,3,3,0.\n"
        + "ANCHOR,1,2,0.\n"
        + "ANCHOR_TWO,2,2,0.\n"
        + "*NODE FILE\n"
        + "U\n"
        + "*END STEP\n"
    )


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def parse_modal_dat(path: Path) -> list[Dict[str, float]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    eigen_heading = re.search(
        r"MODE\s+NO\s+EIGENVALUE.*?IMAGINARY\s+PART",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    participation_heading = re.search(
        r"P\s+A\s+R\s+T\s+I\s+C\s+I\s+P\s+A\s+T\s+I\s+O\s+N",
        text,
        flags=re.IGNORECASE,
    )
    if eigen_heading is None or participation_heading is None:
        raise ValueError("CalculiX modal eigenvalue table was not found")
    eigen_text = text[eigen_heading.end() : participation_heading.start()]
    eigen_rows = re.findall(
        rf"^\s*(\d+)\s+({_NUMBER})\s+({_NUMBER})\s+"
        rf"({_NUMBER})\s+({_NUMBER})\s*$",
        eigen_text,
        flags=re.MULTILINE,
    )
    if not eigen_rows:
        raise ValueError("CalculiX modal table contains no modes")

    mass_heading = re.search(
        r"E\s+F\s+F\s+E\s+C\s+T\s+I\s+V\s+E\s+"
        r"M\s+O\s+D\s+A\s+L\s+M\s+A\s+S\s+S",
        text,
        flags=re.IGNORECASE,
    )
    total_heading = re.search(
        r"T\s+O\s+T\s+A\s+L\s+E\s+F\s+F\s+E\s+C\s+T\s+I\s+V\s+E",
        text,
        flags=re.IGNORECASE,
    )
    masses: Dict[int, tuple[float, ...]] = {}
    if mass_heading is not None and total_heading is not None:
        mass_text = text[mass_heading.end() : total_heading.start()]
        for row in re.findall(
            rf"^\s*(\d+)\s+({_NUMBER})\s+({_NUMBER})\s+"
            rf"({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s+"
            rf"({_NUMBER})\s*$",
            mass_text,
            flags=re.MULTILINE,
        ):
            masses[int(row[0])] = tuple(_number(value) for value in row[1:])

    modes = []
    for mode, eigenvalue, angular, cycles, imaginary in eigen_rows:
        mode_number = int(mode)
        modal_mass = masses.get(mode_number, (float("nan"),) * 6)
        modes.append(
            {
                "mode": mode_number,
                "eigenvalue": _number(eigenvalue),
                "angular_frequency_rad_s": _number(angular),
                "frequency_Hz": _number(cycles),
                "imaginary_angular_frequency": _number(imaginary),
                "effective_mass_x": modal_mass[0],
                "effective_mass_y": modal_mass[1],
                "effective_mass_z": modal_mass[2],
                "effective_rotational_mass_x": modal_mass[3],
                "effective_rotational_mass_y": modal_mass[4],
                "effective_rotational_mass_z": modal_mass[5],
            }
        )
    return modes


def select_out_of_plane_mode(
    modes: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    finite = [
        dict(mode)
        for mode in modes
        if np.isfinite(float(mode["effective_mass_z"]))
    ]
    if not finite:
        raise ValueError("modal output lacks effective Z mass")
    selected = max(finite, key=lambda mode: mode["effective_mass_z"])
    total = sum(float(mode["effective_mass_z"]) for mode in finite)
    selected["z_mass_fraction_within_extracted_modes"] = (
        float(selected["effective_mass_z"]) / total
        if total > 0.0
        else float("nan")
    )
    return selected


def _run_modal(
    deck_path: Path,
    runs_dir: Path,
    run_id: str,
    timeout_seconds: float = 600.0,
) -> Dict[str, object]:
    executable = find_ccx("ccx")
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists():
        run_dir = Path(runs_dir) / f"{run_id}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "model.inp").write_text(
        Path(deck_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if executable is None:
        return {
            "status": "executable_missing",
            "run_directory": str(run_dir.resolve()),
            "error": "CalculiX executable was not found",
            "modes": [],
        }
    start = time.monotonic()
    try:
        process = subprocess.run(
            [str(executable), "model"],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "run_directory": str(run_dir.resolve()),
            "error": f"CalculiX exceeded {timeout_seconds:g} seconds",
            "modes": [],
        }
    runtime = time.monotonic() - start
    (run_dir / "stdout.txt").write_text(
        process.stdout, encoding="utf-8"
    )
    (run_dir / "stderr.txt").write_text(
        process.stderr, encoding="utf-8"
    )
    if process.returncode != 0 or "Job finished" not in process.stdout:
        return {
            "status": "solver_failed",
            "run_directory": str(run_dir.resolve()),
            "runtime_seconds": runtime,
            "error": f"CalculiX returned {process.returncode}",
            "modes": [],
        }
    try:
        modes = parse_modal_dat(run_dir / "model.dat")
    except Exception as error:
        return {
            "status": "parse_failed",
            "run_directory": str(run_dir.resolve()),
            "runtime_seconds": runtime,
            "error": str(error),
            "modes": [],
        }
    return {
        "status": "ok",
        "run_directory": str(run_dir.resolve()),
        "runtime_seconds": runtime,
        "error": None,
        "modes": modes,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_modal_screen(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    gcode_path: Path = DEFAULT_GCODE,
    reference_path: Path = WAVE_REFERENCE / "provenance.json",
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    mode_count: int = MODE_COUNT,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    resolutions = tuple(sorted(set(int(value) for value in resolutions)))
    if not resolutions:
        raise ValueError("at least one modal resolution is required")
    if any(FINE_RASTER_RESOLUTION % value != 0 for value in resolutions):
        raise ValueError("modal resolutions must divide the fine raster")
    toolpath_manifest = _json(TOOLPATH_BENDING / "manifest.json")
    if (
        toolpath_manifest.get("gates", {}).get(
            "relative_toolpath_aware_bending_ranking_admitted"
        )
        is not True
    ):
        raise ValueError("toolpath-aware bending morphology is not admitted")
    profile = build_storage_profile(reference_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decks").mkdir(parents=True, exist_ok=True)
    parsed = parse_deposited_segments(gcode_path)
    selected_objects = {
        pattern: next(
            name
            for name in sorted(parsed)
            if _pattern_for_object(name) == pattern
        )
        for pattern in ("triangular", "graded_hourglass")
    }
    rows: list[Dict[str, object]] = []
    mode_rows: list[Dict[str, object]] = []
    loss_factor = float(profile["loss_factor"])
    damping_ratio = loss_factor / 2.0
    for pattern, object_name in selected_objects.items():
        geometry = build_mass_calibrated_fine_geometry(
            pattern,
            object_name,
            parsed[object_name],
            resolution=FINE_RASTER_RESOLUTION,
        )
        for resolution in resolutions:
            mesh = coarsen_to_voxel_mesh(geometry, resolution)
            deck = output_dir / "decks" / f"{pattern}-{resolution}.inp"
            deck.write_text(
                render_modal_deck(mesh, profile, mode_count),
                encoding="utf-8",
            )
            solver = _run_modal(
                deck,
                runs_dir,
                f"modal-{pattern}-{resolution}",
            )
            selected_mode = None
            if solver["status"] == "ok":
                selected_mode = select_out_of_plane_mode(solver["modes"])
                for mode in solver["modes"]:
                    mode_rows.append(
                        {
                            "pattern": pattern,
                            "resolution_xy": resolution,
                            **mode,
                        }
                    )
            frequency = (
                None
                if selected_mode is None
                else float(selected_mode["frequency_Hz"])
            )
            rows.append(
                {
                    "pattern": pattern,
                    "resolution_xy": resolution,
                    "nodes": len(mesh.node_tags),
                    "elements": len(mesh.element_tags),
                    "represented_volume_mm3": mesh.represented_volume_mm3,
                    "represented_mass_g": (
                        mesh.represented_volume_mm3
                        * float(profile["density_tonnes_mm3"])
                        * 1.0e6
                    ),
                    "status": solver["status"],
                    "selected_out_of_plane_mode": (
                        None
                        if selected_mode is None
                        else int(selected_mode["mode"])
                    ),
                    "out_of_plane_frequency_Hz": frequency,
                    "z_mass_fraction_within_extracted_modes": (
                        None
                        if selected_mode is None
                        else selected_mode[
                            "z_mass_fraction_within_extracted_modes"
                        ]
                    ),
                    "loss_factor": loss_factor,
                    "small_damping_ratio_approximation": damping_ratio,
                    "approximate_damped_frequency_Hz": (
                        None
                        if frequency is None
                        else frequency
                        * np.sqrt(max(0.0, 1.0 - damping_ratio**2))
                    ),
                    "run_directory": solver["run_directory"],
                    "runtime_seconds": solver.get("runtime_seconds"),
                    "deck": str(deck.resolve()),
                    "deck_sha256": _sha256(deck),
                    "error": solver.get("error"),
                }
            )
    _write_csv(output_dir / "results.csv", rows)
    _write_csv(output_dir / "modes.csv", mode_rows)
    lookup = {
        (str(row["pattern"]), int(row["resolution_xy"])): row
        for row in rows
    }
    fine = resolutions[-1]
    previous = resolutions[-2] if len(resolutions) > 1 else fine
    convergence = {}
    for pattern in ("triangular", "graded_hourglass"):
        coarse_value = lookup[(pattern, previous)][
            "out_of_plane_frequency_Hz"
        ]
        fine_value = lookup[(pattern, fine)]["out_of_plane_frequency_Hz"]
        convergence[pattern] = (
            None
            if coarse_value is None or fine_value is None
            else abs(float(coarse_value) / float(fine_value) - 1.0)
        )
    triangular = lookup[("triangular", fine)]["out_of_plane_frequency_Hz"]
    custom = lookup[("graded_hourglass", fine)][
        "out_of_plane_frequency_Hz"
    ]
    ratio = (
        None
        if triangular is None or custom is None
        else float(custom) / float(triangular)
    )
    all_ok = all(row["status"] == "ok" for row in rows)
    converged = all(
        value is not None
        and float(value) <= MAXIMUM_FREQUENCY_CONVERGENCE_CHANGE
        for value in convergence.values()
    )
    within_reference = all(
        row["out_of_plane_frequency_Hz"] is not None
        and float(profile["minimum_reduced_frequency_Hz"])
        <= float(row["out_of_plane_frequency_Hz"])
        <= float(profile["maximum_reduced_frequency_Hz"])
        for row in rows
    )
    gate = all_ok and converged and within_reference
    payload: Dict[str, object] = {
        "study_id": "graded_hourglass_18C_viscoelastic_modal_screen_v1",
        "material": "Bambu PLA Basic",
        "status": (
            "bounded_relative_modal_screen_admitted"
            if gate
            else "modal_screen_not_admitted"
        ),
        "source_gcode": str(Path(gcode_path).resolve()),
        "source_gcode_sha256": _sha256(gcode_path),
        "source_viscoelastic_reference": str(Path(reference_path).resolve()),
        "source_viscoelastic_reference_sha256": _sha256(reference_path),
        "source_toolpath_manifest": str(
            (TOOLPATH_BENDING / "manifest.json").resolve()
        ),
        "source_toolpath_manifest_sha256": _sha256(
            TOOLPATH_BENDING / "manifest.json"
        ),
        "storage_profile": profile,
        "modal_resolutions_xy": list(resolutions),
        "mode_count": mode_count,
        "boundary_condition": (
            "simply supported in Z at the registered three-point-bend "
            "supports, with minimal in-plane anchors"
        ),
        "selected_mode_rule": (
            "largest effective Z modal mass among the extracted modes"
        ),
        "fine_out_of_plane_frequency_Hz": {
            "triangular": triangular,
            "graded_hourglass": custom,
        },
        "graded_hourglass_to_triangular_frequency_ratio": ratio,
        "frequency_convergence_change": convergence,
        "loss_factor": loss_factor,
        "small_damping_ratio_approximation": damping_ratio,
        "gates": {
            "all_calculix_cases_passed": all_ok,
            "frequency_change_at_most_10_percent": converged,
            "all_selected_modes_within_1Hz_to_25kHz_reference": (
                within_reference
            ),
            "bounded_relative_modal_screen_admitted": gate,
            "absolute_resonance_physically_validated": False,
            "frequency_response_or_attenuation_validated": False,
        },
        "restrictions": [
            "The geometry is the existing mass-calibrated voxel replay of "
            "the frozen Orca G-code, not a resolved bead/contact mesh.",
            "The real storage modulus is used for eigenfrequencies; the "
            "reported loss factor is only a small-damping estimate.",
            "No harmonic forcing, support compliance, fixture mass, or "
            "frequency-dependent curve fitting is included.",
            "The isotropic storage proxy does not close G13/G23 or "
            "out-of-plane Poisson measurements.",
            "Use this to compare pattern resonance shifts before printing, "
            "not as an absolute vibration certification.",
        ],
        "results_csv": str((output_dir / "results.csv").resolve()),
        "modes_csv": str((output_dir / "modes.csv").resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# 18 C viscoelastic modal pattern screen

The frozen Orca toolpaths were replayed as mass-calibrated voxel solids using
the exact-brand printed density, measured Poisson ratio, reported broadband
shear storage modulus, and reported loss factor.

| Pattern | Selected out-of-plane frequency at {fine} x {fine} (Hz) |
|---|---:|
| Triangular | {float(triangular) if triangular is not None else float('nan'):.3f} |
| Graded hourglass | {float(custom) if custom is not None else float('nan'):.3f} |

The graded-hourglass/triangular frequency ratio is
**{float(ratio) if ratio is not None else float('nan'):.4f}**. The loss factor
is {loss_factor:.5f}, corresponding to a small-damping ratio estimate of
{damping_ratio:.5f}. Resolution changes are
{', '.join(f"{key} {float(value):.2%}" for key, value in convergence.items() if value is not None)}.

Decision: **{payload['status']}**.

This admits a comparative, small-strain modal screen inside the 18 C,
1 Hz-25 kHz reduced-frequency evidence window. Absolute resonance, forced
response, attenuation, fixture effects, fatigue, and failure still require
physical tests.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the exact-brand viscoelastic modal pattern screen"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
    )
    args = parser.parse_args()
    result = run_modal_screen(
        output_dir=args.output,
        runs_dir=args.runs,
        resolutions=args.resolutions,
    )
    print(
        f"modal screen: {result['status']}; "
        f"frequency ratio="
        f"{result['graded_hourglass_to_triangular_frequency_ratio']:.4f}; "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
