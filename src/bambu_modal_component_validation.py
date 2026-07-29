from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .viscoelastic_pattern_modal_screen import _run_modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    PROJECT_ROOT
    / "data/public/mendeley-8jxw5533py-v2/modal_component"
)
MODAL_WORKBOOK = (
    SOURCE_ROOT / "1-1_Modal_Test_Experiment_and_Numerical_Results.xlsx"
)
TOLERANCE_WORKBOOK = SOURCE_ROOT / "2-2_Specimen_tolerance.xlsx"
SOURCE_DECKS = {
    2.0: SOURCE_ROOT / "AR2d5_SOLID2.inp",
    1.0: SOURCE_ROOT / "AR2d5_SOLID1.inp",
}
SOURCE_STEP = SOURCE_ROOT / "Test2_AR2.5.STEP"
FROZEN_PROFILE = (
    PROJECT_ROOT / "data/reported/bambu_pla_raw_curve_elastic.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/bambu_pla_heldout_modal_validation"
)
DEFAULT_RUNS = PROJECT_ROOT / "runs/bambu_pla_heldout_modal_validation"
SOURCE_CHECKSUMS = {
    "1-1_Modal_Test_Experiment_and_Numerical_Results.xlsx": (
        "34545c9f937404864ad34a4d6dd5c03c0f23b20138115a5bf3dc61fea5749408"
    ),
    "2-2_Specimen_tolerance.xlsx": (
        "80470d582a0bbc8cf6392352bb4583e7b468fb8fa0986fef32706744e7548c59"
    ),
    "AR2d5_SOLID2.inp": (
        "048eb3b4548e292fd12cf3d98310a09d17b01a77a7fd7ed920002553e710a8bc"
    ),
    "AR2d5_SOLID1.inp": (
        "a2c5e4c6215909a6639423f9e0ba107a9a39c26854d06de53ed3d38753897833"
    ),
    "Test2_AR2.5.STEP": (
        "e9c816732a3a8b511dc2a68c473516a2d2deaf7fc5f1a89d0693855cd669c6f2"
    ),
    "Figure3_print_orientations.jpg": (
        "ada3d306fbc63e96f07b491f41c57dca0ce478bab4cf67207ec92589dc53129a"
    ),
}
MODE_LABELS = ("1-B", "1-T", "2-B", "2-T", "3-B")
MAXIMUM_BACKEND_FREQUENCY_ERROR = 0.02
MAXIMUM_PUBLISHED_FINE_MESH_CHANGE = 0.02
MAXIMUM_HELDOUT_RELATIVE_RMSE = 0.15


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_sources() -> Dict[str, str]:
    observed = {}
    for filename, expected in SOURCE_CHECKSUMS.items():
        path = SOURCE_ROOT / filename
        if not path.exists():
            raise FileNotFoundError(f"held-out modal source missing: {path}")
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(
                f"held-out modal source checksum changed: {filename}"
            )
        observed[filename] = digest
    return observed


def extract_test2_modal_targets(
    workbook_path: Path = MODAL_WORKBOOK,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    frame = pd.read_excel(workbook_path, header=None)
    matches = np.argwhere(frame.to_numpy(object) == "Test2-1")
    if len(matches) != 1:
        raise ValueError("modal workbook does not contain one Test2-1 block")
    start = int(matches[0, 0])
    header = frame.iloc[start + 1]
    mesh_columns = {}
    for mesh, label in (
        (2.0, "Solid_Mesh_2"),
        (1.0, "Solid_Mesh_1"),
        (0.5, "Solid_Mesh_0.5"),
    ):
        found = np.flatnonzero(header.to_numpy(object) == label)
        if len(found) != 1:
            raise ValueError(f"modal workbook lacks one {label} column")
        mesh_columns[mesh] = int(found[0])

    target_rows: List[Dict[str, object]] = []
    author_rows: List[Dict[str, object]] = []
    mode_row_lookup = {}
    for row_index in range(start + 2, len(frame)):
        label = str(frame.iat[row_index, 0]).strip()
        if label == "Frequency":
            break
        if label not in MODE_LABELS:
            continue
        mode_row_lookup[label] = row_index
        for replicate, column in enumerate((0, 4, 8), start=1):
            observed_label = str(frame.iat[row_index, column]).strip()
            if observed_label != label:
                raise ValueError(
                    f"Test2 replicate {replicate} mode order changed"
                )
            target_rows.append(
                {
                    "specimen_id": f"Test2-{replicate}",
                    "replicate": replicate,
                    "mode": label,
                    "physical_frequency_Hz": float(
                        frame.iat[row_index, column + 1]
                    ),
                    "physical_dimensionless_frequency": float(
                        frame.iat[row_index, column + 3]
                    ),
                    "group_mean_frequency_Hz": float(
                        frame.iat[row_index, 12]
                    ),
                    "group_mean_dimensionless_frequency": float(
                        frame.iat[row_index, 13]
                    ),
                }
            )
        for mesh, column in mesh_columns.items():
            author_rows.append(
                {
                    "mesh_size_mm": mesh,
                    "mode": label,
                    "author_abaqus_frequency_Hz": float(
                        frame.iat[row_index, column]
                    ),
                }
            )
    if set(mode_row_lookup) != set(MODE_LABELS):
        raise ValueError("Test2 modal block does not contain five common modes")
    if len(target_rows) != 15 or len(author_rows) != 15:
        raise ValueError("Test2 modal extraction count changed")
    return target_rows, author_rows


def extract_test2_dimensions(
    workbook_path: Path = TOLERANCE_WORKBOOK,
) -> List[Dict[str, object]]:
    frame = pd.read_excel(workbook_path, header=None)
    matches = np.argwhere(frame.to_numpy(object) == "Test 2-1")
    if len(matches) != 1:
        raise ValueError("tolerance workbook does not contain one Test 2-1")
    start = int(matches[0, 0])
    rows = []
    for replicate, column in enumerate((0, 5, 10), start=1):
        expected = f"Test 2-{replicate}"
        if str(frame.iat[start, column]).strip() != expected:
            raise ValueError(f"tolerance block changed at {expected}")
        if str(frame.iat[start + 8, column]).strip() != "Mean":
            raise ValueError(f"tolerance mean row missing for {expected}")
        rows.append(
            {
                "specimen_id": f"Test2-{replicate}",
                "replicate": replicate,
                "source_print_orientation": str(
                    frame.iat[start, column + 3]
                ).strip(),
                "mean_length_mm": float(frame.iat[start + 8, column + 1]),
                "mean_thickness_mm": float(
                    frame.iat[start + 8, column + 2]
                ),
                "mean_width_mm": float(frame.iat[start + 8, column + 3]),
            }
        )
    if {row["source_print_orientation"] for row in rows} != {
        "Vertical Print",
        "Horizontal Print",
    }:
        raise ValueError("Test2 print-orientation labels changed")
    return rows


def _source_mesh_blocks(path: Path) -> Dict[str, object]:
    text = Path(path).read_text(
        encoding="utf-8", errors="strict"
    ).replace("\r", "")
    node_match = re.search(
        r"^\*Node\s*\n(.*?)^\*Element,\s*type=(\w+)\s*\n",
        text,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if node_match is None:
        raise ValueError("source Abaqus deck lacks a node/element boundary")
    element_start = node_match.end()
    element_end = re.search(
        r"^\*Nset,", text[element_start:], flags=re.MULTILINE | re.IGNORECASE
    )
    if element_end is None:
        raise ValueError("source Abaqus deck lacks the part node set")
    element_text = text[
        element_start : element_start + element_end.start()
    ].strip()
    fixed_match = re.search(
        r"^\*Nset,\s*nset=Set-3,\s*instance=Part-1-1\s*\n"
        r"(.*?)^\*Elset,",
        text,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if fixed_match is None:
        raise ValueError("source Abaqus deck lacks the clamped Set-3")
    material_match = re.search(
        r"^\*Density\s*\n\s*([^,\n]+),?.*?"
        r"^\*Elastic\s*\n\s*([^,\n]+),\s*([^,\n]+)",
        text,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if material_match is None:
        raise ValueError("source Abaqus deck lacks isotropic material data")
    nodes = node_match.group(1).strip()
    elements = element_text
    fixed = fixed_match.group(1).strip()
    node_count = len(nodes.splitlines())
    element_count = len(elements.splitlines())
    fixed_ids = [
        int(value)
        for value in re.split(r"[,\s]+", fixed)
        if value.strip()
    ]
    return {
        "nodes": nodes,
        "elements": elements,
        "fixed": fixed,
        "element_type": node_match.group(2).upper(),
        "node_count": node_count,
        "element_count": element_count,
        "fixed_node_count": len(fixed_ids),
        "source_density_tonnes_mm3": float(material_match.group(1)),
        "source_E_MPa": float(material_match.group(2)),
        "source_nu": float(material_match.group(3)),
    }


def render_calculix_modal_deck(
    source_path: Path,
    profile: Mapping[str, float],
    mode_count: int = 10,
) -> Tuple[str, Dict[str, object]]:
    if mode_count < len(MODE_LABELS):
        raise ValueError("modal deck must extract at least five modes")
    blocks = _source_mesh_blocks(source_path)
    deck = (
        "*HEADING\n"
        "Bambu PLA Basic held-out Test2 AR2.5 modal replay\n"
        "*NODE,NSET=NALL\n"
        f"{blocks['nodes']}\n"
        f"*ELEMENT,TYPE={blocks['element_type']},ELSET=EALL\n"
        f"{blocks['elements']}\n"
        "*NSET,NSET=FIXED\n"
        f"{blocks['fixed']}\n"
        "*MATERIAL,NAME=PLA\n"
        "*DENSITY\n"
        f"{float(profile['density_tonnes_mm3']):.12g}\n"
        "*ELASTIC\n"
        f"{float(profile['E_MPa']):.12g},"
        f"{float(profile['nu']):.12g}\n"
        "*SOLID SECTION,ELSET=EALL,MATERIAL=PLA\n"
        "*BOUNDARY\n"
        "FIXED,1,3,0.\n"
        "*STEP\n"
        "*FREQUENCY\n"
        f"{mode_count}\n"
        "*NODE FILE\n"
        "U\n"
        "*END STEP\n"
    )
    metadata = {
        key: value
        for key, value in blocks.items()
        if key not in {"nodes", "elements", "fixed"}
    }
    return deck, metadata


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classify_common_plate_modes(
    modes: Sequence[Mapping[str, float]],
) -> Dict[str, Mapping[str, float]]:
    ordered = sorted(modes, key=lambda row: float(row["frequency_Hz"]))
    bending = [
        row
        for row in ordered
        if float(row["effective_mass_x"]) > 1.0e-8
        and float(row["effective_mass_x"])
        > 10.0 * float(row["effective_mass_y"])
    ][:3]
    used = {int(row["mode"]) for row in bending}
    torsion = [
        row
        for row in ordered
        if int(row["mode"]) not in used
        and float(row["effective_rotational_mass_z"]) > 1.0e-5
        and float(row["effective_mass_y"]) < 1.0e-8
    ][:2]
    if len(bending) != 3 or len(torsion) != 2:
        raise ValueError(
            "CalculiX modes cannot be classified into 1/2/3-B and 1/2-T"
        )
    return {
        "1-B": bending[0],
        "2-B": bending[1],
        "3-B": bending[2],
        "1-T": torsion[0],
        "2-T": torsion[1],
    }


def _frozen_profile(path: Path = FROZEN_PROFILE) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    elastic = payload["screening_isotropic"]
    return {
        "profile_id": payload["profile_id"],
        "E_MPa": float(elastic["E_MPa"]),
        "nu": float(elastic["nu"]),
        "density_tonnes_mm3": float(payload["density_tonnes_mm3"]),
        "source_path": str(Path(path).resolve()),
        "source_sha256": _sha256(path),
    }


def run_validation(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
) -> Dict[str, object]:
    source_hashes = _verify_sources()
    targets, author_rows = extract_test2_modal_targets()
    dimensions = extract_test2_dimensions()
    dimension_lookup = {row["specimen_id"]: row for row in dimensions}
    for row in targets:
        row.update(dimension_lookup[row["specimen_id"]])

    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    deck_dir = output_dir / "decks"
    deck_dir.mkdir(parents=True, exist_ok=True)
    author_lookup = {
        (float(row["mesh_size_mm"]), str(row["mode"])): float(
            row["author_abaqus_frequency_Hz"]
        )
        for row in author_rows
    }
    source_profile = _source_mesh_blocks(SOURCE_DECKS[1.0])
    profiles = {
        "source_author_isotropic": {
            "profile_id": "source_author_isotropic",
            "E_MPa": float(source_profile["source_E_MPa"]),
            "nu": float(source_profile["source_nu"]),
            "density_tonnes_mm3": float(
                source_profile["source_density_tonnes_mm3"]
            ),
        },
        "frozen_raw_curve_isotropic": _frozen_profile(),
    }
    solver_rows: List[Dict[str, object]] = []
    mesh_metadata = {}
    for profile_name, profile in profiles.items():
        for mesh_size, source_deck in SOURCE_DECKS.items():
            deck_text, metadata = render_calculix_modal_deck(
                source_deck, profile
            )
            deck_path = deck_dir / (
                f"test2-ar2p5-{profile_name}-h{mesh_size:g}.inp"
            )
            deck_path.write_text(deck_text, encoding="utf-8")
            mesh_metadata[str(mesh_size)] = {
                **metadata,
                "source_deck": str(source_deck.resolve()),
                "source_deck_sha256": _sha256(source_deck),
                "rendered_deck": str(deck_path.resolve()),
                "rendered_deck_sha256": _sha256(deck_path),
            }
            solver = _run_modal(
                deck_path,
                runs_dir,
                f"heldout-test2-{profile_name}-h{mesh_size:g}",
            )
            if solver["status"] != "ok":
                raise RuntimeError(
                    f"held-out modal solve failed: {profile_name}, "
                    f"h={mesh_size:g}: {solver.get('error')}"
                )
            modes = solver["modes"]
            if len(modes) < len(MODE_LABELS):
                raise RuntimeError("held-out modal solve returned too few modes")
            classified = classify_common_plate_modes(modes)
            for label in MODE_LABELS:
                mode = classified[label]
                solver_rows.append(
                    {
                        "profile": profile_name,
                        "mesh_size_mm": mesh_size,
                        "mode": label,
                        "calculix_frequency_Hz": float(
                            mode["frequency_Hz"]
                        ),
                        "author_abaqus_frequency_Hz": author_lookup[
                            (mesh_size, label)
                        ],
                        "author_converged_half_mm_frequency_Hz": (
                            author_lookup[(0.5, label)]
                        ),
                        "calculix_to_author_relative_error": (
                            float(mode["frequency_Hz"])
                            / author_lookup[(mesh_size, label)]
                            - 1.0
                        ),
                        "calculix_to_author_converged_relative_error": (
                            float(mode["frequency_Hz"])
                            / author_lookup[(0.5, label)]
                            - 1.0
                        ),
                        "calculix_mode_number": int(mode["mode"]),
                        "run_directory": solver["run_directory"],
                    }
                )

    source_backend_rows = [
        row
        for row in solver_rows
        if row["profile"] == "source_author_isotropic"
        and math.isclose(float(row["mesh_size_mm"]), 1.0)
    ]
    maximum_backend_error = max(
        abs(float(row["calculix_to_author_converged_relative_error"]))
        for row in source_backend_rows
    )
    published_fine_changes = {}
    for mode in MODE_LABELS:
        published_fine_changes[mode] = abs(
            author_lookup[(1.0, mode)] / author_lookup[(0.5, mode)] - 1.0
        )
    maximum_published_fine_change = max(published_fine_changes.values())

    heldout_predictions = {
        str(row["mode"]): float(row["calculix_frequency_Hz"])
        for row in solver_rows
        if row["profile"] == "frozen_raw_curve_isotropic"
        and math.isclose(float(row["mesh_size_mm"]), 1.0)
    }
    heldout_rows = []
    for mode in MODE_LABELS:
        physical = next(
            float(row["group_mean_frequency_Hz"])
            for row in targets
            if row["mode"] == mode
        )
        predicted = heldout_predictions[mode]
        heldout_rows.append(
            {
                "mode": mode,
                "physical_group_mean_frequency_Hz": physical,
                "frozen_profile_calculix_frequency_Hz": predicted,
                "relative_error": predicted / physical - 1.0,
                "absolute_relative_error": abs(predicted / physical - 1.0),
            }
        )
    heldout_relative_rmse = float(
        np.sqrt(
            np.mean(
                [
                    float(row["relative_error"]) ** 2
                    for row in heldout_rows
                ]
            )
        )
    )
    gates = {
        "source_checksums_passed": True,
        "physical_specimen_mode_count_passed": (
            len(targets) == 15
            and len({row["specimen_id"] for row in targets}) == 3
        ),
        "calculix_author_backend_gate_passed": (
            maximum_backend_error
            <= MAXIMUM_BACKEND_FREQUENCY_ERROR
        ),
        "published_author_mesh_gate_passed": (
            maximum_published_fine_change
            <= MAXIMUM_PUBLISHED_FINE_MESH_CHANGE
        ),
        "heldout_small_strain_modal_gate_passed": (
            heldout_relative_rmse <= MAXIMUM_HELDOUT_RELATIVE_RMSE
        ),
        "component_modal_data_used_for_calibration": False,
        "local_axis_directional_calibration_released": False,
    }
    gate = all(
        value is True
        for key, value in gates.items()
        if key
        not in {
            "component_modal_data_used_for_calibration",
            "local_axis_directional_calibration_released",
        }
    ) and gates["component_modal_data_used_for_calibration"] is False

    _write_csv(output_dir / "physical_modal_targets.csv", targets)
    _write_csv(output_dir / "author_fea_reference.csv", author_rows)
    _write_csv(output_dir / "calculix_solver_rows.csv", solver_rows)
    _write_csv(output_dir / "heldout_comparison.csv", heldout_rows)
    provenance = {
        "validation_id": "bambu_pla_test2_ar2p5_heldout_modal_v1",
        "status": (
            "heldout_small_strain_modal_gate_passed"
            if gate
            else "heldout_modal_gate_failed"
        ),
        "material": "Bambu PLA Basic",
        "source_dataset_doi": "10.17632/8jxw5533py.2",
        "source_article_doi": "10.1016/j.dib.2026.113015",
        "source_license": "CC BY 4.0",
        "source_hashes": source_hashes,
        "component": {
            "source_id": "Test2",
            "geometry": "uniform cantilever plate",
            "aspect_ratio": 2.5,
            "nominal_model_bounds_mm": [4.0, 50.0, 125.0],
            "physical_specimens": 3,
            "physical_modes_per_specimen": 5,
            "source_print_orientations": sorted(
                {row["source_print_orientation"] for row in dimensions}
            ),
            "step_path": str(SOURCE_STEP.resolve()),
            "step_sha256": _sha256(SOURCE_STEP),
        },
        "profiles": profiles,
        "mesh_metadata": mesh_metadata,
        "maximum_converged_calculix_to_author_backend_relative_error": (
            maximum_backend_error
        ),
        "published_author_one_to_half_mm_frequency_changes": (
            published_fine_changes
        ),
        "maximum_published_author_one_to_half_mm_frequency_change": (
            maximum_published_fine_change
        ),
        "heldout_relative_rmse": heldout_relative_rmse,
        "heldout_maximum_absolute_relative_error": max(
            float(row["absolute_relative_error"]) for row in heldout_rows
        ),
        "thresholds": {
            "maximum_backend_frequency_error": (
                MAXIMUM_BACKEND_FREQUENCY_ERROR
            ),
            "maximum_published_fine_mesh_change": (
                MAXIMUM_PUBLISHED_FINE_MESH_CHANGE
            ),
            "maximum_heldout_relative_rmse": (
                MAXIMUM_HELDOUT_RELATIVE_RMSE
            ),
        },
        "gates": gates,
        "allowed_use": (
            "External held-out small-strain modal validation only. Component "
            "frequencies are prohibited from material fitting, profile "
            "selection, or threshold tuning."
        ),
        "directional_evidence_decision": (
            "The source build-platform labels and Figure 3 do not provide "
            "a unique local bead-axis/in-layer-transverse mapping for the "
            "rectilinear toolpaths, and the raw tensile workbook has no "
            "lateral strain. No E1/E2 or Poisson/shear calibration is released."
        ),
        "limitations": [
            "The paper treats the printed component as isotropic in the elastic regime.",
            "The component response is modal frequency, not a static force-displacement curve.",
            "The three physical specimens mix vertical and horizontal print placements.",
            "Mode order is cross-checked against the source Abaqus solid results.",
            "The source workbook has documented tensile label/count conflicts.",
            "No modal target is used to tune the frozen raw-curve material profile.",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Exact-brand held-out component modal validation

Three physical Bambu PLA Basic Test2 AR=2.5 cantilever specimens contribute
five common modes each. Their geometry, dimensional measurements, source
Abaqus meshes, and physical modal frequencies are checksum-pinned from
Mendeley Data version 2.

The CalculiX conversion reproduces the source Abaqus backend at 2 mm and 1 mm.
The maximum backend frequency difference is {maximum_backend_error:.2%}.
The source-published 1 mm to 0.5 mm maximum change is
{maximum_published_fine_change:.2%}.

The Bambu raw-curve isotropic profile was frozen before reading any component
frequency. Across the five held-out modal means, its relative RMSE is
{heldout_relative_rmse:.2%}; the maximum individual absolute relative error is
{provenance['heldout_maximum_absolute_relative_error']:.2%}. The fixed
small-strain modal gate is {'passed' if gate else 'failed'}.

This is a component-level modal validation, not a static curve validation and
not a directional-material calibration. The source does not uniquely map the
rectilinear bead direction to local axes 1/2 for all placement labels and does
not publish lateral strain, so it cannot close the nine-constant orthotropic
campaign.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the exact-brand held-out cantilever modal validation"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    args = parser.parse_args()
    result = run_validation(Path(args.output), Path(args.runs))
    print(
        f"{result['status']}; held-out relative RMSE="
        f"{result['heldout_relative_rmse']:.3%}"
    )


if __name__ == "__main__":
    main()
