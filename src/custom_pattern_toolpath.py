from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .custom_pattern_slicer import (
    FILAMENT_DENSITY_G_CM3,
    FILAMENT_DIAMETER_MM,
    MAXIMUM_TOOLPATH_MASS_RATIO_ERROR,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ROOT = PROJECT_ROOT / "results/custom_infill/graded_hourglass"
DEFAULT_PACKAGE = CUSTOM_ROOT / "physical_validation"
DEFAULT_GCODE = DEFAULT_PACKAGE / "manufacturing/sliced/plate_1.gcode"
DEFAULT_MANUFACTURING_MANIFEST = (
    DEFAULT_PACKAGE / "manufacturing/manifest.json"
)
DEFAULT_PRINT_MESH_MANIFEST = (
    DEFAULT_PACKAGE / "printable_stl/manifest.json"
)
DEFAULT_MATERIAL_PROFILE = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_tds_transverse_sensitivity.json"
)
DEFAULT_OUTPUT = DEFAULT_PACKAGE / "toolpath_morphology"

MAXIMUM_ABSOLUTE_UNDERFILL_FRACTION = 0.10
MAXIMUM_DEPOSITION_FRACTION_DIFFERENCE = 0.02
EXPECTED_LAYER_COUNT = 20
VOLUME_TOLERANCE_CM3 = 1e-9

_NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
_AXIS = re.compile(rf"(?:^|\s)([XYZE])({_NUMBER})")
_STOP = re.compile(r"^; stop printing object (.+?) id:")
_START = "; start printing object, unique label id:"
_FEATURE = "; FEATURE:"
_LINE_WIDTH = "; LINE_WIDTH:"
_Z_HEIGHT = "; Z_HEIGHT:"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pattern_for_object(
    name: str, custom_pattern: str = "graded_hourglass"
) -> str:
    if name.startswith("triangular.stl"):
        return "triangular"
    if name.startswith(f"{custom_pattern}.stl"):
        return custom_pattern
    raise ValueError(f"unexpected sliced object name: {name}")


def parse_deposited_segments(path: Path) -> Dict[str, List[Dict[str, object]]]:
    """Parse positive-extrusion XY segments inside Orca object sections.

    Orca labels the beginning of an object section with a numeric ID but only
    provides its model name at the matching stop marker. Segments are therefore
    buffered until the stop marker and then assigned to that object.
    """

    x = 0.0
    y = 0.0
    z = 0.0
    extrusion_position = 0.0
    position_absolute = True
    extrusion_relative = True
    layer_z: float | None = None
    feature = "Unspecified"
    line_width: float | None = None
    active = False
    buffered: List[Dict[str, object]] = []
    by_object: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for line_number, raw in enumerate(
        Path(path).open(encoding="utf-8", errors="ignore"), start=1
    ):
        line = raw.strip()
        if line.startswith(_Z_HEIGHT):
            layer_z = float(line.split(":", 1)[1].strip())
            continue
        if line.startswith(_FEATURE):
            feature = line.split(":", 1)[1].strip() or "Unspecified"
            continue
        if line.startswith(_LINE_WIDTH):
            line_width = float(line.split(":", 1)[1].strip())
            continue
        if line.startswith(_START):
            if active:
                raise ValueError(
                    f"nested Orca object section at G-code line {line_number}"
                )
            active = True
            buffered = []
            continue
        stop_match = _STOP.match(line)
        if stop_match:
            if not active:
                raise ValueError(
                    f"orphan Orca object stop at G-code line {line_number}"
                )
            by_object[stop_match.group(1)].extend(buffered)
            buffered = []
            active = False
            continue
        if line == "G90" or line.startswith("G90 "):
            position_absolute = True
            continue
        if line == "G91" or line.startswith("G91 "):
            position_absolute = False
            continue
        if line == "M82" or line.startswith("M82 "):
            extrusion_relative = False
            continue
        if line == "M83" or line.startswith("M83 "):
            extrusion_relative = True
            continue
        if line.startswith("G92"):
            axes = {axis: float(value) for axis, value in _AXIS.findall(line)}
            if "X" in axes:
                x = axes["X"]
            if "Y" in axes:
                y = axes["Y"]
            if "Z" in axes:
                z = axes["Z"]
            if "E" in axes:
                extrusion_position = axes["E"]
            continue
        if not line.startswith(("G0 ", "G1 ")):
            continue

        command = line.split(";", 1)[0]
        axes = {axis: float(value) for axis, value in _AXIS.findall(command)}
        old_x, old_y, old_z = x, y, z
        if "X" in axes:
            x = axes["X"] if position_absolute else x + axes["X"]
        if "Y" in axes:
            y = axes["Y"] if position_absolute else y + axes["Y"]
        if "Z" in axes:
            z = axes["Z"] if position_absolute else z + axes["Z"]

        deposited_filament = 0.0
        if "E" in axes:
            if extrusion_relative:
                deposited_filament = axes["E"]
                extrusion_position += axes["E"]
            else:
                deposited_filament = axes["E"] - extrusion_position
                extrusion_position = axes["E"]

        dx = x - old_x
        dy = y - old_y
        length_xy = math.hypot(dx, dy)
        if active and deposited_filament > 0.0 and length_xy > 0.0:
            angle = math.atan2(dy, dx) % math.pi
            buffered.append(
                {
                    "line_number": line_number,
                    "layer_z_mm": (
                        float(layer_z) if layer_z is not None else float(z)
                    ),
                    "feature": feature,
                    "line_width_mm": line_width,
                    "x0_mm": old_x,
                    "y0_mm": old_y,
                    "z0_mm": old_z,
                    "x1_mm": x,
                    "y1_mm": y,
                    "z1_mm": z,
                    "length_xy_mm": length_xy,
                    "filament_length_mm": deposited_filament,
                    "angle_rad_mod_pi": angle,
                }
            )

    if active:
        raise ValueError("unterminated Orca object section")
    return {name: rows for name, rows in sorted(by_object.items())}


def _segment_volume_cm3(segment: Mapping[str, object]) -> float:
    area_mm2 = math.pi * (FILAMENT_DIAMETER_MM / 2.0) ** 2
    return float(segment["filament_length_mm"]) * area_mm2 / 1000.0


def _orientation_summary(
    segments: Sequence[Mapping[str, object]],
) -> Dict[str, float]:
    total = sum(_segment_volume_cm3(segment) for segment in segments)
    if total <= 0.0:
        raise ValueError("cannot calculate orientation for an empty toolpath")
    a11 = 0.0
    a22 = 0.0
    a12 = 0.0
    c4 = 0.0
    s4 = 0.0
    for segment in segments:
        weight = _segment_volume_cm3(segment) / total
        angle = float(segment["angle_rad_mod_pi"])
        cosine = math.cos(angle)
        sine = math.sin(angle)
        a11 += weight * cosine * cosine
        a22 += weight * sine * sine
        a12 += weight * sine * cosine
        c4 += weight * math.cos(4.0 * angle)
        s4 += weight * math.sin(4.0 * angle)
    trace = a11 + a22
    discriminant = math.sqrt((a11 - a22) ** 2 + 4.0 * a12**2)
    principal = 0.5 * math.atan2(2.0 * a12, a11 - a22)
    return {
        "A11": a11,
        "A22": a22,
        "A12": a12,
        "trace": trace,
        "major_eigenvalue": 0.5 * (trace + discriminant),
        "minor_eigenvalue": 0.5 * (trace - discriminant),
        "principal_angle_degrees_mod_180": math.degrees(principal) % 180.0,
        "fourth_harmonic_cosine": c4,
        "fourth_harmonic_sine": s4,
        "fourth_harmonic_magnitude": math.hypot(c4, s4),
    }


def _summarize_object(
    name: str,
    segments: Sequence[Mapping[str, object]],
    custom_pattern: str = "graded_hourglass",
) -> Dict[str, object]:
    volume = sum(_segment_volume_cm3(segment) for segment in segments)
    filament = sum(float(segment["filament_length_mm"]) for segment in segments)
    path_length = sum(float(segment["length_xy_mm"]) for segment in segments)
    layers = sorted({float(segment["layer_z_mm"]) for segment in segments})
    widths = [
        (float(segment["line_width_mm"]), _segment_volume_cm3(segment))
        for segment in segments
        if segment["line_width_mm"] is not None
    ]
    weighted_width = (
        sum(width * weight for width, weight in widths)
        / sum(weight for _, weight in widths)
        if widths
        else None
    )
    return {
        "object_name": name,
        "pattern": _pattern_for_object(name, custom_pattern),
        "segment_count": len(segments),
        "layer_count": len(layers),
        "minimum_layer_z_mm": min(layers),
        "maximum_layer_z_mm": max(layers),
        "deposited_filament_length_mm": filament,
        "deposited_volume_cm3": volume,
        "predicted_deposited_mass_g": volume * FILAMENT_DENSITY_G_CM3,
        "toolpath_xy_length_mm": path_length,
        "volume_weighted_line_width_mm": weighted_width,
        **_orientation_summary(segments),
    }


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _population_cv(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def _feature_rows(
    by_object: Mapping[str, Sequence[Mapping[str, object]]],
    custom_pattern: str = "graded_hourglass",
) -> List[Dict[str, object]]:
    accum: MutableMapping[tuple[str, str], List[float]] = defaultdict(list)
    object_names_by_pattern: MutableMapping[str, set[str]] = defaultdict(set)
    for name, segments in by_object.items():
        pattern = _pattern_for_object(name, custom_pattern)
        object_names_by_pattern[pattern].add(name)
        local: MutableMapping[str, float] = defaultdict(float)
        for segment in segments:
            local[str(segment["feature"])] += _segment_volume_cm3(segment)
        for feature, volume in local.items():
            accum[(pattern, feature)].append(volume)
    rows: List[Dict[str, object]] = []
    for (pattern, feature), values in sorted(accum.items()):
        count = len(object_names_by_pattern[pattern])
        padded = values + [0.0] * (count - len(values))
        mean_volume = sum(padded) / count
        total_pattern = sum(
            sum(_segment_volume_cm3(segment) for segment in segments)
            for name, segments in by_object.items()
            if _pattern_for_object(name, custom_pattern) == pattern
        ) / count
        rows.append(
            {
                "pattern": pattern,
                "feature": feature,
                "mean_volume_per_object_cm3": mean_volume,
                "volume_fraction": mean_volume / total_pattern,
            }
        )
    return rows


def _layer_rows(
    by_object: Mapping[str, Sequence[Mapping[str, object]]],
    custom_pattern: str = "graded_hourglass",
) -> List[Dict[str, object]]:
    local: MutableMapping[tuple[str, float, str], float] = defaultdict(float)
    object_names_by_pattern: MutableMapping[str, set[str]] = defaultdict(set)
    for name, segments in by_object.items():
        pattern = _pattern_for_object(name, custom_pattern)
        object_names_by_pattern[pattern].add(name)
        for segment in segments:
            key = (name, float(segment["layer_z_mm"]), str(segment["feature"]))
            local[key] += _segment_volume_cm3(segment)
    layers = sorted({layer for _, layer, _ in local})
    features = sorted({feature for _, _, feature in local})
    rows: List[Dict[str, object]] = []
    for pattern, names in sorted(object_names_by_pattern.items()):
        for layer in layers:
            feature_means = {}
            for feature in features:
                feature_means[feature] = sum(
                    local.get((name, layer, feature), 0.0) for name in names
                ) / len(names)
            volume = sum(feature_means.values())
            rows.append(
                {
                    "pattern": pattern,
                    "layer_z_mm": layer,
                    "mean_volume_per_object_cm3": volume,
                    **{
                        f"{feature}_volume_cm3": value
                        for feature, value in feature_means.items()
                        if value > 0.0
                    },
                }
            )
    return rows


def _material_is_in_plane_isotropic(
    profile: Mapping[str, object],
) -> tuple[bool, Dict[str, float]]:
    constants = profile["screening_engineering_constants"]
    if not isinstance(constants, Mapping):
        raise ValueError("material profile lacks engineering constants")
    e1 = float(constants["E1"])
    e2 = float(constants["E2"])
    nu12 = float(constants["nu12"])
    g12 = float(constants["G12"])
    derived_g12 = e1 / (2.0 * (1.0 + nu12))
    gate = (
        math.isclose(e1, e2, rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(g12, derived_g12, rel_tol=1e-12, abs_tol=1e-12)
    )
    return gate, {
        "E1_MPa": e1,
        "E2_MPa": e2,
        "nu12": nu12,
        "G12_MPa": g12,
        "isotropic_derived_G12_MPa": derived_g12,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_toolpath_audit(
    output_dir: Path = DEFAULT_OUTPUT,
    gcode_path: Path = DEFAULT_GCODE,
    manufacturing_manifest_path: Path = DEFAULT_MANUFACTURING_MANIFEST,
    print_mesh_manifest_path: Path = DEFAULT_PRINT_MESH_MANIFEST,
    material_profile_path: Path = DEFAULT_MATERIAL_PROFILE,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    gcode_path = Path(gcode_path)
    manufacturing_manifest_path = Path(manufacturing_manifest_path)
    print_mesh_manifest_path = Path(print_mesh_manifest_path)
    material_profile_path = Path(material_profile_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    manufacturing = _json(manufacturing_manifest_path)
    print_mesh = _json(print_mesh_manifest_path)
    material = _json(material_profile_path)
    if manufacturing.get("gcode_sha256") != _sha256(gcode_path):
        raise ValueError("G-code checksum does not match manufacturing manifest")
    if manufacturing.get("all_manufacturing_gates_passed") is not True:
        raise ValueError("manufacturing manifest is not admitted")
    if print_mesh.get("all_print_mesh_gates_passed") is not True:
        raise ValueError("print-mesh manifest is not admitted")
    custom_pattern = str(
        manufacturing.get("custom_pattern", "graded_hourglass")
    )
    expected_patterns = {"triangular", custom_pattern}
    mesh_patterns = {
        str(row["pattern"]) for row in print_mesh.get("meshes", [])
    }
    if mesh_patterns != expected_patterns:
        raise ValueError(
            "print-mesh patterns do not match the manufacturing manifest"
        )

    by_object = parse_deposited_segments(gcode_path)
    if len(by_object) != 6:
        raise ValueError(f"expected six sliced objects, found {len(by_object)}")
    object_rows = [
        _summarize_object(name, segments, custom_pattern)
        for name, segments in by_object.items()
    ]
    feature_rows = _feature_rows(by_object, custom_pattern)
    layer_rows = _layer_rows(by_object, custom_pattern)

    cad_volume_by_pattern = {
        str(row["pattern"]): float(row["volume_mm3"]) / 1000.0
        for row in print_mesh["meshes"]
    }
    pattern_rows: List[Dict[str, object]] = []
    for pattern in ("triangular", custom_pattern):
        objects = [row for row in object_rows if row["pattern"] == pattern]
        if len(objects) != 3:
            raise ValueError(f"{pattern} does not have three sliced replicates")
        mean_volume = _mean(objects, "deposited_volume_cm3")
        cad_volume = cad_volume_by_pattern[pattern]
        fraction = mean_volume / cad_volume
        pattern_layer_volumes = [
            float(row["mean_volume_per_object_cm3"])
            for row in layer_rows
            if row["pattern"] == pattern
        ]
        pattern_rows.append(
            {
                "pattern": pattern,
                "replicate_count": len(objects),
                "replicate_volume_range_cm3": (
                    max(float(row["deposited_volume_cm3"]) for row in objects)
                    - min(
                        float(row["deposited_volume_cm3"]) for row in objects
                    )
                ),
                "segment_count_per_object": _mean(objects, "segment_count"),
                "layer_count": int(round(_mean(objects, "layer_count"))),
                "mean_deposited_volume_cm3": mean_volume,
                "mean_predicted_deposited_mass_g": (
                    mean_volume * FILAMENT_DENSITY_G_CM3
                ),
                "continuum_cad_volume_cm3": cad_volume,
                "deposited_to_continuum_volume_fraction": fraction,
                "effective_underfill_fraction": 1.0 - fraction,
                "mean_toolpath_xy_length_mm": _mean(
                    objects, "toolpath_xy_length_mm"
                ),
                "volume_weighted_line_width_mm": _mean(
                    objects, "volume_weighted_line_width_mm"
                ),
                "layer_volume_population_cv": _population_cv(
                    pattern_layer_volumes
                ),
                **{
                    key: _mean(objects, key)
                    for key in (
                        "A11",
                        "A22",
                        "A12",
                        "major_eigenvalue",
                        "minor_eigenvalue",
                        "principal_angle_degrees_mod_180",
                        "fourth_harmonic_cosine",
                        "fourth_harmonic_sine",
                        "fourth_harmonic_magnitude",
                    )
                },
            }
        )

    control = next(row for row in pattern_rows if row["pattern"] == "triangular")
    custom = next(
        row for row in pattern_rows if row["pattern"] == custom_pattern
    )
    toolpath_ratio = (
        float(custom["mean_deposited_volume_cm3"])
        / float(control["mean_deposited_volume_cm3"])
    )
    fraction_difference = abs(
        float(custom["deposited_to_continuum_volume_fraction"])
        - float(control["deposited_to_continuum_volume_fraction"])
    )
    orientation_distance = math.sqrt(
        (
            float(custom["A11"]) - float(control["A11"])
        )
        ** 2
        + (
            float(custom["A22"]) - float(control["A22"])
        )
        ** 2
        + 2.0
        * (
            float(custom["A12"]) - float(control["A12"])
        )
        ** 2
    )
    in_plane_isotropic, constants = _material_is_in_plane_isotropic(material)
    volume_matches_manufacturing = all(
        abs(
            float(row["deposited_volume_cm3"])
            - float(
                manufacturing["object_toolpaths"][row["object_name"]][
                    "deposited_volume_cm3"
                ]
            )
        )
        <= VOLUME_TOLERANCE_CM3
        for row in object_rows
    )
    replicate_gate = all(
        float(row["replicate_volume_range_cm3"]) <= VOLUME_TOLERANCE_CM3
        for row in pattern_rows
    )
    layer_gate = all(
        int(row["layer_count"]) == EXPECTED_LAYER_COUNT for row in pattern_rows
    )
    underfill_gate = all(
        abs(float(row["effective_underfill_fraction"]))
        <= MAXIMUM_ABSOLUTE_UNDERFILL_FRACTION
        for row in pattern_rows
    )
    registered_mass_limit = float(
        manufacturing.get(
            "maximum_allowed_toolpath_mass_ratio_error",
            MAXIMUM_TOOLPATH_MASS_RATIO_ERROR,
        )
    )
    mass_balance_gate = (
        abs(toolpath_ratio - 1.0) <= registered_mass_limit
    )
    fraction_difference_gate = (
        fraction_difference <= MAXIMUM_DEPOSITION_FRACTION_DIFFERENCE
    )
    relative_gate = all(
        (
            volume_matches_manufacturing,
            replicate_gate,
            layer_gate,
            underfill_gate,
            mass_balance_gate,
            fraction_difference_gate,
            in_plane_isotropic,
        )
    )
    status = (
        "relative_ranking_admitted_with_restrictions"
        if relative_gate
        else "toolpath_resolved_resimulation_required_before_relative_ranking"
    )

    _write_csv(output_dir / "object_summary.csv", object_rows)
    _write_csv(output_dir / "pattern_summary.csv", pattern_rows)
    _write_csv(output_dir / "feature_summary.csv", feature_rows)
    _write_csv(output_dir / "layer_summary.csv", layer_rows)
    manifest: Dict[str, object] = {
        "audit_id": f"{custom_pattern}_orca_toolpath_morphology_v1",
        "custom_pattern": custom_pattern,
        "material": "Bambu PLA Basic",
        "status": status,
        "interpretation_scope": (
            "relative linear-elastic stiffness ranking for the frozen "
            f"{custom_pattern} versus triangular bending comparison"
        ),
        "source_gcode": str(gcode_path.resolve()),
        "source_gcode_sha256": _sha256(gcode_path),
        "manufacturing_manifest": str(
            manufacturing_manifest_path.resolve()
        ),
        "manufacturing_manifest_sha256": _sha256(
            manufacturing_manifest_path
        ),
        "print_mesh_manifest": str(print_mesh_manifest_path.resolve()),
        "print_mesh_manifest_sha256": _sha256(print_mesh_manifest_path),
        "material_profile": str(material_profile_path.resolve()),
        "material_profile_sha256": _sha256(material_profile_path),
        "object_count": len(object_rows),
        "pattern_summaries": {
            str(row["pattern"]): {
                key: value for key, value in row.items() if key != "pattern"
            }
            for row in pattern_rows
        },
        "comparison": {
            "custom_to_triangular_toolpath_volume_ratio": toolpath_ratio,
            "graded_to_triangular_toolpath_volume_ratio": (
                toolpath_ratio
                if custom_pattern == "graded_hourglass"
                else None
            ),
            "toolpath_mass_ratio_error": toolpath_ratio - 1.0,
            "deposition_fraction_absolute_difference": fraction_difference,
            "orientation_tensor_frobenius_distance": orientation_distance,
        },
        "material_orientation_assumption": {
            "in_plane_isotropic": in_plane_isotropic,
            **constants,
            "consequence": (
                "The current linear material law is invariant to bead "
                "rotation within the XY plane. The measured orientation "
                "difference is retained for future orthotropic or failure "
                "models but does not alter the present linear ranking."
            ),
        },
        "gates": {
            "parsed_volumes_match_manufacturing_manifest": (
                volume_matches_manufacturing
            ),
            "replicate_toolpath_volumes_identical": replicate_gate,
            "expected_20_layers_per_object": layer_gate,
            "absolute_underfill_at_most_10_percent": underfill_gate,
            "custom_control_toolpath_mass_imbalance_at_most_1_percent": (
                abs(toolpath_ratio - 1.0)
                <= MAXIMUM_TOOLPATH_MASS_RATIO_ERROR
            ),
            "custom_control_toolpath_mass_imbalance_within_registered_limit": (
                mass_balance_gate
            ),
            "registered_mass_imbalance_limit_fraction": registered_mass_limit,
            "custom_control_deposition_fraction_difference_at_most_2pp": (
                fraction_difference_gate
            ),
            "current_material_law_is_in_plane_isotropic": in_plane_isotropic,
            "relative_linear_stiffness_ranking_admitted": relative_gate,
            "absolute_stiffness_validated": False,
            "local_bead_stress_validated": False,
            "failure_or_fracture_validated": False,
        },
        "restrictions": [
            "The CalculiX solids represent fully fused continuum geometry; "
            "the slicer deposits less polymer than the CAD envelope.",
            "Physical coupon mass normalization remains mandatory.",
            "Absolute stiffness requires measured coupons or a calibrated "
            "bead/void representation.",
            "Local bead stress, interlayer failure, fracture, and "
            "post-collapse predictions require toolpath-resolved modelling "
            "and additional constitutive data.",
        ],
        "toolpath_resolved_reanalysis_required_for": [
            "absolute stiffness precision",
            "bead-scale stress concentrations",
            "interlayer initiation and delamination",
            "strength, fracture, fatigue, and post-collapse response",
        ],
        "outputs": {
            name: {
                "path": str((output_dir / name).resolve()),
                "sha256": _sha256(output_dir / name),
            }
            for name in (
                "object_summary.csv",
                "pattern_summary.csv",
                "feature_summary.csv",
                "layer_summary.csv",
            )
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    control_underfill = float(control["effective_underfill_fraction"])
    custom_underfill = float(custom["effective_underfill_fraction"])
    report = f"""# Frozen Orca toolpath morphology audit

The sliced bead paths support using the existing continuum result as a
**restricted relative stiffness screen**, not as an absolute or failure
prediction.

| Quantity | Triangular | {custom_pattern} |
|---|---:|---:|
| Mean deposited volume (cm³) | {float(control['mean_deposited_volume_cm3']):.6f} | {float(custom['mean_deposited_volume_cm3']):.6f} |
| Continuum CAD volume (cm³) | {float(control['continuum_cad_volume_cm3']):.6f} | {float(custom['continuum_cad_volume_cm3']):.6f} |
| Effective underfill | {control_underfill:.2%} | {custom_underfill:.2%} |
| Deposited mass (g) | {float(control['mean_predicted_deposited_mass_g']):.4f} | {float(custom['mean_predicted_deposited_mass_g']):.4f} |
| XY orientation A11 / A22 / A12 | {float(control['A11']):.3f} / {float(control['A22']):.3f} / {float(control['A12']):.3f} | {float(custom['A11']):.3f} / {float(custom['A22']):.3f} / {float(custom['A12']):.3f} |

The custom/control deposited-volume ratio is **{toolpath_ratio:.5f}**
({toolpath_ratio - 1.0:+.2%}), and their deposited-to-CAD fractions differ by
**{fraction_difference:.2%}**. All three replicas of each topology have
identical parsed volume and all objects contain {EXPECTED_LAYER_COUNT} layers.
The registered pre-test mass-imbalance limit is
**{registered_mass_limit:.2%}**; measured-mass normalization remains required.

The Bambu screening card is isotropic within the print plane
(E1 = E2 and G12 = E/[2(1+nu12)]), so XY bead-angle differences cannot change
the current linear-elastic prediction. They would matter to a future
directional strength or bead-resolved model.

Admission decision: **{status}**.

This gate does not validate absolute stiffness, local bead stresses,
interlayer failure, fracture, fatigue, or post-collapse response. Those remain
physical-test or toolpath-resolved-model questions.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    if not relative_gate:
        raise RuntimeError("toolpath morphology gate rejected relative ranking")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit frozen Orca bead paths against continuum FEA solids"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gcode", default=str(DEFAULT_GCODE))
    parser.add_argument(
        "--manufacturing-manifest",
        default=str(DEFAULT_MANUFACTURING_MANIFEST),
    )
    parser.add_argument(
        "--print-mesh-manifest", default=str(DEFAULT_PRINT_MESH_MANIFEST)
    )
    parser.add_argument(
        "--material-profile", default=str(DEFAULT_MATERIAL_PROFILE)
    )
    args = parser.parse_args()
    manifest = write_toolpath_audit(
        Path(args.output),
        Path(args.gcode),
        Path(args.manufacturing_manifest),
        Path(args.print_mesh_manifest),
        Path(args.material_profile),
    )
    print(
        f"toolpath morphology: status={manifest['status']}; "
        f"objects={manifest['object_count']}; "
        f"output: {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
