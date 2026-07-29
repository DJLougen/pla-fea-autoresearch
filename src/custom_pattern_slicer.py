from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Mapping, Sequence
from xml.etree import ElementTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ROOT = (
    PROJECT_ROOT / "results/custom_infill/graded_hourglass"
)
DEFAULT_STL_DIR = CUSTOM_ROOT / "physical_validation/printable_stl"
DEFAULT_OUTPUT = CUSTOM_ROOT / "physical_validation/manufacturing"
PROCESS_OVERRIDE = (
    PROJECT_ROOT
    / "configs/printing/bambu_pla_pattern_validation_process.json"
)
FILAMENT_OVERRIDE = (
    PROJECT_ROOT
    / "configs/printing/bambu_pla_pattern_validation_filament.json"
)
ORCA_CANDIDATES = (
    Path("/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer"),
    Path.home()
    / "Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer",
)
ORCA_PROFILE_ROOT = (
    Path.home()
    / "Library/Application Support/OrcaSlicer/system/BBL"
)
OFFICIAL_MACHINE = (
    ORCA_PROFILE_ROOT / "machine/Bambu Lab X1 Carbon 0.4 nozzle.json"
)
OFFICIAL_PROCESS = (
    ORCA_PROFILE_ROOT / "process/0.20mm Standard @BBL X1C.json"
)
OFFICIAL_FILAMENT = (
    ORCA_PROFILE_ROOT / "filament/Bambu PLA Basic @BBL X1C.json"
)
MAXIMUM_TOOLPATH_MASS_RATIO_ERROR = 0.01
FILAMENT_DIAMETER_MM = 1.75
FILAMENT_DENSITY_G_CM3 = 1.24


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_orca_slicer() -> Path:
    for path in ORCA_CANDIDATES:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "OrcaSlicer was not found in the supported application locations"
    )


def _run(command: Sequence[str], log_path: Path) -> None:
    completed = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600.0,
        check=False,
    )
    log_path.write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"OrcaSlicer exited {completed.returncode}; see {log_path}"
        )


def _config_from_gcode(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    inside = False
    for raw in Path(path).open(encoding="utf-8", errors="ignore"):
        line = raw.rstrip("\n")
        if line == "; CONFIG_BLOCK_START":
            inside = True
            continue
        if line == "; CONFIG_BLOCK_END":
            break
        if inside and line.startswith("; ") and " = " in line:
            key, value = line[2:].split(" = ", 1)
            values[key] = value
    return values


def _header_from_gcode(path: Path) -> Dict[str, str]:
    values = {}
    for raw in Path(path).open(encoding="utf-8", errors="ignore"):
        line = raw.strip()
        if line == "; HEADER_BLOCK_END":
            break
        match = re.match(r"^; ([^:]+): (.+)$", line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def parse_object_extrusion(path: Path) -> Dict[str, Dict[str, float]]:
    """Sum deposited filament moves inside Orca object-label sections."""
    active = False
    segment_filament_mm = 0.0
    filament_by_object: Dict[str, float] = {}
    start = "; start printing object, unique label id:"
    stop = re.compile(r"^; stop printing object (.+?) id:")
    extrusion = re.compile(
        r"(?:^| )E(-?(?:\d+(?:\.\d*)?|\.\d+))"
    )
    for raw in Path(path).open(encoding="utf-8", errors="ignore"):
        line = raw.rstrip("\n")
        if line.startswith(start):
            active = True
            segment_filament_mm = 0.0
            continue
        match = stop.match(line)
        if match and active:
            name = match.group(1)
            filament_by_object[name] = (
                filament_by_object.get(name, 0.0)
                + segment_filament_mm
            )
            active = False
            continue
        if (
            active
            and line.startswith(("G0 ", "G1 "))
            and (" X" in line or " Y" in line)
        ):
            match = extrusion.search(line)
            if match and float(match.group(1)) > 0.0:
                segment_filament_mm += float(match.group(1))
    area_mm2 = math.pi * (FILAMENT_DIAMETER_MM / 2.0) ** 2
    return {
        name: {
            "deposited_filament_length_mm": length,
            "deposited_volume_cm3": length * area_mm2 / 1000.0,
            "predicted_deposited_mass_g": (
                length
                * area_mm2
                / 1000.0
                * FILAMENT_DENSITY_G_CM3
            ),
        }
        for name, length in sorted(filament_by_object.items())
    }


def _three_mf_inspection(
    path: Path, custom_pattern: str = "graded_hourglass"
) -> Dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(
            archive.read("3D/3dmodel.model")
        )
        model_settings = ElementTree.fromstring(
            archive.read("Metadata/model_settings.config")
        )
        project = json.loads(
            archive.read("Metadata/project_settings.config")
        )
    namespace = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    items = root.findall(".//m:build/m:item", namespace)
    objects = model_settings.findall("./object")
    names = [
        node.attrib["value"]
        for obj in objects
        for node in obj.findall("./metadata")
        if node.attrib.get("key") == "name"
    ]
    linear_identity = True
    positions = []
    for item in items:
        transform = [
            float(value)
            for value in item.attrib.get(
                "transform", "1 0 0 0 1 0 0 0 1 0 0 0"
            ).split()
        ]
        if len(transform) != 12:
            linear_identity = False
            continue
        expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        linear_identity = linear_identity and all(
            math.isclose(transform[index], expected[index], abs_tol=1e-12)
            for index in range(9)
        )
        positions.append(transform[9:12])
    custom_object_count = sum(
        name.startswith(f"{custom_pattern}.stl") for name in names
    )
    return {
        "build_item_count": len(items),
        "model_object_count": len(objects),
        "triangular_object_count": sum(
            name.startswith("triangular.stl") for name in names
        ),
        "graded_hourglass_object_count": sum(
            name.startswith("graded_hourglass.stl") for name in names
        ),
        "custom_pattern": custom_pattern,
        "custom_object_count": custom_object_count,
        "all_build_transforms_unscaled_and_unrotated": linear_identity,
        "build_positions_mm": positions,
        "embedded_printer_settings_id": project.get(
            "printer_settings_id"
        ),
        "embedded_print_settings_id": project.get(
            "print_settings_id"
        ),
        "embedded_filament_settings_id": project.get(
            "filament_settings_id"
        ),
    }


def _float_config(
    config: Mapping[str, str], name: str
) -> float:
    return float(str(config[name]).strip().strip('"'))


def _expected_config_gate(config: Mapping[str, str]) -> bool:
    exact = {
        "curr_bed_type": "Textured PEI Plate",
        "brim_type": "no_brim",
        "enable_support": "0",
        "sparse_infill_density": "100%",
        "sparse_infill_pattern": "rectilinear",
        "print_settings_id": (
            "0.20mm Bambu PLA Pattern Validation @BBL X1C"
        ),
        "printer_settings_id": "Bambu Lab X1 Carbon 0.4 nozzle",
    }
    if any(config.get(key) != value for key, value in exact.items()):
        return False
    filament_id = config.get("filament_settings_id", "")
    if "Bambu PLA Basic Pattern Validation @BBL X1C" not in filament_id:
        return False
    numeric = {
        "layer_height": 0.2,
        "wall_loops": 3.0,
        "nozzle_temperature": 220.0,
        "textured_plate_temp": 55.0,
        "filament_density": 1.24,
        "outer_wall_speed": 50.0,
        "inner_wall_speed": 50.0,
        "sparse_infill_speed": 50.0,
        "internal_solid_infill_speed": 50.0,
        "top_surface_speed": 50.0,
        "gap_infill_speed": 50.0,
    }
    try:
        return all(
            math.isclose(
                _float_config(config, key), value, abs_tol=1e-9
            )
            for key, value in numeric.items()
        )
    except (KeyError, ValueError):
        return False


def write_manufacturing_package(
    output_dir: Path = DEFAULT_OUTPUT,
    stl_dir: Path = DEFAULT_STL_DIR,
    orca_path: Path | None = None,
    custom_pattern: str | None = None,
    maximum_toolpath_mass_ratio_error: float = (
        MAXIMUM_TOOLPATH_MASS_RATIO_ERROR
    ),
) -> Dict[str, object]:
    # OrcaSlicer's CLI project export can fail with exit 243 when its output
    # or model paths are relative, even though they resolve from the current
    # working directory. Normalize them before constructing either command.
    output_dir = Path(output_dir).resolve()
    stl_dir = Path(stl_dir).resolve()
    if not 0.0 < maximum_toolpath_mass_ratio_error <= 0.03:
        raise ValueError(
            "maximum toolpath mass-ratio error must lie in (0, 0.03]"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "profile_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    orca = (
        Path(orca_path).resolve()
        if orca_path is not None
        else find_orca_slicer().resolve()
    )
    required = {
        "official_machine": OFFICIAL_MACHINE,
        "official_process": OFFICIAL_PROCESS,
        "official_filament": OFFICIAL_FILAMENT,
        "experiment_process_override": PROCESS_OVERRIDE,
        "experiment_filament_override": FILAMENT_OVERRIDE,
    }
    snapshots = {}
    for name, source in required.items():
        if not source.exists():
            raise FileNotFoundError(f"required slicer profile missing: {source}")
        destination = snapshot_dir / f"{name}.json"
        destination.write_bytes(source.read_bytes())
        snapshots[name] = {
            "source": str(source.resolve()),
            "snapshot": str(destination.resolve()),
            "sha256": _sha256(destination),
        }
    triangular = stl_dir / "triangular.stl"
    if custom_pattern is None:
        custom_candidates = sorted(
            path.stem
            for path in stl_dir.glob("*.stl")
            if path.stem != "triangular"
        )
        if len(custom_candidates) != 1:
            raise ValueError(
                "STL directory must contain triangular.stl and exactly one "
                "custom-pattern STL"
            )
        custom_pattern = custom_candidates[0]
    custom = stl_dir / f"{custom_pattern}.stl"
    for path in (triangular, custom):
        if not path.exists():
            raise FileNotFoundError(f"printable STL missing: {path}")
    plate = output_dir / "bambu-pla-pattern-validation.3mf"
    # OrcaSlicer refuses to overwrite an existing project or G-code file.
    # Both paths are generated artifacts, so remove only those exact targets
    # before refreshing an existing validation package.
    if plate.exists():
        plate.unlink()
    settings = (
        f"{OFFICIAL_MACHINE};{PROCESS_OVERRIDE}"
    )
    _run(
        (
            str(orca),
            "--debug",
            "2",
            "--load-settings",
            settings,
            "--load-filaments",
            str(FILAMENT_OVERRIDE),
            "--clone-objects",
            "3,3",
            "--arrange",
            "1",
            "--ensure-on-bed",
            "--export-3mf",
            str(plate),
            str(triangular),
            str(custom),
        ),
        output_dir / "export.log",
    )
    sliced_dir = output_dir / "sliced"
    sliced_dir.mkdir(parents=True, exist_ok=True)
    gcode = sliced_dir / "plate_1.gcode"
    if gcode.exists():
        gcode.unlink()
    _run(
        (
            str(orca),
            "--debug",
            "2",
            "--slice",
            "0",
            "--outputdir",
            str(sliced_dir),
            str(plate),
        ),
        output_dir / "slice.log",
    )
    if not gcode.exists():
        raise RuntimeError("OrcaSlicer did not produce plate_1.gcode")
    config = _config_from_gcode(gcode)
    header = _header_from_gcode(gcode)
    three_mf = _three_mf_inspection(plate, custom_pattern)
    objects = parse_object_extrusion(gcode)
    triangular_rows = [
        row for name, row in objects.items()
        if name.startswith("triangular.stl")
    ]
    custom_rows = [
        row for name, row in objects.items()
        if name.startswith(f"{custom_pattern}.stl")
    ]
    replicate_equal = (
        len(triangular_rows) == 3
        and len(custom_rows) == 3
        and max(
            row["deposited_volume_cm3"] for row in triangular_rows
        )
        - min(
            row["deposited_volume_cm3"] for row in triangular_rows
        )
        <= 1e-9
        and max(
            row["deposited_volume_cm3"] for row in custom_rows
        )
        - min(
            row["deposited_volume_cm3"] for row in custom_rows
        )
        <= 1e-9
    )
    triangular_mean = float(
        np_mean(
            [row["deposited_volume_cm3"] for row in triangular_rows]
        )
    )
    custom_mean = float(
        np_mean(
            [row["deposited_volume_cm3"] for row in custom_rows]
        )
    )
    toolpath_ratio = custom_mean / triangular_mean
    ratio_error = toolpath_ratio - 1.0
    config_gate = _expected_config_gate(config)
    plate_gate = (
        three_mf["build_item_count"] == 6
        and three_mf["model_object_count"] == 6
        and three_mf["triangular_object_count"] == 3
        and three_mf["custom_object_count"] == 3
        and three_mf[
            "all_build_transforms_unscaled_and_unrotated"
        ]
        is True
    )
    header_gate = (
        header.get("total layer number") == "20"
        and header.get("max_z_height") == "4.00"
        and len(header.get("model label id", "").split(",")) == 6
    )
    toolpath_gate = (
        replicate_equal
        and abs(ratio_error) <= maximum_toolpath_mass_ratio_error
    )
    gate = config_gate and plate_gate and header_gate and toolpath_gate
    version_output = subprocess.run(
        [str(orca), "--help"],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    ).stdout.splitlines()
    version = version_output[0].rstrip(":") if version_output else "unknown"
    manifest = {
        "manufacturing_id": "bambu_pla_basic_pattern_plate_v1",
        "slicer": version,
        "slicer_executable": str(orca.resolve()),
        "printer": "Bambu Lab X1 Carbon 0.4 nozzle",
        "material": "Bambu PLA Basic",
        "custom_pattern": custom_pattern,
        "plate_3mf": str(plate.resolve()),
        "plate_3mf_sha256": _sha256(plate),
        "gcode": str(gcode.resolve()),
        "gcode_sha256": _sha256(gcode),
        "source_stls": {
            "triangular": {
                "path": str(triangular.resolve()),
                "sha256": _sha256(triangular),
            },
            custom_pattern: {
                "path": str(custom.resolve()),
                "sha256": _sha256(custom),
            },
        },
        "profile_snapshots": snapshots,
        "three_mf_inspection": three_mf,
        "gcode_header": header,
        "registered_config": {
            key: config.get(key)
            for key in (
                "layer_height",
                "wall_loops",
                "sparse_infill_density",
                "sparse_infill_pattern",
                "nozzle_temperature",
                "textured_plate_temp",
                "curr_bed_type",
                "filament_density",
                "filament_flow_ratio",
                "outer_wall_speed",
                "inner_wall_speed",
                "sparse_infill_speed",
                "internal_solid_infill_speed",
                "top_surface_speed",
                "gap_infill_speed",
                "brim_type",
                "enable_support",
                "print_settings_id",
                "printer_settings_id",
                "filament_settings_id",
            )
        },
        "object_toolpaths": objects,
        "replicates_have_identical_toolpath_volume": replicate_equal,
        "mean_deposited_volume_cm3": {
            "triangular": triangular_mean,
            custom_pattern: custom_mean,
        },
        "predicted_deposited_mass_g": {
            "triangular": (
                triangular_mean * FILAMENT_DENSITY_G_CM3
            ),
            custom_pattern: (
                custom_mean * FILAMENT_DENSITY_G_CM3
            ),
        },
        "custom_to_triangular_toolpath_mass_ratio": toolpath_ratio,
        "graded_to_triangular_toolpath_mass_ratio": (
            toolpath_ratio
            if custom_pattern == "graded_hourglass"
            else None
        ),
        "toolpath_mass_ratio_error": ratio_error,
        "maximum_allowed_toolpath_mass_ratio_error": (
            maximum_toolpath_mass_ratio_error
        ),
        "config_gate_passed": config_gate,
        "plate_gate_passed": plate_gate,
        "gcode_header_gate_passed": header_gate,
        "toolpath_mass_gate_passed": toolpath_gate,
        "all_manufacturing_gates_passed": gate,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Bambu PLA Basic pattern-validation build plate

The plate contains three triangular controls and three {custom_pattern}
specimens. OrcaSlicer embedded the registered X1 Carbon process and filament
settings into the 3MF and produced the checksum-frozen G-code.

| Check | Result |
|---|---|
| Slicer | {version} |
| Printer | Bambu Lab X1 Carbon, 0.4 mm nozzle |
| Objects | 6 (3 control + 3 custom) |
| Layers | {header.get('total layer number')} at 0.20 mm |
| Nozzle / textured plate | 220 / 55 °C |
| Core / walls | 100% rectilinear / 3 |
| Deposition speed ceiling | 50 mm/s |
| Supports / brim | disabled / disabled |
| Control deposited mass | {triangular_mean * FILAMENT_DENSITY_G_CM3:.4f} g each |
| Custom deposited mass | {custom_mean * FILAMENT_DENSITY_G_CM3:.4f} g each |
| Custom/control toolpath mass ratio | {toolpath_ratio:.5f} |
| Allowed toolpath mass-ratio error | {maximum_toolpath_mass_ratio_error:.2%} |

All manufacturing gates: **{'passed' if gate else 'failed'}**.

Open `bambu-pla-pattern-validation.3mf` in OrcaSlicer or Bambu Studio and
confirm the intended physical spool before printing. Do not rescale, rearrange,
or change the embedded process. Weigh each cooled coupon and use the measured
mass in the frozen capture template.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    if not gate:
        raise RuntimeError("one or more manufacturing gates failed")
    return manifest


def np_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and inspect the six-coupon OrcaSlicer plate"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stl-dir", default=str(DEFAULT_STL_DIR))
    parser.add_argument("--orca")
    parser.add_argument(
        "--maximum-toolpath-mass-ratio-error",
        type=float,
        default=MAXIMUM_TOOLPATH_MASS_RATIO_ERROR,
    )
    args = parser.parse_args()
    manifest = write_manufacturing_package(
        Path(args.output),
        Path(args.stl_dir),
        None if args.orca is None else Path(args.orca),
        maximum_toolpath_mass_ratio_error=(
            args.maximum_toolpath_mass_ratio_error
        ),
    )
    print(
        f"manufacturing plate: gate="
        f"{manifest['all_manufacturing_gates_passed']}; "
        f"objects={manifest['three_mf_inspection']['build_item_count']}; "
        f"output: {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
