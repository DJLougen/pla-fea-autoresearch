from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


ALIASES = {
    "time": "time_s",
    "time_s": "time_s",
    "force": "force_N",
    "force_n": "force_N",
    "displacement": "displacement_mm",
    "displacement_mm": "displacement_mm",
    "strain": "engineering_strain",
    "engineering_strain": "engineering_strain",
    "stress": "engineering_stress_MPa",
    "engineering_stress": "engineering_stress_MPa",
    "engineering_stress_mpa": "engineering_stress_MPa",
    "transverse_strain": "transverse_strain",
    "lateral_strain": "transverse_strain",
    "width": "specimen_width_mm",
    "specimen_width": "specimen_width_mm",
    "specimen_width_mm": "specimen_width_mm",
    "thickness": "specimen_thickness_mm",
    "specimen_thickness": "specimen_thickness_mm",
    "specimen_thickness_mm": "specimen_thickness_mm",
    "gauge_length": "gauge_length_mm",
    "gauge_length_mm": "gauge_length_mm",
    "print_direction": "print_direction",
    "raster_angle": "raster_angle_deg",
    "raster_angle_deg": "raster_angle_deg",
    "layer_height": "layer_height_mm",
    "layer_height_mm": "layer_height_mm",
    "nozzle_temperature": "nozzle_temperature_C",
    "nozzle_temperature_c": "nozzle_temperature_C",
    "print_speed": "print_speed_mm_s",
    "print_speed_mm_s": "print_speed_mm_s",
    "infill_percentage": "infill_percent",
    "infill_percent": "infill_percent",
    "test_id": "test_id",
    "partition": "partition",
    "test_type": "test_type",
}
PARTITIONS = {"calibration", "validation", "heldout"}


def _canonical_name(name: str) -> str:
    key = re.sub(r"_+", "_", name.strip().lower().replace("-", "_").replace(" ", "_"))
    return ALIASES.get(key, name.strip())


def _constant(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{column} must have one non-missing value per specimen")
    return float(values[0])


def force_displacement_to_stress_strain(
    force_n: np.ndarray,
    displacement_mm: np.ndarray,
    width_mm: float,
    thickness_mm: float,
    gauge_length_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if width_mm <= 0 or thickness_mm <= 0 or gauge_length_mm <= 0:
        raise ValueError("specimen dimensions must be positive")
    # N/mm² is exactly MPa in the fixed unit system.
    stress_mpa = np.asarray(force_n, dtype=float) / (width_mm * thickness_mm)
    strain = np.asarray(displacement_mm, dtype=float) / gauge_length_mm
    return stress_mpa, strain


def _detect_unit_errors(frame: pd.DataFrame) -> None:
    checks = {
        "specimen_width_mm": (0.05, 500.0),
        "specimen_thickness_mm": (0.02, 100.0),
        "gauge_length_mm": (0.5, 2000.0),
        "force_N": (-1.0e7, 1.0e7),
        "engineering_strain": (-0.2, 2.0),
        "transverse_strain": (-1.0, 1.0),
        "engineering_stress_MPa": (-1000.0, 10000.0),
    }
    for column, (lower, upper) in checks.items():
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty and ((values < lower) | (values > upper)).any():
            raise ValueError(
                f"{column} contains implausible values; check units (expected N, mm, MPa)"
            )


def preprocess_frame(frame: pd.DataFrame, source_name: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    frame = frame.rename(columns={column: _canonical_name(column) for column in frame.columns})
    frame = frame.drop_duplicates().copy()
    test_type = (
        str(frame["test_type"].dropna().iloc[0])
        if "test_type" in frame and not frame["test_type"].dropna().empty
        else "tensile"
    )
    has_stress_strain = {
        "engineering_strain",
        "engineering_stress_MPa",
    }.issubset(frame.columns)
    has_force_displacement = {"force_N", "displacement_mm"}.issubset(frame.columns)
    if not has_stress_strain and not has_force_displacement:
        raise ValueError(
            f"{source_name}: provide stress+strain or force+displacement"
        )
    if not has_stress_strain and has_force_displacement:
        if test_type == "tensile":
            required_geometry = {
                "specimen_width_mm",
                "specimen_thickness_mm",
                "gauge_length_mm",
            }
            missing = required_geometry - set(frame)
            if missing:
                raise ValueError(
                    f"{source_name}: tensile force-displacement conversion "
                    f"requires {sorted(missing)}"
                )
            width = _constant(frame, "specimen_width_mm")
            thickness = _constant(frame, "specimen_thickness_mm")
            gauge = _constant(frame, "gauge_length_mm")
            stress, strain = force_displacement_to_stress_strain(
                pd.to_numeric(frame["force_N"]).to_numpy(),
                pd.to_numeric(frame["displacement_mm"]).to_numpy(),
                width,
                thickness,
                gauge,
            )
            frame["engineering_stress_MPa"] = stress
            frame["engineering_strain"] = strain
            has_stress_strain = True
    numeric = (
        ["engineering_strain", "engineering_stress_MPa"]
        if has_stress_strain
        else ["displacement_mm", "force_N"]
    )
    if "transverse_strain" in frame:
        numeric.append("transverse_strain")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required_numeric = (
        ["engineering_strain", "engineering_stress_MPa"]
        if has_stress_strain
        else ["displacement_mm", "force_N"]
    )
    frame = frame.dropna(subset=required_numeric)
    if len(frame) < 3:
        raise ValueError(f"{source_name}: fewer than three usable curve points")
    _detect_unit_errors(frame)
    sort_column = (
        "engineering_strain" if has_stress_strain else "displacement_mm"
    )
    frame = frame.sort_values(sort_column, kind="mergesort").reset_index(
        drop=True
    )
    if not np.all(np.diff(frame[sort_column].to_numpy()) >= 0):
        raise ValueError(f"{source_name}: {sort_column} could not be sorted")
    metadata = {
        "test_id": str(frame["test_id"].iloc[0]) if "test_id" in frame else Path(source_name).stem,
        "partition": str(frame["partition"].iloc[0]) if "partition" in frame else "calibration",
        "test_type": test_type,
        "direction": str(frame["print_direction"].iloc[0]) if "print_direction" in frame else "1",
        "n_points": int(len(frame)),
    }
    if metadata["partition"] not in PARTITIONS:
        raise ValueError(f"{source_name}: invalid partition {metadata['partition']!r}")
    return frame, metadata


def prepare_directory(
    input_dir: Union[str, Path], output_dir: Union[str, Path]
) -> pd.DataFrame:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    sources = sorted(
        path for path in input_dir.glob("*.csv") if path.name != "synthetic_truth.csv"
    )
    if not sources:
        raise ValueError(f"no CSV files found in {input_dir}")
    for source in sources:
        raw = source.read_bytes()
        frame = pd.read_csv(source)
        processed, metadata = preprocess_frame(frame, source.name)
        destination = output_dir / source.name
        processed.to_csv(destination, index=False, float_format="%.10g")
        metadata.update(
            {
                "file": destination.name,
                "source_file": str(source.resolve()),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        records.append(metadata)
    manifest = pd.DataFrame(records).sort_values(["partition", "test_id"])
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    provenance = {
        "input": str(input_dir.resolve()),
        "files": len(records),
        "deterministic_sort": ["partition", "test_id"],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and preprocess experimental curves")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = prepare_directory(args.input, args.output)
    counts = manifest.groupby("partition").size().to_dict()
    print(f"processed {len(manifest)} specimens: {counts}")


if __name__ == "__main__":
    main()
