from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .custom_pattern_fixture import (
    MAXIMUM_FIXTURE_COMPLIANCE_MM_N,
)
from .custom_pattern_validation import (
    CAPTURE_FIELDS,
    DEFAULT_PACKAGE,
    validate_submission,
)


STANDARD_GRAVITY_M_S2 = 9.80665
MINIMUM_FORCE_CAPACITY_N = 50.0
MAXIMUM_FORCE_RESOLUTION_N = 0.05
MAXIMUM_FORCE_UNCERTAINTY_N = 0.25
MAXIMUM_DISPLACEMENT_RESOLUTION_MM = 0.005
MAXIMUM_DISPLACEMENT_UNCERTAINTY_MM = 0.01
MAXIMUM_COMPLIANCE_UNCERTAINTY_MM_N = 0.0001
MINIMUM_ACQUISITION_RATE_HZ = 10.0
CURVE_FIELDS = ("time_s", "displacement_mm", "force_N")

MANUAL_SPECIFIC_FIELDS = (
    "force_measurement_method",
    "force_calibration_id",
    "force_zero_input",
    "force_resolution_N",
    "force_calibration_max_error_N",
    "force_capacity_N",
    "displacement_measurement_method",
    "displacement_calibration_id",
    "displacement_zero_mm",
    "displacement_resolution_mm",
    "displacement_calibration_max_error_mm",
    "fixture_compliance_calibration_id",
    "fixture_compliance_uncertainty_mm_per_N",
    "raw_time_s",
    "raw_force_input",
    "raw_displacement_mm",
)
MANUAL_CAPTURE_FIELDS = tuple(
    field for field in CAPTURE_FIELDS if field not in CURVE_FIELDS
) + MANUAL_SPECIFIC_FIELDS


@dataclass(frozen=True)
class ConversionDiagnostic:
    specimen_id: str
    force_measurement_method: str
    displacement_measurement_method: str
    curve_points: int
    acquisition_rate_floor_hz: float
    force_conversion_N_per_input_unit: float
    maximum_force_N: float
    maximum_corrected_displacement_mm: float
    force_uncertainty_bound_N: float
    displacement_uncertainty_bound_mm: float
    fixture_compliance_mm_per_N: float
    fixture_compliance_uncertainty_mm_per_N: float
    input_file: str
    input_sha256: str
    output_file: str
    output_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_plan(package_dir: Path) -> List[Dict[str, str]]:
    package_dir = Path(package_dir)
    manifest_path = package_dir / "manifest.json"
    plan_path = package_dir / "frozen_plan.csv"
    if not manifest_path.exists() or not plan_path.exists():
        raise FileNotFoundError(
            "capture package must contain manifest.json and frozen_plan.csv"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = str(manifest.get("plan_sha256", ""))
    observed_hash = _sha256(plan_path)
    if not expected_hash or observed_hash != expected_hash:
        raise ValueError("frozen plan checksum does not match package manifest")
    with plan_path.open(encoding="utf-8", newline="") as handle:
        plan = list(csv.DictReader(handle))
    if len(plan) != 6:
        raise ValueError(
            f"manual adapter requires the frozen six-coupon plan, found {len(plan)}"
        )
    return plan


def _constant_text(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        raise ValueError(f"missing column {column}")
    values = frame[column].fillna("").astype(str).str.strip()
    if (values == "").any():
        raise ValueError(f"{column} must be repeated on every row")
    unique = values.unique()
    if len(unique) != 1:
        raise ValueError(f"{column} must be constant within a specimen")
    return str(unique[0])


def _constant_number(
    frame: pd.DataFrame,
    column: str,
    *,
    allow_zero: bool = False,
) -> float:
    if column not in frame:
        raise ValueError(f"missing column {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{column} must be finite and numeric on every row")
    unique = np.unique(values.to_numpy(float))
    if len(unique) != 1:
        raise ValueError(f"{column} must be constant within a specimen")
    value = float(unique[0])
    if value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{column} must be {qualifier}")
    return value


def initialize_manual_templates(
    package_dir: Path = DEFAULT_PACKAGE,
    output_dir: Path | None = None,
) -> Dict[str, object]:
    package_dir = Path(package_dir)
    output_dir = (
        package_dir / "manual_capture_templates"
        if output_dir is None
        else Path(output_dir)
    )
    plan = _read_plan(package_dir)
    canonical_dir = package_dir / "capture_templates"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for expected in plan:
        specimen_id = expected["specimen_id"]
        canonical_path = canonical_dir / f"{specimen_id}.csv"
        if not canonical_path.exists():
            raise FileNotFoundError(
                f"canonical capture template missing: {canonical_path}"
            )
        canonical = pd.read_csv(canonical_path, keep_default_na=False)
        if len(canonical) != 1:
            raise ValueError(
                f"canonical template must contain one row: {canonical_path}"
            )
        row = {
            field: canonical.iloc[0].get(field, "")
            for field in MANUAL_CAPTURE_FIELDS
        }
        row.update(
            {
                "force_measurement_method": "scale_mass_g",
                "force_zero_input": 0.0,
                "force_resolution_N": STANDARD_GRAVITY_M_S2 / 1000.0,
                "force_capacity_N": MINIMUM_FORCE_CAPACITY_N,
                "displacement_measurement_method": "local_indicator",
                "displacement_zero_mm": 0.0,
                "displacement_resolution_mm": (
                    MAXIMUM_DISPLACEMENT_RESOLUTION_MM
                ),
            }
        )
        output_path = output_dir / f"{specimen_id}.csv"
        pd.DataFrame([row], columns=MANUAL_CAPTURE_FIELDS).to_csv(
            output_path, index=False
        )
        files.append(
            {
                "specimen_id": specimen_id,
                "path": str(output_path.resolve()),
                "sha256": _sha256(output_path),
            }
        )
    manifest = {
        "adapter_id": "bambu-pattern-bend-manual-v1",
        "status": "blank_templates_only",
        "package_dir": str(package_dir.resolve()),
        "frozen_plan_sha256": _sha256(package_dir / "frozen_plan.csv"),
        "specimen_count": len(files),
        "files": files,
        "supported_force_methods": ["scale_mass_g", "load_cell_N"],
        "supported_displacement_methods": [
            "local_indicator",
            "actuator_corrected",
        ],
        "fixed_standard_gravity_m_s2": STANDARD_GRAVITY_M_S2,
        "quality_gates": {
            "minimum_force_capacity_N": MINIMUM_FORCE_CAPACITY_N,
            "maximum_force_resolution_N": MAXIMUM_FORCE_RESOLUTION_N,
            "maximum_force_uncertainty_N": MAXIMUM_FORCE_UNCERTAINTY_N,
            "maximum_displacement_resolution_mm": (
                MAXIMUM_DISPLACEMENT_RESOLUTION_MM
            ),
            "maximum_displacement_uncertainty_mm": (
                MAXIMUM_DISPLACEMENT_UNCERTAINTY_MM
            ),
            "maximum_fixture_compliance_mm_per_N": (
                MAXIMUM_FIXTURE_COMPLIANCE_MM_N
            ),
            "maximum_fixture_compliance_uncertainty_mm_per_N": (
                MAXIMUM_COMPLIANCE_UNCERTAINTY_MM_N
            ),
            "minimum_acquisition_rate_Hz": MINIMUM_ACQUISITION_RATE_HZ,
        },
    }
    (output_dir / "template_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _numeric_curve(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"missing measurement columns: {sorted(missing)}")
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("raw curve contains blank, nonnumeric, NaN, or infinite values")
    return values


def _convert_specimen(
    path: Path,
    expected: Mapping[str, str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    frame = pd.read_csv(path, keep_default_na=False)
    if len(frame) < 20:
        raise ValueError(
            f"only {len(frame)} raw curve rows; minimum is 20"
        )
    missing = set(MANUAL_CAPTURE_FIELDS) - set(frame)
    if missing:
        raise ValueError(f"missing manual capture columns: {sorted(missing)}")

    for column, expected_value in (
        ("specimen_id", expected["specimen_id"]),
        ("pattern", expected["pattern"]),
        ("load_case", expected["load_case"]),
        ("material", "Bambu PLA Basic"),
        ("fixture_id", "bambu-pattern-bend-v1"),
    ):
        observed = _constant_text(frame, column)
        if observed != expected_value:
            raise ValueError(
                f"{column}={observed!r} does not match frozen "
                f"{expected_value!r}"
            )

    force_method = _constant_text(frame, "force_measurement_method")
    if force_method == "scale_mass_g":
        force_scale = STANDARD_GRAVITY_M_S2 / 1000.0
    elif force_method == "load_cell_N":
        force_scale = 1.0
    else:
        raise ValueError(
            "force_measurement_method must be scale_mass_g or load_cell_N"
        )
    displacement_method = _constant_text(
        frame, "displacement_measurement_method"
    )
    if displacement_method not in {
        "local_indicator",
        "actuator_corrected",
    }:
        raise ValueError(
            "displacement_measurement_method must be local_indicator "
            "or actuator_corrected"
        )
    for calibration_field in (
        "force_calibration_id",
        "displacement_calibration_id",
        "fixture_compliance_calibration_id",
    ):
        _constant_text(frame, calibration_field)

    force_zero = _constant_number(
        frame, "force_zero_input", allow_zero=True
    )
    force_resolution = _constant_number(frame, "force_resolution_N")
    force_calibration_error = _constant_number(
        frame, "force_calibration_max_error_N", allow_zero=True
    )
    force_capacity = _constant_number(frame, "force_capacity_N")
    displacement_zero = _constant_number(
        frame, "displacement_zero_mm", allow_zero=True
    )
    displacement_resolution = _constant_number(
        frame, "displacement_resolution_mm"
    )
    displacement_calibration_error = _constant_number(
        frame,
        "displacement_calibration_max_error_mm",
        allow_zero=True,
    )
    compliance = _constant_number(
        frame, "fixture_compliance_mm_per_N", allow_zero=True
    )
    compliance_uncertainty = _constant_number(
        frame,
        "fixture_compliance_uncertainty_mm_per_N",
        allow_zero=True,
    )

    if force_capacity < MINIMUM_FORCE_CAPACITY_N:
        raise ValueError(
            f"force capacity {force_capacity:g} N is below "
            f"{MINIMUM_FORCE_CAPACITY_N:g} N"
        )
    if force_resolution > MAXIMUM_FORCE_RESOLUTION_N:
        raise ValueError(
            f"force resolution {force_resolution:g} N exceeds "
            f"{MAXIMUM_FORCE_RESOLUTION_N:g} N"
        )
    force_uncertainty = force_calibration_error + 0.5 * force_resolution
    if force_uncertainty > MAXIMUM_FORCE_UNCERTAINTY_N:
        raise ValueError(
            f"force uncertainty bound {force_uncertainty:g} N exceeds "
            f"{MAXIMUM_FORCE_UNCERTAINTY_N:g} N"
        )
    if displacement_resolution > MAXIMUM_DISPLACEMENT_RESOLUTION_MM:
        raise ValueError(
            f"displacement resolution {displacement_resolution:g} mm exceeds "
            f"{MAXIMUM_DISPLACEMENT_RESOLUTION_MM:g} mm"
        )
    if compliance > MAXIMUM_FIXTURE_COMPLIANCE_MM_N:
        raise ValueError(
            f"fixture compliance {compliance:g} mm/N exceeds "
            f"{MAXIMUM_FIXTURE_COMPLIANCE_MM_N:g} mm/N"
        )
    if compliance_uncertainty > MAXIMUM_COMPLIANCE_UNCERTAINTY_MM_N:
        raise ValueError(
            "fixture compliance uncertainty "
            f"{compliance_uncertainty:g} mm/N exceeds "
            f"{MAXIMUM_COMPLIANCE_UNCERTAINTY_MM_N:g} mm/N"
        )

    raw = _numeric_curve(
        frame, ("raw_time_s", "raw_force_input", "raw_displacement_mm")
    )
    time = raw[:, 0]
    force = (raw[:, 1] - force_zero) * force_scale
    measured_displacement = raw[:, 2] - displacement_zero
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("raw_time_s must be strictly increasing")
    acquisition_rate_floor = 1.0 / float(np.max(np.diff(time)))
    if acquisition_rate_floor + 1e-9 < MINIMUM_ACQUISITION_RATE_HZ:
        raise ValueError(
            f"acquisition rate floor {acquisition_rate_floor:g} Hz is below "
            f"{MINIMUM_ACQUISITION_RATE_HZ:g} Hz"
        )
    if float(np.max(force)) > force_capacity:
        raise ValueError("measured force exceeds registered instrument capacity")
    if float(np.min(force)) < -force_uncertainty:
        raise ValueError("force is negative beyond its uncertainty bound")

    if displacement_method == "local_indicator":
        corrected_displacement = measured_displacement
        compliance_term = 0.0
        force_compliance_term = 0.0
    else:
        corrected_displacement = measured_displacement - compliance * force
        compliance_term = float(np.max(np.abs(force))) * compliance_uncertainty
        force_compliance_term = compliance * force_uncertainty
    displacement_uncertainty = (
        displacement_calibration_error
        + 0.5 * displacement_resolution
        + compliance_term
        + force_compliance_term
    )
    if displacement_uncertainty > MAXIMUM_DISPLACEMENT_UNCERTAINTY_MM:
        raise ValueError(
            "displacement uncertainty bound "
            f"{displacement_uncertainty:g} mm exceeds "
            f"{MAXIMUM_DISPLACEMENT_UNCERTAINTY_MM:g} mm"
        )
    if float(np.min(corrected_displacement)) < -displacement_uncertainty:
        raise ValueError(
            "corrected displacement is negative beyond its uncertainty bound"
        )
    if np.any(np.diff(corrected_displacement) < -displacement_uncertainty):
        raise ValueError(
            "corrected displacement is not monotonic within uncertainty"
        )

    output = frame.loc[
        :, [field for field in CAPTURE_FIELDS if field not in CURVE_FIELDS]
    ].copy()
    output["time_s"] = time - time[0]
    output["displacement_mm"] = corrected_displacement
    output["force_N"] = force
    output = output.reindex(columns=CAPTURE_FIELDS)
    diagnostic = {
        "specimen_id": expected["specimen_id"],
        "force_measurement_method": force_method,
        "displacement_measurement_method": displacement_method,
        "curve_points": len(frame),
        "acquisition_rate_floor_hz": acquisition_rate_floor,
        "force_conversion_N_per_input_unit": force_scale,
        "maximum_force_N": float(np.max(force)),
        "maximum_corrected_displacement_mm": float(
            np.max(corrected_displacement)
        ),
        "force_uncertainty_bound_N": force_uncertainty,
        "displacement_uncertainty_bound_mm": displacement_uncertainty,
        "fixture_compliance_mm_per_N": compliance,
        "fixture_compliance_uncertainty_mm_per_N": compliance_uncertainty,
    }
    return output, diagnostic


def convert_manual_capture(
    input_dir: Path,
    package_dir: Path = DEFAULT_PACKAGE,
    output_dir: Path | None = None,
) -> Dict[str, object]:
    input_dir = Path(input_dir)
    package_dir = Path(package_dir)
    output_dir = (
        package_dir / "converted_manual_submission"
        if output_dir is None
        else Path(output_dir)
    )
    plan = _read_plan(package_dir)
    expected_names = {
        f"{row['specimen_id']}.csv": row for row in plan
    }
    observed_names = {path.name for path in input_dir.glob("*.csv")}
    unexpected = sorted(observed_names - set(expected_names))
    missing = sorted(set(expected_names) - observed_names)
    if unexpected or missing:
        raise ValueError(
            f"manual specimen set mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )

    converted: Dict[str, pd.DataFrame] = {}
    diagnostics: Dict[str, Dict[str, object]] = {}
    for filename, expected in expected_names.items():
        try:
            converted[filename], diagnostics[filename] = _convert_specimen(
                input_dir / filename, expected
            )
        except ValueError as exc:
            raise ValueError(f"{expected['specimen_id']}: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_rows: List[ConversionDiagnostic] = []
    for filename, expected in expected_names.items():
        input_path = input_dir / filename
        output_path = output_dir / filename
        converted[filename].to_csv(output_path, index=False)
        diagnostic_rows.append(
            ConversionDiagnostic(
                **diagnostics[filename],
                input_file=str(input_path.resolve()),
                input_sha256=_sha256(input_path),
                output_file=str(output_path.resolve()),
                output_sha256=_sha256(output_path),
            )
        )

    canonical_audit = validate_submission(
        output_dir,
        package_dir / "frozen_plan.csv",
        Path(json.loads(
            (package_dir / "manifest.json").read_text(encoding="utf-8")
        )["study_path"]),
        package_dir / "manifest.json",
    )
    manifest = {
        "adapter_id": "bambu-pattern-bend-manual-v1",
        "status": (
            "converted_and_canonical_gate_passed"
            if canonical_audit["ready_for_analysis"]
            else "converted_but_canonical_gate_failed"
        ),
        "software_verification_only": False,
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "package_dir": str(package_dir.resolve()),
        "frozen_plan_sha256": _sha256(package_dir / "frozen_plan.csv"),
        "formulas": {
            "scale_mass_g": (
                "force_N=(raw_force_input-force_zero_input)"
                "*9.80665/1000"
            ),
            "load_cell_N": (
                "force_N=raw_force_input-force_zero_input"
            ),
            "local_indicator": (
                "displacement_mm=raw_displacement_mm-displacement_zero_mm"
            ),
            "actuator_corrected": (
                "displacement_mm=raw_displacement_mm-displacement_zero_mm"
                "-fixture_compliance_mm_per_N*force_N"
            ),
        },
        "diagnostics": [asdict(row) for row in diagnostic_rows],
        "canonical_validation": canonical_audit,
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not canonical_audit["ready_for_analysis"]:
        raise ValueError(
            "converted curves failed the unchanged canonical validation gate; "
            "inspect conversion_manifest.json"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize or convert calibrated low-cost bending capture CSVs"
        )
    )
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    package = Path(args.package)
    if args.initialize:
        if args.input:
            parser.error("--input cannot be used with --initialize")
        payload = initialize_manual_templates(
            package,
            None if args.output is None else Path(args.output),
        )
        print(
            f"initialized {payload['specimen_count']} manual templates"
        )
        return
    if not args.input:
        parser.error("--input is required unless --initialize is used")
    payload = convert_manual_capture(
        Path(args.input),
        package,
        None if args.output is None else Path(args.output),
    )
    print(
        f"{payload['status']}: "
        f"{len(payload['diagnostics'])} specimens"
    )


if __name__ == "__main__":
    main()
