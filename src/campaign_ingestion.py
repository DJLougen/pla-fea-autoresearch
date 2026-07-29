from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .characterization_campaign import DEFAULT_OUTPUT as DEFAULT_CAMPAIGN
from .prepare import prepare_directory, preprocess_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/campaign_ingestion_audit"
MIN_CURVE_POINTS = 10
MIN_LATERAL_COVERAGE = 0.80


@dataclass(frozen=True)
class SpecimenAudit:
    specimen_id: str
    expected_partition: str
    test_type: str
    print_direction: str
    status: str
    source_file: str | None
    source_sha256: str | None
    source_rows: int
    usable_curve_points: int
    lateral_coverage: float | None
    issues: tuple[str, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan(plan_path: Path) -> Dict[str, Dict[str, str]]:
    with Path(plan_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {row["specimen_id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("test plan contains duplicate specimen IDs")
    if len(rows) != 43:
        raise ValueError(f"expected 43 planned specimens, found {len(rows)}")
    return lookup


def _verify_plan_manifest(
    plan_path: Path, manifest_path: Path | None
) -> None:
    if manifest_path is None:
        return
    manifest = json.loads(
        Path(manifest_path).read_text(encoding="utf-8")
    )
    expected = manifest.get("test_plan_sha256")
    actual = _sha256(Path(plan_path))
    if expected != actual:
        raise ValueError(
            "test plan checksum does not match the frozen campaign manifest"
        )


def _strict_constant_text(
    frame: pd.DataFrame, column: str
) -> tuple[str | None, str | None]:
    if column not in frame:
        return None, f"missing column {column}"
    values = frame[column].dropna().astype(str).str.strip()
    values = values.loc[values != ""]
    if len(values) != len(frame):
        return None, f"{column} must be repeated on every source row"
    unique = values.unique()
    if len(unique) != 1:
        return None, f"{column} must be constant within a specimen"
    return str(unique[0]), None


def _numeric_constant(
    frame: pd.DataFrame, column: str
) -> tuple[float | None, str | None]:
    if column not in frame:
        return None, f"missing column {column}"
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        return None, f"{column} must be repeated and numeric on every row"
    unique = np.unique(values.to_numpy(float))
    if len(unique) != 1 or unique[0] <= 0.0:
        return None, f"{column} must be one positive constant per specimen"
    return float(unique[0]), None


def _usable_mask(
    frame: pd.DataFrame, test_type: str
) -> tuple[np.ndarray, str]:
    if test_type == "tensile" and {
        "engineering_strain",
        "engineering_stress_MPa",
    }.issubset(frame):
        columns = ["engineering_strain", "engineering_stress_MPa"]
        mode = "stress_strain"
    else:
        columns = ["displacement_mm", "force_N"]
        mode = "force_displacement"
    if not set(columns).issubset(frame):
        return np.zeros(len(frame), dtype=bool), mode
    values = np.column_stack(
        [
            pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            for column in columns
        ]
    )
    return np.isfinite(values).all(axis=1), mode


def _audit_file(
    path: Path, expected: Mapping[str, str]
) -> SpecimenAudit:
    issues: List[str] = []
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return SpecimenAudit(
            expected["specimen_id"],
            expected["partition"],
            expected["test_type"],
            expected["print_direction"],
            "invalid",
            str(path.resolve()),
            _sha256(path),
            0,
            0,
            None,
            (f"CSV parse failed: {exc}",),
        )
    for column, expected_value in (
        ("test_id", expected["specimen_id"]),
        ("partition", expected["partition"]),
        ("test_type", expected["test_type"]),
        ("print_direction", expected["print_direction"]),
        ("material_name", "Bambu PLA Basic"),
    ):
        observed, error = _strict_constant_text(frame, column)
        if error:
            issues.append(error)
        elif observed != expected_value:
            issues.append(
                f"{column}={observed!r} does not match frozen value "
                f"{expected_value!r}"
            )
    for column in (
        "material_lot",
        "material_color",
        "machine_id",
        "slicer_profile",
        "drying_protocol",
    ):
        _, error = _strict_constant_text(frame, column)
        if error:
            issues.append(error)
    for column in (
        "layer_height_mm",
        "nozzle_temperature_C",
        "print_speed_mm_s",
        "infill_percent",
        "specimen_width_mm",
        "specimen_thickness_mm",
        "specimen_mass_g",
    ):
        _, error = _numeric_constant(frame, column)
        if error:
            issues.append(error)
    if expected["test_type"] == "tensile":
        _, error = _numeric_constant(frame, "gauge_length_mm")
        if error:
            issues.append(error)
    elif expected["test_type"] == "flexural":
        _, error = _numeric_constant(frame, "support_span_mm")
        if error:
            issues.append(error)

    usable, mode = _usable_mask(frame, expected["test_type"])
    usable_count = int(usable.sum())
    if usable_count < MIN_CURVE_POINTS:
        issues.append(
            f"only {usable_count} usable curve points; "
            f"minimum is {MIN_CURVE_POINTS}"
        )
    elif mode == "stress_strain":
        x = pd.to_numeric(
            frame.loc[usable, "engineering_strain"], errors="coerce"
        ).to_numpy(float)
        if np.ptp(x) < 0.002:
            issues.append("engineering-strain range is below 0.2%")
        if np.any(np.diff(x) < -1.0e-8):
            issues.append("engineering strain is not monotonic in acquisition order")
    else:
        x = pd.to_numeric(
            frame.loc[usable, "displacement_mm"], errors="coerce"
        ).to_numpy(float)
        if np.ptp(x) <= 0.0:
            issues.append("displacement has no positive range")
        if np.any(np.diff(x) < -1.0e-8):
            issues.append("displacement is not monotonic in acquisition order")
    duplicate_rate = float(frame.duplicated().mean()) if len(frame) else 1.0
    if duplicate_rate > 0.05:
        issues.append(
            f"exact duplicate-row rate {duplicate_rate:.1%} exceeds 5%"
        )

    lateral_coverage = None
    if expected["test_type"] == "tensile":
        if "transverse_strain" not in frame or usable_count == 0:
            lateral_coverage = 0.0
        else:
            lateral = pd.to_numeric(
                frame.loc[usable, "transverse_strain"], errors="coerce"
            )
            lateral_coverage = float(lateral.notna().mean())
        if lateral_coverage < MIN_LATERAL_COVERAGE:
            issues.append(
                f"transverse-strain coverage {lateral_coverage:.1%} is below "
                f"{MIN_LATERAL_COVERAGE:.0%}"
            )
    try:
        preprocess_frame(frame, path.name)
    except Exception as exc:
        issues.append(f"fixed preprocessor rejected file: {exc}")
    return SpecimenAudit(
        expected["specimen_id"],
        expected["partition"],
        expected["test_type"],
        expected["print_direction"],
        "valid" if not issues else "invalid",
        str(path.resolve()),
        _sha256(path),
        len(frame),
        usable_count,
        lateral_coverage,
        tuple(issues),
    )


def validate_campaign_submission(
    input_dir: Path,
    plan_path: Path = DEFAULT_CAMPAIGN / "test_plan.csv",
    manifest_path: Path | None = (
        DEFAULT_CAMPAIGN / "campaign_manifest.json"
    ),
) -> Dict[str, object]:
    input_dir = Path(input_dir)
    plan_path = Path(plan_path)
    _verify_plan_manifest(plan_path, manifest_path)
    plan = _load_plan(plan_path)
    expected_files = {
        f"{specimen_id}.csv": specimen_id for specimen_id in plan
    }
    actual = {path.name: path for path in input_dir.glob("*.csv")}
    unexpected = sorted(set(actual) - set(expected_files))
    audits: List[SpecimenAudit] = []
    for filename, specimen_id in expected_files.items():
        path = actual.get(filename)
        expected = plan[specimen_id]
        if path is None:
            audits.append(
                SpecimenAudit(
                    specimen_id,
                    expected["partition"],
                    expected["test_type"],
                    expected["print_direction"],
                    "missing",
                    None,
                    None,
                    0,
                    0,
                    None,
                    ("expected specimen CSV is absent",),
                )
            )
        else:
            audits.append(_audit_file(path, expected))
    counts = {
        status: sum(audit.status == status for audit in audits)
        for status in ("valid", "invalid", "missing")
    }
    ready = (
        counts == {"valid": 43, "invalid": 0, "missing": 0}
        and not unexpected
    )
    return {
        "campaign_id": "bambu_pla_basic_directional_v1",
        "input_directory": str(input_dir.resolve()),
        "plan_sha256": _sha256(plan_path),
        "ready_for_fixed_preprocessing": ready,
        "counts": counts,
        "unexpected_csv_files": unexpected,
        "specimens": [asdict(audit) for audit in audits],
    }


def write_campaign_audit(
    input_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    plan_path: Path = DEFAULT_CAMPAIGN / "test_plan.csv",
    manifest_path: Path | None = (
        DEFAULT_CAMPAIGN / "campaign_manifest.json"
    ),
    processed_output: Path | None = None,
) -> Dict[str, object]:
    payload = validate_campaign_submission(
        input_dir, plan_path, manifest_path
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    specimens = payload["specimens"]
    flat_rows = [
        {
            **{key: value for key, value in row.items() if key != "issues"},
            "issues": " | ".join(row["issues"]),
        }
        for row in specimens
    ]
    with (output_dir / "specimen_status.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    counts = payload["counts"]
    lines = [
        "# Bambu PLA Basic campaign ingestion audit",
        "",
        (
            "## Ready for calibration"
            if payload["ready_for_fixed_preprocessing"]
            else "## Not ready for calibration"
        ),
        "",
        (
            f"Valid: {counts['valid']}/43; invalid: {counts['invalid']}; "
            f"missing: {counts['missing']}; unexpected CSVs: "
            f"{len(payload['unexpected_csv_files'])}."
        ),
        "",
        "Partition, test type, direction, and specimen ID are compared with "
        "the checksum-frozen test plan rather than trusted from submitted files.",
        "",
        "| Specimen | Partition | Mode | Direction | Status | Points | "
        "Lateral coverage | Issues |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in specimens:
        lateral = row["lateral_coverage"]
        lines.append(
            f"| {row['specimen_id']} | {row['expected_partition']} | "
            f"{row['test_type']} | {row['print_direction']} | "
            f"{row['status']} | {row['usable_curve_points']} | "
            f"{'n/a' if lateral is None else f'{float(lateral):.1%}'} | "
            f"{'; '.join(row['issues'])} |"
        )
    lines.extend(
        [
            "",
            "## Stable automated gates",
            "",
            "- Exactly 43 planned specimen IDs and no unplanned CSVs.",
            "- Immutable calibration/validation/held-out labels.",
            "- At least 10 usable points and a nonzero monotonic loading range.",
            "- At least 80% transverse-strain coverage for tensile specimens.",
            "- Positive constant actual dimensions, mass, and fixture geometry.",
            "- Repeated material lot, color, machine, slicer, drying, and print "
            "controls on every acquisition row.",
            "- No more than 5% exact duplicate rows and acceptance by the fixed "
            "preprocessor.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if processed_output is not None:
        if not payload["ready_for_fixed_preprocessing"]:
            raise ValueError(
                "campaign is not ready; processed data were not written"
            )
        prepare_directory(input_dir, processed_output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a filled Bambu PLA characterization campaign"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--plan", default=str(DEFAULT_CAMPAIGN / "test_plan.csv")
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_CAMPAIGN / "campaign_manifest.json"),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--processed-output")
    args = parser.parse_args()
    payload = write_campaign_audit(
        Path(args.input),
        Path(args.output),
        Path(args.plan),
        Path(args.manifest),
        (
            None
            if args.processed_output is None
            else Path(args.processed_output)
        ),
    )
    counts = payload["counts"]
    print(
        f"campaign audit: ready={payload['ready_for_fixed_preprocessing']}; "
        f"valid={counts['valid']}, invalid={counts['invalid']}, "
        f"missing={counts['missing']}; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )
    if not payload["ready_for_fixed_preprocessing"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
