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
from scipy.stats import t

from .custom_infill import admit_pattern_study
from .custom_pattern_fixture import (
    CROSSHEAD_RATE_MM_MIN,
    MAXIMUM_FIXTURE_COMPLIANCE_MM_N,
    ROLLER_DIAMETER_MM,
    SUPPORT_SPAN_MM,
    write_fixture_package,
)
from .custom_pattern_print_mesh import write_print_meshes
from .custom_pattern_slicer import write_manufacturing_package
from .custom_pattern_slicer import MAXIMUM_TOOLPATH_MASS_RATIO_ERROR
from .custom_pattern_toolpath import write_toolpath_audit
from .custom_pattern_toolpath_fea import (
    DEFAULT_ANALYSIS_RESOLUTIONS,
    run_toolpath_bending_study,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ROOT = (
    PROJECT_ROOT / "results/custom_infill/graded_hourglass"
)
DEFAULT_PLAN = CUSTOM_ROOT / "first_physical_test_plan.csv"
DEFAULT_STUDY = CUSTOM_ROOT / "study.json"
DEFAULT_PACKAGE = CUSTOM_ROOT / "physical_validation"
DEFAULT_ANALYSIS = DEFAULT_PACKAGE / "analysis"
MIN_CURVE_POINTS = 20
MIN_FIT_POINTS = 8
MIN_FIT_R2 = 0.98
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260728

CAPTURE_FIELDS = (
    "specimen_id",
    "pattern",
    "load_case",
    "replicate",
    "material",
    "material_lot",
    "material_color",
    "machine_id",
    "slicer_profile",
    "drying_protocol",
    "conditioning",
    "fixture_id",
    "nozzle_temperature_C",
    "bed_temperature_C",
    "layer_height_mm",
    "actual_mass_g",
    "test_temperature_C",
    "test_relative_humidity_percent",
    "support_span_mm",
    "support_roller_diameter_mm",
    "loading_roller_diameter_mm",
    "crosshead_rate_mm_min",
    "fixture_compliance_mm_per_N",
    "time_s",
    "displacement_mm",
    "force_N",
    "failure_mode",
    "notes",
)


@dataclass(frozen=True)
class PhysicalSpecimenResult:
    specimen_id: str
    pattern: str
    load_case: str
    replicate: int
    status: str
    source_file: str | None
    source_sha256: str | None
    usable_curve_points: int
    fit_curve_points: int
    actual_mass_g: float | None
    target_mass_g: float
    mass_deviation_fraction: float | None
    slicer_expected_mass_g: float
    slicer_mass_deviation_fraction: float | None
    stiffness_N_per_mm: float | None
    stiffness_fit_r2: float | None
    mass_normalized_stiffness_N_per_mm: float | None
    issues: tuple[str, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_plan(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise ValueError(
            f"first physical plan must contain 6 specimens, found {len(rows)}"
        )
    ids = [row["specimen_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("first physical plan contains duplicate specimen IDs")
    patterns = {row["pattern"] for row in rows}
    if len(patterns) != 2:
        raise ValueError("first physical plan must contain two patterns")
    if {row["load_case"] for row in rows} != {"bending"}:
        raise ValueError("first physical plan must be a bending comparison")
    for pattern in patterns:
        replicates = {
            int(row["replicate"])
            for row in rows
            if row["pattern"] == pattern
        }
        if replicates != {1, 2, 3}:
            raise ValueError(
                f"{pattern} must contain replicates 1, 2, and 3"
            )
    return rows


def _load_study(path: Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("pattern") is None or payload.get("control") is None:
        raise ValueError("custom study does not identify pattern and control")
    if (
        payload.get("material_admission", {}).get("gate_passed")
        is not True
    ):
        raise ValueError("custom study lacks a passing material admission")
    return payload


def _plan_prediction(
    rows: Sequence[Mapping[str, str]], pattern: str
) -> float:
    values = {
        float(row["predicted_value"])
        for row in rows
        if row["pattern"] == pattern
    }
    if len(values) != 1:
        raise ValueError(
            f"planned prediction is not constant for {pattern}"
        )
    return values.pop()


def write_capture_package(
    output_dir: Path = DEFAULT_PACKAGE,
    plan_path: Path = DEFAULT_PLAN,
    study_path: Path = DEFAULT_STUDY,
    *,
    include_manufacturing: bool = True,
    include_fixture: bool = True,
    maximum_toolpath_mass_ratio_error: float = (
        MAXIMUM_TOOLPATH_MASS_RATIO_ERROR
    ),
    toolpath_resolutions: Sequence[int] = DEFAULT_ANALYSIS_RESOLUTIONS,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    plan_path = Path(plan_path)
    study_path = Path(study_path)
    plan = _load_plan(plan_path)
    study = _load_study(study_path)
    study_root = study_path.parent
    patterns = [str(study["control"]), str(study["pattern"])]
    if set(patterns) != {row["pattern"] for row in plan}:
        raise ValueError("study pattern/control do not match the first plan")
    output_dir.mkdir(parents=True, exist_ok=True)
    template_dir = output_dir / "capture_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    frozen_plan = output_dir / "frozen_plan.csv"
    frozen_plan.write_bytes(plan_path.read_bytes())
    print_mesh_manifest = write_print_meshes(
        output_dir / "printable_stl",
        frozen_plan,
        study_root / "solid/meshes",
    )
    manufacturing_manifest = (
        write_manufacturing_package(
            output_dir / "manufacturing",
            output_dir / "printable_stl",
            maximum_toolpath_mass_ratio_error=(
                maximum_toolpath_mass_ratio_error
            ),
        )
        if include_manufacturing
        else None
    )
    toolpath_manifest = (
        write_toolpath_audit(
            output_dir / "toolpath_morphology",
            output_dir / "manufacturing/sliced/plate_1.gcode",
            output_dir / "manufacturing/manifest.json",
            output_dir / "printable_stl/manifest.json",
        )
        if manufacturing_manifest is not None
        else None
    )
    toolpath_fea_manifest = (
        run_toolpath_bending_study(
            output_dir / "toolpath_resolved_bending",
            PROJECT_ROOT / "runs/custom_infill/toolpath_resolved_bending",
            output_dir / "manufacturing/sliced/plate_1.gcode",
            toolpath_audit_manifest_path=(
                output_dir / "toolpath_morphology/manifest.json"
            ),
            study_path=study_path,
            resolutions=toolpath_resolutions,
        )
        if toolpath_manifest is not None
        else None
    )
    fixture_manifest = (
        write_fixture_package(output_dir / "fixture")
        if include_fixture
        else None
    )
    for row in plan:
        template = {
            field: "" for field in CAPTURE_FIELDS
        }
        template.update(
            {
                "specimen_id": row["specimen_id"],
                "pattern": row["pattern"],
                "load_case": row["load_case"],
                "replicate": row["replicate"],
                "material": row["material"],
                "conditioning": "as_printed_dry",
                "fixture_id": "bambu-pattern-bend-v1",
                "support_span_mm": SUPPORT_SPAN_MM,
                "support_roller_diameter_mm": ROLLER_DIAMETER_MM,
                "loading_roller_diameter_mm": ROLLER_DIAMETER_MM,
                "crosshead_rate_mm_min": CROSSHEAD_RATE_MM_MIN,
            }
        )
        with (
            template_dir / f"{row['specimen_id']}.csv"
        ).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CAPTURE_FIELDS)
            writer.writeheader()
            writer.writerow(template)
    scad_rows = {}
    for pattern in patterns:
        scad = study_root / "solid/patterns" / f"{pattern}.scad"
        if not scad.exists():
            raise FileNotFoundError(f"printable source missing: {scad}")
        scad_rows[pattern] = {
            "path": str(scad.resolve()),
            "sha256": _sha256(scad),
        }
    predicted_control = _plan_prediction(plan, patterns[0])
    predicted_custom = _plan_prediction(plan, patterns[1])
    manifest = {
        "validation_id": f"{study['pattern']}_bending_physical_v1",
        "material": "Bambu PLA Basic",
        "status": "awaiting_measurements",
        "specimen_count": len(plan),
        "replicates_per_pattern": 3,
        "patterns": patterns,
        "load_case": "bending",
        "plan_path": str(frozen_plan.resolve()),
        "plan_sha256": _sha256(frozen_plan),
        "source_plan_sha256": _sha256(plan_path),
        "study_path": str(study_path.resolve()),
        "study_sha256": _sha256(study_path),
        "predicted_stiffness_N_per_mm": {
            patterns[0]: predicted_control,
            patterns[1]: predicted_custom,
        },
        "predicted_custom_to_control_ratio": (
            predicted_custom / predicted_control
        ),
        "printable_sources": scad_rows,
        "printable_mesh_manifest": str(
            (output_dir / "printable_stl/manifest.json").resolve()
        ),
        "printable_mesh_manifest_sha256": _sha256(
            output_dir / "printable_stl/manifest.json"
        ),
        "printable_mesh_gate_passed": print_mesh_manifest[
            "all_print_mesh_gates_passed"
        ],
        "printable_meshes": {
            row["pattern"]: {
                "path": row["printable_stl"],
                "sha256": row["printable_stl_sha256"],
                "watertight": row["watertight"],
                "calculated_mass_g": row["calculated_mass_g"],
                "mass_relative_error": row["mass_relative_error"],
            }
            for row in print_mesh_manifest["meshes"]
        },
        "manufacturing_manifest": (
            str((output_dir / "manufacturing/manifest.json").resolve())
            if manufacturing_manifest is not None
            else None
        ),
        "manufacturing_manifest_sha256": (
            _sha256(output_dir / "manufacturing/manifest.json")
            if manufacturing_manifest is not None
            else None
        ),
        "manufacturing_gate_passed": (
            manufacturing_manifest["all_manufacturing_gates_passed"]
            if manufacturing_manifest is not None
            else None
        ),
        "slicer_predicted_deposited_mass_g": (
            manufacturing_manifest["predicted_deposited_mass_g"]
            if manufacturing_manifest is not None
            else None
        ),
        "toolpath_morphology_manifest": (
            str((output_dir / "toolpath_morphology/manifest.json").resolve())
            if toolpath_manifest is not None
            else None
        ),
        "toolpath_morphology_manifest_sha256": (
            _sha256(output_dir / "toolpath_morphology/manifest.json")
            if toolpath_manifest is not None
            else None
        ),
        "toolpath_relative_ranking_gate_passed": (
            toolpath_manifest["gates"][
                "relative_linear_stiffness_ranking_admitted"
            ]
            if toolpath_manifest is not None
            else None
        ),
        "toolpath_model_status": (
            toolpath_manifest["status"]
            if toolpath_manifest is not None
            else None
        ),
        "toolpath_resolved_bending_manifest": (
            str(
                (
                    output_dir
                    / "toolpath_resolved_bending/manifest.json"
                ).resolve()
            )
            if toolpath_fea_manifest is not None
            else None
        ),
        "toolpath_resolved_bending_manifest_sha256": (
            _sha256(output_dir / "toolpath_resolved_bending/manifest.json")
            if toolpath_fea_manifest is not None
            else None
        ),
        "toolpath_resolved_bending_gate_passed": (
            toolpath_fea_manifest["gates"][
                "relative_toolpath_aware_bending_ranking_admitted"
            ]
            if toolpath_fea_manifest is not None
            else None
        ),
        "toolpath_resolved_bending_custom_to_control_ratio": (
            toolpath_fea_manifest[
                "toolpath_fine_custom_to_control_ratio"
            ]
            if toolpath_fea_manifest is not None
            else None
        ),
        "fixture_manifest": (
            str((output_dir / "fixture/manifest.json").resolve())
            if fixture_manifest is not None
            else None
        ),
        "fixture_manifest_sha256": (
            _sha256(output_dir / "fixture/manifest.json")
            if fixture_manifest is not None
            else None
        ),
        "fixture_gate_passed": (
            fixture_manifest["all_fixture_gates_passed"]
            if fixture_manifest is not None
            else None
        ),
        "capture_fields": list(CAPTURE_FIELDS),
        "quality_gates": {
            "minimum_curve_points": MIN_CURVE_POINTS,
            "minimum_linear_fit_points": MIN_FIT_POINTS,
            "minimum_linear_fit_r2": MIN_FIT_R2,
            "maximum_absolute_mass_deviation_fraction": 0.15,
            "minimum_fraction_of_recommended_displacement": 0.80,
            "identical_print_controls_required": True,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""# {study['pattern']} physical validation capture

This package freezes the six-coupon Bambu PLA Basic bending comparison.

The frozen G-code has also passed a three-resolution, mass-calibrated
toolpath-voxel CalculiX replay. That sensitivity confirms the simulated
{study['pattern']} bending advantage survives the deposited bead morphology.
It remains a relative-ranking result: do not use it as an absolute stiffness
or failure prediction.

1. Open the checksum-listed six-object 3MF, confirm the intended physical
   Bambu PLA Basic spool, and print the plate without rescaling, rearranging,
   or changing its embedded process.
2. Install the registered 4 mm steel rods in the fixture and verify its
   40.00 mm support span. Measure fixture compliance before coupon testing.
3. Keep spool, drying, printer, slicer, nozzle temperature, bed temperature,
   and layer height identical.
4. Weigh every coupon.
5. Copy each matching template into a submission directory and repeat all
   metadata on every force-displacement row.
6. Record at least {MIN_CURVE_POINTS} points through the compliance-corrected
   recommended displacement. Use positive displacement and positive force.
7. Analyze the filled directory with:

```bash
python -m src.custom_pattern_validation --input path/to/submission
```

Templates are deliberately invalid until real curve rows and print controls
are supplied. The analysis reports raw and mass-normalized stiffness, replicate
uncertainty, the measured custom/control ratio, and whether the simulated
ranking survives. It never silently updates the canonical PLA card.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def _constant_text(
    frame: pd.DataFrame, column: str
) -> Tuple[str | None, str | None]:
    if column not in frame:
        return None, f"missing column {column}"
    values = frame[column].fillna("").astype(str).str.strip()
    if (values == "").any():
        return None, f"{column} must be repeated on every row"
    unique = values.unique()
    if len(unique) != 1:
        return None, f"{column} must be constant within a specimen"
    return str(unique[0]), None


def _constant_number(
    frame: pd.DataFrame, column: str, *, allow_zero: bool = False
) -> Tuple[float | None, str | None]:
    if column not in frame:
        return None, f"missing column {column}"
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        return None, f"{column} must be numeric on every row"
    unique = np.unique(values.to_numpy(float))
    if len(unique) != 1:
        return None, f"{column} must be constant within a specimen"
    value = float(unique[0])
    if value < 0.0 or (not allow_zero and value == 0.0):
        return None, f"{column} must be positive"
    return value, None


def _invalid_result(
    expected: Mapping[str, str],
    path: Path | None,
    issues: Sequence[str],
) -> PhysicalSpecimenResult:
    return PhysicalSpecimenResult(
        specimen_id=expected["specimen_id"],
        pattern=expected["pattern"],
        load_case=expected["load_case"],
        replicate=int(expected["replicate"]),
        status="missing" if path is None else "invalid",
        source_file=None if path is None else str(path.resolve()),
        source_sha256=None if path is None else _sha256(path),
        usable_curve_points=0,
        fit_curve_points=0,
        actual_mass_g=None,
        target_mass_g=float(expected["target_actual_mass_g"]),
        mass_deviation_fraction=None,
        slicer_expected_mass_g=float(
            expected.get(
                "slicer_expected_mass_g",
                expected["target_actual_mass_g"],
            )
        ),
        slicer_mass_deviation_fraction=None,
        stiffness_N_per_mm=None,
        stiffness_fit_r2=None,
        mass_normalized_stiffness_N_per_mm=None,
        issues=tuple(issues),
    )


def _audit_specimen(
    path: Path, expected: Mapping[str, str]
) -> Tuple[PhysicalSpecimenResult, Dict[str, object] | None]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return _invalid_result(
            expected, path, [f"CSV parse failed: {exc}"]
        ), None
    issues: List[str] = []
    observed_text: Dict[str, str] = {}
    for column in (
        "specimen_id",
        "pattern",
        "load_case",
        "material",
        "material_lot",
        "material_color",
        "machine_id",
        "slicer_profile",
        "drying_protocol",
        "conditioning",
        "fixture_id",
    ):
        value, issue = _constant_text(frame, column)
        if issue:
            issues.append(issue)
        elif value is not None:
            observed_text[column] = value
    for column, expected_value in (
        ("specimen_id", expected["specimen_id"]),
        ("pattern", expected["pattern"]),
        ("load_case", expected["load_case"]),
        ("material", "Bambu PLA Basic"),
        ("fixture_id", "bambu-pattern-bend-v1"),
    ):
        value = observed_text.get(column)
        if value is not None and value != expected_value:
            issues.append(
                f"{column}={value!r} does not match frozen "
                f"{expected_value!r}"
            )

    observed_numeric: Dict[str, float] = {}
    for column, allow_zero in (
        ("replicate", False),
        ("actual_mass_g", False),
        ("nozzle_temperature_C", False),
        ("bed_temperature_C", False),
        ("layer_height_mm", False),
        ("test_temperature_C", False),
        ("test_relative_humidity_percent", True),
        ("support_span_mm", False),
        ("support_roller_diameter_mm", False),
        ("loading_roller_diameter_mm", False),
        ("crosshead_rate_mm_min", False),
        ("fixture_compliance_mm_per_N", True),
    ):
        value, issue = _constant_number(
            frame, column, allow_zero=allow_zero
        )
        if issue:
            issues.append(issue)
        elif value is not None:
            observed_numeric[column] = value
    replicate = observed_numeric.get("replicate")
    if (
        replicate is not None
        and not math.isclose(
            replicate, float(expected["replicate"]), abs_tol=1e-12
        )
    ):
        issues.append("replicate does not match the frozen plan")
    fixture_expectations = {
        "support_span_mm": (SUPPORT_SPAN_MM, 0.05),
        "support_roller_diameter_mm": (ROLLER_DIAMETER_MM, 0.02),
        "loading_roller_diameter_mm": (ROLLER_DIAMETER_MM, 0.02),
        "crosshead_rate_mm_min": (CROSSHEAD_RATE_MM_MIN, 0.05),
    }
    for field, (expected_value, tolerance) in fixture_expectations.items():
        observed = observed_numeric.get(field)
        if (
            observed is not None
            and not math.isclose(
                observed, expected_value, abs_tol=tolerance
            )
        ):
            issues.append(
                f"{field}={observed:g} is outside the registered "
                f"{expected_value:g} ± {tolerance:g}"
            )
    compliance = observed_numeric.get(
        "fixture_compliance_mm_per_N"
    )
    if (
        compliance is not None
        and compliance > MAXIMUM_FIXTURE_COMPLIANCE_MM_N
    ):
        issues.append(
            f"fixture compliance {compliance:.6g} mm/N exceeds "
            f"{MAXIMUM_FIXTURE_COMPLIANCE_MM_N:.6g} mm/N"
        )

    target_mass = float(expected["target_actual_mass_g"])
    slicer_expected_mass = float(
        expected.get("slicer_expected_mass_g", target_mass)
    )
    actual_mass = observed_numeric.get("actual_mass_g")
    mass_deviation = (
        None
        if actual_mass is None
        else actual_mass / target_mass - 1.0
    )
    slicer_mass_deviation = (
        None
        if actual_mass is None
        else actual_mass / slicer_expected_mass - 1.0
    )
    if (
        slicer_mass_deviation is not None
        and abs(slicer_mass_deviation) > 0.15
    ):
        issues.append(
            "actual mass differs from the frozen toolpath prediction by "
            f"{slicer_mass_deviation:.1%}; maximum is 15%"
        )

    curve_columns = ("time_s", "displacement_mm", "force_N")
    usable = np.zeros(len(frame), dtype=bool)
    time = np.array([], dtype=float)
    displacement = np.array([], dtype=float)
    force = np.array([], dtype=float)
    if not set(curve_columns).issubset(frame):
        issues.append(
            "missing one or more curve columns: time_s, displacement_mm, force_N"
        )
    else:
        numeric = frame[list(curve_columns)].apply(
            pd.to_numeric, errors="coerce"
        )
        usable = np.isfinite(numeric.to_numpy(float)).all(axis=1)
        time = pd.to_numeric(
            frame.loc[usable, "time_s"], errors="coerce"
        ).to_numpy(float)
        displacement = pd.to_numeric(
            frame.loc[usable, "displacement_mm"], errors="coerce"
        ).to_numpy(float)
        force = pd.to_numeric(
            frame.loc[usable, "force_N"], errors="coerce"
        ).to_numpy(float)
    usable_count = int(usable.sum())
    if usable_count < MIN_CURVE_POINTS:
        issues.append(
            f"only {usable_count} usable curve points; "
            f"minimum is {MIN_CURVE_POINTS}"
        )
    if usable_count:
        if np.any(np.diff(time) < -1e-8):
            issues.append("time is not monotonic")
        if np.any(np.diff(displacement) < -1e-8):
            issues.append("displacement is not monotonic")
        if np.ptp(displacement) <= 0.0:
            issues.append("displacement has no positive range")
        if np.ptp(force) <= 0.0:
            issues.append("force has no positive range")
        duplicate_rate = float(frame.duplicated().mean())
        if duplicate_rate > 0.05:
            issues.append(
                f"exact duplicate-row rate {duplicate_rate:.1%} exceeds 5%"
            )
    recommended = float(
        expected["recommended_first_displacement_mm"]
    )
    if usable_count and np.max(displacement) < 0.80 * recommended:
        issues.append(
            "curve does not reach 80% of the recommended first displacement"
        )

    slope = None
    r2 = None
    fit_count = 0
    if usable_count >= MIN_CURVE_POINTS:
        lower = 0.10 * recommended
        upper = 0.70 * recommended
        fit_mask = (
            (displacement >= lower)
            & (displacement <= upper)
        )
        fit_count = int(fit_mask.sum())
        if fit_count < MIN_FIT_POINTS:
            issues.append(
                f"only {fit_count} points in the stiffness-fit window; "
                f"minimum is {MIN_FIT_POINTS}"
            )
        else:
            x = displacement[fit_mask]
            y = force[fit_mask]
            slope_value, intercept = np.polyfit(x, y, 1)
            predicted = slope_value * x + intercept
            residual = float(np.sum((y - predicted) ** 2))
            total = float(np.sum((y - np.mean(y)) ** 2))
            r2_value = 1.0 - residual / total if total > 0.0 else 0.0
            slope = float(slope_value)
            r2 = float(r2_value)
            if slope <= 0.0:
                issues.append("fitted stiffness is not positive")
            if r2 < MIN_FIT_R2:
                issues.append(
                    f"linear stiffness fit R²={r2:.4f} is below "
                    f"{MIN_FIT_R2:.2f}"
                )
    normalized = (
        None
        if slope is None or actual_mass is None
        else slope * target_mass / actual_mass
    )
    controls = None
    required_controls = (
        "material_lot",
        "material_color",
        "machine_id",
        "slicer_profile",
        "drying_protocol",
        "conditioning",
        "fixture_id",
    )
    required_numeric = (
        "nozzle_temperature_C",
        "bed_temperature_C",
        "layer_height_mm",
        "test_temperature_C",
        "test_relative_humidity_percent",
        "support_span_mm",
        "support_roller_diameter_mm",
        "loading_roller_diameter_mm",
        "crosshead_rate_mm_min",
        "fixture_compliance_mm_per_N",
    )
    if all(name in observed_text for name in required_controls) and all(
        name in observed_numeric for name in required_numeric
    ):
        controls = {
            **{name: observed_text[name] for name in required_controls},
            **{name: observed_numeric[name] for name in required_numeric},
        }
    result = PhysicalSpecimenResult(
        specimen_id=expected["specimen_id"],
        pattern=expected["pattern"],
        load_case=expected["load_case"],
        replicate=int(expected["replicate"]),
        status="valid" if not issues else "invalid",
        source_file=str(path.resolve()),
        source_sha256=_sha256(path),
        usable_curve_points=usable_count,
        fit_curve_points=fit_count,
        actual_mass_g=actual_mass,
        target_mass_g=target_mass,
        mass_deviation_fraction=mass_deviation,
        slicer_expected_mass_g=slicer_expected_mass,
        slicer_mass_deviation_fraction=slicer_mass_deviation,
        stiffness_N_per_mm=slope,
        stiffness_fit_r2=r2,
        mass_normalized_stiffness_N_per_mm=normalized,
        issues=tuple(issues),
    )
    return result, controls


def _verify_package_manifest(
    manifest_path: Path | None,
    plan_path: Path,
    study_path: Path,
) -> Dict[str, object]:
    if manifest_path is None:
        return {}
    manifest = json.loads(
        Path(manifest_path).read_text(encoding="utf-8")
    )
    if manifest.get("plan_sha256") != _sha256(plan_path):
        raise ValueError("physical validation plan checksum does not match")
    if manifest.get("study_sha256") != _sha256(study_path):
        raise ValueError("physical validation study checksum does not match")
    manufacturing_path = manifest.get("manufacturing_manifest")
    if manufacturing_path is not None:
        manufacturing_path = Path(str(manufacturing_path))
        if (
            not manufacturing_path.exists()
            or manifest.get("manufacturing_manifest_sha256")
            != _sha256(manufacturing_path)
        ):
            raise ValueError(
                "physical validation manufacturing checksum does not match"
            )
        manufacturing = json.loads(
            manufacturing_path.read_text(encoding="utf-8")
        )
        if (
            manufacturing.get("all_manufacturing_gates_passed")
            is not True
        ):
            raise ValueError("physical validation manufacturing gate failed")
    toolpath_path = manifest.get("toolpath_morphology_manifest")
    if toolpath_path is not None:
        toolpath_path = Path(str(toolpath_path))
        if (
            not toolpath_path.exists()
            or manifest.get("toolpath_morphology_manifest_sha256")
            != _sha256(toolpath_path)
        ):
            raise ValueError(
                "physical validation toolpath checksum does not match"
            )
        toolpath = json.loads(toolpath_path.read_text(encoding="utf-8"))
        if (
            toolpath.get("gates", {}).get(
                "relative_linear_stiffness_ranking_admitted"
            )
            is not True
        ):
            raise ValueError(
                "physical validation toolpath morphology gate failed"
            )
    toolpath_fea_path = manifest.get(
        "toolpath_resolved_bending_manifest"
    )
    if toolpath_fea_path is not None:
        toolpath_fea_path = Path(str(toolpath_fea_path))
        if (
            not toolpath_fea_path.exists()
            or manifest.get(
                "toolpath_resolved_bending_manifest_sha256"
            )
            != _sha256(toolpath_fea_path)
        ):
            raise ValueError(
                "physical validation toolpath FEA checksum does not match"
            )
        toolpath_fea = json.loads(
            toolpath_fea_path.read_text(encoding="utf-8")
        )
        if (
            toolpath_fea.get("gates", {}).get(
                "relative_toolpath_aware_bending_ranking_admitted"
            )
            is not True
        ):
            raise ValueError(
                "physical validation toolpath FEA gate failed"
            )
    fixture_path = manifest.get("fixture_manifest")
    if fixture_path is not None:
        fixture_path = Path(str(fixture_path))
        if (
            not fixture_path.exists()
            or manifest.get("fixture_manifest_sha256")
            != _sha256(fixture_path)
        ):
            raise ValueError(
                "physical validation fixture checksum does not match"
            )
        fixture = json.loads(
            fixture_path.read_text(encoding="utf-8")
        )
        if fixture.get("all_fixture_gates_passed") is not True:
            raise ValueError("physical validation fixture gate failed")
    return manifest


def validate_submission(
    input_dir: Path,
    plan_path: Path = DEFAULT_PACKAGE / "frozen_plan.csv",
    study_path: Path = DEFAULT_STUDY,
    manifest_path: Path | None = DEFAULT_PACKAGE / "manifest.json",
) -> Dict[str, object]:
    input_dir = Path(input_dir)
    plan_path = Path(plan_path)
    study_path = Path(study_path)
    plan = _load_plan(plan_path)
    study = _load_study(study_path)
    package_manifest = _verify_package_manifest(
        manifest_path, plan_path, study_path
    )
    slicer_masses = (
        package_manifest.get("slicer_predicted_deposited_mass_g")
        or {}
    )
    expected = {
        row["specimen_id"]: {
            **row,
            "slicer_expected_mass_g": slicer_masses.get(
                row["pattern"], row["target_actual_mass_g"]
            ),
        }
        for row in plan
    }
    actual = {path.stem: path for path in input_dir.glob("*.csv")}
    unexpected = sorted(set(actual) - set(expected))
    results: List[PhysicalSpecimenResult] = []
    controls: List[Dict[str, object]] = []
    for specimen_id, row in expected.items():
        path = actual.get(specimen_id)
        if path is None:
            results.append(
                _invalid_result(
                    row, None, ["expected specimen CSV is absent"]
                )
            )
            continue
        result, control = _audit_specimen(path, row)
        results.append(result)
        if control is not None:
            controls.append(control)
    global_issues: List[str] = []
    if unexpected:
        global_issues.append(
            "unexpected CSV files: " + ", ".join(unexpected)
        )
    if len(controls) == len(plan):
        text_fields = (
            "material_lot",
            "material_color",
            "machine_id",
            "slicer_profile",
            "drying_protocol",
            "conditioning",
        )
        numeric_fields = (
            "nozzle_temperature_C",
            "bed_temperature_C",
            "layer_height_mm",
        )
        for field in text_fields + numeric_fields:
            values = {control[field] for control in controls}
            if len(values) != 1:
                global_issues.append(
                    f"print control {field} differs across specimens"
                )
        temperatures = [
            float(control["test_temperature_C"])
            for control in controls
        ]
        humidities = [
            float(control["test_relative_humidity_percent"])
            for control in controls
        ]
        if max(temperatures) - min(temperatures) > 2.0:
            global_issues.append(
                "test temperature spread exceeds 2 C"
            )
        if max(humidities) - min(humidities) > 10.0:
            global_issues.append(
                "test humidity spread exceeds 10 percentage points"
            )
        if not global_issues:
            try:
                admit_pattern_study(
                    str(study["pattern"]),
                    temperature_C=float(np.mean(temperatures)),
                    relative_humidity_percent=float(np.mean(humidities)),
                    conditioning=str(controls[0]["conditioning"]),
                )
            except ValueError as exc:
                global_issues.append(
                    f"material service gate rejected measurements: {exc}"
                )
    counts = {
        status: sum(result.status == status for result in results)
        for status in ("valid", "invalid", "missing")
    }
    ready = (
        counts == {"valid": 6, "invalid": 0, "missing": 0}
        and not global_issues
    )
    return {
        "validation_id": f"{study['pattern']}_bending_physical_v1",
        "input_directory": str(input_dir.resolve()),
        "plan_sha256": _sha256(plan_path),
        "study_sha256": _sha256(study_path),
        "ready_for_analysis": ready,
        "counts": counts,
        "unexpected_csv_files": unexpected,
        "global_issues": global_issues,
        "specimens": [asdict(result) for result in results],
    }


def _mean_ci(values: np.ndarray) -> Tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean
    half = float(
        t.ppf(0.975, len(values) - 1)
        * np.std(values, ddof=1)
        / math.sqrt(len(values))
    )
    return mean - half, mean + half


def _bootstrap_ratio(
    custom: np.ndarray, control: np.ndarray
) -> Tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    custom_draws = rng.choice(
        custom, size=(BOOTSTRAP_SAMPLES, len(custom)), replace=True
    ).mean(axis=1)
    control_draws = rng.choice(
        control, size=(BOOTSTRAP_SAMPLES, len(control)), replace=True
    ).mean(axis=1)
    ratios = custom_draws / control_draws
    lower, upper = np.percentile(ratios, [2.5, 97.5])
    return float(lower), float(upper)


def analyze_submission(
    input_dir: Path,
    plan_path: Path = DEFAULT_PACKAGE / "frozen_plan.csv",
    study_path: Path = DEFAULT_STUDY,
    manifest_path: Path | None = DEFAULT_PACKAGE / "manifest.json",
) -> Dict[str, object]:
    audit = validate_submission(
        input_dir, plan_path, study_path, manifest_path
    )
    if not audit["ready_for_analysis"]:
        return {
            "validation_id": audit["validation_id"],
            "status": "awaiting_valid_measurements",
            "audit": audit,
            "result": None,
        }
    plan = _load_plan(plan_path)
    study = _load_study(study_path)
    custom = str(study["pattern"])
    control = str(study["control"])
    specimen_rows = audit["specimens"]
    groups: Dict[str, np.ndarray] = {}
    raw_groups: Dict[str, np.ndarray] = {}
    summaries = {}
    for pattern in (control, custom):
        selected = [
            row for row in specimen_rows if row["pattern"] == pattern
        ]
        normalized = np.array(
            [
                float(row["mass_normalized_stiffness_N_per_mm"])
                for row in selected
            ]
        )
        raw = np.array(
            [float(row["stiffness_N_per_mm"]) for row in selected]
        )
        groups[pattern] = normalized
        raw_groups[pattern] = raw
        lower, upper = _mean_ci(normalized)
        predicted = _plan_prediction(plan, pattern)
        summaries[pattern] = {
            "replicates": len(normalized),
            "raw_mean_stiffness_N_per_mm": float(np.mean(raw)),
            "mass_normalized_mean_stiffness_N_per_mm": float(
                np.mean(normalized)
            ),
            "sample_standard_deviation_N_per_mm": float(
                np.std(normalized, ddof=1)
            ),
            "mean_95pct_t_interval_N_per_mm": [lower, upper],
            "predicted_stiffness_N_per_mm": predicted,
            "measured_to_predicted_scale": float(
                np.mean(normalized) / predicted
            ),
        }
    measured_ratio = float(
        np.mean(groups[custom]) / np.mean(groups[control])
    )
    raw_ratio = float(
        np.mean(raw_groups[custom]) / np.mean(raw_groups[control])
    )
    ratio_lower, ratio_upper = _bootstrap_ratio(
        groups[custom], groups[control]
    )
    predicted_ratio = (
        _plan_prediction(plan, custom)
        / _plan_prediction(plan, control)
    )
    ratio_relative_error = abs(measured_ratio / predicted_ratio - 1.0)
    topology_bias = (
        summaries[custom]["measured_to_predicted_scale"]
        / summaries[control]["measured_to_predicted_scale"]
    )
    if ratio_lower > 1.0 and ratio_relative_error <= 0.20:
        decision = "validated_specialist_advantage"
    elif ratio_lower > 1.0:
        decision = "ranking_confirmed_recalibration_required"
    elif ratio_upper < 1.0:
        decision = "ranking_rejected"
    else:
        decision = "ranking_not_distinguishable"
    common_scale = math.sqrt(
        summaries[custom]["measured_to_predicted_scale"]
        * summaries[control]["measured_to_predicted_scale"]
    )
    result = {
        "decision": decision,
        "custom_pattern": custom,
        "control_pattern": control,
        "group_summaries": summaries,
        "raw_custom_to_control_ratio": raw_ratio,
        "mass_normalized_custom_to_control_ratio": measured_ratio,
        "bootstrap_95pct_ratio_interval": [
            ratio_lower,
            ratio_upper,
        ],
        "predicted_custom_to_control_ratio": predicted_ratio,
        "ratio_relative_error": ratio_relative_error,
        "shared_stiffness_scale_candidate": common_scale,
        "topology_specific_bias_ratio": topology_bias,
        "canonical_material_card_updated": False,
        "interpretation": (
            "A common stiffness scale may inform later material calibration, "
            "but the custom/control ratio is geometry evidence. A topology "
            "mismatch cannot be repaired by scaling one shared elastic modulus."
        ),
    }
    return {
        "validation_id": audit["validation_id"],
        "status": "analyzed",
        "audit": audit,
        "result": result,
    }


def write_analysis(
    input_dir: Path,
    output_dir: Path = DEFAULT_ANALYSIS,
    plan_path: Path = DEFAULT_PACKAGE / "frozen_plan.csv",
    study_path: Path = DEFAULT_STUDY,
    manifest_path: Path | None = DEFAULT_PACKAGE / "manifest.json",
) -> Dict[str, object]:
    study = _load_study(study_path)
    payload = analyze_submission(
        input_dir, plan_path, study_path, manifest_path
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    specimen_rows = payload["audit"]["specimens"]
    flat = [
        {
            **{key: value for key, value in row.items() if key != "issues"},
            "issues": " | ".join(row["issues"]),
        }
        for row in specimen_rows
    ]
    with (output_dir / "specimen_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    counts = payload["audit"]["counts"]
    lines = [
        f"# {study['pattern']} physical validation",
        "",
        (
            "## Analysis complete"
            if payload["status"] == "analyzed"
            else "## Awaiting valid measurements"
        ),
        "",
        f"Valid: {counts['valid']}/6; invalid: {counts['invalid']}; "
        f"missing: {counts['missing']}.",
    ]
    if payload["audit"]["global_issues"]:
        lines.extend(
            [
                "",
                "Global issues:",
                *[
                    f"- {issue}"
                    for issue in payload["audit"]["global_issues"]
                ],
            ]
        )
    if payload["result"] is not None:
        result = payload["result"]
        lower, upper = result["bootstrap_95pct_ratio_interval"]
        lines.extend(
            [
                "",
                f"Decision: **{result['decision']}**.",
                "",
                f"Predicted custom/control ratio: "
                f"{result['predicted_custom_to_control_ratio']:.3f}.",
                f"Measured mass-normalized ratio: "
                f"{result['mass_normalized_custom_to_control_ratio']:.3f} "
                f"(bootstrap 95% interval {lower:.3f}–{upper:.3f}).",
                f"Relative ratio error: "
                f"{result['ratio_relative_error']:.1%}.",
                "",
                "The canonical material card was not changed. The shared "
                "stiffness scale is retained only as a future calibration "
                "candidate; topology-specific error remains geometry evidence.",
            ]
        )
    lines.extend(
        [
            "",
            "## Stable gates",
            "",
            f"- Exactly six frozen specimen IDs and at least "
            f"{MIN_CURVE_POINTS} usable points per curve.",
            f"- Linear-region fit R² at least {MIN_FIT_R2:.2f}.",
            "- Actual mass within 15% of the exact-mass target.",
            "- 40.00 mm support span, 4.00 mm steel rollers, registered "
            "crosshead rate, and fixture compliance no greater than "
            f"{MAXIMUM_FIXTURE_COMPLIANCE_MM_N:.4f} mm/N.",
            "- Identical spool, machine, slicer, drying, and print settings.",
            "- Test environment must remain inside the admitted Bambu PLA "
            "service condition.",
            "- Three independent replicates per topology.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture and analyze the custom-pattern physical test"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--input")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--study", default=str(DEFAULT_STUDY))
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--output", default=str(DEFAULT_ANALYSIS))
    parser.add_argument(
        "--maximum-toolpath-mass-ratio-error",
        type=float,
        default=MAXIMUM_TOOLPATH_MASS_RATIO_ERROR,
    )
    parser.add_argument(
        "--toolpath-resolutions",
        default=",".join(
            str(value) for value in DEFAULT_ANALYSIS_RESOLUTIONS
        ),
    )
    args = parser.parse_args()
    if args.initialize:
        payload = write_capture_package(
            Path(args.package),
            Path(args.plan),
            Path(args.study),
            maximum_toolpath_mass_ratio_error=(
                args.maximum_toolpath_mass_ratio_error
            ),
            toolpath_resolutions=tuple(
                int(value)
                for value in args.toolpath_resolutions.split(",")
            ),
        )
        print(
            f"physical capture package: {payload['specimen_count']} "
            f"specimens; status={payload['status']}; "
            f"output: {Path(args.package).resolve()}"
        )
        return
    package = Path(args.package)
    payload = write_analysis(
        Path(args.input),
        Path(args.output),
        package / "frozen_plan.csv",
        Path(args.study),
        package / "manifest.json",
    )
    print(
        f"physical validation: status={payload['status']}; "
        f"valid={payload['audit']['counts']['valid']}/6; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )
    if payload["status"] != "analyzed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
