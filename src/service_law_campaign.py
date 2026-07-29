from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_service_envelope_v3"
    / "missing_law_experiment_plan.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/bambu_pla_service_law_campaign"
RANDOMIZATION_SEED = 20260729
FATIGUE_RUNOUT_CYCLES = 100_000
MIN_CURVE_POINTS = 20
MIN_RELAXATION_POINTS = 25

STAGE_LAWS = {
    1: "temperature_frequency_shift_and_dynamic_modulus",
    2: "long_time_stress_relaxation",
    3: "humidity_conditioned_mechanical_retention",
    4: "directional_tension_tension_fatigue",
    5: "controlled_aging_mechanical_retention",
}

CAPTURE_FIELDS = (
    "test_id",
    "stage",
    "method",
    "orientation",
    "replicate",
    "material",
    "material_lot",
    "material_color",
    "machine_id",
    "print_profile",
    "drying_protocol",
    "conditioning",
    "ambient_temperature_C",
    "ambient_relative_humidity_percent",
    "test_temperature_C",
    "frequency_Hz",
    "target_strain",
    "time_s",
    "cycle_count",
    "storage_modulus_MPa",
    "loss_modulus_MPa",
    "tan_delta",
    "engineering_strain",
    "engineering_stress_MPa",
    "directional_UTS_MPa",
    "maximum_stress_fraction_of_directional_UTS",
    "maximum_stress_MPa",
    "minimum_stress_MPa",
    "secant_stiffness_MPa",
    "hysteresis_energy_MJ_m3",
    "pre_condition_mass_g",
    "post_condition_mass_g",
    "equilibrium_mass_change_percent_per_24h",
    "outcome",
    "failure_mode",
    "exposure_elapsed_h",
    "exposure_cycles",
    "surface_observation",
    "operator_notes",
)


@dataclass(frozen=True)
class ServiceSpecimenAudit:
    test_id: str
    stage: int
    law: str
    status: str
    source_file: str | None
    source_sha256: str | None
    source_rows: int
    usable_measurement_points: int
    issues: tuple[str, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_plan(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 90:
        raise ValueError(f"expected 90 service-law specimens, found {len(rows)}")
    ids = [row["test_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("service-law plan contains duplicate test IDs")
    expected_counts = {1: 6, 2: 18, 3: 12, 4: 30, 5: 24}
    observed_counts = {
        stage: sum(int(row["stage"]) == stage for row in rows)
        for stage in expected_counts
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"service-law stage counts changed: {observed_counts}"
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _constant_template_values(row: Mapping[str, str]) -> Dict[str, object]:
    return {
        "test_id": row["test_id"],
        "stage": int(row["stage"]),
        "method": row["method"],
        "orientation": row["orientation"],
        "replicate": int(row["replicate"]),
        "material": "Bambu PLA Basic",
        "conditioning": row["conditioning"],
        "target_strain": 0.005 if int(row["stage"]) == 2 else "",
        "maximum_stress_fraction_of_directional_UTS": (
            row.get("maximum_stress_fraction_of_directional_UTS", "")
            if int(row["stage"]) == 4
            else ""
        ),
    }


def _template_rows(row: Mapping[str, str]) -> List[Dict[str, object]]:
    stage = int(row["stage"])
    base = {field: "" for field in CAPTURE_FIELDS}
    base.update(_constant_template_values(row))
    if stage == 1:
        temperatures = [
            float(value) for value in row["temperatures_C"].split(",")
        ]
        frequencies = [
            float(value) for value in row["frequencies_Hz"].split(",")
        ]
        return [
            {
                **base,
                "test_temperature_C": temperature,
                "frequency_Hz": frequency,
            }
            for temperature in temperatures
            for frequency in frequencies
        ]
    if stage == 2:
        temperature = float(row["temperatures_C"])
        times = np.geomspace(1.0, 14_400.0, MIN_RELAXATION_POINTS)
        return [
            {
                **base,
                "test_temperature_C": temperature,
                "time_s": float(time),
            }
            for time in times
        ]
    if stage == 4:
        return [
            {
                **base,
                "test_temperature_C": 23.0,
                "frequency_Hz": 5.0,
                "cycle_count": cycle,
            }
            for cycle in (1, 10, 100, 1_000, 10_000, FATIGUE_RUNOUT_CYCLES)
        ]
    if stage == 3:
        base["test_temperature_C"] = 25.0
    elif stage == 5:
        base["test_temperature_C"] = 23.0
        if row["conditioning"] == "unaged_control":
            base["exposure_elapsed_h"] = 0.0
            base["exposure_cycles"] = 0
        elif row["conditioning"] in {"UVA_500h", "40C_75RH_500h"}:
            base["exposure_elapsed_h"] = 500.0
        else:
            base["exposure_cycles"] = 100
    return [base]


def write_campaign(
    output_dir: Path = DEFAULT_OUTPUT,
    plan_path: Path = DEFAULT_PLAN,
    seed: int = RANDOMIZATION_SEED,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    plan_path = Path(plan_path)
    rows = _read_plan(plan_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_plan = output_dir / "frozen_plan.csv"
    frozen_plan.write_bytes(plan_path.read_bytes())
    captures = output_dir / "capture_templates"
    captures.mkdir(parents=True, exist_ok=True)
    for row in rows:
        with (captures / f"{row['test_id']}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=CAPTURE_FIELDS)
            writer.writeheader()
            writer.writerows(_template_rows(row))

    rng = random.Random(seed)
    run_rows: List[Dict[str, object]] = []
    global_order = 0
    for stage in sorted(STAGE_LAWS):
        selected = [dict(row) for row in rows if int(row["stage"]) == stage]
        rng.shuffle(selected)
        for stage_order, row in enumerate(selected, start=1):
            global_order += 1
            run_rows.append(
                {
                    "global_randomized_order": global_order,
                    "randomized_order_within_stage": stage_order,
                    "print_block": 1 + (global_order - 1) // 6,
                    **row,
                }
            )
    _write_csv(output_dir / "randomized_run_order.csv", run_rows)

    manifest = {
        "campaign_id": "bambu_pla_basic_service_laws_v1",
        "material": "Bambu PLA Basic",
        "status": "awaiting_measurements",
        "randomization_seed": seed,
        "specimen_count": len(rows),
        "stage_counts": {
            str(stage): sum(int(row["stage"]) == stage for row in rows)
            for stage in STAGE_LAWS
        },
        "stage_laws": {str(key): value for key, value in STAGE_LAWS.items()},
        "source_plan": str(plan_path.resolve()),
        "source_plan_sha256": _sha256(plan_path),
        "frozen_plan": str(frozen_plan.resolve()),
        "frozen_plan_sha256": _sha256(frozen_plan),
        "capture_template_count": len(rows),
        "capture_fields": list(CAPTURE_FIELDS),
        "release_policy": (
            "a law remains unavailable until every specimen in its frozen "
            "stage passes the stage-specific ingestion gate"
        ),
        "fatigue_runout_cycles": FATIGUE_RUNOUT_CYCLES,
        "minimum_curve_points": MIN_CURVE_POINTS,
        "minimum_relaxation_points": MIN_RELAXATION_POINTS,
        "external_source_screening": [
            {
                "title": (
                    "Investigation of the Effect of Reinforcement Type and "
                    "Ratio on the Fatigue Performance of PLA Matrix Polymer "
                    "Composites Produced by Additive Manufacturing"
                ),
                "url": (
                    "https://dergipark.org.tr/en/download/"
                    "article-file/5497521"
                ),
                "material_identity_gate_passed": False,
                "reason": (
                    "the X1 Carbon was only the printer; the study dried, "
                    "mixed, granulated, and twin-screw-extruded its own PLA "
                    "and composite filaments, so its S-N curve is not Bambu "
                    "PLA Basic evidence"
                ),
            }
        ],
    }
    (output_dir / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    protocol = """# Bambu PLA Basic service-law campaign

This package converts the 90-specimen missing-law plan into checksum-frozen,
stage-specific capture files. Templates contain only planned coordinates or
checkpoints; blank response cells are never interpreted as measurements.

## Stage release gates

1. DMA: all 6 specimens, all 18 planned temperature-frequency points, positive
   storage/loss modulus, and tan-delta consistency.
2. Stress relaxation: all 18 specimens, at least 25 increasing time points
   spanning 1 s to 4 h at 0.5% strain.
3. Humidity: all 12 specimens, equilibrium mass criterion plus raw tensile
   curves with at least 20 points.
4. Fatigue: all 30 specimens, R=0.1 and 5 Hz controls, cycle history, failure
   or runout status, and at least 100,000 cycles for a runout.
5. Aging: all 24 specimens, completed exposure controls plus matched raw
   tensile curves and mass/surface observations.

Use one material lot and retain spool position, print block, specimen failures,
and censored fatigue runouts. Do not replace runouts with invented failures or
fit an environmental correction from mass uptake alone.

After measurements, put the filled files in a separate directory and run:

```bash
python -m src.service_law_campaign --input path/to/filled-csvs
```
"""
    (output_dir / "protocol.md").write_text(protocol, encoding="utf-8")
    return manifest


def _strict_text(
    frame: pd.DataFrame, column: str
) -> tuple[str | None, str | None]:
    if column not in frame:
        return None, f"missing column {column}"
    values = frame[column].fillna("").astype(str).str.strip()
    if (values == "").any():
        return None, f"{column} must be repeated on every row"
    unique = values.unique()
    if len(unique) != 1:
        return None, f"{column} must be constant within a specimen"
    return str(unique[0]), None


def _strict_number(
    frame: pd.DataFrame,
    column: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[float | None, str | None]:
    if column not in frame:
        return None, f"missing column {column}"
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        return None, f"{column} must be numeric on every row"
    unique = np.unique(values.to_numpy(float))
    if len(unique) != 1:
        return None, f"{column} must be constant within a specimen"
    value = float(unique[0])
    if positive and value <= 0.0:
        return None, f"{column} must be positive"
    if nonnegative and value < 0.0:
        return None, f"{column} must be nonnegative"
    return value, None


def _finite_rows(
    frame: pd.DataFrame, columns: Sequence[str]
) -> tuple[pd.DataFrame, List[str]]:
    missing = [column for column in columns if column not in frame]
    if missing:
        return frame.iloc[0:0], [
            "missing measurement columns: " + ", ".join(missing)
        ]
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    usable = numeric.loc[np.isfinite(numeric.to_numpy(float)).all(axis=1)]
    return usable, []


def _curve_checks(
    usable: pd.DataFrame,
    *,
    minimum_points: int = MIN_CURVE_POINTS,
    minimum_strain_range: float = 0.02,
) -> List[str]:
    issues: List[str] = []
    if len(usable) < minimum_points:
        issues.append(
            f"only {len(usable)} usable tensile points; "
            f"minimum is {minimum_points}"
        )
        return issues
    strain = usable["engineering_strain"].to_numpy(float)
    stress = usable["engineering_stress_MPa"].to_numpy(float)
    if np.any(np.diff(strain) < -1.0e-8):
        issues.append("engineering strain is not monotonic")
    if float(np.ptp(strain)) < minimum_strain_range:
        issues.append(
            f"engineering strain range is below {minimum_strain_range:g}"
        )
    if np.any(stress < -0.1):
        issues.append("engineering stress contains negative values")
    if float(np.nanmax(stress)) <= 1.0:
        issues.append("engineering stress never exceeds 1 MPa")
    return issues


def _audit_stage_one(
    frame: pd.DataFrame, expected: Mapping[str, str]
) -> tuple[int, List[str]]:
    columns = (
        "test_temperature_C",
        "frequency_Hz",
        "storage_modulus_MPa",
        "loss_modulus_MPa",
        "tan_delta",
    )
    usable, issues = _finite_rows(frame, columns)
    temperatures = [
        float(value) for value in expected["temperatures_C"].split(",")
    ]
    frequencies = [
        float(value) for value in expected["frequencies_Hz"].split(",")
    ]
    expected_pairs = {
        (temperature, frequency)
        for temperature in temperatures
        for frequency in frequencies
    }
    if len(usable) != len(expected_pairs):
        issues.append(
            f"expected {len(expected_pairs)} complete DMA points, "
            f"found {len(usable)}"
        )
        return len(usable), issues
    observed_pairs = {
        (
            float(row["test_temperature_C"]),
            float(row["frequency_Hz"]),
        )
        for _, row in usable.iterrows()
    }
    if observed_pairs != expected_pairs:
        issues.append("DMA temperature-frequency grid differs from the plan")
    storage = usable["storage_modulus_MPa"].to_numpy(float)
    loss = usable["loss_modulus_MPa"].to_numpy(float)
    tan_delta = usable["tan_delta"].to_numpy(float)
    if np.any(storage <= 0.0) or np.any(loss <= 0.0) or np.any(tan_delta <= 0.0):
        issues.append("DMA moduli and tan delta must be positive")
    predicted = loss / storage
    relative_error = np.abs(tan_delta / predicted - 1.0)
    if np.any(relative_error > 0.05):
        issues.append("tan delta differs from loss/storage by more than 5%")
    return len(usable), issues


def _audit_stage_two(
    frame: pd.DataFrame, expected: Mapping[str, str]
) -> tuple[int, List[str]]:
    usable, issues = _finite_rows(
        frame,
        ("time_s", "target_strain", "engineering_stress_MPa"),
    )
    if len(usable) < MIN_RELAXATION_POINTS:
        issues.append(
            f"only {len(usable)} relaxation points; "
            f"minimum is {MIN_RELAXATION_POINTS}"
        )
        return len(usable), issues
    time = usable["time_s"].to_numpy(float)
    strain = usable["target_strain"].to_numpy(float)
    stress = usable["engineering_stress_MPa"].to_numpy(float)
    if np.any(np.diff(time) <= 0.0):
        issues.append("relaxation time must increase strictly")
    if float(np.min(time)) > 1.05 or float(np.max(time)) < 14_000.0:
        issues.append("relaxation record must span approximately 1 s to 4 h")
    if np.any((strain < 0.00475) | (strain > 0.00525)):
        issues.append("relaxation strain must remain within 0.5% +/-5%")
    if np.any(stress <= 0.0):
        issues.append("relaxation stress must remain positive")
    if stress[-1] > stress[0] * 1.05:
        issues.append("final relaxation stress exceeds initial stress by >5%")
    temperature, error = _strict_number(frame, "test_temperature_C")
    if error:
        issues.append(error)
    elif not math.isclose(
        float(temperature), float(expected["temperatures_C"]), abs_tol=0.5
    ):
        issues.append("relaxation temperature differs from the frozen plan")
    return len(usable), issues


def _audit_stage_three(
    frame: pd.DataFrame, expected: Mapping[str, str]
) -> tuple[int, List[str]]:
    usable, issues = _finite_rows(
        frame, ("engineering_strain", "engineering_stress_MPa")
    )
    issues.extend(_curve_checks(usable))
    for field in ("pre_condition_mass_g", "post_condition_mass_g"):
        _, error = _strict_number(frame, field, positive=True)
        if error:
            issues.append(error)
    equilibrium, error = _strict_number(
        frame,
        "equilibrium_mass_change_percent_per_24h",
        nonnegative=True,
    )
    if error:
        issues.append(error)
    elif float(equilibrium) > 0.10:
        issues.append("conditioning mass has not reached the 0.10%/24 h gate")
    temperature, error = _strict_number(frame, "test_temperature_C")
    if error:
        issues.append(error)
    elif not math.isclose(float(temperature), 25.0, abs_tol=1.0):
        issues.append("humidity tensile test temperature is not 25 C +/-1 C")
    return len(usable), issues


def _audit_stage_four(
    frame: pd.DataFrame, expected: Mapping[str, str]
) -> tuple[int, List[str]]:
    usable, issues = _finite_rows(
        frame,
        (
            "cycle_count",
            "secant_stiffness_MPa",
            "hysteresis_energy_MJ_m3",
        ),
    )
    if len(usable) < 3:
        issues.append("fatigue history needs at least three cycle checkpoints")
        return len(usable), issues
    cycles = usable["cycle_count"].to_numpy(float)
    stiffness = usable["secant_stiffness_MPa"].to_numpy(float)
    energy = usable["hysteresis_energy_MJ_m3"].to_numpy(float)
    if np.any(np.diff(cycles) <= 0.0):
        issues.append("fatigue cycle checkpoints must increase strictly")
    if np.any(stiffness <= 0.0) or np.any(energy < 0.0):
        issues.append("fatigue stiffness must be positive and energy nonnegative")
    expected_fraction = float(
        expected["maximum_stress_fraction_of_directional_UTS"]
    )
    fraction, error = _strict_number(
        frame,
        "maximum_stress_fraction_of_directional_UTS",
        positive=True,
    )
    if error:
        issues.append(error)
    elif not math.isclose(float(fraction), expected_fraction, abs_tol=0.005):
        issues.append("fatigue stress fraction differs from the frozen plan")
    uts, error = _strict_number(frame, "directional_UTS_MPa", positive=True)
    if error:
        issues.append(error)
    maximum, error = _strict_number(frame, "maximum_stress_MPa", positive=True)
    if error:
        issues.append(error)
    minimum, error = _strict_number(frame, "minimum_stress_MPa", positive=True)
    if error:
        issues.append(error)
    if uts is not None and maximum is not None:
        if not math.isclose(
            maximum / uts, expected_fraction, rel_tol=0.02, abs_tol=0.005
        ):
            issues.append("maximum stress is not the planned UTS fraction")
    if minimum is not None and maximum is not None:
        if not math.isclose(minimum / maximum, 0.1, abs_tol=0.01):
            issues.append("fatigue stress ratio is not R=0.1 +/-0.01")
    frequency, error = _strict_number(frame, "frequency_Hz", positive=True)
    if error:
        issues.append(error)
    elif not math.isclose(float(frequency), 5.0, abs_tol=0.25):
        issues.append("fatigue frequency is not 5 Hz +/-0.25 Hz")
    temperature, error = _strict_number(frame, "test_temperature_C")
    if error:
        issues.append(error)
    elif not math.isclose(float(temperature), 23.0, abs_tol=1.0):
        issues.append("fatigue temperature is not 23 C +/-1 C")
    outcome, error = _strict_text(frame, "outcome")
    if error:
        issues.append(error)
    elif outcome not in {"failure", "runout"}:
        issues.append("fatigue outcome must be failure or runout")
    elif outcome == "runout" and float(cycles[-1]) < FATIGUE_RUNOUT_CYCLES:
        issues.append(
            f"runout must reach at least {FATIGUE_RUNOUT_CYCLES} cycles"
        )
    return len(usable), issues


def _audit_stage_five(
    frame: pd.DataFrame, expected: Mapping[str, str]
) -> tuple[int, List[str]]:
    usable, issues = _finite_rows(
        frame, ("engineering_strain", "engineering_stress_MPa")
    )
    issues.extend(_curve_checks(usable))
    for field in ("pre_condition_mass_g", "post_condition_mass_g"):
        _, error = _strict_number(frame, field, positive=True)
        if error:
            issues.append(error)
    observation, error = _strict_text(frame, "surface_observation")
    if error:
        issues.append(error)
    elif not observation:
        issues.append("surface observation is empty")
    exposure = expected["conditioning"]
    temperature, error = _strict_number(frame, "test_temperature_C")
    if error:
        issues.append(error)
    elif not math.isclose(float(temperature), 23.0, abs_tol=1.0):
        issues.append("post-aging tensile temperature is not 23 C +/-1 C")
    if exposure == "unaged_control":
        elapsed, error = _strict_number(
            frame, "exposure_elapsed_h", nonnegative=True
        )
        if error:
            issues.append(error)
        elif not math.isclose(float(elapsed), 0.0, abs_tol=0.1):
            issues.append("unaged control must have zero exposure hours")
    elif exposure in {"UVA_500h", "40C_75RH_500h"}:
        elapsed, error = _strict_number(
            frame, "exposure_elapsed_h", nonnegative=True
        )
        if error:
            issues.append(error)
        elif float(elapsed) < 500.0:
            issues.append("aging exposure did not reach 500 h")
    else:
        cycles, error = _strict_number(
            frame, "exposure_cycles", nonnegative=True
        )
        if error:
            issues.append(error)
        elif float(cycles) < 100.0:
            issues.append("thermal-cycle exposure did not reach 100 cycles")
    return len(usable), issues


def _has_any_measurement(frame: pd.DataFrame, stage: int) -> bool:
    measurement_columns = {
        1: ("storage_modulus_MPa", "loss_modulus_MPa", "tan_delta"),
        2: ("engineering_stress_MPa",),
        3: ("engineering_stress_MPa",),
        4: ("secant_stiffness_MPa", "outcome"),
        5: ("engineering_stress_MPa",),
    }[stage]
    for column in measurement_columns:
        if column not in frame:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        if (values != "").any():
            return True
    return False


def _audit_file(
    path: Path, expected: Mapping[str, str]
) -> ServiceSpecimenAudit:
    stage = int(expected["stage"])
    issues: List[str] = []
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return ServiceSpecimenAudit(
            expected["test_id"],
            stage,
            STAGE_LAWS[stage],
            "invalid",
            str(path.resolve()),
            _sha256(path),
            0,
            0,
            (f"CSV parse failed: {exc}",),
        )
    if len(frame) == 0:
        return ServiceSpecimenAudit(
            expected["test_id"],
            stage,
            STAGE_LAWS[stage],
            "awaiting_measurements",
            str(path.resolve()),
            _sha256(path),
            0,
            0,
            ("capture file contains no rows",),
        )
    for field, expected_value in (
        ("test_id", expected["test_id"]),
        ("method", expected["method"]),
        ("orientation", expected["orientation"]),
        ("conditioning", expected["conditioning"]),
        ("material", "Bambu PLA Basic"),
    ):
        observed, error = _strict_text(frame, field)
        if error:
            issues.append(error)
        elif observed != expected_value:
            issues.append(
                f"{field}={observed!r} differs from {expected_value!r}"
            )
    for field, expected_value in (
        ("stage", stage),
        ("replicate", int(expected["replicate"])),
    ):
        observed, error = _strict_number(frame, field)
        if error:
            issues.append(error)
        elif not math.isclose(float(observed), float(expected_value)):
            issues.append(f"{field} differs from the frozen plan")
    for field in (
        "material_lot",
        "material_color",
        "machine_id",
        "print_profile",
        "drying_protocol",
    ):
        _, error = _strict_text(frame, field)
        if error:
            issues.append(error)
    ambient_temperature, error = _strict_number(
        frame, "ambient_temperature_C"
    )
    if error:
        issues.append(error)
    elif not 10.0 <= float(ambient_temperature) <= 35.0:
        issues.append("ambient temperature is outside 10-35 C")
    ambient_humidity, error = _strict_number(
        frame, "ambient_relative_humidity_percent"
    )
    if error:
        issues.append(error)
    elif not 0.0 <= float(ambient_humidity) <= 100.0:
        issues.append("ambient relative humidity is outside 0-100%")

    audit_functions = {
        1: _audit_stage_one,
        2: _audit_stage_two,
        3: _audit_stage_three,
        4: _audit_stage_four,
        5: _audit_stage_five,
    }
    usable, stage_issues = audit_functions[stage](frame, expected)
    issues.extend(stage_issues)
    any_measurement = _has_any_measurement(frame, stage)
    status = (
        "valid"
        if not issues
        else "invalid" if any_measurement else "awaiting_measurements"
    )
    return ServiceSpecimenAudit(
        expected["test_id"],
        stage,
        STAGE_LAWS[stage],
        status,
        str(path.resolve()),
        _sha256(path),
        len(frame),
        usable,
        tuple(issues),
    )


def validate_submission(
    input_dir: Path,
    plan_path: Path = DEFAULT_OUTPUT / "frozen_plan.csv",
    manifest_path: Path | None = DEFAULT_OUTPUT / "campaign_manifest.json",
) -> Dict[str, object]:
    input_dir = Path(input_dir)
    plan_path = Path(plan_path)
    rows = _read_plan(plan_path)
    if manifest_path is not None:
        manifest = json.loads(
            Path(manifest_path).read_text(encoding="utf-8")
        )
        if manifest.get("frozen_plan_sha256") != _sha256(plan_path):
            raise ValueError("frozen service-law plan checksum changed")
    expected = {row["test_id"]: row for row in rows}
    actual = {path.stem: path for path in input_dir.glob("*.csv")}
    unexpected = sorted(set(actual) - set(expected))
    audits: List[ServiceSpecimenAudit] = []
    for test_id, row in expected.items():
        path = actual.get(test_id)
        if path is None:
            stage = int(row["stage"])
            audits.append(
                ServiceSpecimenAudit(
                    test_id,
                    stage,
                    STAGE_LAWS[stage],
                    "missing",
                    None,
                    None,
                    0,
                    0,
                    ("expected capture CSV is absent",),
                )
            )
        else:
            audits.append(_audit_file(path, row))
    counts = {
        status: sum(audit.status == status for audit in audits)
        for status in ("valid", "invalid", "awaiting_measurements", "missing")
    }
    stage_status: Dict[str, Dict[str, object]] = {}
    for stage, law in STAGE_LAWS.items():
        selected = [audit for audit in audits if audit.stage == stage]
        stage_counts = {
            status: sum(audit.status == status for audit in selected)
            for status in (
                "valid",
                "invalid",
                "awaiting_measurements",
                "missing",
            )
        }
        if stage_counts["valid"] == len(selected):
            status = "ready_for_calibration"
        elif stage_counts["invalid"] > 0:
            status = "invalid_submission"
        else:
            status = "awaiting_measurements"
        stage_status[str(stage)] = {
            "law": law,
            "status": status,
            "planned_specimens": len(selected),
            "counts": stage_counts,
            "law_released": status == "ready_for_calibration",
        }
    all_ready = all(
        item["law_released"] for item in stage_status.values()
    )
    any_invalid = any(
        item["status"] == "invalid_submission"
        for item in stage_status.values()
    )
    overall_status = (
        "ready_for_service_law_calibration"
        if all_ready
        else "invalid_submission" if any_invalid else "awaiting_measurements"
    )
    return {
        "campaign_id": "bambu_pla_basic_service_laws_v1",
        "status": overall_status,
        "counts": counts,
        "unexpected_csv_files": unexpected,
        "stage_status": stage_status,
        "all_laws_released": all_ready,
        "specimens": [asdict(audit) for audit in audits],
    }


def write_validation(
    input_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT / "ingestion",
    plan_path: Path = DEFAULT_OUTPUT / "frozen_plan.csv",
    manifest_path: Path | None = DEFAULT_OUTPUT / "campaign_manifest.json",
) -> Dict[str, object]:
    payload = validate_submission(input_dir, plan_path, manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    flat = [
        {
            **{key: value for key, value in row.items() if key != "issues"},
            "issues": " | ".join(row["issues"]),
        }
        for row in payload["specimens"]
    ]
    _write_csv(output_dir / "specimen_audit.csv", flat)
    lines = [
        "# Bambu PLA Basic service-law ingestion",
        "",
        f"Status: **{payload['status']}**.",
        "",
        "| Stage | Law | Valid | Invalid | Awaiting | Missing | Released |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for stage, item in payload["stage_status"].items():
        counts = item["counts"]
        lines.append(
            f"| {stage} | {item['law']} | {counts['valid']} | "
            f"{counts['invalid']} | {counts['awaiting_measurements']} | "
            f"{counts['missing']} | {'yes' if item['law_released'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "No service law is written or admitted from incomplete stages. "
            "Fatigue runouts remain right-censored observations and are not "
            "converted into invented failures.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or validate the Bambu PLA service-law campaign"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.input is None:
        manifest = write_campaign(args.output, args.plan)
        payload = write_validation(
            args.output / "capture_templates",
            args.output / "ingestion",
            args.output / "frozen_plan.csv",
            args.output / "campaign_manifest.json",
        )
        print(
            f"service-law campaign: {manifest['specimen_count']} specimens; "
            f"status={payload['status']}; output={args.output.resolve()}"
        )
        return
    payload = write_validation(
        args.input,
        args.output / "ingestion",
        args.output / "frozen_plan.csv",
        args.output / "campaign_manifest.json",
    )
    print(
        f"service-law ingestion: status={payload['status']}; "
        f"valid={payload['counts']['valid']}/90; "
        f"output={(args.output / 'ingestion').resolve()}"
    )
    if payload["status"] == "invalid_submission":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
