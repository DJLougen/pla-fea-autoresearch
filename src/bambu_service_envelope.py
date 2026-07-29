from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/public/bambu-pla-basic-tds-v3"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_service_envelope_v3"
)
TDS_FILENAME = "Bambu_PLA_Basic_TDS_V3.0.pdf"
TDS_SHA256 = (
    "44c00d8398cba7fbdb8fdcd13886ab904f78826d9eefa531a95118b6891201b3"
)
TDS_URL = (
    "https://store.bblcdn.com/6bd7b337c4234e7d8a72997824d01bcc.pdf"
)
DMA_REFERENCE = (
    PROJECT_ROOT / "data/reported/bambu_pla_published_dma_220C"
)
WAVE_REFERENCE = (
    PROJECT_ROOT / "data/reported/bambu_pla_viscoelastic_wave_2026"
)
SHORT_TIME_RELAXATION_REFERENCE = (
    PROJECT_ROOT / "data/reported/bambu_pla_short_time_relaxation_18C"
)
SHORT_TIME_RELAXATION_MINIMUM_S = 1.0 / (2.0 * 3.141592653589793 * 25_000.0)
SHORT_TIME_RELAXATION_MAXIMUM_S = 1.0 / (2.0 * 3.141592653589793)


@dataclass(frozen=True)
class ServiceCondition:
    scenario_id: str
    temperature_C: float
    load_kind: str
    duration_s: float = 1.0
    cycles: int = 1
    frequency_Hz: float | None = None
    relative_humidity_percent: float = 20.0
    conditioning: str = "as_printed_dry"


@dataclass(frozen=True)
class ServiceDecision:
    scenario_id: str
    status: str
    admitted_model: str
    evidence: str
    reason: str
    maximum_temperature_guardrail_C: float
    requires_new_measurement: bool


MANUFACTURER_PROPERTIES = (
    {
        "property": "density",
        "value": 1.24,
        "units": "g/cm3",
        "test_method": "ISO 1183",
        "interpretation": "material density",
    },
    {
        "property": "glass_transition_temperature",
        "value": 60.0,
        "units": "degC",
        "test_method": "DSC, 10 degC/min",
        "interpretation": "transition marker, not allowable service temperature",
    },
    {
        "property": "vicat_softening_temperature",
        "value": 57.0,
        "units": "degC",
        "test_method": "ISO 306 / GB/T 1633",
        "interpretation": "softening guardrail",
    },
    {
        "property": "heat_deflection_temperature_1p8MPa",
        "value": 54.0,
        "units": "degC",
        "test_method": "ISO 75 at 1.8 MPa",
        "interpretation": "lowest manufacturer structural-temperature guardrail",
    },
    {
        "property": "heat_deflection_temperature_0p45MPa",
        "value": 57.0,
        "units": "degC",
        "test_method": "ISO 75 at 0.45 MPa",
        "interpretation": "load-dependent deflection threshold",
    },
    {
        "property": "saturated_water_absorption_25C_55RH",
        "value": 0.43,
        "units": "percent",
        "test_method": "25 degC, 55% RH",
        "interpretation": (
            "mass uptake only; no mechanical-property correction is published"
        ),
    },
    {
        "property": "youngs_modulus_XY",
        "value": 2580.0,
        "uncertainty": 220.0,
        "units": "MPa",
        "test_method": "ISO 527 / GB/T 1040",
        "interpretation": "annealed and dried manufacturer specimen",
    },
    {
        "property": "youngs_modulus_Z",
        "value": 2060.0,
        "uncertainty": 170.0,
        "units": "MPa",
        "test_method": "ISO 527 / GB/T 1040",
        "interpretation": "annealed and dried manufacturer specimen",
    },
)


DEFAULT_SCENARIOS = (
    ServiceCondition(
        "room_monotonic",
        23.0,
        "monotonic",
        duration_s=30.0,
    ),
    ServiceCondition(
        "published_dma_point",
        30.0,
        "small_amplitude_dynamic",
        duration_s=1.0,
        frequency_Hz=1.0,
    ),
    ServiceCondition(
        "broadband_wave_point",
        18.0,
        "small_amplitude_dynamic",
        duration_s=1.0,
        frequency_Hz=1_000.0,
        relative_humidity_percent=40.0,
        conditioning="ambient_lab_18C_40RH",
    ),
    ServiceCondition(
        "short_time_relaxation_10ms",
        18.0,
        "short_time_relaxation",
        duration_s=0.01,
        relative_humidity_percent=40.0,
        conditioning="ambient_lab_18C_40RH",
    ),
    ServiceCondition(
        "warm_static_40C",
        40.0,
        "monotonic",
        duration_s=30.0,
    ),
    ServiceCondition(
        "manufacturer_hdt_54C",
        54.0,
        "monotonic",
        duration_s=30.0,
    ),
    ServiceCondition(
        "manufacturer_tg_60C",
        60.0,
        "monotonic",
        duration_s=30.0,
    ),
    ServiceCondition(
        "room_creep_24h",
        23.0,
        "creep",
        duration_s=86400.0,
    ),
    ServiceCondition(
        "room_fatigue_100k",
        23.0,
        "fatigue",
        cycles=100_000,
    ),
    ServiceCondition(
        "humid_55RH",
        25.0,
        "monotonic",
        duration_s=30.0,
        relative_humidity_percent=55.0,
        conditioning="saturated",
    ),
    ServiceCondition(
        "room_impact",
        23.0,
        "impact",
        duration_s=0.001,
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_tds(source_dir: Path = DEFAULT_SOURCE) -> Dict[str, object]:
    path = Path(source_dir) / TDS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing manufacturer TDS: {path}")
    observed = _sha256(path)
    if observed != TDS_SHA256:
        raise ValueError(
            f"manufacturer TDS checksum changed: expected {TDS_SHA256}, "
            f"got {observed}"
        )
    return {
        "filename": TDS_FILENAME,
        "sha256": observed,
        "url": TDS_URL,
        "version": "3.0",
        "pages": 5,
    }


def evaluate_condition(condition: ServiceCondition) -> ServiceDecision:
    if condition.load_kind not in {
        "monotonic",
        "small_amplitude_dynamic",
        "short_time_relaxation",
        "creep",
        "fatigue",
        "impact",
    }:
        raise ValueError(f"unknown load kind {condition.load_kind!r}")
    if not -100.0 <= condition.temperature_C <= 250.0:
        raise ValueError("temperature is outside the input sanity range")
    if condition.duration_s <= 0.0:
        raise ValueError("duration must be positive")
    if condition.cycles < 1:
        raise ValueError("cycles must be at least one")
    if not 0.0 <= condition.relative_humidity_percent <= 100.0:
        raise ValueError("relative humidity must lie in [0, 100]")

    common = {
        "scenario_id": condition.scenario_id,
        "maximum_temperature_guardrail_C": 54.0,
    }
    if condition.temperature_C >= 54.0:
        return ServiceDecision(
            **common,
            status="rejected",
            admitted_model="none",
            evidence=(
                "Bambu TDS V3.0: HDT 54 degC at 1.8 MPa, "
                "57 degC at 0.45 MPa; Tg 60 degC"
            ),
            reason=(
                "The condition reaches or exceeds the lowest published "
                "heat-deflection boundary. The room-temperature material "
                "cards must not be extrapolated here."
            ),
            requires_new_measurement=True,
        )
    if (
        condition.load_kind == "small_amplitude_dynamic"
        and abs(condition.temperature_C - 18.0) <= 0.5
        and condition.frequency_Hz is not None
        and 1.0 <= condition.frequency_Hz <= 25_000.0
        and 35.0 <= condition.relative_humidity_percent <= 45.0
        and condition.conditioning == "ambient_lab_18C_40RH"
    ):
        return ServiceDecision(
            **common,
            status="admitted_with_restrictions",
            admitted_model="18C_1Hz_25kHz_viscoelastic_wave_screen",
            evidence=(
                "exact-brand raw DMA master curve and reported G'/G'' at "
                "18 degC over 1 Hz to 25 kHz"
            ),
            reason=(
                "Small-strain storage stiffness and a representative loss "
                "factor are supported for bounded wave/modal screening."
            ),
            requires_new_measurement=False,
        )
    if condition.load_kind == "short_time_relaxation":
        exact_environment = (
            abs(condition.temperature_C - 18.0) <= 0.5
            and 35.0 <= condition.relative_humidity_percent <= 45.0
            and condition.conditioning == "ambient_lab_18C_40RH"
        )
        bounded_duration = (
            SHORT_TIME_RELAXATION_MINIMUM_S
            <= condition.duration_s
            <= SHORT_TIME_RELAXATION_MAXIMUM_S
        )
        if exact_environment and bounded_duration:
            return ServiceDecision(
                **common,
                status="admitted_with_restrictions",
                admitted_model="18C_positive_short_time_maxwell_sensitivity",
                evidence=(
                    "positive generalized-Maxwell fit to the exact-brand "
                    "18 C reduced-frequency master curve"
                ),
                reason=(
                    "Storage fit and spectrum positivity pass within the "
                    "inverse measured-frequency window; loss precision does "
                    "not pass, so this is a sensitivity rather than a creep "
                    "validation."
                ),
                requires_new_measurement=False,
            )
        return ServiceDecision(
            **common,
            status="unsupported",
            admitted_model="none",
            evidence=(
                "the inferred short-time spectrum is restricted to "
                "18 degC / 40% RH and 6.37 microseconds to 0.159 seconds"
            ),
            reason=(
                "The request lies outside the bounded environment or "
                "sub-second time window. Frequency-domain inference cannot "
                "be extrapolated into long-term creep."
            ),
            requires_new_measurement=True,
        )
    if condition.temperature_C < 20.0:
        return ServiceDecision(
            **common,
            status="unsupported",
            admitted_model="none",
            evidence="no exact-brand sub-room-temperature mechanical curve",
            reason=(
                "Low-temperature modulus, strength, and fracture response "
                "are unmeasured."
            ),
            requires_new_measurement=True,
        )
    if (
        condition.conditioning == "saturated"
        or condition.relative_humidity_percent > 20.0
    ):
        return ServiceDecision(
            **common,
            status="unsupported",
            admitted_model="none",
            evidence=(
                "Bambu TDS V3.0 reports 0.43% saturated absorption at "
                "25 degC/55% RH but no mechanical correction"
            ),
            reason=(
                "Water uptake is known, but its effect on modulus, strength, "
                "interfaces, and creep is not identified."
            ),
            requires_new_measurement=True,
        )
    if condition.load_kind == "creep":
        return ServiceDecision(
            **common,
            status="unsupported",
            admitted_model="none",
            evidence="no exact-brand creep compliance or Prony series",
            reason=(
                "A rate-independent elastic/plastic card cannot predict "
                "time-dependent deformation."
            ),
            requires_new_measurement=True,
        )
    if condition.load_kind == "fatigue" or condition.cycles > 1:
        return ServiceDecision(
            **common,
            status="unsupported",
            admitted_model="none",
            evidence="no exact-brand S-N, strain-life, or cyclic-damage law",
            reason="Monotonic strength is not a fatigue-life model.",
            requires_new_measurement=True,
        )
    if condition.load_kind == "small_amplitude_dynamic":
        if (
            abs(condition.temperature_C - 30.0) <= 0.5
            and condition.frequency_Hz is not None
            and abs(condition.frequency_Hz - 1.0) <= 1.0e-9
        ):
            return ServiceDecision(
                **common,
                status="admitted_with_restrictions",
                admitted_model="30C_1Hz_storage_stiffness_only",
                evidence=(
                    "peer-reviewed exact-brand DMA at 30 degC and 1 Hz"
                ),
                reason=(
                    "Storage modulus is supported; loss modulus, damping, "
                    "resonance, strength, and long-time response are not."
                ),
                requires_new_measurement=False,
            )
        return ServiceDecision(
            **common,
            status="unsupported",
            admitted_model="none",
            evidence=(
                "exact-brand dynamic data exist only at the admitted 30 C / "
                "1 Hz summary point and the 18 C / 1 Hz-25 kHz reduced-"
                "frequency wave window"
            ),
            reason=(
                "The requested dynamic condition is outside both measured "
                "windows."
            ),
            requires_new_measurement=True,
        )
    if condition.load_kind == "impact":
        return ServiceDecision(
            **common,
            status="guardrail_only",
            admitted_model="published_impact_and_ballistic_screen_only",
            evidence=(
                "manufacturer impact values and a peer-reviewed narrow "
                "354-368 m/s exact-brand ballistic validation"
            ),
            reason=(
                "Evidence can rank the published cases but does not define "
                "a general strain-rate law."
            ),
            requires_new_measurement=True,
        )
    if condition.temperature_C <= 30.0:
        return ServiceDecision(
            **common,
            status="admitted_with_restrictions",
            admitted_model="room_temperature_short_term_monotonic",
            evidence=(
                "exact-brand tensile, flexural, compression, TDS, and "
                "room-temperature topology references"
            ),
            reason=(
                "Use only the documented pre-peak/short-term envelope; no "
                "creep, fatigue, or environmental extrapolation."
            ),
            requires_new_measurement=False,
        )
    return ServiceDecision(
        **common,
        status="guardrail_only",
        admitted_model="none",
        evidence=(
            "exact-brand Tg/HDT thresholds exist, but no service-temperature "
            "constitutive curve"
        ),
        reason=(
            "The temperature is below HDT but outside the measured "
            "room-temperature material-card envelope."
        ),
        requires_new_measurement=True,
    )


def _experiment_plan() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    test_order = 0
    for orientation in ("XY", "Z"):
        for replicate in range(1, 4):
            test_order += 1
            rows.append(
                {
                    "stage": 1,
                    "test_id": f"DMA-{orientation}-R{replicate}",
                    "method": "DMA temperature/frequency sweep",
                    "orientation": orientation,
                    "replicate": replicate,
                    "temperatures_C": "20,30,40,50,60,70",
                    "frequencies_Hz": "0.1,1,10",
                    "conditioning": "dry, <20% RH",
                    "required_output": "storage modulus, loss modulus, tan delta",
                    "priority_reason": (
                        "fit temperature shift and relaxation spectrum"
                    ),
                    "test_order": test_order,
                }
            )
    for orientation in ("XY", "Z"):
        for temperature in (23, 40, 50):
            for replicate in range(1, 4):
                test_order += 1
                rows.append(
                    {
                        "stage": 2,
                        "test_id": (
                            f"RELAX-{orientation}-{temperature}C-"
                            f"R{replicate}"
                        ),
                        "method": "tensile stress relaxation at 0.5% strain",
                        "orientation": orientation,
                        "replicate": replicate,
                        "temperatures_C": str(temperature),
                        "frequencies_Hz": "",
                        "conditioning": "dry, <20% RH",
                        "required_output": "stress versus time, 1 s to 4 h",
                        "priority_reason": (
                            "validate Prony series outside DMA oscillation"
                        ),
                        "test_order": test_order,
                    }
                )
    for orientation in ("XY", "Z"):
        for humidity in ("dry_<20RH", "equilibrated_55RH"):
            for replicate in range(1, 4):
                test_order += 1
                rows.append(
                    {
                        "stage": 3,
                        "test_id": (
                            f"HUMID-{orientation}-{humidity}-R{replicate}"
                        ),
                        "method": "tensile plus mass conditioning",
                        "orientation": orientation,
                        "replicate": replicate,
                        "temperatures_C": "25",
                        "frequencies_Hz": "",
                        "conditioning": humidity,
                        "required_output": (
                            "mass uptake, modulus, yield, strength, elongation"
                        ),
                        "priority_reason": (
                            "identify mechanical effect of reported 0.43% uptake"
                        ),
                        "test_order": test_order,
                    }
                )
    for orientation in ("XY", "Z"):
        for maximum_stress_fraction in (0.40, 0.60, 0.75):
            for replicate in range(1, 6):
                test_order += 1
                rows.append(
                    {
                        "stage": 4,
                        "test_id": (
                            f"FATIGUE-{orientation}-"
                            f"{int(100 * maximum_stress_fraction):02d}UTS-"
                            f"R{replicate}"
                        ),
                        "method": "tension-tension fatigue, stress ratio R=0.1",
                        "orientation": orientation,
                        "replicate": replicate,
                        "temperatures_C": "23",
                        "frequencies_Hz": "5",
                        "conditioning": "dry, <20% RH",
                        "required_output": (
                            "cycles to failure or runout, hysteresis, "
                            "stiffness every decade"
                        ),
                        "priority_reason": (
                            "identify directional S-N screening law and "
                            "runout boundary"
                        ),
                        "maximum_stress_fraction_of_directional_UTS": (
                            maximum_stress_fraction
                        ),
                        "test_order": test_order,
                    }
                )
    aging_exposures = (
        "unaged_control",
        "UVA_500h",
        "40C_75RH_500h",
        "thermal_cycle_5C_45C_100cycles",
    )
    for orientation in ("XY", "Z"):
        for exposure in aging_exposures:
            for replicate in range(1, 4):
                test_order += 1
                rows.append(
                    {
                        "stage": 5,
                        "test_id": (
                            f"AGING-{orientation}-{exposure}-R{replicate}"
                        ),
                        "method": "controlled aging followed by tensile test",
                        "orientation": orientation,
                        "replicate": replicate,
                        "temperatures_C": "23 tensile after exposure",
                        "frequencies_Hz": "",
                        "conditioning": exposure,
                        "required_output": (
                            "mass, modulus, yield, strength, elongation, "
                            "surface observation"
                        ),
                        "priority_reason": (
                            "identify UV, hygrothermal, and thermal-cycle "
                            "retention factors"
                        ),
                        "test_order": test_order,
                    }
                )
    return rows


def write_service_envelope(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = verify_tds(source_dir)
    decisions = [
        evaluate_condition(condition) for condition in DEFAULT_SCENARIOS
    ]
    scenario_rows = [
        {**asdict(condition), **asdict(decision)}
        for condition, decision in zip(DEFAULT_SCENARIOS, decisions)
    ]
    _write_csv(
        output_dir / "manufacturer_properties.csv",
        MANUFACTURER_PROPERTIES,
    )
    _write_csv(output_dir / "service_condition_matrix.csv", scenario_rows)
    plan = _experiment_plan()
    _write_csv(output_dir / "missing_law_experiment_plan.csv", plan)
    counts = {
        status: sum(decision.status == status for decision in decisions)
        for status in (
            "admitted_with_restrictions",
            "guardrail_only",
            "unsupported",
            "rejected",
        )
    }
    payload = {
        "material": "Bambu PLA Basic",
        "source": source,
        "lowest_structural_temperature_guardrail_C": 54.0,
        "glass_transition_temperature_C": 60.0,
        "saturated_water_absorption_percent_at_25C_55RH": 0.43,
        "condition_counts": counts,
        "scenarios": scenario_rows,
        "measurement_plan_specimens": len(plan),
        "supported_contract": [
            (
                "short-term monotonic simulation at 20-30 degC in the "
                "documented dry/as-printed or manufacturer-conditioned state"
            ),
            "30 degC, 1 Hz storage-stiffness sensitivity only",
            (
                "18 degC / 40% RH small-strain relaxation sensitivity from "
                "6.37 microseconds to 0.159 seconds"
            ),
            "published impact/ballistic cases as bounded screens only",
        ],
        "prohibited_contract": [
            "no extrapolation at or above 54 degC",
            (
                "no constant-load creep or stress relaxation beyond the "
                "bounded sub-second sensitivity"
            ),
            "no fatigue-life prediction",
            "no humidity or water-uptake mechanical correction",
            "no aging law",
            "no general strain-rate law",
        ],
    }
    (output_dir / "service_contract.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Bambu PLA Basic service-condition envelope

The checksum-pinned manufacturer V3.0 TDS adds exact-brand environmental
thresholds to the simulator:

- glass transition: **60 degC**;
- heat deflection: **54 degC at 1.8 MPa** and **57 degC at 0.45 MPa**;
- saturated water absorption: **0.43% at 25 degC / 55% RH**;
- manufacturer mechanical specimens were annealed and dried at 55 degC for
  eight hours.

The gate admits short-term monotonic room-temperature use, the published
30 degC / 1 Hz storage-stiffness point, and a positive-spectrum sub-second
relaxation sensitivity at 18 degC / 40% RH with restrictions. It rejects
structural extrapolation at or above 54 degC. Constant-load creep, fatigue,
humid conditioning, aging, and arbitrary dynamic conditions remain
unsupported because the exact brand lacks the necessary curves.

This is an enforceable simulation boundary, not a fabricated temperature or
service-life law. The exported {len(plan)}-specimen staged plan identifies the
staged DMA, relaxation, humidity, fatigue, and aging measurements needed to
replace those rejections with calibrated behavior.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the exact-brand service-condition admission gate"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = write_service_envelope(args.source, args.output)
    print(
        json.dumps(
            {
                "condition_counts": result["condition_counts"],
                "measurement_plan_specimens": result[
                    "measurement_plan_specimens"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
