from pathlib import Path

from src.bambu_service_envelope import (
    DEFAULT_SOURCE,
    ServiceCondition,
    evaluate_condition,
    verify_tds,
    write_service_envelope,
)


def test_manufacturer_tds_is_checksum_pinned() -> None:
    evidence = verify_tds(DEFAULT_SOURCE)
    assert evidence["version"] == "3.0"
    assert evidence["pages"] == 5


def test_condition_gate_admits_only_supported_envelope() -> None:
    room = evaluate_condition(
        ServiceCondition("room", 23.0, "monotonic")
    )
    dma = evaluate_condition(
        ServiceCondition(
            "dma", 30.0, "small_amplitude_dynamic", frequency_Hz=1.0
        )
    )
    wave = evaluate_condition(
        ServiceCondition(
            "wave",
            18.0,
            "small_amplitude_dynamic",
            frequency_Hz=1_000.0,
            relative_humidity_percent=40.0,
            conditioning="ambient_lab_18C_40RH",
        )
    )
    short_relaxation = evaluate_condition(
        ServiceCondition(
            "short-relaxation",
            18.0,
            "short_time_relaxation",
            duration_s=0.01,
            relative_humidity_percent=40.0,
            conditioning="ambient_lab_18C_40RH",
        )
    )
    long_relaxation = evaluate_condition(
        ServiceCondition(
            "long-relaxation",
            18.0,
            "short_time_relaxation",
            duration_s=1.0,
            relative_humidity_percent=40.0,
            conditioning="ambient_lab_18C_40RH",
        )
    )
    warm = evaluate_condition(
        ServiceCondition("warm", 40.0, "monotonic")
    )
    hdt = evaluate_condition(
        ServiceCondition("hdt", 54.0, "monotonic")
    )
    creep = evaluate_condition(
        ServiceCondition("creep", 23.0, "creep", duration_s=86400.0)
    )
    humid = evaluate_condition(
        ServiceCondition(
            "humid",
            25.0,
            "monotonic",
            relative_humidity_percent=55.0,
            conditioning="saturated",
        )
    )
    actual_minus80 = evaluate_condition(
        ServiceCondition("actual-minus80", -80.0, "monotonic")
    )
    assert room.status == "admitted_with_restrictions"
    assert dma.status == "admitted_with_restrictions"
    assert wave.status == "admitted_with_restrictions"
    assert short_relaxation.status == "admitted_with_restrictions"
    assert long_relaxation.status == "unsupported"
    assert warm.status == "guardrail_only"
    assert hdt.status == "rejected"
    assert creep.status == "unsupported"
    assert humid.status == "unsupported"
    assert actual_minus80.status == "unsupported"


def test_service_envelope_exports_contract_and_measurement_plan(
    tmp_path: Path,
) -> None:
    result = write_service_envelope(DEFAULT_SOURCE, tmp_path)
    assert result["lowest_structural_temperature_guardrail_C"] == 54.0
    assert result["measurement_plan_specimens"] == 90
    assert result["condition_counts"]["rejected"] == 2
    assert (tmp_path / "service_contract.json").exists()
    assert (tmp_path / "missing_law_experiment_plan.csv").exists()
