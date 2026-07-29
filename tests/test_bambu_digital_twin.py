from pathlib import Path

from src.bambu_digital_twin import (
    TwinRequest,
    select_model,
    write_digital_twin,
)


def test_selector_never_falls_back_for_unsupported_conditions() -> None:
    for request in (
        TwinRequest("hot", "static_elastic", temperature_C=54.0),
        TwinRequest("creep", "creep"),
        TwinRequest("fatigue", "fatigue"),
        TwinRequest(
            "humid",
            "static_elastic",
            temperature_C=25.0,
            relative_humidity_percent=55.0,
            conditioning="saturated",
        ),
    ):
        selection = select_model(request)
        assert selection.status in {"unsupported", "rejected"}
        assert selection.model_id is None
        assert selection.canonical_card is None
        assert selection.evidence_files == (
            "data/reported/bambu_pla_service_envelope_v3/"
            "service_contract.json",
        )


def test_prepeak_selector_enforces_direction_and_strain_limit() -> None:
    admitted = select_model(
        TwinRequest(
            "in-plane",
            "prepeak_tension",
            material_direction="in_plane",
            maximum_strain=0.015,
        )
    )
    build = select_model(
        TwinRequest(
            "build",
            "prepeak_tension",
            material_direction="build",
            maximum_strain=0.01,
        )
    )
    beyond = select_model(
        TwinRequest(
            "beyond",
            "prepeak_tension",
            material_direction="in_plane",
            maximum_strain=0.03,
        )
    )
    assert admitted.status == "admitted_with_restrictions"
    assert build.status == "unsupported"
    assert beyond.status == "rejected"


def test_directional_static_selector_carries_interval_evidence() -> None:
    selection = select_model(
        TwinRequest(
            "directional",
            "static_elastic",
            directionality="directional",
            material_direction="mixed",
        )
    )
    assert selection.model_id == "tds_transverse_elastic_sensitivity_v2"
    assert selection.status == "sensitivity_only"
    assert (
        "results/directional_interval_robustness/summary.json"
        in selection.evidence_files
    )
    assert any(
        "not a confidence interval" in limitation
        for limitation in selection.limitations
    )


def test_dma_selector_requires_exact_published_condition() -> None:
    admitted = select_model(
        TwinRequest(
            "dma",
            "dma_storage",
            temperature_C=30.0,
            frequency_Hz=1.0,
            nozzle_temperature_C=220,
        )
    )
    wrong_frequency = select_model(
        TwinRequest(
            "dma-10hz",
            "dma_storage",
            temperature_C=30.0,
            frequency_Hz=10.0,
            nozzle_temperature_C=220,
        )
    )
    assert admitted.status == "admitted_with_restrictions"
    assert wrong_frequency.status == "unsupported"


def test_wave_selector_requires_exact_broadband_condition() -> None:
    admitted = select_model(
        TwinRequest(
            "wave",
            "viscoelastic_wave",
            temperature_C=18.0,
            frequency_Hz=1_000.0,
            relative_humidity_percent=40.0,
            conditioning="ambient_lab_18C_40RH",
        )
    )
    outside = select_model(
        TwinRequest(
            "wave-outside",
            "viscoelastic_wave",
            temperature_C=18.0,
            frequency_Hz=30_000.0,
            relative_humidity_percent=40.0,
            conditioning="ambient_lab_18C_40RH",
        )
    )
    assert admitted.status == "bounded_screen"
    assert admitted.model_id == "bambu_pla_18C_broadband_storage_proxy_v1"
    assert outside.status == "unsupported"


def test_short_time_relaxation_selector_is_strictly_bounded() -> None:
    admitted = select_model(
        TwinRequest(
            "short",
            "short_time_relaxation",
            temperature_C=18.0,
            relative_humidity_percent=40.0,
            conditioning="ambient_lab_18C_40RH",
            duration_s=0.01,
        )
    )
    outside = select_model(
        TwinRequest(
            "long",
            "short_time_relaxation",
            temperature_C=18.0,
            relative_humidity_percent=40.0,
            conditioning="ambient_lab_18C_40RH",
            duration_s=1.0,
        )
    )
    assert admitted.status == "sensitivity_only"
    assert admitted.model_id == (
        "bambu_pla_18C_short_time_maxwell_sensitivity_v1"
    )
    assert admitted.canonical_card is None
    assert outside.status == "unsupported"


def test_cryo_selector_is_a_bounded_screen_without_material_card() -> None:
    admitted = select_model(
        TwinRequest(
            "cryo",
            "cryo_recovery_sensitivity",
            conditioning="minus80C_24h_then_room_temperature_24h",
        )
    )
    wrong_protocol = select_model(
        TwinRequest("wrong", "cryo_recovery_sensitivity")
    )
    actual_low_temperature = select_model(
        TwinRequest(
            "actual-minus80",
            "static_elastic",
            temperature_C=-80.0,
        )
    )
    assert admitted.status == "bounded_screen"
    assert admitted.canonical_card is None
    assert wrong_protocol.status == "unsupported"
    assert actual_low_temperature.status == "unsupported"


def test_digital_twin_exports_cards_and_checksum_inventory(
    tmp_path: Path,
) -> None:
    payload = write_digital_twin(tmp_path)
    assert payload["selection_gate_passed"] is True
    assert payload["canonical_card_count"] == 9
    assert payload["selection_count"] == 18
    handoff = payload["calibration_handoffs"]["service_laws"]
    directional = payload["calibration_handoffs"]["directional_material"]
    modal = payload["heldout_validations"][
        "exact_geometry_small_strain_modal"
    ]
    assert modal["status"] == "heldout_small_strain_modal_gate_passed"
    assert modal["physical_specimens"] == 3
    assert modal["physical_modes_per_specimen"] == 5
    assert modal["heldout_relative_rmse"] < 0.15
    assert modal["component_modal_data_used_for_calibration"] is False
    assert modal["local_axis_directional_calibration_released"] is False
    assert modal["manifest_sha256"]
    assert "nine-constant local-axis" in directional["purpose"]
    if directional["calibration_id"] is not None:
        assert (
            directional["calibration_id"]
            == "bambu_pla_basic_directional_calibration_v1"
        )
        assert directional["status"] == "awaiting_measurements"
        assert directional["campaign_ready"] is False
        assert directional["card_released"] is False
        assert directional["heldout_used_for_fitting"] is False
        assert directional["synthetic_recovery_gate_passed"] is True
        assert directional["manifest_sha256"]
    assert "calibrate currently rejected" in handoff["purpose"]
    if handoff["campaign_id"] is not None:
        assert handoff["campaign_id"] == "bambu_pla_basic_service_laws_v1"
        assert handoff["specimen_count"] == 90
        assert handoff["status"] == "awaiting_measurements"
        assert handoff["manifest_sha256"]
        if handoff["calibration_manifest"] is not None:
            assert handoff["calibration_status"] == "awaiting_measurements"
            assert handoff["released_calibration_stages"] == []
            assert handoff["synthetic_recovery_gate_passed"] is True
            assert handoff["calibration_manifest_sha256"]
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "selection_matrix.csv").exists()
    assert (tmp_path / "evidence_inventory.csv").exists()
    assert len(list((tmp_path / "cards").glob("*.inp"))) == 9
