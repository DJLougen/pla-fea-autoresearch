from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .bambu_service_envelope import ServiceCondition, evaluate_condition
from .materials import material_card


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/bambu_pla_digital_twin"
REPORTED = PROJECT_ROOT / "data/reported"


@dataclass(frozen=True)
class TwinRequest:
    request_id: str
    analysis_kind: str
    temperature_C: float = 23.0
    frequency_Hz: float | None = None
    relative_humidity_percent: float = 20.0
    conditioning: str = "as_printed_dry"
    nozzle_temperature_C: int = 220
    directionality: str = "directional"
    material_direction: str = "in_plane"
    maximum_strain: float | None = None
    duration_s: float = 1.0


@dataclass(frozen=True)
class ModelSelection:
    request_id: str
    status: str
    model_id: str | None
    fidelity: str
    solver_compatibility: str
    canonical_card: str | None
    evidence_files: tuple[str, ...]
    reason: str
    limitations: tuple[str, ...]
    service_status: str


ANALYSIS_KINDS = {
    "static_elastic",
    "prepeak_tension",
    "topology_compression",
    "flexural",
    "dma_storage",
    "viscoelastic_wave",
    "short_time_relaxation",
    "impact_ballistic",
    "interlayer_initiation",
    "cryo_recovery_sensitivity",
    "creep",
    "fatigue",
}


def _json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _service_kind(analysis_kind: str) -> str:
    if analysis_kind in {"dma_storage", "viscoelastic_wave"}:
        return "small_amplitude_dynamic"
    if analysis_kind == "impact_ballistic":
        return "impact"
    if analysis_kind == "short_time_relaxation":
        return "short_time_relaxation"
    if analysis_kind in {"creep", "fatigue"}:
        return analysis_kind
    return "monotonic"


def _service_decision(request: TwinRequest):
    return evaluate_condition(
        ServiceCondition(
            scenario_id=request.request_id,
            temperature_C=request.temperature_C,
            load_kind=_service_kind(request.analysis_kind),
            duration_s=(
                86400.0
                if request.analysis_kind == "creep"
                else request.duration_s
            ),
            cycles=(
                100_000 if request.analysis_kind == "fatigue" else 1
            ),
            frequency_Hz=request.frequency_Hz,
            relative_humidity_percent=request.relative_humidity_percent,
            conditioning=request.conditioning,
        )
    )


def select_model(
    request: TwinRequest,
    canonical_cards: Mapping[str, str] | None = None,
) -> ModelSelection:
    if request.analysis_kind not in ANALYSIS_KINDS:
        raise ValueError(
            f"unknown analysis kind {request.analysis_kind!r}"
        )
    if request.directionality not in {"isotropic", "directional"}:
        raise ValueError("directionality must be isotropic or directional")
    if request.material_direction not in {"in_plane", "build", "mixed"}:
        raise ValueError(
            "material_direction must be in_plane, build, or mixed"
        )
    service = _service_decision(request)
    base = {
        "request_id": request.request_id,
        "service_status": service.status,
    }
    if service.status in {"rejected", "unsupported"}:
        return ModelSelection(
            **base,
            status=service.status,
            model_id=None,
            fidelity="none",
            solver_compatibility="none",
            canonical_card=None,
            evidence_files=(
                "data/reported/bambu_pla_service_envelope_v3/"
                "service_contract.json",
            ),
            reason=service.reason,
            limitations=(
                "No material card may be substituted for this condition.",
            ),
        )
    if (
        service.status == "guardrail_only"
        and request.analysis_kind != "impact_ballistic"
    ):
        return ModelSelection(
            **base,
            status="guardrail_only",
            model_id=None,
            fidelity="threshold_only",
            solver_compatibility="none",
            canonical_card=None,
            evidence_files=(
                "data/reported/bambu_pla_service_envelope_v3/"
                "service_contract.json",
            ),
            reason=service.reason,
            limitations=(
                "A threshold is available, but no constitutive interpolation.",
            ),
        )

    cards = dict(canonical_cards or {})
    if request.analysis_kind == "static_elastic":
        if request.directionality == "isotropic":
            return ModelSelection(
                **base,
                status="admitted_with_restrictions",
                model_id="raw_curve_isotropic_elastic_v1",
                fidelity="raw-curve-calibrated isotropic screening",
                solver_compatibility="CalculiX",
                canonical_card=cards.get("room_isotropic_elastic"),
                evidence_files=(
                    "data/reported/bambu_pla_raw_curve_elastic.json",
                    "results/mendeley_tensile_reconciliation_audit/"
                    "provenance.json",
                    "results/raw_specimen_infill_uncertainty/"
                    "summary.json",
                    "results/raw_specimen_infill_uncertainty/"
                    "prediction_intervals.csv",
                    "results/bambu_print_tolerance_infill/"
                    "summary.json",
                    "results/bambu_print_tolerance_infill/"
                    "solid_load_ratios.csv",
                    "results/bambu_pla_heldout_modal_validation/"
                    "provenance.json",
                    "results/bambu_pla_heldout_modal_validation/"
                    "heldout_comparison.csv",
                    "data/reported/bambu_pla_uav_arm_2026/"
                    "provenance.json",
                    "data/reported/bambu_pla_uav_arm_2026/"
                    "curve_validation.csv",
                ),
                reason=(
                    "Use when local material axes are unknown or deliberately "
                    "homogenized."
                ),
                limitations=(
                    "Poisson ratio is a prior.",
                    "Linear elastic only; no failure prediction.",
                    "The exact-brand tensile labels fail a checksum-pinned "
                    "one-to-one reconciliation, so only pooled isotropic "
                    "screening is admitted.",
                    "The 22-specimen modulus ensemble quantifies pooled "
                    "scalar-E variation only; it does not supply directional "
                    "or defect uncertainty.",
                    "The 25-cube X1C study bounds whole-geometry dimensional "
                    "variation but not local bead or void defects.",
                    "The exact-geometry cantilever check validates held-out "
                    "small-strain modal frequencies only.",
                    "The external UAV-arm curves are held-out targets; "
                    "their exact CAD is not public, so no replay is claimed.",
                ),
            )
        return ModelSelection(
            **base,
            status="sensitivity_only",
            model_id="tds_transverse_elastic_sensitivity_v2",
            fidelity=(
                "reported XY/Z modulus plus measured in-plane Poisson and "
                "remaining out-of-plane/shear priors"
            ),
            solver_compatibility="CalculiX",
            canonical_card=cards.get("room_tds_transverse"),
            evidence_files=(
                "data/reported/bambu_pla_tds_transverse_sensitivity.json",
                "data/reported/bambu_pla_viscoelastic_wave_2026/"
                "provenance.json",
                "results/static_poisson_propagation/summary.json",
                "results/directional_interval_robustness/summary.json",
                "data/reported/bambu_pla_uav_arm_2026/"
                "provenance.json",
                "data/reported/bambu_pla_uav_arm_2026/"
                "curve_validation.csv",
            ),
            reason=(
                "Direction-aware screening is useful, but the six shear/"
                "Poisson constants are not all measured."
            ),
            limitations=(
                "E1/E2 and E3 are reported summary values.",
                "nu12 uses one exact-brand measurement from a separate "
                "Bambu-X1C process.",
                "Out-of-plane Poisson and quasistatic shear values remain "
                "priors.",
                "The directional interval study is a deterministic "
                "sensitivity screen, not a confidence interval.",
                "The external UAV-arm curves are held-out targets; "
                "their exact CAD is not public, so no replay is claimed.",
            ),
        )
    if request.analysis_kind == "prepeak_tension":
        profile_path = REPORTED / "bambu_pla_raw_curve_prepeak_plastic.json"
        profile = _json(profile_path)
        maximum_supported = float(
            profile["plastic_isotropic"][
                "maximum_supported_engineering_strain"
            ]
        )
        if request.material_direction != "in_plane":
            return ModelSelection(
                **base,
                status="unsupported",
                model_id=None,
                fidelity="none",
                solver_compatibility="none",
                canonical_card=None,
                evidence_files=(_relative(profile_path),),
                reason=(
                    "The pre-peak aggregate excludes Z-upright specimens and "
                    "cannot be promoted to build-direction plasticity."
                ),
                limitations=(
                    "Acquire matched build-direction plastic curves.",
                ),
            )
        if (
            request.maximum_strain is not None
            and request.maximum_strain > maximum_supported
        ):
            return ModelSelection(
                **base,
                status="rejected",
                model_id=None,
                fidelity="none",
                solver_compatibility="none",
                canonical_card=None,
                evidence_files=(_relative(profile_path),),
                reason=(
                    f"Requested strain {request.maximum_strain:.5g} exceeds "
                    f"the supported {maximum_supported:.5g} pre-peak limit."
                ),
                limitations=(
                    "The card has no post-peak damage or fracture response.",
                ),
            )
        return ModelSelection(
            **base,
            status="admitted_with_restrictions",
            model_id="raw_curve_in_plane_prepeak_J2_v1",
            fidelity="19-curve in-plane pre-peak plastic calibration",
            solver_compatibility="CalculiX",
            canonical_card=cards.get("room_prepeak_tension"),
            evidence_files=(_relative(profile_path),),
            reason=(
                f"Admitted only through engineering strain "
                f"{maximum_supported:.5g}."
            ),
            limitations=tuple(profile["limitations"][-6:]),
        )
    if request.analysis_kind == "topology_compression":
        return ModelSelection(
            **base,
            status="sensitivity_only",
            model_id="pvh_author_compression_transfer_v1",
            fidelity=(
                "independent exact-brand elastic/yield card transferred to "
                "explicit topology"
            ),
            solver_compatibility="CalculiX",
            canonical_card=cards.get("pvh_compression_sensitivity"),
            evidence_files=(
                "data/reported/bambu_pla_pvh_metamaterial_2025/"
                "author_model.json",
                "results/common_infill_compression_validation/summary.json",
                "results/pvh_material_transfer_sensitivity/summary.json",
            ),
            reason=(
                "Appropriate for relative short-term compression ranking, "
                "not precision crush or failure prediction."
            ),
            limitations=(
                "Author post-yield hardening table is unpublished.",
                "Elastic-perfectly-plastic completion is a sensitivity.",
                "Self-contact, crushing damage, and exact toolpaths are absent.",
            ),
        )
    if request.analysis_kind == "flexural":
        return ModelSelection(
            **base,
            status="bounded_screen",
            model_id="published_flexural_process_reference_v1",
            fidelity="published 0/45/90 degree modulus and strength screen",
            solver_compatibility="CalculiX fields plus external strength gate",
            canonical_card=cards.get("room_tds_transverse"),
            evidence_files=(
                "data/reported/bambu_pla_published_flexural_process/"
                "flexural_process_matrix.json",
                "data/reported/bambu_pla_published_flexural_220C/"
                "provenance.json",
                "results/custom_infill/weird_kite_search/"
                "toolpath_first_limit_screen/summary.json",
            ),
            reason=(
                "Use the elastic card for fields and the published process "
                "matrix as a separate strength threshold."
            ),
            limitations=(
                "Flexural strength is not a fracture law.",
                "Build-direction process scaling remains a proxy.",
            ),
        )
    if request.analysis_kind == "dma_storage":
        if request.nozzle_temperature_C not in {210, 220, 230, 240}:
            return ModelSelection(
                **base,
                status="unsupported",
                model_id=None,
                fidelity="none",
                solver_compatibility="none",
                canonical_card=None,
                evidence_files=(
                    "data/reported/bambu_pla_nozzle_temperature_sensitivity/"
                    "manifest.json",
                ),
                reason=(
                    "DMA storage profiles exist only for 210/220/230/240 C "
                    "nozzle settings."
                ),
                limitations=("No interpolation across print temperature.",),
            )
        key = f"dma_print_{request.nozzle_temperature_C}C"
        return ModelSelection(
            **base,
            status="admitted_with_restrictions",
            model_id=(
                "dma_30C_1Hz_print"
                f"{request.nozzle_temperature_C}C_sensitivity_v1"
            ),
            fidelity="published 30 C / 1 Hz storage stiffness",
            solver_compatibility="CalculiX in-phase elastic proxy",
            canonical_card=cards.get(key),
            evidence_files=(
                "data/reported/bambu_pla_nozzle_temperature_sensitivity/"
                f"print_{request.nozzle_temperature_C}C.json",
            ),
            reason="Exact published temperature/frequency summary point.",
            limitations=(
                "No loss modulus, phase, damping, resonance, or Prony law.",
                "No service-temperature extrapolation.",
            ),
        )
    if request.analysis_kind == "viscoelastic_wave":
        return ModelSelection(
            **base,
            status="bounded_screen",
            model_id="bambu_pla_18C_broadband_storage_proxy_v1",
            fidelity=(
                "raw-DMA-reconstructed 18 C reduced-frequency storage/loss "
                "screen"
            ),
            solver_compatibility=(
                "CalculiX in-phase eigenfrequency proxy plus external "
                "loss-factor estimate"
            ),
            canonical_card=cards.get("wave_18C_storage_proxy"),
            evidence_files=(
                "data/reported/bambu_pla_viscoelastic_wave_2026/"
                "provenance.json",
                "data/reported/bambu_pla_viscoelastic_wave_2026/"
                "dma_master_curve_18C.csv",
                "results/custom_infill/graded_hourglass/"
                "viscoelastic_modal_screen/summary.json",
                "results/custom_infill/weird_kite_search/"
                "frequency_dependent_modal_screen/summary.json",
            ),
            reason=(
                "Exact-brand printed density, in-plane Poisson ratio, and "
                "broadband G'/G'' support a small-strain 18 C modal screen; "
                "the promoted sliced patterns also close the storage-"
                "frequency loop iteratively."
            ),
            limitations=(
                "The CalculiX card uses E'=2G'(1+nu) as an isotropic "
                "in-phase storage proxy.",
                "The deposited source uses interactive outlier selection; "
                "the no-click raw reconstruction is retained separately.",
                "No harmonic response, fixture compliance, fatigue, failure, "
                "or arbitrary-temperature extrapolation.",
            ),
        )
    if request.analysis_kind == "short_time_relaxation":
        return ModelSelection(
            **base,
            status="sensitivity_only",
            model_id="bambu_pla_18C_short_time_maxwell_sensitivity_v1",
            fidelity=(
                "positive five-branch spectrum inferred from exact-brand "
                "broadband DMA"
            ),
            solver_compatibility=(
                "external analytic relaxation plus frequency-specific "
                "in-phase CalculiX elastic cards"
            ),
            canonical_card=None,
            evidence_files=(
                "data/reported/bambu_pla_short_time_relaxation_18C/"
                "summary.json",
                "data/reported/bambu_pla_short_time_relaxation_18C/"
                "maxwell_spectrum.csv",
                "results/custom_infill/weird_kite_search/"
                "frequency_dependent_modal_screen/summary.json",
            ),
            reason=(
                "The positive storage spectrum is bounded to the inverse "
                "measured-frequency window at 18 C / 40% RH."
            ),
            limitations=(
                "The loss-modulus precision gate fails.",
                "CalculiX anchor cards are in-phase elastic proxies, not a "
                "native time-domain Prony material.",
                "No constant-load creep, fatigue, moisture, failure, or "
                "response beyond 0.159 s.",
            ),
        )
    if request.analysis_kind == "impact_ballistic":
        return ModelSelection(
            **base,
            status="bounded_screen",
            model_id="author_ballistic_damage_v1",
            fidelity="author-calibrated narrow ballistic validation",
            solver_compatibility="Abaqus author model; CalculiX curves only",
            canonical_card=None,
            evidence_files=(
                "data/reported/bambu_pla_ballistic_damage_2025/"
                "author_model.json",
                "data/reported/bambu_pla_ballistic_damage_2025/"
                "provenance.json",
                "data/reported/bambu_pla_ballistic_damage_2025/"
                "curve_validation.csv",
                "results/ballistic_pattern_validation/summary.json",
            ),
            reason=(
                "Use only in the reported 354-368 m/s projectile/target "
                "window and documented damage sensitivity."
            ),
            limitations=(
                "No explicit strain-rate hardening law.",
                "Only four of five reported tensile/flexural specimen "
                "traces are independently rendered; machine raw curves "
                "are unavailable.",
                "Internal infill was homogenized in the author FE model.",
                "Not armor certification.",
            ),
        )
    if request.analysis_kind == "interlayer_initiation":
        return ModelSelection(
            **base,
            status="bounded_screen",
            model_id="interlayer_opening_tear_initiation_v1",
            fidelity="process-conditioned initiation threshold",
            solver_compatibility="CalculiX fields plus external initiation gate",
            canonical_card=cards.get("room_tds_transverse"),
            evidence_files=(
                "data/reported/bambu_pla_interlayer_strength_2026/"
                "interface_initiation_profiles.json",
                "results/custom_infill/weird_kite_search/"
                "toolpath_first_limit_screen/summary.json",
            ),
            reason=(
                "Screens local-3 opening and apparent tear initiation."
            ),
            limitations=(
                "Tear is not a pure shear test.",
                "No traction-separation or damage-evolution law.",
                "Source used a modified non-Bambu printer.",
            ),
        )
    if request.analysis_kind == "cryo_recovery_sensitivity":
        expected_conditioning = (
            "minus80C_24h_then_room_temperature_24h"
        )
        evidence = (
            "data/reported/bambu_pla_cryogenic_preconditioning_2025/"
            "provenance.json",
            "data/reported/bambu_pla_cryogenic_preconditioning_2025/"
            "conditioning_effects.csv",
            "results/custom_infill/graded_hourglass/"
            "cryogenic_recovery_sensitivity/summary.json",
        )
        if request.conditioning != expected_conditioning:
            return ModelSelection(
                **base,
                status="unsupported",
                model_id=None,
                fidelity="none",
                solver_compatibility="none",
                canonical_card=None,
                evidence_files=evidence,
                reason=(
                    "The exact-brand matrix supports only -80 C for 24 h "
                    "followed by 24 h recovery at room temperature."
                ),
                limitations=(
                    "Choose the exact published conditioning protocol.",
                ),
            )
        return ModelSelection(
            **base,
            status="bounded_screen",
            model_id="cryo_recovery_cubic_matrix_transfer_screen_v1",
            fidelity=(
                "160 figure-labeled means and 16-cell flexural-modulus "
                "correction envelope"
            ),
            solver_compatibility=(
                "external scalar sensitivity around room-temperature "
                "toolpath CalculiX result"
            ),
            canonical_card=None,
            evidence_files=evidence,
            reason=(
                "The published freeze/recovery flexural-modulus envelope "
                "can stress-test a relative pattern ranking."
            ),
            limitations=(
                "Published specimens used cubic infill, not the custom pattern.",
                "No raw curves or numeric specimen dispersion are available.",
                "The test occurred after room-temperature recovery, not at -80 C.",
                "No cryogenic constitutive or interface-damage card is admitted.",
            ),
        )
    raise RuntimeError("unreachable model-selection branch")


def _isotropic_card(E_MPa: float, nu: float, name: str) -> str:
    return f"""*MATERIAL, NAME={name}
*DENSITY
1.24e-09
*ELASTIC
{E_MPa:.12g}, {nu:.12g}
"""


def _plastic_card(
    E_MPa: float,
    nu: float,
    points: Sequence[Mapping[str, float]],
    name: str,
) -> str:
    rows = "\n".join(
        f"{float(point['stress_MPa']):.12g}, "
        f"{float(point['equivalent_plastic_strain']):.12g}"
        for point in points
    )
    return (
        _isotropic_card(E_MPa, nu, name)
        + "*PLASTIC\n"
        + rows
        + "\n"
    )


def write_canonical_cards(output_dir: Path) -> Dict[str, str]:
    card_dir = Path(output_dir) / "cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    raw = _json(REPORTED / "bambu_pla_raw_curve_elastic.json")
    transverse = _json(
        REPORTED / "bambu_pla_tds_transverse_sensitivity.json"
    )
    plastic = _json(
        REPORTED / "bambu_pla_raw_curve_prepeak_plastic.json"
    )
    pvh = _json(
        REPORTED
        / "bambu_pla_pvh_metamaterial_2025"
        / "author_model.json"
    )["material_card"]
    contents = {
        "room_isotropic_elastic": _isotropic_card(
            float(raw["screening_isotropic"]["E_MPa"]),
            float(raw["screening_isotropic"]["nu"]),
            "BAMBU_PLA_ROOM_ISO",
        ),
        "room_tds_transverse": material_card(
            transverse["screening_engineering_constants"],
            name="BAMBU_PLA_TDS_TRANSVERSE",
        ),
        "room_prepeak_tension": _plastic_card(
            float(plastic["screening_isotropic"]["E_MPa"]),
            float(plastic["screening_isotropic"]["nu"]),
            [
                {
                    "stress_MPa": point["true_stress_MPa"],
                    "equivalent_plastic_strain": point[
                        "equivalent_plastic_strain"
                    ],
                }
                for point in plastic["plastic_isotropic"]["points"]
            ],
            "BAMBU_PLA_INPLANE_PREPEAK",
        ),
        "pvh_compression_sensitivity": _plastic_card(
            float(pvh["youngs_modulus_MPa"]),
            float(pvh["poisson_ratio"]),
            [
                {
                    "stress_MPa": pvh["yield_stress_MPa"],
                    "equivalent_plastic_strain": 0.0,
                }
            ],
            "BAMBU_PLA_PVH_COMPRESSION",
        ),
    }
    for temperature in (210, 220, 230, 240):
        source = (
            REPORTED
            / "bambu_pla_nozzle_temperature_sensitivity"
            / f"print_{temperature}C.inp"
        )
        contents[f"dma_print_{temperature}C"] = source.read_text(
            encoding="utf-8"
        )
    contents["wave_18C_storage_proxy"] = (
        REPORTED
        / "bambu_pla_viscoelastic_wave_2026"
        / "storage_proxy.inp"
    ).read_text(encoding="utf-8")
    mapping: Dict[str, str] = {}
    manifest_rows = []
    for card_id, text in contents.items():
        path = card_dir / f"{card_id}.inp"
        path.write_text(text, encoding="utf-8")
        mapping[card_id] = _relative(path)
        manifest_rows.append(
            {
                "card_id": card_id,
                "path": mapping[card_id],
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    with (output_dir / "canonical_cards.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(manifest_rows[0])
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return mapping


DEFAULT_REQUESTS = (
    TwinRequest(
        "room_static_isotropic",
        "static_elastic",
        directionality="isotropic",
    ),
    TwinRequest("room_static_directional", "static_elastic"),
    TwinRequest(
        "room_prepeak_in_plane",
        "prepeak_tension",
        material_direction="in_plane",
        maximum_strain=0.015,
    ),
    TwinRequest(
        "room_prepeak_build",
        "prepeak_tension",
        material_direction="build",
        maximum_strain=0.01,
    ),
    TwinRequest(
        "prepeak_beyond_limit",
        "prepeak_tension",
        maximum_strain=0.03,
    ),
    TwinRequest("room_topology_compression", "topology_compression"),
    TwinRequest("room_flexural", "flexural"),
    TwinRequest(
        "dma_30C_1Hz",
        "dma_storage",
        temperature_C=30.0,
        frequency_Hz=1.0,
    ),
    TwinRequest(
        "wave_18C_1kHz",
        "viscoelastic_wave",
        temperature_C=18.0,
        frequency_Hz=1_000.0,
        relative_humidity_percent=40.0,
        conditioning="ambient_lab_18C_40RH",
    ),
    TwinRequest(
        "short_relaxation_18C_10ms",
        "short_time_relaxation",
        temperature_C=18.0,
        relative_humidity_percent=40.0,
        conditioning="ambient_lab_18C_40RH",
        duration_s=0.01,
    ),
    TwinRequest("room_impact", "impact_ballistic"),
    TwinRequest("room_interlayer", "interlayer_initiation"),
    TwinRequest(
        "cryo_recovery_pattern_screen",
        "cryo_recovery_sensitivity",
        conditioning="minus80C_24h_then_room_temperature_24h",
    ),
    TwinRequest("warm_static_40C", "static_elastic", temperature_C=40.0),
    TwinRequest("hot_static_54C", "static_elastic", temperature_C=54.0),
    TwinRequest("room_creep", "creep"),
    TwinRequest("room_fatigue", "fatigue"),
    TwinRequest(
        "humid_static",
        "static_elastic",
        temperature_C=25.0,
        relative_humidity_percent=55.0,
        conditioning="saturated",
    ),
)


def _source_inventory(selections: Sequence[ModelSelection]):
    paths = sorted(
        {
            PROJECT_ROOT / path
            for selection in selections
            for path in selection.evidence_files
        }
    )
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"digital-twin evidence file is missing: {path}"
            )
        rows.append(
            {
                "path": _relative(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def write_digital_twin(
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = write_canonical_cards(output_dir)
    selections = [
        select_model(request, cards) for request in DEFAULT_REQUESTS
    ]
    selection_rows = [
        {
            **asdict(request),
            **{
                **asdict(selection),
                "evidence_files": " | ".join(selection.evidence_files),
                "limitations": " | ".join(selection.limitations),
            },
        }
        for request, selection in zip(DEFAULT_REQUESTS, selections)
    ]
    fields = list(
        dict.fromkeys(key for row in selection_rows for key in row)
    )
    with (output_dir / "selection_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selection_rows)
    source_rows = _source_inventory(selections)
    with (output_dir / "evidence_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(source_rows[0])
        )
        writer.writeheader()
        writer.writerows(source_rows)
    service_campaign_path = (
        PROJECT_ROOT
        / "results/bambu_pla_service_law_campaign/campaign_manifest.json"
    )
    service_campaign = (
        json.loads(service_campaign_path.read_text(encoding="utf-8"))
        if service_campaign_path.exists()
        else {}
    )
    service_calibration_path = (
        PROJECT_ROOT
        / "results/bambu_pla_service_law_calibration/"
        "calibration_manifest.json"
    )
    service_calibration = (
        json.loads(service_calibration_path.read_text(encoding="utf-8"))
        if service_calibration_path.exists()
        else {}
    )
    service_handoff = {
        "campaign_id": service_campaign.get("campaign_id"),
        "status": service_campaign.get("status", "not_generated"),
        "specimen_count": service_campaign.get("specimen_count", 0),
        "manifest": (
            _relative(service_campaign_path)
            if service_campaign_path.exists()
            else None
        ),
        "manifest_sha256": (
            _sha256(service_campaign_path)
            if service_campaign_path.exists()
            else None
        ),
        "calibration_status": service_calibration.get(
            "status", "not_generated"
        ),
        "released_calibration_stages": service_calibration.get(
            "released_stages", []
        ),
        "calibration_manifest": (
            _relative(service_calibration_path)
            if service_calibration_path.exists()
            else None
        ),
        "calibration_manifest_sha256": (
            _sha256(service_calibration_path)
            if service_calibration_path.exists()
            else None
        ),
        "synthetic_recovery_gate_passed": service_calibration.get(
            "synthetic_recovery", {}
        ).get("all_synthetic_recovery_gates_passed"),
        "purpose": (
            "calibrate currently rejected temperature-frequency, long-time "
            "relaxation, humidity, fatigue, and aging behavior"
        ),
    }
    directional_calibration_path = (
        PROJECT_ROOT
        / "results/directional_material_calibration/"
        "calibration_manifest.json"
    )
    directional_calibration = (
        json.loads(
            directional_calibration_path.read_text(encoding="utf-8")
        )
        if directional_calibration_path.exists()
        else {}
    )
    directional_handoff = {
        "calibration_id": directional_calibration.get("calibration_id"),
        "status": directional_calibration.get(
            "status", "not_generated"
        ),
        "campaign_ready": directional_calibration.get("campaign_ready"),
        "card_released": directional_calibration.get(
            "card_released", False
        ),
        "heldout_used_for_fitting": directional_calibration.get(
            "heldout_used_for_fitting"
        ),
        "synthetic_recovery_gate_passed": directional_calibration.get(
            "synthetic_recovery", {}
        ).get("recovery_gate_passed"),
        "manifest": (
            _relative(directional_calibration_path)
            if directional_calibration_path.exists()
            else None
        ),
        "manifest_sha256": (
            _sha256(directional_calibration_path)
            if directional_calibration_path.exists()
            else None
        ),
        "purpose": (
            "release a nine-constant local-axis CalculiX card only after "
            "the frozen calibration and independent coupon-validation "
            "partitions pass; preserve the L-bracket as a post-release "
            "held-out component check"
        ),
    }
    modal_validation_path = (
        PROJECT_ROOT
        / "results/bambu_pla_heldout_modal_validation/provenance.json"
    )
    modal_validation = (
        json.loads(modal_validation_path.read_text(encoding="utf-8"))
        if modal_validation_path.exists()
        else {}
    )
    heldout_modal_validation = {
        "validation_id": modal_validation.get("validation_id"),
        "status": modal_validation.get("status", "not_generated"),
        "physical_specimens": modal_validation.get("component", {}).get(
            "physical_specimens", 0
        ),
        "physical_modes_per_specimen": modal_validation.get(
            "component", {}
        ).get("physical_modes_per_specimen", 0),
        "heldout_relative_rmse": modal_validation.get(
            "heldout_relative_rmse"
        ),
        "maximum_absolute_relative_error": modal_validation.get(
            "heldout_maximum_absolute_relative_error"
        ),
        "component_modal_data_used_for_calibration": modal_validation.get(
            "gates", {}
        ).get("component_modal_data_used_for_calibration"),
        "local_axis_directional_calibration_released": modal_validation.get(
            "gates", {}
        ).get("local_axis_directional_calibration_released"),
        "manifest": (
            _relative(modal_validation_path)
            if modal_validation_path.exists()
            else None
        ),
        "manifest_sha256": (
            _sha256(modal_validation_path)
            if modal_validation_path.exists()
            else None
        ),
        "allowed_use": modal_validation.get("allowed_use"),
    }

    selected_with_card = [
        selection
        for selection in selections
        if selection.canonical_card is not None
    ]
    excluded = [
        selection
        for selection in selections
        if selection.status in {
            "unsupported",
            "rejected",
            "guardrail_only",
        }
    ]
    gate_ok = (
        len(cards) == 9
        and all(
            (PROJECT_ROOT / str(selection.canonical_card)).exists()
            for selection in selected_with_card
        )
        and all(
            selection.canonical_card is None for selection in excluded
        )
        and len({request.request_id for request in DEFAULT_REQUESTS})
        == len(DEFAULT_REQUESTS)
    )
    payload = {
        "material": "Bambu PLA Basic",
        "twin_id": "bambu_pla_basic_evidence_twin_v1",
        "scope": (
            "condition-aware selection across exact-brand static, pre-peak, "
            "DMA, broadband wave/modal, flexural, compression, interlayer, "
            "freeze/recovery, and ballistic evidence"
        ),
        "canonical_cards": cards,
        "canonical_card_count": len(cards),
        "selection_count": len(selections),
        "selection_status_counts": {
            status: sum(selection.status == status for selection in selections)
            for status in sorted({selection.status for selection in selections})
        },
        "evidence_file_count": len(source_rows),
        "calibration_handoffs": {
            "directional_material": directional_handoff,
            "service_laws": service_handoff,
        },
        "heldout_validations": {
            "exact_geometry_small_strain_modal": heldout_modal_validation,
        },
        "selection_gate_passed": gate_ok,
        "selections": [asdict(selection) for selection in selections],
        "unresolved": [
            "verified bead/local-axis tensile mapping and independent curves",
            "measured out-of-plane Poisson and quasistatic shear constants",
            "general service-temperature viscoelastic constitutive law",
            "creep, fatigue, moisture, and aging laws",
            "validated general strain-rate law",
            "matched fracture and interlayer traction-separation calibration",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Bambu PLA Basic evidence digital twin

This manifest is the canonical condition-to-model selector for the project.
It consolidates **{len(cards)} ready-to-use CalculiX material cards** and
**{len(source_rows)} checksum-inventoried evidence artifacts**.

The default matrix exercises {len(selections)} representative requests:
room-temperature isotropic and directional elasticity, in-plane pre-peak
plasticity, explicit-topology compression, flexural screening, 30 C / 1 Hz
storage stiffness, the 18 C / 1 Hz-25 kHz wave/modal screen, interlayer
initiation, the narrow ballistic screen, the published -80 C/24 h plus
room-recovery sensitivity, and unsupported warm, hot, humid, creep, fatigue,
build-plastic, and post-limit requests.

Every selection reports fidelity, solver compatibility, evidence files, and
limitations. Unsupported or rejected conditions never receive a fallback
material card. This prevents a valid room-temperature card from being reused
as an implicit heat, lifetime, humidity, or impact law.

Selection gate: **{'passed' if gate_ok else 'failed'}**.

This integration layer does not erase the unresolved measurements listed in
the manifest; it makes their absence enforceable at model-selection time.
The 90-specimen service-law campaign and its bounded calibrators are recorded
as a calibration handoff, not as evidence or released material cards.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the canonical Bambu PLA Basic digital twin"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = write_digital_twin(args.output)
    print(
        json.dumps(
            {
                "twin_id": payload["twin_id"],
                "canonical_card_count": payload[
                    "canonical_card_count"
                ],
                "selection_count": payload["selection_count"],
                "selection_status_counts": payload[
                    "selection_status_counts"
                ],
                "selection_gate_passed": payload[
                    "selection_gate_passed"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
