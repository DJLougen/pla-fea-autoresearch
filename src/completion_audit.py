from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from .calculix import find_ccx
from .materials import check_admissibility
from .parsing import parse_frd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/completion_audit"


@dataclass(frozen=True)
class AuditItem:
    category: str
    requirement: str
    status: str
    evidence: str


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def audit_project(
    root: Path = PROJECT_ROOT, run_tests: bool = True
) -> List[AuditItem]:
    root = Path(root)
    items: List[AuditItem] = []

    required = [
        "README.md",
        "program.md",
        "pyproject.toml",
        "src/prepare.py",
        "src/evaluate.py",
        "src/optimize.py",
        "src/autoresearch.py",
        "src/report.py",
        "src/bambu_flexural_digitization.py",
        "src/bambu_flexural_process_reference.py",
        "src/published_flexural_validation.py",
        "src/flexural_process_strength_screen.py",
        "src/adaptive_infill_external_validation.py",
        "src/bambu_interlayer_reference.py",
        "src/interlayer_initiation_screen.py",
        "src/bambu_print_speed_reference.py",
        "src/print_speed_infill_sensitivity.py",
        "src/bambu_ballistic_reference.py",
        "src/ballistic_pattern_validation.py",
        "src/volumetric_infill.py",
        "src/volumetric_pattern_validation.py",
        "src/bambu_common_infill_reference.py",
        "src/common_infill_compression_validation.py",
        "src/bambu_pvh_reference.py",
        "src/pvh_material_transfer_sensitivity.py",
        "src/multiaxial_failure_sensitivity.py",
        "src/bambu_modal_component_validation.py",
        "src/mendeley_tensile_reconciliation_audit.py",
        "src/raw_specimen_infill_uncertainty.py",
        "src/bambu_print_tolerance_infill.py",
        "src/campaign_ingestion.py",
        "src/characterization_identifiability.py",
        "src/directional_material_calibration.py",
        "src/static_poisson_propagation.py",
        "src/directional_interval_robustness.py",
        "src/custom_infill.py",
        "src/custom_pattern_fixture.py",
        "src/custom_pattern_print_mesh.py",
        "src/manual_bending_capture.py",
        "src/custom_pattern_slicer.py",
        "src/custom_pattern_toolpath.py",
        "src/custom_pattern_toolpath_fea.py",
        "src/custom_pattern_validation.py",
        "src/bambu_dma_reference.py",
        "src/bambu_dynamic_profile.py",
        "src/bambu_viscoelastic_wave_reference.py",
        "src/bambu_short_time_relaxation.py",
        "src/frequency_dependent_modal_screen.py",
        "src/toolpath_first_limit_screen.py",
        "src/viscoelastic_pattern_modal_screen.py",
        "src/weird_infill_search.py",
        "src/weird_infill_toolpath_validation.py",
        "src/weird_infill_physical_validation.py",
        "src/pattern_robustness_campaign.py",
        "src/bambu_service_envelope.py",
        "src/bambu_digital_twin.py",
        "src/bambu_cryogenic_reference.py",
        "src/cryogenic_pattern_sensitivity.py",
        "src/bambu_nozzle_temperature_profiles.py",
        "src/dma_storage_sensitivity.py",
        "src/nozzle_temperature_sensitivity.py",
        "ideas/README.md",
        "ideas/graded_hourglass.json",
        "configs/printing/bambu_pla_pattern_validation_process.json",
        "configs/printing/bambu_pla_pattern_validation_filament.json",
        "meshes/tensile_x.inp.template",
        "meshes/tensile_y.inp.template",
        "meshes/tensile_z.inp.template",
        "meshes/flexural.inp.template",
        "meshes/heldout_component.inp.template",
    ]
    missing = [path for path in required if not (root / path).exists()]
    items.append(
        AuditItem(
            "software",
            "Repository scaffold and required fixed workflow files",
            "proven" if not missing else "missing",
            "all required files present"
            if not missing
            else f"missing: {', '.join(missing)}",
        )
    )

    executable = find_ccx("ccx")
    items.append(
        AuditItem(
            "software",
            "CalculiX executable detected",
            "proven" if executable else "missing",
            str(executable) if executable else "ccx not found",
        )
    )

    if run_tests:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        match = re.search(r"(\d+) passed", output)
        evidence = (
            f"{match.group(1)} passed"
            if completed.returncode == 0 and match
            else output[-1000:]
        )
        items.append(
            AuditItem(
                "software",
                "Automated unit and real-CalculiX integration tests",
                "proven" if completed.returncode == 0 else "missing",
                evidence,
            )
        )

    results_file = root / "results/results.tsv"
    model_classes = set()
    if results_file.exists():
        with results_file.open(encoding="utf-8", newline="") as handle:
            model_classes = {
                row["model_class"] for row in csv.DictReader(handle, delimiter="\t")
            }
    expected_models = {"isotropic", "transverse_isotropic", "orthotropic"}
    items.append(
        AuditItem(
            "elastic autoresearch",
            "Isotropic, transverse-isotropic, and orthotropic hypotheses run",
            "proven" if expected_models <= model_classes else "missing",
            f"logged model classes: {sorted(model_classes)}",
        )
    )
    best_files = [
        root / "results/best/result.json",
        root / "results/best/config.yaml",
        root / "results/best/material.inp",
    ]
    items.append(
        AuditItem(
            "elastic autoresearch",
            "Best configuration and CalculiX material card exported",
            "proven" if all(path.exists() for path in best_files) else "missing",
            ", ".join(str(path.relative_to(root)) for path in best_files),
        )
    )
    best_result_path = root / "results/best/result.json"
    best_result = (
        _json(best_result_path) if best_result_path.exists() else {}
    )
    lateral_score_count = sum(
        bool(metric.get("lateral_strain_used"))
        for partition in ("calibration", "validation")
        for metric in (
            best_result.get(partition, {}).get("per_test", {}).values()
        )
    )
    lateral_solver_count = 0
    for direction in ("1", "2", "3", "12_45", "13_45", "23_45"):
        path = (
            root
            / "runs/lateral_strain_validation"
            / f"direction-{direction}"
            / "model.frd"
        )
        if path.exists() and "frd_tostrain" in parse_frd(path):
            lateral_solver_count += 1
    items.append(
        AuditItem(
            "elastic autoresearch",
            "Lateral-strain/Poisson data enter the fixed score",
            "proven"
            if lateral_score_count == 12 and lateral_solver_count == 6
            else "partial",
            (
                f"{lateral_score_count} synthetic tensile curve scores use "
                f"lateral strain; {lateral_solver_count}/6 real CalculiX "
                "orientation outputs contain transverse strain"
            ),
        )
    )
    processed_manifest = root / "data/processed/manifest.csv"
    partitions = set()
    source_roots = set()
    if processed_manifest.exists():
        with processed_manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                partitions.add(row["partition"])
                source_roots.add(
                    "synthetic"
                    if "/data/example/" in row["source_file"]
                    else "real_or_external"
                )
    items.append(
        AuditItem(
            "elastic autoresearch",
            "Explicit calibration, validation, and held-out partitions",
            "proven"
            if {"calibration", "validation", "heldout"} <= partitions
            else "missing",
            f"partitions={sorted(partitions)}; sources={sorted(source_roots)}",
        )
    )

    public_manifest = (
        root / "data/public/mendeley-8jxw5533py-v2/manifest.csv"
    )
    curve_count = (
        _count_csv_rows(public_manifest) if public_manifest.exists() else 0
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Literal public brand-specific tensile curves imported",
            "proven" if curve_count == 22 else "partial",
            f"{curve_count} curve records; source workbook claims 21",
        )
    )

    tensile_reconciliation_path = (
        root
        / "results/mendeley_tensile_reconciliation_audit/provenance.json"
    )
    tensile_reconciliation = (
        _json(tensile_reconciliation_path)
        if tensile_reconciliation_path.exists()
        else {}
    )
    reconciliation_counts = tensile_reconciliation.get("counts", {})
    reconciliation_gates = tensile_reconciliation.get("gates", {})
    reconciliation_ok = (
        tensile_reconciliation.get("status")
        == "source_reconciliation_failed_non_bijective"
        and reconciliation_counts.get("literal_raw_curve_blocks") == 22
        and reconciliation_counts.get("summary_specimen_cells") == 21
        and reconciliation_counts.get("same_label_peak_matches") == 12
        and reconciliation_counts.get(
            "unique_raw_curves_referenced_by_summary"
        )
        == 13
        and reconciliation_counts.get(
            "raw_curves_omitted_from_summary_peak_values"
        )
        == 9
        and reconciliation_gates.get("source_checksums_passed") is True
        and reconciliation_gates.get(
            "summary_to_raw_mapping_is_one_to_one"
        )
        is False
        and reconciliation_gates.get(
            "directional_calibration_release_allowed"
        )
        is False
    )
    items.append(
        AuditItem(
            "data quality",
            "Exact-brand tensile source reconciliation release gate",
            "proven" if reconciliation_ok else "missing",
            (
                "immutable workbooks verified; 12/21 summary labels match "
                "their literal raw curve, the summary references only "
                "13/22 unique raw curves and omits 9; the non-bijective "
                "mapping is explicitly blocked from directional calibration"
                if reconciliation_ok
                else "tensile reconciliation audit or hard release gate incomplete"
            ),
        )
    )

    raw_profile_path = (
        root / "data/reported/bambu_pla_raw_curve_elastic.json"
    )
    raw_profile = _json(raw_profile_path) if raw_profile_path.exists() else {}
    raw_specimens = (
        raw_profile.get("raw_curve_extraction", {}).get("specimens")
        if raw_profile
        else None
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Brand-specific elastic modulus calibrated from raw curves",
            "proven" if raw_specimens == 22 else "missing",
            (
                f"Huber E={raw_profile['screening_isotropic']['E_MPa']:.3f} "
                f"MPa from {raw_specimens} curves"
                if raw_profile
                else "profile missing"
            ),
        )
    )

    specimen_uncertainty_path = (
        root / "results/raw_specimen_infill_uncertainty/summary.json"
    )
    specimen_uncertainty = (
        _json(specimen_uncertainty_path)
        if specimen_uncertainty_path.exists()
        else {}
    )
    specimen_uncertainty_gates = specimen_uncertainty.get("gates", {})
    specimen_solver_cases = specimen_uncertainty.get("solver_cases", {})
    specimen_uncertainty_ok = (
        specimen_uncertainty.get("status")
        == "empirical_scalar_modulus_ensemble_passed"
        and specimen_uncertainty.get("raw_specimen_modulus_count") == 22
        and specimen_solver_cases.get("baseline_finalist_cases") == 20
        and specimen_solver_cases.get("direct_endpoint_check_cases") == 10
        and float(
            specimen_uncertainty.get(
                "maximum_scalar_linearity_relative_error", 1.0
            )
        )
        <= 0.005
        and specimen_uncertainty.get("balanced_winner") == "triangular"
        and specimen_uncertainty_gates.get(
            "source_files_present_and_checksum_locked"
        )
        is True
        and specimen_uncertainty_gates.get(
            "baseline_solver_matrix_passed"
        )
        is True
        and specimen_uncertainty_gates.get(
            "endpoint_direct_solver_checks_passed"
        )
        is True
        and specimen_uncertainty_gates.get(
            "scalar_linearity_gate_passed"
        )
        is True
        and specimen_uncertainty_gates.get(
            "directional_calibration_release_allowed"
        )
        is False
        and specimen_uncertainty_gates.get(
            "absolute_failure_prediction_allowed"
        )
        is False
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Raw-specimen modulus uncertainty propagated through infill finalists",
            "proven" if specimen_uncertainty_ok else "missing",
            (
                "22 exact-brand specimen moduli propagated through four "
                "equal-mass finalists and five loads; 20 baseline plus 10 "
                "direct endpoint CalculiX cases; scalar propagation error="
                f"{float(specimen_uncertainty.get('maximum_scalar_linearity_relative_error', 0.0)):.3%}; "
                "all empirical ranks stable, triangular balanced winner; "
                "directional and failure uncertainty remain explicitly blocked"
                if specimen_uncertainty_ok
                else "raw-specimen finalist uncertainty matrix incomplete"
            ),
        )
    )

    print_tolerance_path = (
        root / "results/bambu_print_tolerance_infill/summary.json"
    )
    print_tolerance = (
        _json(print_tolerance_path)
        if print_tolerance_path.exists()
        else {}
    )
    print_tolerance_gates = print_tolerance.get("gates", {})
    print_tolerance_cases = print_tolerance.get("solver_cases", {})
    solid_bending_tolerance = print_tolerance.get(
        "solid_uniform_kite_to_triangular_bending_ratio", {}
    )
    print_tolerance_ok = (
        print_tolerance.get("status")
        == "measured_print_tolerance_sensitivity_passed"
        and print_tolerance.get("cube_count") == 25
        and print_tolerance.get("measurements_per_axis") == 75
        and print_tolerance_cases.get("beam_bending_cases") == 200
        and print_tolerance_cases.get(
            "selected_solid_multiload_cases"
        )
        == 60
        and print_tolerance.get("solid_balanced_winners")
        == ["triangular", "triangular", "triangular"]
        and float(solid_bending_tolerance.get("minimum", 0.0)) > 1.38
        and float(solid_bending_tolerance.get("maximum", 2.0)) < 1.40
        and print_tolerance_gates.get("source_checksum_passed") is True
        and print_tolerance_gates.get(
            "twenty_five_cube_measurements_passed"
        )
        is True
        and print_tolerance_gates.get(
            "reported_axis_statistics_crosscheck_passed"
        )
        is True
        and print_tolerance_gates.get(
            "beam_scenario_matrix_passed"
        )
        is True
        and print_tolerance_gates.get(
            "solid_selected_matrix_passed"
        )
        is True
        and print_tolerance_gates.get("balanced_winner_stable") is True
        and print_tolerance_gates.get(
            "absolute_failure_prediction_allowed"
        )
        is False
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Measured Bambu X1C dimensional tolerance propagated through finalists",
            "proven" if print_tolerance_ok else "missing",
            (
                "25 exact-brand 20 mm cubes, 75 measurements/axis, and "
                "published X/Y/Z statistics checksum-crosschecked; 200 beam "
                "plus 60 selected solid cases bracket additive-offset and "
                "multiplicative-scale transfer; triangular remains balanced "
                "winner and uniform-kite bending ratio spans "
                f"{float(solid_bending_tolerance.get('minimum', 0.0)):.3f}–"
                f"{float(solid_bending_tolerance.get('maximum', 0.0)):.3f}; "
                "failure prediction remains blocked"
                if print_tolerance_ok
                else "measured print-tolerance propagation incomplete"
            ),
        )
    )

    transverse_path = (
        root / "data/reported/bambu_pla_tds_transverse_sensitivity.json"
    )
    transverse = _json(transverse_path) if transverse_path.exists() else {}
    constants = transverse.get("screening_engineering_constants", {})
    admissible = check_admissibility(constants).valid if constants else False
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Local-axis transverse-isotropic sensitivity is admissible",
            "partial" if admissible else "missing",
            (
                "E1/E2 and E3 are reported; nu12 is a separate exact-brand "
                "measurement; nu13/nu23 and G13/G23 remain priors"
                if admissible
                else "profile missing or inadmissible"
            ),
        )
    )
    poisson_propagation_path = (
        root / "results/static_poisson_propagation/summary.json"
    )
    poisson_propagation = (
        _json(poisson_propagation_path)
        if poisson_propagation_path.exists()
        else {}
    )
    poisson_reference = transverse.get(
        "supplemental_in_plane_poisson_evidence", {}
    )
    poisson_update_ok = (
        transverse.get("profile_id")
        == "bambu_pla_tds_transverse_sensitivity_v2"
        and abs(float(constants.get("nu12", 0.0)) - 0.3459) <= 1.0e-12
        and poisson_reference.get("reference_id")
        == "bambu_pla_viscoelastic_wave_2026_v1"
        and poisson_propagation.get("study_id")
        == "bambu_pla_static_inplane_poisson_propagation_v1"
        and poisson_propagation.get("solver_case_count") == 50
        and poisson_propagation.get("all_solver_cases_passed") is True
        and poisson_propagation.get("rank_order_preserved") is True
        and poisson_propagation.get("propagation_gate_passed") is True
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Exact-brand measured in-plane Poisson integration",
            "proven" if poisson_update_ok else "partial",
            (
                "canonical transverse profile v2 uses measured nu12=0.3459; "
                "50/50 matched prior-versus-measured 3D cases pass, all "
                "five-pattern rankings are preserved, and maximum individual "
                "metric change is "
                f"{float(poisson_propagation.get('maximum_absolute_metric_change', 0.0)):.3%}; "
                "out-of-plane Poisson and direct quasistatic shear remain "
                "explicitly unresolved"
                if poisson_update_ok
                else "measured in-plane Poisson propagation is incomplete"
            ),
        )
    )

    plastic_path = (
        root / "data/reported/bambu_pla_raw_curve_prepeak_plastic.json"
    )
    plastic = _json(plastic_path) if plastic_path.exists() else {}
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Pre-peak nonlinear tensile sensitivity",
            "partial" if plastic.get("plastic_isotropic") else "missing",
            (
                "0.2%-offset, pre-peak, in-layer J2 card; no post-peak failure"
                if plastic
                else "profile missing"
            ),
        )
    )

    flexural_root = (
        root / "data/reported/bambu_pla_published_flexural_220C"
    )
    flexural_provenance_path = flexural_root / "provenance.json"
    flexural_provenance = (
        _json(flexural_provenance_path)
        if flexural_provenance_path.exists()
        else {}
    )
    digitization_path = flexural_root / "digitization_validation.csv"
    digitization_rows = []
    if digitization_path.exists():
        with digitization_path.open(
            encoding="utf-8", newline=""
        ) as handle:
            digitization_rows = list(csv.DictReader(handle))
    digitization_ok = (
        len(digitization_rows) == 3
        and all(
            row.get("passes_table_crosscheck", "").lower() == "true"
            for row in digitization_rows
        )
        and flexural_provenance.get("quality_gate_passed") is True
    )
    flexural_comparison_path = (
        root / "results/published_flexural_validation/comparison.json"
    )
    flexural_comparison = (
        _json(flexural_comparison_path)
        if flexural_comparison_path.exists()
        else []
    )
    flexural_solver_ok = (
        len(flexural_comparison) == 3
        and all(row.get("status") == "ok" for row in flexural_comparison)
    )
    flexural_mae = (
        float(
            sum(
                abs(float(row["relative_error_to_table"]))
                for row in flexural_comparison
            )
            / len(flexural_comparison)
        )
        if flexural_solver_ok
        else float("nan")
    )
    mesh_path = (
        root / "results/published_flexural_validation/mesh_convergence.csv"
    )
    mesh_rows = []
    if mesh_path.exists():
        with mesh_path.open(encoding="utf-8", newline="") as handle:
            mesh_rows = list(csv.DictReader(handle))
    mesh_modulus = {
        row["mesh_level"]: float(row["apparent_modulus_MPa"])
        for row in mesh_rows
        if row.get("status") == "ok"
    }
    mesh_change = (
        abs(mesh_modulus["fine"] / mesh_modulus["medium"] - 1.0)
        if {"medium", "fine"} <= mesh_modulus.keys()
        else float("inf")
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Published 220 C flexural evidence and matched FEA comparison",
            "partial"
            if digitization_ok
            and flexural_solver_ok
            and mesh_change <= 0.02
            else "missing",
            (
                f"3/3 digitizations cross-check Table 3; 3/3 CalculiX "
                f"directions; modulus MAE={flexural_mae:.1%}; "
                f"medium-to-fine change={mesh_change:.2%}; plotted curve "
                "aggregation and raw specimens are unavailable"
            ),
        )
    )
    dma_root = root / "data/reported/bambu_pla_published_dma_220C"
    dma_provenance_path = dma_root / "provenance.json"
    dma_provenance = (
        _json(dma_provenance_path)
        if dma_provenance_path.exists()
        else {}
    )
    dma_rows_path = dma_root / "dma_30C_1Hz.csv"
    dma_rows = []
    if dma_rows_path.exists():
        with dma_rows_path.open(encoding="utf-8", newline="") as handle:
            dma_rows = list(csv.DictReader(handle))
    dynamic_profile_path = (
        root / "data/reported/bambu_pla_dma_30C_1Hz_sensitivity.json"
    )
    dynamic_profile = (
        _json(dynamic_profile_path)
        if dynamic_profile_path.exists()
        else {}
    )
    dma_sensitivity_path = (
        root / "results/dma_storage_sensitivity/summary.json"
    )
    dma_sensitivity = (
        _json(dma_sensitivity_path)
        if dma_sensitivity_path.exists()
        else {}
    )
    dma_ok = (
        dma_provenance.get("quality_gate_passed") is True
        and len(
            [
                row
                for row in dma_rows
                if int(float(row["nozzle_temperature_C"])) == 220
            ]
        )
        == 3
        and all(
            row.get("passes_independent_crossplot", "").lower()
            == "true"
            for row in dma_rows
            if int(float(row["nozzle_temperature_C"])) == 220
        )
        and dynamic_profile.get("admissibility", {}).get("valid") is True
        and dma_sensitivity.get("successful_cases") == 10
        and all(
            check.get("preserved") is True
            for check in dma_sensitivity.get(
                "ranking_checks", {}
            ).values()
        )
        and len(dma_sensitivity.get("ranking_checks", {})) == 5
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Published 30 C / 1 Hz DMA storage-stiffness sensitivity",
            "partial" if dma_ok else "missing",
            (
                "3/3 Figure 7/9 points cross-check; admissible orthotropic "
                "storage proxy; 10/10 solid cases and 5/5 topology rankings "
                "preserved; loss modulus/frequency sweep/raw DMA unavailable"
                if dma_ok
                else "DMA reference or sensitivity run missing/inconsistent"
            ),
        )
    )
    nozzle_profile_manifest_path = (
        root
        / "data/reported/bambu_pla_nozzle_temperature_sensitivity"
        / "manifest.json"
    )
    nozzle_profiles = (
        _json(nozzle_profile_manifest_path)
        if nozzle_profile_manifest_path.exists()
        else []
    )
    nozzle_summary_path = (
        root / "results/nozzle_temperature_sensitivity/summary.json"
    )
    nozzle_summary = (
        _json(nozzle_summary_path)
        if nozzle_summary_path.exists()
        else {}
    )
    nozzle_rankings = nozzle_summary.get("ranking_checks", {})
    nozzle_ok = (
        len(dma_rows) == 12
        and sum(
            row.get("passes_independent_crossplot", "").lower()
            == "true"
            for row in dma_rows
        )
        == 10
        and len(nozzle_profiles) == 4
        and all(
            (root / Path(str(row["profile_path"]))).exists()
            for row in nozzle_profiles
        )
        and nozzle_summary.get("successful_cases") == 40
        and len(nozzle_rankings) == 5
        and all(
            check.get("preserved") is True
            for check in nozzle_rankings.values()
        )
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Published nozzle-temperature storage-process sensitivity",
            "partial" if nozzle_ok else "missing",
            (
                "12 plotted points, 10/12 strict crossplot matches; "
                "4/4 admissible process profiles; 40/40 solid cases and "
                "5/5 topology rankings preserved; flexural strength is "
                "screened separately; no flow/void/tensile/shear law"
                if nozzle_ok
                else "nozzle-temperature profiles or sensitivity incomplete"
            ),
        )
    )
    flexural_process_root = (
        root / "data/reported/bambu_pla_published_flexural_process"
    )
    flexural_process_provenance_path = (
        flexural_process_root / "provenance.json"
    )
    flexural_process_provenance = (
        _json(flexural_process_provenance_path)
        if flexural_process_provenance_path.exists()
        else {}
    )
    flexural_process_rows_path = (
        flexural_process_root / "flexural_process_matrix.json"
    )
    flexural_process_rows = (
        _json(flexural_process_rows_path)
        if flexural_process_rows_path.exists()
        else []
    )
    flexural_screen_path = (
        root / "results/flexural_process_strength_screen/screen.json"
    )
    flexural_screen_rows = (
        _json(flexural_screen_path) if flexural_screen_path.exists() else []
    )
    flexural_process_ok = (
        flexural_process_provenance.get("quality_gate_passed") is True
        and len(flexural_process_rows) == 12
        and all(
            row.get("passes_crossplot_and_table_gate") is True
            for row in flexural_process_rows
        )
        and len(flexural_screen_rows) == 8
        and all(
            row.get("screen_status") == "ok"
            for row in flexural_screen_rows
        )
    )
    flexural_peak_crossings = sum(
        row.get("peak_crosses_mean_flexural_reference") is True
        for row in flexural_screen_rows
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Published process-adjusted flexural strength sensitivity",
            "partial" if flexural_process_ok else "missing",
            (
                "12/12 exact Table 3 rows pass Figure 5/6 checks; "
                f"8/8 bending fields screened; mean-reference crossings="
                f"{flexural_peak_crossings}; build-Z remains a scaled proxy "
                "and no crack-growth/final-load law exists"
                if flexural_process_ok
                else "flexural process reference or field screen incomplete"
            ),
        )
    )

    solid_summary = root / "results/solid_multiload_ideas/summary.json"
    solid_rows = _json(solid_summary) if solid_summary.exists() else []
    solid_ok = sum(row.get("status") == "ok" for row in solid_rows)
    items.append(
        AuditItem(
            "infill simulation",
            "Equal-actual-mass 3D solid multi-load pattern matrix",
            "proven" if solid_ok == 20 else "partial",
            f"{solid_ok}/20 tension, shear, bending, compression, buckling cases",
        )
    )

    volumetric_summary_path = (
        root / "results/volumetric_pattern_validation/summary.json"
    )
    volumetric_summary = (
        _json(volumetric_summary_path)
        if volumetric_summary_path.exists()
        else {}
    )
    volumetric_plan_path = (
        root
        / "results/volumetric_pattern_validation"
        / "physical_compression_test_plan.csv"
    )
    volumetric_plan_rows = (
        _count_csv_rows(volumetric_plan_path)
        if volumetric_plan_path.exists()
        else 0
    )
    volumetric_ok = (
        volumetric_summary.get("successful_solver_cases") == 24
        and volumetric_summary.get("total_solver_cases") == 24
        and volumetric_summary.get("watertight_printable_stl_count") == 4
        and volumetric_summary.get("physical_compression_specimens") == 36
        and volumetric_plan_rows == 36
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Explicit 3D cellular patterns with physical ranking challenge",
            "partial" if volumetric_ok else "missing",
            (
                "24/24 equal-mass honeycomb/cubic/grid/gyroid cases; "
                "24/32/40 voxel mesh check <= "
                f"{float(volumetric_summary.get('maximum_medium_to_fine_change', 0.0)):.1%}; "
                "4/4 volume-matched watertight STLs; 36 randomized "
                "compression coupons; physical ballistic ranking reproduced "
                f"{volumetric_summary.get('full_physical_ranking_matches')}/4 "
                "times with "
                f"{volumetric_summary.get('pairwise_matches')}/24 pairwise "
                "orders, so the proxy is not a validated impact predictor"
                if volumetric_ok
                else "explicit volumetric validation or physical handoff incomplete"
            ),
        )
    )

    common_reference_root = (
        root
        / "data/reported/bambu_pla_common_infill_compression_2025"
    )
    common_provenance_path = common_reference_root / "provenance.json"
    common_provenance = (
        _json(common_provenance_path)
        if common_provenance_path.exists()
        else {}
    )
    common_curve_path = common_reference_root / "curve_validation.csv"
    common_curve_rows = []
    if common_curve_path.exists():
        with common_curve_path.open(
            encoding="utf-8", newline=""
        ) as handle:
            common_curve_rows = list(csv.DictReader(handle))
    common_summary_path = (
        root / "results/common_infill_compression_validation/summary.json"
    )
    common_summary = (
        _json(common_summary_path)
        if common_summary_path.exists()
        else {}
    )
    common_ok = (
        common_provenance.get("quality_gate_passed") is True
        and len(common_curve_rows) == 9
        and all(
            row.get("quality_gate_passed", "").lower() == "true"
            for row in common_curve_rows
        )
        and common_summary.get("solver_cases_successful") == 18
        and common_summary.get("solver_cases_total") == 18
        and common_summary.get("full_ranking_matches") == 3
        and common_summary.get("mesh_stable_ranking_comparisons") == 3
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Exact-brand quasi-static compression topology benchmark",
            "partial" if common_ok else "missing",
            (
                "27 Bambu PLA Basic physical tests represented by 9/9 "
                "quality-gated aggregate curves; 18/18 mass-matched "
                "CalculiX cases; grid > gyroid > cubic reproduced at "
                "10/20/30% on both 32- and 40-voxel meshes; Spearman="
                f"{float(common_summary.get('spearman_rank_correlation_9_conditions', 0.0)):.3f}; "
                "literal modulus scale conflicts by "
                f"{float(common_summary.get('raw_simulation_to_literal_source_ratio_min', 0.0)):.0f}-"
                f"{float(common_summary.get('raw_simulation_to_literal_source_ratio_max', 0.0)):.0f}x, "
                "while the publication-internal x1000 sensitivity gives "
                f"{float(common_summary.get('raw_simulation_to_x1000_source_ratio_min', 0.0)):.3f}-"
                f"{float(common_summary.get('raw_simulation_to_x1000_source_ratio_max', 0.0)):.3f}x; "
                "preprint aggregate curves and idealized wall-free geometry"
                if common_ok
                else "exact-brand compression reference or matched simulation incomplete"
            ),
        )
    )

    pvh_reference_root = (
        root / "data/reported/bambu_pla_pvh_metamaterial_2025"
    )
    pvh_provenance_path = pvh_reference_root / "provenance.json"
    pvh_provenance = (
        _json(pvh_provenance_path)
        if pvh_provenance_path.exists()
        else {}
    )
    pvh_optimum_path = (
        pvh_reference_root / "reported_optimum_validation.csv"
    )
    pvh_optimum_rows = (
        _count_csv_rows(pvh_optimum_path)
        if pvh_optimum_path.exists()
        else 0
    )
    pvh_transfer_path = (
        root / "results/pvh_material_transfer_sensitivity/summary.json"
    )
    pvh_transfer = (
        _json(pvh_transfer_path) if pvh_transfer_path.exists() else {}
    )
    pvh_ok = (
        pvh_provenance.get("quality_gate_passed") is True
        and pvh_provenance.get("exact_specimen_replay_allowed") is False
        and pvh_optimum_rows == 6
        and pvh_transfer.get("solver_cases_successful") == 8
        and pvh_transfer.get("solver_cases_total") == 8
        and pvh_transfer.get("full_ranking_stable") is True
    )
    pvh_digitization = pvh_provenance.get(
        "digitization_validation", {}
    )
    items.append(
        AuditItem(
            "infill simulation",
            (
                "Peer-reviewed exact-brand metamaterial validation and "
                "material-card transfer"
            ),
            "partial" if pvh_ok else "missing",
            (
                "checksum-pinned CC BY supplement; author simulation versus "
                "physical PVH curve digitized with first-peak error "
                f"{float(pvh_digitization.get('simulation_to_experiment_initial_peak_error', 0.0)):+.1%} "
                "and energy-to-0.45 error "
                f"{float(pvh_digitization.get('simulation_to_experiment_energy_to_0p45_error', 0.0)):+.1%}; "
                "6 reported optimum/baseline FEA-to-test comparisons; "
                "8/8 CalculiX material-transfer cases; two exact-brand cards "
                "give identical four-pattern energy ranking; exact PVH replay "
                "excluded because the random realization, model files, and "
                "post-yield table are unpublished"
                if pvh_ok
                else "PVH reference import or material-transfer screen incomplete"
            ),
        )
    )

    anisotropy_comparison = (
        root / "results/tds_anisotropy_sensitivity/comparison.json"
    )
    anisotropy_rows = (
        _json(anisotropy_comparison)
        if anisotropy_comparison.exists()
        else []
    )
    paired_ok = sum(
        row.get("isotropic_status") == "ok"
        and row.get("transverse_status") == "ok"
        for row in anisotropy_rows
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Paired isotropic/transverse 3D sensitivity",
            "proven" if paired_ok == 20 else "partial",
            f"{paired_ok}/20 paired comparisons",
        )
    )
    parametric_beam = (
        root / "results/parametric_infill_search/beam/summary.json"
    )
    parametric_solid = (
        root / "results/parametric_infill_search/solid/summary.json"
    )
    parametric_nonlinear = (
        root / "results/parametric_infill_search/nonlinear/summary.json"
    )
    beam_rows = _json(parametric_beam) if parametric_beam.exists() else []
    search_solid_rows = (
        _json(parametric_solid) if parametric_solid.exists() else []
    )
    search_nonlinear_rows = (
        _json(parametric_nonlinear)
        if parametric_nonlinear.exists()
        else []
    )
    search_ok = (
        sum(row.get("status") == "ok" for row in beam_rows) == 115
        and sum(
            row.get("status") == "ok" for row in search_solid_rows
        )
        == 35
        and sum(
            row.get("status") == "ok"
            for row in search_nonlinear_rows
        )
        == 2
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Parametric weird-infill search with nonlinear promotion gate",
            "proven" if search_ok else "partial",
            (
                f"{len(beam_rows)} beam, {len(search_solid_rows)} solid, "
                f"{len(search_nonlinear_rows)} nonlinear result rows"
            ),
        )
    )
    custom_root = root / "results/custom_infill/graded_hourglass"
    custom_validation_path = custom_root / "input_validation.json"
    custom_study_path = custom_root / "study.json"
    custom_validation = (
        _json(custom_validation_path)
        if custom_validation_path.exists()
        else {}
    )
    custom_study = (
        _json(custom_study_path) if custom_study_path.exists() else {}
    )
    custom_plan_path = (
        custom_root / "solid/physical_multiload_test_plan.csv"
    )
    custom_plan_rows = (
        _count_csv_rows(custom_plan_path)
        if custom_plan_path.exists()
        else 0
    )
    custom_first_plan_path = (
        custom_root / "first_physical_test_plan.csv"
    )
    custom_first_plan_rows = (
        _count_csv_rows(custom_first_plan_path)
        if custom_first_plan_path.exists()
        else 0
    )
    custom_decision_path = custom_root / "promotion_decision.json"
    custom_decision = (
        _json(custom_decision_path)
        if custom_decision_path.exists()
        else {}
    )
    custom_admission = custom_study.get("material_admission", {})
    convergence = custom_study.get(
        "bending_medium_to_fine_change", {}
    )
    custom_ok = (
        custom_validation.get("status") == "valid"
        and custom_study.get("solid_run_included") is True
        and custom_study.get("successful_beam_cases") == 10
        and custom_study.get("successful_solid_cases") == 10
        and custom_plan_rows == 30
        and custom_admission.get("gate_passed") is True
        and custom_admission.get("twin_id")
        == "bambu_pla_basic_evidence_twin_v1"
        and custom_admission.get("selection", {}).get("model_id")
        == "tds_transverse_elastic_sensitivity_v2"
        and custom_decision.get("specialist_load_cases") == ["bending"]
        and custom_decision.get("mesh_convergence_gate_passed") is True
        and custom_decision.get("first_test_specimen_count") == 6
        and custom_first_plan_rows == 6
        and len(convergence) == 2
        and max(float(value) for value in convergence.values()) <= 0.02
    )
    items.append(
        AuditItem(
            "infill simulation",
            "User-defined pattern gate through printable 3D handoff",
            "proven" if custom_ok else "partial",
            (
                "source hash and graph/manufacturability gates pass; 10/10 "
                "beam plus 10/10 solid cases; bending convergence <=2%; "
                "canonical exact-brand card checksum and service condition "
                "admitted; 30-case matrix reduced to a 6-coupon first "
                "bending validation"
                if custom_ok
                else "custom study missing, incomplete, or unconverged"
            ),
        )
    )
    bowtie_root = root / "results/custom_infill/bowtie_fan"
    bowtie_validation_path = bowtie_root / "input_validation.json"
    bowtie_study_path = bowtie_root / "study.json"
    bowtie_decision_path = bowtie_root / "promotion_decision.json"
    bowtie_print_path = (
        bowtie_root / "physical_validation/printable_stl/manifest.json"
    )
    bowtie_manufacturing_path = (
        bowtie_root / "physical_validation/manufacturing/manifest.json"
    )
    bowtie_morphology_path = (
        bowtie_root / "physical_validation/toolpath_morphology/manifest.json"
    )
    bowtie_toolpath_fea_path = (
        bowtie_root
        / "physical_validation/toolpath_resolved_bending/manifest.json"
    )
    bowtie_package_path = (
        bowtie_root / "physical_validation/manifest.json"
    )
    bowtie_validation = (
        _json(bowtie_validation_path)
        if bowtie_validation_path.exists()
        else {}
    )
    bowtie_study = (
        _json(bowtie_study_path) if bowtie_study_path.exists() else {}
    )
    bowtie_decision = (
        _json(bowtie_decision_path)
        if bowtie_decision_path.exists()
        else {}
    )
    bowtie_print = (
        _json(bowtie_print_path) if bowtie_print_path.exists() else {}
    )
    bowtie_manufacturing = (
        _json(bowtie_manufacturing_path)
        if bowtie_manufacturing_path.exists()
        else {}
    )
    bowtie_morphology = (
        _json(bowtie_morphology_path)
        if bowtie_morphology_path.exists()
        else {}
    )
    bowtie_toolpath_fea = (
        _json(bowtie_toolpath_fea_path)
        if bowtie_toolpath_fea_path.exists()
        else {}
    )
    bowtie_package = (
        _json(bowtie_package_path)
        if bowtie_package_path.exists()
        else {}
    )
    bowtie_templates = len(
        list(
            (
                bowtie_root / "physical_validation/capture_templates"
            ).glob("*.csv")
        )
    )
    bowtie_ok = (
        bowtie_validation.get("status") == "valid"
        and bowtie_study.get("successful_beam_cases") == 10
        and bowtie_study.get("successful_solid_cases") == 10
        and bowtie_decision.get("specialist_load_cases") == ["bending"]
        and bowtie_decision.get("mesh_convergence_gate_passed") is True
        and bowtie_print.get("all_print_mesh_gates_passed") is True
        and bowtie_manufacturing.get("all_manufacturing_gates_passed")
        is True
        and abs(
            float(
                bowtie_manufacturing.get(
                    "toolpath_mass_ratio_error", 1.0
                )
            )
        )
        <= 0.02
        and bowtie_morphology.get("gates", {}).get(
            "relative_linear_stiffness_ranking_admitted"
        )
        is True
        and bowtie_toolpath_fea.get("status")
        == "toolpath_aware_relative_bending_ranking_admitted"
        and bowtie_toolpath_fea.get("successful_solver_case_count") == 6
        and float(
            bowtie_toolpath_fea.get(
                "toolpath_fine_custom_to_control_ratio", 0.0
            )
        )
        >= 1.30
        and bowtie_package.get("validation_id")
        == "bowtie_fan_bending_physical_v1"
        and bowtie_package.get("status") == "awaiting_measurements"
        and bowtie_templates == 6
    )
    items.append(
        AuditItem(
            "infill simulation",
            (
                "Mechanism-inspired bow-tie pattern through "
                "frozen-toolpath handoff"
            ),
            "proven" if bowtie_ok else "partial",
            (
                "10/10 beam plus 10/10 equal-mass solid cases; bending-only "
                f"continuum ratio={float(bowtie_toolpath_fea.get('continuum_custom_to_control_ratio', 0.0)):.3f}; "
                f"120x120x20 frozen-toolpath ratio={float(bowtie_toolpath_fea.get('toolpath_fine_custom_to_control_ratio', 0.0)):.3f}; "
                "6/6 toolpath solves, <=5% convergence changes, watertight "
                "STLs, registered six-object plate, and six raw-curve "
                "templates; real measurements still required"
                if bowtie_ok
                else "bow-tie simulation or printable handoff incomplete"
            ),
        )
    )
    robustness_root = root / "results/pattern_robustness_campaign"
    robustness_path = robustness_root / "summary.json"
    robustness = (
        _json(robustness_path) if robustness_path.exists() else {}
    )
    robustness_physical = robustness.get("physical_validation", {})
    robustness_toolpath = robustness.get("toolpath_replay", {})
    robustness_measurement = robustness.get(
        "measurement_evaluation", {}
    )
    robustness_plate = robustness_root / (
        "physical_validation/bambu-pla-robust-finalist-showdown.3mf"
    )
    robustness_ok = (
        robustness.get("study_id")
        == "bambu_pla_pattern_robustness_campaign_v1"
        and robustness.get("solver_variants") == 146
        and robustness.get("solver_cases") == 730
        and len(robustness.get("ranking", [])) == 4
        and robustness.get("ranking", [{}])[0].get("pattern")
        == "triangular"
        and robustness.get("gates", {}).get(
            "all_nominal_cases_solved"
        )
        is True
        and robustness.get("gates", {}).get(
            "all_patterns_have_single_strut_envelope"
        )
        is True
        and robustness_physical.get("manufacturing_gate_passed") is True
        and robustness_physical.get("build_item_count") == 12
        and robustness_plate.exists()
        and robustness_toolpath.get("solver_case_count") == 12
        and robustness_toolpath.get("gates", {}).get(
            "candidate_set_admitted_for_physical_testing"
        )
        is True
        and robustness_toolpath.get("gates", {}).get(
            "absolute_toolpath_replay_gate_passed"
        )
        is False
        and robustness_measurement.get("status")
        == "awaiting_measurements"
        and robustness_measurement.get(
            "physical_validation_gate_passed"
        )
        is False
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Print-defect and orientation robustness finalist tournament",
            "proven" if robustness_ok else "partial",
            (
                "146 variants and 730 five-load CalculiX cases cover "
                "0/90 orientation, +/-10% bead width, and every removable "
                "interior strut; exact 12-object G-code replay admits the "
                "four-pattern candidate set from converged relative ratios "
                "while correctly refusing absolute convergence and holding "
                "blank physical capture at awaiting_measurements"
                if robustness_ok
                else "pattern robustness campaign or physical handoff incomplete"
            ),
        )
    )
    custom_physical_root = custom_root / "physical_validation"
    custom_physical_manifest_path = (
        custom_physical_root / "manifest.json"
    )
    custom_physical_manifest = (
        _json(custom_physical_manifest_path)
        if custom_physical_manifest_path.exists()
        else {}
    )
    custom_frozen_plan_path = custom_physical_root / "frozen_plan.csv"
    custom_template_count = len(
        list((custom_physical_root / "capture_templates").glob("*.csv"))
    )
    custom_print_mesh_manifest_path = (
        custom_physical_root / "printable_stl/manifest.json"
    )
    custom_print_mesh_manifest = (
        _json(custom_print_mesh_manifest_path)
        if custom_print_mesh_manifest_path.exists()
        else {}
    )
    custom_manufacturing_manifest_path = (
        custom_physical_root / "manufacturing/manifest.json"
    )
    custom_manufacturing_manifest = (
        _json(custom_manufacturing_manifest_path)
        if custom_manufacturing_manifest_path.exists()
        else {}
    )
    custom_toolpath_manifest_path = (
        custom_physical_root / "toolpath_morphology/manifest.json"
    )
    custom_toolpath_manifest = (
        _json(custom_toolpath_manifest_path)
        if custom_toolpath_manifest_path.exists()
        else {}
    )
    custom_toolpath_fea_manifest_path = (
        custom_physical_root
        / "toolpath_resolved_bending/manifest.json"
    )
    custom_toolpath_fea_manifest = (
        _json(custom_toolpath_fea_manifest_path)
        if custom_toolpath_fea_manifest_path.exists()
        else {}
    )
    custom_fixture_manifest_path = (
        custom_physical_root / "fixture/manifest.json"
    )
    custom_fixture_manifest = (
        _json(custom_fixture_manifest_path)
        if custom_fixture_manifest_path.exists()
        else {}
    )
    toolpath_fea_ok = (
        custom_toolpath_fea_manifest.get("status")
        == "toolpath_aware_relative_bending_ranking_admitted"
        and custom_toolpath_fea_manifest.get("solver_case_count") == 6
        and custom_toolpath_fea_manifest.get(
            "successful_solver_case_count"
        )
        == 6
        and custom_toolpath_fea_manifest.get("gates", {}).get(
            "all_rasters_and_voxel_meshes_connected"
        )
        is True
        and custom_toolpath_fea_manifest.get("gates", {}).get(
            "mass_representation_error_at_most_0p1_percent"
        )
        is True
        and custom_toolpath_fea_manifest.get("gates", {}).get(
            "relative_toolpath_aware_bending_ranking_admitted"
        )
        is True
        and custom_toolpath_fea_manifest.get("gates", {}).get(
            "absolute_stiffness_validated"
        )
        is False
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Frozen-slicer toolpath-aware custom-pattern replay",
            "proven" if toolpath_fea_ok else "partial",
            (
                "6/6 mass-calibrated C3D8 solves at 40/60/80 XY cells "
                "with all 20 printed layers; connected morphology and "
                "<=0.1% mass representation error; fine custom/control "
                f"bending ratio="
                f"{float(custom_toolpath_fea_manifest.get('toolpath_fine_custom_to_control_ratio', 0.0)):.3f} "
                "versus continuum "
                f"{float(custom_toolpath_fea_manifest.get('continuum_custom_to_control_ratio', 0.0)):.3f}; "
                "relative ranking admitted, absolute and failure claims "
                "explicitly excluded"
                if toolpath_fea_ok
                else "toolpath-aware bending replay absent or unconverged"
            ),
        )
    )
    custom_feedback_ready = (
        custom_physical_manifest.get("validation_id")
        == "graded_hourglass_bending_physical_v1"
        and custom_physical_manifest.get("status")
        == "awaiting_measurements"
        and custom_physical_manifest.get("specimen_count") == 6
        and custom_physical_manifest.get("replicates_per_pattern") == 3
        and custom_physical_manifest.get("plan_sha256")
        == (
            hashlib.sha256(custom_frozen_plan_path.read_bytes()).hexdigest()
            if custom_frozen_plan_path.exists()
            else None
        )
        and custom_template_count == 6
        and custom_physical_manifest.get("printable_mesh_gate_passed")
        is True
        and custom_print_mesh_manifest.get(
            "all_print_mesh_gates_passed"
        )
        is True
        and custom_print_mesh_manifest.get("mesh_count") == 2
        and custom_print_mesh_manifest.get(
            "watertight_mesh_count"
        )
        == 2
        and custom_physical_manifest.get(
            "manufacturing_gate_passed"
        )
        is True
        and custom_manufacturing_manifest.get(
            "all_manufacturing_gates_passed"
        )
        is True
        and custom_manufacturing_manifest.get(
            "three_mf_inspection", {}
        ).get("build_item_count")
        == 6
        and custom_manufacturing_manifest.get(
            "toolpath_mass_gate_passed"
        )
        is True
        and custom_physical_manifest.get(
            "toolpath_relative_ranking_gate_passed"
        )
        is True
        and custom_toolpath_manifest.get("status")
        == "relative_ranking_admitted_with_restrictions"
        and custom_toolpath_manifest.get("gates", {}).get(
            "relative_linear_stiffness_ranking_admitted"
        )
        is True
        and custom_toolpath_manifest.get("gates", {}).get(
            "absolute_stiffness_validated"
        )
        is False
        and custom_physical_manifest.get(
            "toolpath_resolved_bending_gate_passed"
        )
        is True
        and custom_toolpath_fea_manifest.get("status")
        == "toolpath_aware_relative_bending_ranking_admitted"
        and custom_toolpath_fea_manifest.get(
            "successful_solver_case_count"
        )
        == 6
        and custom_toolpath_fea_manifest.get("gates", {}).get(
            "relative_toolpath_aware_bending_ranking_admitted"
        )
        is True
        and custom_toolpath_fea_manifest.get("gates", {}).get(
            "absolute_stiffness_validated"
        )
        is False
        and custom_physical_manifest.get("fixture_gate_passed")
        is True
        and custom_fixture_manifest.get("all_fixture_gates_passed")
        is True
        and custom_fixture_manifest.get(
            "solver_trace", {}
        ).get("solver_boundary_geometry_gate_passed")
        is True
    )
    items.append(
        AuditItem(
            "physical handoff",
            "Custom-pattern raw-curve feedback and decision pipeline",
            "proven" if custom_feedback_ready else "missing",
            (
                "two watertight exact-simulation STLs, a six-object "
                "checksum-frozen X1C 3MF/G-code plate, and six "
                "checksum-frozen capture templates; a traced 40 mm "
                "steel-roller fixture and compliance protocol; the parsed "
                "bead-volume, feature, layer, and orientation gate admits "
                "relative linear ranking; 6/6 mass-calibrated "
                "40/60/80-cell toolpath-voxel bending solves retain the "
                "custom advantage while excluding absolute and failure "
                "claims; curve-point, "
                "linearity, mass, print-control, service-condition, "
                "three-replicate, bootstrap-ratio, and topology-bias gates "
                "implemented; templates intentionally remain invalid until "
                "real measurements are supplied"
                if custom_feedback_ready
                else "custom-pattern physical feedback package incomplete"
            ),
        )
    )

    weird_root = root / "results/custom_infill/weird_kite_search"
    weird_search_path = weird_root / "summary.json"
    weird_search = (
        _json(weird_search_path) if weird_search_path.exists() else {}
    )
    weird_toolpath_path = weird_root / "toolpath_validation/summary.json"
    weird_toolpath = (
        _json(weird_toolpath_path) if weird_toolpath_path.exists() else {}
    )
    weird_manufacturing_path = (
        weird_root
        / "physical_validation/manufacturing/manifest.json"
    )
    weird_manufacturing = (
        _json(weird_manufacturing_path)
        if weird_manufacturing_path.exists()
        else {}
    )
    weird_evaluation_path = (
        weird_root / "physical_validation/evaluation/summary.json"
    )
    weird_evaluation = (
        _json(weird_evaluation_path)
        if weird_evaluation_path.exists()
        else {}
    )
    weird_ratios = weird_toolpath.get("gate", {}).get("fine_ratios", {})
    weird_ok = (
        weird_search.get("candidate_count") == 82
        and weird_search.get("beam_solver_case_count") == 415
        and weird_search.get("gates", {}).get(
            "candidate_search_admitted_for_physical_testing"
        )
        is True
        and weird_manufacturing.get("build_item_count") == 9
        and weird_manufacturing.get("manufacturing_gate_passed") is True
        and weird_toolpath.get("status")
        == "frozen_toolpath_candidates_admitted_for_physical_testing"
        and weird_toolpath.get("gate", {}).get(
            "toolpath_candidate_gate_passed"
        )
        is True
        and weird_toolpath.get("gate", {}).get(
            "all_static_and_modal_changes_at_most_5_percent"
        )
        is True
        and weird_evaluation.get("status") == "awaiting_measurements"
        and weird_evaluation.get("physical_validation_gate_passed") is False
    )
    best_weird = weird_ratios.get("kite_08_08_08_08", {})
    second_weird = weird_ratios.get("kite_08_08_12_08", {})
    items.append(
        AuditItem(
            "infill simulation",
            "Multiobjective weird-pattern search and frozen physical handoff",
            "partial" if weird_ok else "missing",
            (
                "82 kite variants / 415 beam cases; six promoted to "
                "equal-mass C3D6 static and modal screening; fine frozen-"
                "toolpath ratios versus triangular are "
                f"{float(best_weird.get('bending_to_triangular', 0.0)):.3f} bending/"
                f"{float(best_weird.get('modal_to_triangular', 0.0)):.3f} modal "
                "for kite_08_08_08_08 and "
                f"{float(second_weird.get('bending_to_triangular', 0.0)):.3f}/"
                f"{float(second_weird.get('modal_to_triangular', 0.0)):.3f} "
                "for kite_08_08_12_08; all toolpath convergence changes "
                "<=1.5%; slicer-audited 9-coupon 3MF and strict modal/bending "
                "measurement intake ready; physical rows remain blank"
                if weird_ok
                else "weird-pattern search, toolpath, or physical handoff incomplete"
            ),
        )
    )

    first_limit_path = (
        weird_root / "toolpath_first_limit_screen/summary.json"
    )
    first_limit = (
        _json(first_limit_path) if first_limit_path.exists() else {}
    )
    first_limit_gate = first_limit.get("gate", {})
    first_limit_results = first_limit.get("fine_results", {})
    first_limit_ok = (
        first_limit.get("status")
        == "bounded_toolpath_first_limit_screen_admitted"
        and first_limit.get("solver_case_count") == 6
        and first_limit_gate.get("all_calculix_cases_passed") is True
        and first_limit_gate.get(
            "all_endpoint_strain_fields_present"
        )
        is True
        and first_limit_gate.get(
            "all_first_limit_bands_ordered_and_positive"
        )
        is True
        and first_limit_gate.get(
            "all_60_to_80_limit_changes_at_most_10_percent"
        )
        is True
        and first_limit_gate.get(
            "absolute_failure_physically_validated"
        )
        is False
        and set(first_limit_results) == {
            "triangular",
            "kite_08_08_08_08",
            "kite_08_08_12_08",
        }
    )
    best_first_limit = first_limit_results.get(
        "kite_08_08_08_08", {}
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Frozen-toolpath local-stress first-limit bending screen",
            "partial" if first_limit_ok else "missing",
            (
                "6/6 sliced-path 60/80-cell CalculiX fields pass; "
                "fill-fraction voxel strains are converted to solid-material "
                "local stress; directional tensile/shear and measured "
                "interface sensitivities form ordered p99 first-limit bands "
                "with all mesh changes <=4.2%; planned 0.4 mm ramp is below "
                "every conservative band; uniform-kite conservative/"
                "optimistic limit load="
                f"{float(best_first_limit.get('conservative_p99_first_limit_force_N', 0.0)):.1f}/"
                f"{float(best_first_limit.get('optimistic_p99_first_limit_force_N', 0.0)):.1f} N; "
                "absolute crack initiation and final load remain unvalidated"
                if first_limit_ok
                else "toolpath stress fields or first-limit gates incomplete"
            ),
        )
    )

    custom_analysis_path = (
        custom_physical_root / "analysis/analysis.json"
    )
    custom_analysis = (
        _json(custom_analysis_path)
        if custom_analysis_path.exists()
        else {}
    )
    custom_physical_decision = (
        custom_analysis.get("result", {}).get("decision")
        if custom_analysis.get("result") is not None
        else None
    )
    if custom_physical_decision == "validated_specialist_advantage":
        custom_physical_status = "proven"
        custom_physical_evidence = (
            "three real replicates per topology validate the predicted "
            "bending advantage within the registered agreement gate"
        )
    elif custom_analysis.get("status") == "analyzed":
        custom_physical_status = "partial"
        custom_physical_evidence = (
            "real curves were analyzed but the registered validation "
            f"decision is {custom_physical_decision}"
        )
    else:
        custom_physical_status = "missing"
        custom_physical_evidence = (
            "awaiting six real Bambu PLA Basic bending force-displacement "
            "curves; generated templates are not measurements"
        )
    items.append(
        AuditItem(
            "missing evidence",
            "Matched graded-hourglass versus triangular physical validation",
            custom_physical_status,
            custom_physical_evidence,
        )
    )
    directional_path = (
        root
        / "results/directional_strength_screening"
        / "directional_strength_rows.json"
    )
    directional_rows = (
        _json(directional_path) if directional_path.exists() else []
    )
    directional_ok = sum(
        row.get("screen_status") == "ok" for row in directional_rows
    )
    fresh_directional_path = (
        root
        / "results/directional_strength_screening"
        / "fresh_validation"
        / "summary.json"
    )
    fresh_directional_rows = (
        _json(fresh_directional_path)
        if fresh_directional_path.exists()
        else []
    )
    fresh_directional_ok = any(
        row.get("status") == "ok"
        and row.get("directional_normal_utilization_mean") is not None
        for row in fresh_directional_rows
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Local-axis directional normal-strength hotspot screen",
            "proven"
            if directional_ok == 28 and fresh_directional_ok
            else "partial",
            (
                f"{directional_ok}/28 saved static fields screened; "
                f"fresh CalculiX field path={'ok' if fresh_directional_ok else 'missing'}; "
                "shear/compression/fracture remain outside criterion"
            ),
        )
    )
    multiaxial_path = (
        root
        / "results/multiaxial_failure_sensitivity"
        / "failure_sensitivity_rows.json"
    )
    multiaxial_method_path = (
        root / "results/multiaxial_failure_sensitivity/method.json"
    )
    multiaxial_rows = (
        _json(multiaxial_path) if multiaxial_path.exists() else []
    )
    multiaxial_method = (
        _json(multiaxial_method_path)
        if multiaxial_method_path.exists()
        else {}
    )
    multiaxial_ok = sum(
        row.get("screen_status") == "ok" for row in multiaxial_rows
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Theory-bounded local tension-shear initiation sensitivity",
            "partial" if multiaxial_ok == 36 else "missing",
            (
                f"{multiaxial_ok}/36 static fields screened; "
                f"conservative/optimistic peak crossings="
                f"{multiaxial_method.get('conservative_peak_crossings')}/"
                f"{multiaxial_method.get('optimistic_peak_crossings')}; "
                "shear bookends are theoretical and compression/fracture "
                "remain uncalibrated"
            ),
        )
    )
    uncertainty_root = (
        root / "results/material_uncertainty_sensitivity"
    )
    uncertainty_counts = []
    for scenario in (
        "measured_low",
        "measured_high",
        "shear_low",
        "shear_high",
    ):
        path = uncertainty_root / "scenarios" / scenario / "summary.json"
        rows = _json(path) if path.exists() else []
        uncertainty_counts.append(
            sum(row.get("status") == "ok" for row in rows)
        )
    uncertainty_summary = uncertainty_root / "summary.json"
    uncertainty_payload = (
        _json(uncertainty_summary)
        if uncertainty_summary.exists()
        else {}
    )
    uncertainty_ok = (
        uncertainty_counts == [25, 25, 25, 25]
        and uncertainty_payload.get("winner") == "triangular"
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Reported-modulus and shear-prior robustness envelope",
            "proven" if uncertainty_ok else "partial",
            (
                f"scenario successful cases={uncertainty_counts}; "
                f"maximin winner={uncertainty_payload.get('winner')}"
            ),
        )
    )
    interval_path = (
        root / "results/directional_interval_robustness/summary.json"
    )
    interval = _json(interval_path) if interval_path.exists() else {}
    interval_coarse = interval.get("coarse_solid_candidate_summary", [])
    interval_refined = interval.get(
        "refined_solid_candidate_summary", []
    )
    coarse_ranks = {
        str(row.get("pattern")): (
            int(row.get("best_sample_rank", 0)),
            int(row.get("worst_sample_rank", 0)),
        )
        for row in interval_coarse
    }
    refined_by_pattern = {
        str(row.get("pattern")): row for row in interval_refined
    }
    bowtie_refined = refined_by_pattern.get("bowtie_fan", {})
    bowtie_bending = bowtie_refined.get("load_envelopes", {}).get(
        "bending", {}
    )
    interval_ok = (
        interval.get("study_id")
        == "bambu_pla_directional_interval_robustness_v1"
        and interval.get("status") == "bounded_sensitivity_only"
        and interval.get("material_sample_count") == 44
        and interval.get("beam_solver_case_count") == 880
        and interval.get("beam_interval_response_status")
        == "invariant_not_used_for_material_promotion"
        and interval.get("coarse_solid_solver_case_count") == 880
        and interval.get("promoted_adversarial_sample_count") == 6
        and interval.get("refined_solid_solver_case_count") == 120
        and interval.get("all_solver_cases_passed") is True
        and interval.get("ratio_refinement_gate_passed") is True
        and float(
            interval.get(
                "maximum_absolute_refinement_ratio_change", 1.0
            )
        )
        <= 0.05
        and coarse_ranks.get("triangular") == (1, 1)
        and coarse_ranks.get("bowtie_fan") == (2, 2)
        and float(bowtie_bending.get("minimum_ratio", 0.0)) >= 1.30
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Full directional-constant interval robustness for finalists",
            "proven" if interval_ok else "partial",
            (
                "44 admissible non-probabilistic cards; 880 beam cases "
                "diagnose beam invariance; 880 coarse plus 120 refined "
                "equal-mass solid cases pass; triangular remains rank 1 and "
                "bowtie_fan rank 2 in every sampled card; adverse refined "
                "bow-tie bending ratio="
                f"{float(bowtie_bending.get('minimum_ratio', 0.0)):.3f}; "
                "maximum ratio refinement change="
                f"{float(interval.get('maximum_absolute_refinement_ratio_change', 0.0)):.2%}; "
                "unmeasured ranges remain sensitivity assumptions"
                if interval_ok
                else "directional interval campaign missing or incomplete"
            ),
        )
    )

    physical_plan = (
        root
        / "results/solid_multiload_ideas/physical_multiload_test_plan.csv"
    )
    plan_rows = _count_csv_rows(physical_plan) if physical_plan.exists() else 0
    items.append(
        AuditItem(
            "physical handoff",
            "Printable geometry and replicated test plan",
            "proven" if plan_rows >= 24 else "missing",
            f"{plan_rows} planned specimens plus generated SCAD coupons",
        )
    )
    manual_template_dir = (
        root
        / "results/custom_infill/graded_hourglass/physical_validation/"
        "manual_capture_templates"
    )
    manual_manifest_path = manual_template_dir / "template_manifest.json"
    manual_manifest = (
        _json(manual_manifest_path)
        if manual_manifest_path.exists()
        else {}
    )
    manual_template_count = len(list(manual_template_dir.glob("*.csv")))
    manual_adapter_ok = (
        (root / "src/manual_bending_capture.py").exists()
        and (root / "tests/test_manual_bending_capture.py").exists()
        and manual_template_count == 6
        and manual_manifest.get("specimen_count") == 6
        and manual_manifest.get("quality_gates", {}).get(
            "minimum_force_capacity_N"
        )
        == 50.0
        and manual_manifest.get("quality_gates", {}).get(
            "minimum_acquisition_rate_Hz"
        )
        == 10.0
    )
    items.append(
        AuditItem(
            "physical handoff",
            "Calibrated low-cost bending capture adapter",
            "proven" if manual_adapter_ok else "partial",
            (
                "six checksum-linked raw templates support calibrated scale "
                "or load-cell force and local or compliance-corrected "
                "displacement; uncertainty, capacity, resolution, sampling, "
                "and unchanged canonical-validation gates are tested"
                if manual_adapter_ok
                else "manual adapter or its six generated templates are absent"
            ),
        )
    )
    campaign_audit_path = (
        root / "results/campaign_ingestion_audit/audit.json"
    )
    campaign_audit = (
        _json(campaign_audit_path)
        if campaign_audit_path.exists()
        else {}
    )
    blank_templates_rejected = (
        campaign_audit.get("ready_for_fixed_preprocessing") is False
        and campaign_audit.get("counts", {}).get("invalid") == 43
        and campaign_audit.get("counts", {}).get("missing") == 0
    )
    items.append(
        AuditItem(
            "physical handoff",
            "Checksum-pinned campaign ingestion and quality gates",
            "proven" if blank_templates_rejected else "partial",
            (
                "43/43 metadata-only templates correctly rejected; gates "
                "lock specimen IDs, partitions, process metadata, point "
                "counts, lateral coverage, units, and preprocessing"
                if blank_templates_rejected
                else "campaign ingestion audit missing or inconsistent"
            ),
        )
    )
    campaign_plan = (
        root / "results/characterization_campaign/test_plan.csv"
    )
    campaign_rows = (
        _count_csv_rows(campaign_plan) if campaign_plan.exists() else 0
    )
    capture_count = len(
        list(
            (
                root
                / "results/characterization_campaign/capture_templates"
            ).glob("*.csv")
        )
    )
    items.append(
        AuditItem(
            "physical handoff",
            "Pre-partitioned local-axis/off-axis characterization campaign",
            "proven"
            if campaign_rows == 43 and capture_count == 43
            else "partial",
            (
                f"{campaign_rows} randomized specimens and "
                f"{capture_count} pipeline-ready capture templates"
            ),
        )
    )
    identifiability_path = (
        root / "results/characterization_identifiability/summary.json"
    )
    identifiability = (
        _json(identifiability_path)
        if identifiability_path.exists()
        else {}
    )
    identifiability_matrix = identifiability.get(
        "normalized_jacobian", {}
    )
    identifiability_ok = (
        identifiability.get("full_rank_gate_passed") is True
        and identifiability.get("condition_gate_passed") is True
        and identifiability.get("ablation_gate_passed") is True
        and identifiability.get("all_parameters_have_sensitivity") is True
        and identifiability_matrix.get("rank") == 9
    )
    items.append(
        AuditItem(
            "physical handoff",
            "Nine-constant directional campaign identifiability gate",
            "proven" if identifiability_ok else "partial",
            (
                "12 axial/lateral observables for 9 orthotropic constants; "
                "normalized sensitivity rank=9/9, condition number="
                f"{float(identifiability_matrix.get('condition_number', 0.0)):.3f}; "
                "axial-only, principal-only, missing build-coupled off-axis, "
                "and missing in-plane off-axis ablations all lose full rank; "
                "this proves campaign design, not acquisition"
                if identifiability_ok
                else "campaign identifiability analysis missing or rank deficient"
            ),
        )
    )
    directional_calibration_path = (
        root
        / "results/directional_material_calibration/"
        "calibration_manifest.json"
    )
    directional_calibration = (
        _json(directional_calibration_path)
        if directional_calibration_path.exists()
        else {}
    )
    directional_calibrator_ok = (
        directional_calibration.get("calibration_id")
        == "bambu_pla_basic_directional_calibration_v1"
        and directional_calibration.get("status")
        == "awaiting_measurements"
        and directional_calibration.get("card_released") is False
        and directional_calibration.get("synthetic_recovery", {}).get(
            "recovery_gate_passed"
        )
        is True
        and directional_calibration.get("heldout_used_for_fitting") is False
        and not (
            root / "results/directional_material_calibration/material.inp"
        ).exists()
    )
    items.append(
        AuditItem(
            "physical handoff",
            "Validation-gated nine-constant directional calibrator",
            "proven" if directional_calibrator_ok else "partial",
            (
                "synthetic recovery passes; incomplete real submissions "
                "release no card; r01-r03 fit, r04-r05 independently gate "
                "release, and held-out L-brackets are evaluated only after "
                "the card is frozen"
                if directional_calibrator_ok
                else "directional calibration registry missing or inconsistent"
            ),
        )
    )
    external_infill_path = (
        root
        / "results/adaptive_infill_external_validation"
        / "validation_summary.json"
    )
    external_infill = (
        _json(external_infill_path)
        if external_infill_path.exists()
        else {}
    )
    external_infill_ok = (
        external_infill.get("physical_hilbert_best_cases") == 4
        and external_infill.get("physical_case_count") == 4
        and external_infill.get("simulation_case_count") == 6
    )
    items.append(
        AuditItem(
            "infill simulation",
            "Exact-brand held-out physical topology ranking",
            "partial" if external_infill_ok else "missing",
            (
                "Bambu PLA Basic preprint supplies 4/4 Hilbert-first "
                "physical comparisons; equal-mass beam/solid models match "
                f"{external_infill.get('simulation_full_ranking_matches')}/6 "
                "full rankings and 6/6 zigzag-angle pair rankings; source "
                "has one specimen/case and incomplete geometry"
                if external_infill_ok
                else "checksum-pinned physical ranking audit absent"
            ),
        )
    )
    interface_summary_path = (
        root / "results/interlayer_initiation_screen/summary.json"
    )
    interface_summary = (
        _json(interface_summary_path)
        if interface_summary_path.exists()
        else {}
    )
    interface_reference_path = (
        root
        / "data/reported/bambu_pla_interlayer_strength_2026"
        / "interface_initiation_profiles.json"
    )
    interface_ok = (
        interface_reference_path.exists()
        and interface_summary.get("source_static_fields") == 42
        and interface_summary.get("valid_static_profile_fields") == 84
        and interface_summary.get("heldout_pairwise_comparisons") == 24
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Peer-reviewed interlayer opening/tear initiation sensitivity",
            "partial" if interface_ok else "missing",
            (
                "exact Tables 3/4 means and SDs checksum-pinned; 42 static "
                "fields x 2 process profiles screened; held-out pairwise "
                f"rankings={interface_summary.get('heldout_pairwise_matches')}/24; "
                "tear is not pure shear and fracture energy/damage evolution "
                "remain unavailable"
                if interface_ok
                else "interface reference or field screen missing"
            ),
        )
    )
    print_speed_reference_root = (
        root / "data/reported/bambu_pla_print_speed_2025"
    )
    print_speed_provenance_path = (
        print_speed_reference_root / "provenance.json"
    )
    print_speed_provenance = (
        _json(print_speed_provenance_path)
        if print_speed_provenance_path.exists()
        else {}
    )
    print_speed_profiles_path = (
        print_speed_reference_root / "process_groups.csv"
    )
    print_speed_profiles = (
        _count_csv_rows(print_speed_profiles_path)
        if print_speed_profiles_path.exists()
        else 0
    )
    print_speed_curve_validation_path = (
        print_speed_reference_root / "curve_validation.csv"
    )
    print_speed_curve_validation = []
    if print_speed_curve_validation_path.exists():
        with print_speed_curve_validation_path.open(
            encoding="utf-8", newline=""
        ) as handle:
            print_speed_curve_validation = list(csv.DictReader(handle))
    print_speed_summary_path = (
        root / "results/print_speed_infill_sensitivity/summary.json"
    )
    print_speed_summary = (
        _json(print_speed_summary_path)
        if print_speed_summary_path.exists()
        else {}
    )
    print_speed_ok = (
        print_speed_provenance.get("quality_gate_passed") is True
        and print_speed_profiles == 8
        and len(print_speed_curve_validation) == 8
        and all(
            row.get("quality_gate_passed", "").lower() == "true"
            for row in print_speed_curve_validation
        )
        and print_speed_summary.get("source_static_fields") == 42
        and print_speed_summary.get(
            "valid_field_profile_evaluations"
        ) == 336
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Published print-speed/process and representative post-peak sensitivity",
            "partial" if print_speed_ok else "missing",
            (
                "8 exact-brand groups x 5 repeats; plotted tensile/modulus/"
                "impact mean+/-SD and 8/8 representative peak cross-checks; "
                "42 static fields x 8 profiles screened; pre-peak and "
                "terminal fracture drop are explicit reconstructions, not "
                "raw specimen or validated damage data"
                if print_speed_ok
                else "print-speed reference or infill sensitivity incomplete"
            ),
        )
    )
    ballistic_reference_root = (
        root / "data/reported/bambu_pla_ballistic_damage_2025"
    )
    ballistic_provenance_path = (
        ballistic_reference_root / "provenance.json"
    )
    ballistic_provenance = (
        _json(ballistic_provenance_path)
        if ballistic_provenance_path.exists()
        else {}
    )
    ballistic_curve_path = (
        ballistic_reference_root / "curve_validation.csv"
    )
    ballistic_curve_rows = []
    if ballistic_curve_path.exists():
        with ballistic_curve_path.open(
            encoding="utf-8", newline=""
        ) as handle:
            ballistic_curve_rows = list(csv.DictReader(handle))
    ballistic_summary_path = (
        root / "results/ballistic_pattern_validation/summary.json"
    )
    ballistic_summary = (
        _json(ballistic_summary_path)
        if ballistic_summary_path.exists()
        else {}
    )
    ballistic_ok = (
        ballistic_provenance.get("curve_quality_gate_passed") is True
        and ballistic_provenance.get("damage_energy_gate_passed") is True
        and len(ballistic_curve_rows) == 8
        and all(
            row.get("passes_table_crosscheck", "").lower() == "true"
            and row.get("passes_trace_continuity_gate", "").lower()
            == "true"
            for row in ballistic_curve_rows
        )
        and ballistic_provenance.get(
            "separable_colored_curves_per_response"
        )
        == 4
        and ballistic_provenance.get(
            "figure6_digitization_asset", {}
        ).get("pixel_dimensions")
        == [4262, 1579]
        and ballistic_summary.get("physical_target_tests") == 12
        and ballistic_summary.get("ranking_match") is True
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Peer-reviewed specimen post-peak, damage, and ballistic infill reference",
            "partial" if ballistic_ok else "missing",
            (
                "4/5 independently visible tensile plus 4/5 flexural "
                "specimen traces from the official high-resolution PNG "
                "cross-check reported means; the fifth trace is not "
                "independently rendered; crack-band curves preserve "
                "2.5 N/mm at 0.5/1/2 mm; 12 physical targets rank honeycomb > "
                "cubic > grid > right-censored gyroid; author FE grid "
                f"penetration error={abs(float(ballistic_summary.get('author_FE_grid_relative_error', 0.0))):.1%}; "
                "yield/damage parameters are author model assumptions and "
                "infill was homogenized"
                if ballistic_ok
                else "ballistic/damage reference incomplete"
            ),
        )
    )

    uav_root = root / "data/reported/bambu_pla_uav_arm_2026"
    uav_provenance_path = uav_root / "provenance.json"
    uav_provenance = (
        _json(uav_provenance_path)
        if uav_provenance_path.exists()
        else {}
    )
    uav_validation_path = uav_root / "curve_validation.csv"
    uav_curve_rows = []
    if uav_validation_path.exists():
        with uav_validation_path.open(
            encoding="utf-8", newline=""
        ) as handle:
            uav_curve_rows = list(csv.DictReader(handle))
    uav_ok = (
        uav_provenance.get("curve_quality_gate_passed") is True
        and uav_provenance.get("physical_specimen_curve_count") == 12
        and len(uav_curve_rows) == 12
        and all(
            row.get("passes_visible_trace_gate", "").lower() == "true"
            and row.get("passes_table_crosscheck", "").lower() == "true"
            and abs(float(row["average_peak_relative_error"])) < 0.01
            for row in uav_curve_rows
        )
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Independent held-out UAV-arm component specimen curves",
            "partial" if uav_ok else "missing",
            (
                "12/12 physical Figure 7 traces across topology-optimized, "
                "honeycomb, drilled, and solid arms pass visible-trace and "
                "Table 3 peak checks; maximum group-mean peak error <1%; "
                "all are locked external-heldout-only; machine curves and "
                "reconstructed CAD/STEP are available only on request, so "
                "an exact CalculiX replay is not claimed"
                if uav_ok
                else "exact-brand UAV-arm curve reference incomplete"
            ),
        )
    )

    modal_validation_path = (
        root
        / "results/bambu_pla_heldout_modal_validation/provenance.json"
    )
    modal_validation = (
        _json(modal_validation_path)
        if modal_validation_path.exists()
        else {}
    )
    modal_component = modal_validation.get("component", {})
    modal_gates = modal_validation.get("gates", {})
    modal_ok = (
        modal_validation.get("status")
        == "heldout_small_strain_modal_gate_passed"
        and modal_component.get("physical_specimens") == 3
        and modal_component.get("physical_modes_per_specimen") == 5
        and modal_gates.get("source_checksums_passed") is True
        and modal_gates.get("physical_specimen_mode_count_passed") is True
        and modal_gates.get("calculix_author_backend_gate_passed") is True
        and modal_gates.get("published_author_mesh_gate_passed") is True
        and modal_gates.get(
            "heldout_small_strain_modal_gate_passed"
        ) is True
        and modal_gates.get(
            "component_modal_data_used_for_calibration"
        ) is False
        and modal_gates.get(
            "local_axis_directional_calibration_released"
        ) is False
        and float(modal_validation.get("heldout_relative_rmse", 1.0))
        <= 0.15
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Exact-geometry held-out Bambu cantilever modal validation",
            "proven" if modal_ok else "missing",
            (
                "exact STEP and source solid meshes; 3 physical components "
                "x 5 common modes = 15 held-out targets; converged "
                f"CalculiX/Abaqus error <= {float(modal_validation.get('maximum_converged_calculix_to_author_backend_relative_error', 0.0)):.2%}; "
                f"frozen-profile held-out RMSE={float(modal_validation.get('heldout_relative_rmse', 0.0)):.2%}, "
                f"maximum error={float(modal_validation.get('heldout_maximum_absolute_relative_error', 0.0)):.2%}; "
                "component data are locked external-heldout-only and do not "
                "release local-axis calibration"
                if modal_ok
                else "exact-brand held-out modal replay incomplete"
            ),
        )
    )

    service_contract_path = (
        root
        / "data/reported/bambu_pla_service_envelope_v3"
        / "service_contract.json"
    )
    service_contract = (
        _json(service_contract_path)
        if service_contract_path.exists()
        else {}
    )
    service_plan_path = (
        root
        / "data/reported/bambu_pla_service_envelope_v3"
        / "missing_law_experiment_plan.csv"
    )
    service_plan_rows = (
        _count_csv_rows(service_plan_path)
        if service_plan_path.exists()
        else 0
    )
    service_counts = service_contract.get("condition_counts", {})
    service_gate_ok = (
        service_contract.get(
            "lowest_structural_temperature_guardrail_C"
        )
        == 54.0
        and service_contract.get("glass_transition_temperature_C") == 60.0
        and service_contract.get(
            "saturated_water_absorption_percent_at_25C_55RH"
        )
        == 0.43
        and service_counts.get("admitted_with_restrictions") == 4
        and service_counts.get("unsupported") == 3
        and service_counts.get("rejected") == 2
        and service_plan_rows == 90
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Manufacturer service-condition admission and rejection gate",
            "partial" if service_gate_ok else "missing",
            (
                "checksum-pinned TDS V3.0; exact-brand Tg=60 C, "
                "HDT=54 C at 1.8 MPa / 57 C at 0.45 MPa, saturated water "
                "absorption=0.43% at 25 C/55% RH; room monotonic and "
                "30 C/1 Hz storage and 18 C/1 Hz-25 kHz wave cases admitted "
                "with restrictions; a bounded 18 C sub-second relaxation "
                "sensitivity is admitted; >=54 C rejected; creep, fatigue, "
                "humidity, and aging are routed to a 90-specimen calibration "
                "plan; thresholds are not "
                "constitutive laws"
                if service_gate_ok
                else "manufacturer service envelope or measurement plan incomplete"
            ),
        )
    )
    service_campaign_root = (
        root / "results/bambu_pla_service_law_campaign"
    )
    service_campaign_manifest_path = (
        service_campaign_root / "campaign_manifest.json"
    )
    service_campaign_ingestion_path = (
        service_campaign_root / "ingestion/summary.json"
    )
    service_campaign_manifest = (
        _json(service_campaign_manifest_path)
        if service_campaign_manifest_path.exists()
        else {}
    )
    service_campaign_ingestion = (
        _json(service_campaign_ingestion_path)
        if service_campaign_ingestion_path.exists()
        else {}
    )
    service_campaign_templates = len(
        list((service_campaign_root / "capture_templates").glob("*.csv"))
    )
    stage_status = service_campaign_ingestion.get("stage_status", {})
    screened_sources = service_campaign_manifest.get(
        "external_source_screening", []
    )
    service_campaign_ok = (
        service_campaign_manifest.get("campaign_id")
        == "bambu_pla_basic_service_laws_v1"
        and service_campaign_manifest.get("specimen_count") == 90
        and service_campaign_manifest.get("stage_counts")
        == {"1": 6, "2": 18, "3": 12, "4": 30, "5": 24}
        and service_campaign_manifest.get("capture_template_count") == 90
        and service_campaign_templates == 90
        and service_campaign_ingestion.get("status")
        == "awaiting_measurements"
        and service_campaign_ingestion.get("counts")
        == {
            "valid": 0,
            "invalid": 0,
            "awaiting_measurements": 90,
            "missing": 0,
        }
        and set(stage_status) == {"1", "2", "3", "4", "5"}
        and all(
            item.get("law_released") is False
            for item in stage_status.values()
        )
        and service_campaign_ingestion.get("all_laws_released") is False
        and service_campaign_manifest.get("fatigue_runout_cycles")
        == 100_000
        and len(screened_sources) >= 1
        and screened_sources[0].get("material_identity_gate_passed") is False
    )
    items.append(
        AuditItem(
            "evidence acquisition",
            "Strict staged service-law capture and ingestion",
            "proven" if service_campaign_ok else "partial",
            (
                "90 checksum-frozen randomized templates: 6 DMA, 18 "
                "relaxation, 12 humidity, 30 directional fatigue, and 24 "
                "aging; blank templates are 90/90 awaiting rather than "
                "passing; each law has an all-specimen release gate; "
                "100,000-cycle runouts remain censored; an in-house-extruded "
                "PLA fatigue paper is rejected by the exact-material gate"
                if service_campaign_ok
                else "service-law campaign or its no-release gate is incomplete"
            ),
        )
    )
    service_calibration_root = (
        root / "results/bambu_pla_service_law_calibration"
    )
    service_calibration_manifest_path = (
        service_calibration_root / "calibration_manifest.json"
    )
    service_calibration = (
        _json(service_calibration_manifest_path)
        if service_calibration_manifest_path.exists()
        else {}
    )
    service_calibration_stage_files = list(
        service_calibration_root.glob("stage_*.json")
    )
    calibration_stages = service_calibration.get("stage_results", {})
    synthetic_recovery = service_calibration.get(
        "synthetic_recovery", {}
    )
    recovery_metrics = synthetic_recovery.get("metrics", {})
    service_calibration_ok = (
        service_calibration.get("calibration_id")
        == "bambu_pla_basic_service_law_calibration_v1"
        and service_calibration.get("status") == "awaiting_measurements"
        and service_calibration.get("released_stages") == []
        and set(calibration_stages) == {"1", "2", "3", "4", "5"}
        and all(
            result.get("status") == "not_released"
            and result.get("model_file") is None
            for result in calibration_stages.values()
        )
        and service_calibration_stage_files == []
        and synthetic_recovery.get(
            "all_synthetic_recovery_gates_passed"
        )
        is True
        and recovery_metrics.get(
            "maximum_DMA_leave_one_replicate_out_NRMSE", 1.0
        )
        < 0.01
        and recovery_metrics.get("relaxation_fit_NRMSE", 1.0)
        < 0.01
        and recovery_metrics.get("fatigue_failures", 0) >= 3
        and recovery_metrics.get("fatigue_runouts", 0) >= 1
        and recovery_metrics.get("fatigue_slope", 0.0) < 0.0
        and service_calibration.get("selector_contract", {}).get(
            "no_fallback"
        )
        is True
    )
    items.append(
        AuditItem(
            "evidence acquisition",
            "Bounded service-law calibration and no-release registry",
            "proven" if service_calibration_ok else "partial",
            (
                "synthetic gates recover held-out DMA response (<1% NRMSE), "
                "a positive Maxwell relaxation spectrum, tensile metrics, "
                "and a decreasing censored Basquin law from 9 failures plus "
                "6 runouts; the real-data registry correctly releases 0/5 "
                "laws and writes no stage model files while measurements "
                "remain blank"
                if service_calibration_ok
                else "service-law calibrators or their no-release gate are incomplete"
            ),
        )
    )

    wave_root = (
        root / "data/reported/bambu_pla_viscoelastic_wave_2026"
    )
    wave_provenance_path = wave_root / "provenance.json"
    wave_provenance = (
        _json(wave_provenance_path)
        if wave_provenance_path.exists()
        else {}
    )
    wave_curve_path = wave_root / "dma_master_curve_18C.csv"
    wave_curve_rows = (
        _count_csv_rows(wave_curve_path) if wave_curve_path.exists() else 0
    )
    modal_summary_path = (
        root
        / "results/custom_infill/graded_hourglass"
        / "viscoelastic_modal_screen/summary.json"
    )
    modal_summary = (
        _json(modal_summary_path) if modal_summary_path.exists() else {}
    )
    modal_gates = modal_summary.get("gates", {})
    wave_ok = (
        wave_provenance.get("quality_gate_passed") is True
        and wave_curve_rows == 85
        and modal_summary.get("status")
        == "bounded_relative_modal_screen_admitted"
        and modal_gates.get("all_calculix_cases_passed") is True
        and modal_gates.get(
            "all_selected_modes_within_1Hz_to_25kHz_reference"
        )
        is True
        and modal_gates.get("absolute_resonance_physically_validated")
        is False
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Exact-brand broadband viscoelastic and modal-pattern screen",
            "partial" if wave_ok else "missing",
            (
                "11 deposited files checksum-pinned; 85 raw DMA points "
                "reconstructed at 18 C; measured nu=0.3459 and printed "
                "density=1144 kg/m3; reported G'=714.7 MPa, G''=26.6 MPa; "
                f"graded-hourglass/triangular out-of-plane modal ratio="
                f"{float(modal_summary.get('graded_hourglass_to_triangular_frequency_ratio', 0.0)):.3f}; "
                "relative modal gate passes, but interactive source outlier "
                "selection and absent physical resonance tests prevent an "
                "absolute validation claim"
                if wave_ok
                else "broadband reference or modal-pattern gate incomplete"
            ),
        )
    )

    short_relaxation_root = (
        root / "data/reported/bambu_pla_short_time_relaxation_18C"
    )
    short_relaxation_path = short_relaxation_root / "summary.json"
    short_relaxation = (
        _json(short_relaxation_path)
        if short_relaxation_path.exists()
        else {}
    )
    short_relaxation_spectrum_rows = _count_csv_rows(
        short_relaxation_root / "maxwell_spectrum.csv"
    )
    short_relaxation_card_rows = _count_csv_rows(
        short_relaxation_root / "frequency_card_index.csv"
    )
    frequency_modal_path = (
        root
        / "results/custom_infill/weird_kite_search"
        / "frequency_dependent_modal_screen/summary.json"
    )
    frequency_modal = (
        _json(frequency_modal_path)
        if frequency_modal_path.exists()
        else {}
    )
    frequency_modal_gate = frequency_modal.get("gate", {})
    short_relaxation_ok = (
        short_relaxation.get("overall_status")
        == "storage_and_short_time_relaxation_sensitivity_only"
        and short_relaxation.get("raw_row_count") == 85
        and short_relaxation.get("inlier_row_count") == 83
        and short_relaxation.get("outlier_row_count") == 2
        and short_relaxation.get("selected_term_count") == 5
        and short_relaxation.get("storage_fit_gate_passed") is True
        and short_relaxation.get("loss_fit_gate_passed") is False
        and short_relaxation.get("positive_spectrum_gate_passed") is True
        and short_relaxation_spectrum_rows == 5
        and short_relaxation_card_rows == 6
        and frequency_modal.get("status")
        == "frequency_dependent_relative_modal_screen_admitted"
        and frequency_modal.get("solver_iteration_count") == 12
        and frequency_modal_gate.get(
            "frequency_dependent_modal_gate_passed"
        )
        is True
        and frequency_modal_gate.get(
            "selected_mode_consistent_within_each_iteration"
        )
        is True
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Positive short-time relaxation-spectrum sensitivity",
            "partial" if short_relaxation_ok else "missing",
            (
                "5-branch positive generalized-Maxwell spectrum selected "
                "from 3/5/7/9 branches by source-stratified five-fold "
                "cross-validation and the one-standard-error rule; 83/85 "
                "raw points retained; storage normalized RMSE="
                f"{float(short_relaxation.get('final_storage_normalized_rmse', 0.0)):.2%} "
                "passes, loss precision fails; six frequency-specific "
                "in-phase CalculiX cards cover 1 Hz-25 kHz; 12/12 "
                "self-consistent CalculiX iterations converge for the "
                "triangular and two promoted sliced patterns, retaining "
                f"modal ratios {float(frequency_modal_gate.get('fine_frequency_ratios_to_triangular', {}).get('kite_08_08_08_08', 0.0)):.3f}/"
                f"{float(frequency_modal_gate.get('fine_frequency_ratios_to_triangular', {}).get('kite_08_08_12_08', 0.0)):.3f}; inferred "
                "relaxation is restricted to 6.37 microseconds-0.159 s and "
                "does not unlock creep or fatigue"
                if short_relaxation_ok
                else "short-time spectrum, model-selection, or fit gates incomplete"
            ),
        )
    )

    cryo_root = (
        root
        / "data/reported/bambu_pla_cryogenic_preconditioning_2025"
    )
    cryo_provenance_path = cryo_root / "provenance.json"
    cryo_provenance = (
        _json(cryo_provenance_path)
        if cryo_provenance_path.exists()
        else {}
    )
    cryo_means_path = cryo_root / "summary_means.csv"
    cryo_effects_path = cryo_root / "conditioning_effects.csv"
    cryo_metric_path = cryo_root / "metric_effect_summary.csv"
    cryo_means = (
        _count_csv_rows(cryo_means_path)
        if cryo_means_path.exists()
        else 0
    )
    cryo_effects = (
        _count_csv_rows(cryo_effects_path)
        if cryo_effects_path.exists()
        else 0
    )
    cryo_metrics = (
        _count_csv_rows(cryo_metric_path)
        if cryo_metric_path.exists()
        else 0
    )
    cryo_screen_path = (
        root
        / "results/custom_infill/graded_hourglass"
        / "cryogenic_recovery_sensitivity/summary.json"
    )
    cryo_screen = (
        _json(cryo_screen_path) if cryo_screen_path.exists() else {}
    )
    cryo_ok = (
        cryo_provenance.get("quality_gate_passed") is True
        and cryo_provenance.get("testing", {}).get("total_specimens")
        == 320
        and cryo_means == 160
        and cryo_effects == 80
        and cryo_metrics == 5
        and cryo_screen.get("status")
        == "ranking_survives_published_freeze_recovery_envelope"
        and cryo_screen.get("screens", {}).get(
            "ranking_survives_adversarial_factor_assignment"
        )
        is True
        and cryo_screen.get("gate", {}).get(
            "absolute_freeze_recovery_prediction_validated"
        )
        is False
        and cryo_screen.get("physical_handoff", {}).get(
            "specimen_count"
        )
        == 18
        and cryo_screen.get("physical_handoff", {}).get(
            "time_matched_control_included"
        )
        is True
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Published -80 C freeze/recovery conditioning sensitivity",
            "partial" if cryo_ok else "missing",
            (
                "320 exact-brand specimens represented by 160 chart-labeled "
                "means and 80 paired effects; -80 C for 24 h plus 24 h "
                "room-temperature recovery; the 16-cell flexural-modulus "
                "envelope leaves the frozen-toolpath custom/control ratio at "
                f"{float(cryo_screen.get('screens', {}).get('adversarial_custom_min_control_max_ratio', 0.0)):.3f} "
                "under adversarial factor assignment; an 18-coupon, "
                "three-plate handoff adds a sealed 48 h time control; no raw "
                "curves, numeric dispersion, cubic-to-custom topology "
                "calibration, or in-service low-temperature law"
                if cryo_ok
                else "freeze/recovery reference or pattern sensitivity incomplete"
            ),
        )
    )

    twin_manifest_path = (
        root / "results/bambu_pla_digital_twin/manifest.json"
    )
    twin_manifest = (
        _json(twin_manifest_path) if twin_manifest_path.exists() else {}
    )
    twin_cards_path = (
        root / "results/bambu_pla_digital_twin/canonical_cards.csv"
    )
    twin_selection_path = (
        root / "results/bambu_pla_digital_twin/selection_matrix.csv"
    )
    twin_card_rows = (
        _count_csv_rows(twin_cards_path)
        if twin_cards_path.exists()
        else 0
    )
    twin_selection_rows = (
        _count_csv_rows(twin_selection_path)
        if twin_selection_path.exists()
        else 0
    )
    twin_excluded = [
        selection
        for selection in twin_manifest.get("selections", [])
        if selection.get("status")
        in {"unsupported", "rejected", "guardrail_only"}
    ]
    twin_gate_ok = (
        twin_manifest.get("twin_id")
        == "bambu_pla_basic_evidence_twin_v1"
        and twin_manifest.get("selection_gate_passed") is True
        and twin_manifest.get("canonical_card_count") == 9
        and twin_manifest.get("selection_count") == 18
        and twin_manifest.get("evidence_file_count", 0) >= 19
        and twin_card_rows == 9
        and twin_selection_rows == 18
        and len(twin_excluded) == 7
        and all(
            selection.get("canonical_card") is None
            for selection in twin_excluded
        )
    )
    items.append(
        AuditItem(
            "Bambu PLA Basic",
            "Condition-aware canonical digital-twin model selector",
            "proven" if twin_gate_ok else "missing",
            (
                "9 checksum-tracked canonical CalculiX cards and 18 "
                "representative "
                "requests are integrated behind one evidence-backed selector; "
                "all 7 unsupported, rejected, or guardrail-only requests "
                "receive no fallback material card; this proves selection "
                "behavior, not the unresolved constitutive laws"
                if twin_gate_ok
                else "canonical selector manifest or hard-exclusion gate incomplete"
            ),
        )
    )

    items.extend(
        [
            AuditItem(
                "missing evidence",
                "Matched real local-1/local-2/local-3 calibration and validation curves",
                "missing",
                "the frozen 43-specimen campaign and validation-gated "
                "nine-constant calibrator are ready, but no accepted real "
                "submission exists; public labels do not uniquely map E1/E2",
            ),
            AuditItem(
                "Bambu PLA Basic",
                "Raw specimen-level flexural curves",
                "partial" if ballistic_ok else "missing",
                (
                    "five exact-brand flexural specimens were tested; the "
                    "official high-resolution figure independently renders "
                    "4/5 colored post-peak traces, all continuity-gated and "
                    "digitized; a fifth distinct trace and machine-exported "
                    "numeric curves are unavailable"
                    if ballistic_ok
                    else "published specimen curve reference unavailable"
                ),
            ),
            AuditItem(
                "missing evidence",
                "Measured shear moduli and Poisson ratios",
                "partial",
                "an exact-brand Bambu-X1C study measures in-plane flat "
                "nu=0.3459 and broadband dynamic G'=714.7 MPa / "
                "G''=26.6 MPa at 18 C; quasistatic G12/G13/G23 and "
                "out-of-plane Poisson values remain unmeasured",
            ),
            AuditItem(
                "Bambu PLA Basic",
                "Validated damage evolution, fracture, separation, and post-peak response",
                "partial" if ballistic_ok else "missing",
                (
                    "author Abaqus model supplies 5% PEEQ initiation, linear "
                    "2.5 kJ/m2 softening and deletion; homogenized-grid "
                    "penetration is within 8.7% and damage mode is qualitatively "
                    "similar; parameters are not matched fracture-coupon "
                    "measurements and no interlayer traction-separation law exists"
                    if ballistic_ok
                    else "no validated damage reference available"
                ),
            ),
            AuditItem(
                "missing evidence",
                "Temperature, rate, creep, fatigue, moisture, and aging laws",
                "partial",
                "the TDS/DMA admission gate now rejects unsupported use and "
                "exports a 90-specimen DMA, relaxation, humidity, fatigue, "
                "and aging plan; one exact-brand "
                "-80 C/24 h plus room-recovery summary matrix now exists, "
                "and an exact-brand raw DMA master curve now supports a "
                "bounded 18 C / 1 Hz-25 kHz storage/loss wave screen plus a "
                "positive sub-second relaxation sensitivity; these "
                "do not provide an in-service low-temperature law, general "
                "temperature law, creep, fatigue, moisture, or aging curves",
            ),
        ]
    )
    return items


def write_audit(
    output_dir: Path = DEFAULT_OUTPUT,
    root: Path = PROJECT_ROOT,
    run_tests: bool = True,
) -> List[AuditItem]:
    items = audit_project(root, run_tests=run_tests)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective": "fully simulate 1 brand of pla",
        "material": "Bambu PLA Basic",
        "assessment": "incomplete_physical_validation_required",
        "items": [asdict(item) for item in items],
    }
    (output_dir / "audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    counts = {
        status: sum(item.status == status for item in items)
        for status in ("proven", "partial", "missing")
    }
    lines = [
        "# Bambu PLA Basic simulator completion audit",
        "",
        "## Overall assessment: Needs physical validation",
        "",
        (
            f"Evidence inventory: {counts['proven']} proven, "
            f"{counts['partial']} partial, {counts['missing']} missing."
        ),
        "",
        "The software/autoresearch contract is runnable and the static infill "
        "simulator is decision-useful within its documented pre-peak envelope. "
        "The broader claim of a fully validated brand model is not yet proven.",
        "",
        "| Category | Requirement | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for item in items:
        lines.append(
            f"| {item.category} | {item.requirement} | {item.status} | "
            f"{item.evidence} |"
        )
    lines.extend(
        [
            "",
            "## Required next measurements",
            "",
            "1. Matched local-axis tensile coupons with lateral strain for "
            "E1/E2/E3 and out-of-plane Poisson identification.",
            "2. Quasistatic in-plane and out-of-plane shear coupons for "
            "G12/G13/G23.",
            "3. Raw three-point-bending curves and one held-out printed part.",
            "4. Post-peak tension and interlayer fracture tests if failure, "
            "rather than pre-peak stiffness, is the intended decision.",
            "5. Temperature/rate/creep/fatigue tests only if those service "
            "conditions are in scope.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Bambu PLA simulator against the full goal"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    items = write_audit(
        Path(args.output), run_tests=not args.skip_tests
    )
    counts = {
        status: sum(item.status == status for item in items)
        for status in ("proven", "partial", "missing")
    }
    print(
        f"audit: {counts['proven']} proven, {counts['partial']} partial, "
        f"{counts['missing']} missing; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
