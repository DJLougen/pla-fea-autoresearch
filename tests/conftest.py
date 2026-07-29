"""Repository-wide test categories for optional research artifacts."""

from pathlib import Path

import pytest


# These modules verify checksum-pinned publications, source workbooks, or
# generated study artifacts that are deliberately excluded from the Git
# repository. Keeping the category here makes the clean-clone boundary
# auditable while preserving the full tests for hydrated research workspaces.
EXTERNAL_ASSET_MODULES = {
    "test_adaptive_infill_external_validation.py",
    "test_bambu_ballistic_reference.py",
    "test_bambu_common_infill_reference.py",
    "test_bambu_cryogenic_reference.py",
    "test_bambu_digital_twin.py",
    "test_bambu_dma_reference.py",
    "test_bambu_flexural_digitization.py",
    "test_bambu_flexural_process_reference.py",
    "test_bambu_interlayer_reference.py",
    "test_bambu_modal_component_validation.py",
    "test_bambu_print_speed_reference.py",
    "test_bambu_print_tolerance_infill.py",
    "test_bambu_pvh_reference.py",
    "test_bambu_service_envelope.py",
    "test_bambu_uav_arm_reference.py",
    "test_bambu_viscoelastic_wave_reference.py",
    "test_cryogenic_pattern_sensitivity.py",
    "test_custom_infill.py",
    "test_custom_pattern_fixture.py",
    "test_custom_pattern_print_mesh.py",
    "test_custom_pattern_toolpath.py",
    "test_custom_pattern_toolpath_fea.py",
    "test_custom_pattern_validation.py",
    "test_manual_bending_capture.py",
    "test_mendeley_tensile_reconciliation_audit.py",
    "test_raw_specimen_infill_uncertainty.py",
    "test_results_and_cache.py",
    "test_weird_infill_physical_validation.py",
}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark tests whose default inputs live outside the public Git clone."""
    for item in items:
        if Path(str(item.path)).name in EXTERNAL_ASSET_MODULES:
            item.add_marker(pytest.mark.external_assets)
