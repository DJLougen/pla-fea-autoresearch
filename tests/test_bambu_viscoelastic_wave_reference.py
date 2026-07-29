from pathlib import Path

import pytest

from src.bambu_viscoelastic_wave_reference import (
    DEFAULT_SOURCE,
    build_reference,
    reconstruct_reduced_master_curve,
    verify_source,
    write_reference,
)


def test_deposited_viscoelastic_files_are_checksum_pinned() -> None:
    source = verify_source(DEFAULT_SOURCE)
    assert source["verified_file_count"] == 11


def test_raw_dma_reconstruction_matches_reported_storage_scale() -> None:
    rows = reconstruct_reduced_master_curve(DEFAULT_SOURCE)
    assert len(rows) == 85
    assert rows[0]["reduced_frequency_at_18C_Hz"] >= 1.0
    assert rows[-1]["reduced_frequency_at_18C_Hz"] <= 25_000.0
    reference = build_reference(DEFAULT_SOURCE)
    assert reference["quality_gate_passed"] is True
    reconstruction = reference["raw_reconstruction"]
    assert reconstruction["storage_modulus_MPa"]["mean"] == pytest.approx(
        718.3426283
    )
    assert reconstruction["loss_modulus_MPa"]["mean"] == pytest.approx(
        23.51303837
    )


def test_reference_exports_curve_provenance_and_storage_proxy(
    tmp_path: Path,
) -> None:
    reference = write_reference(DEFAULT_SOURCE, tmp_path)
    dynamic = reference["reported_dynamic_measurements"]
    assert dynamic["loss_factor"] == pytest.approx(26.6 / 714.7)
    assert dynamic["storage_equivalent_youngs_modulus_MPa"] == pytest.approx(
        2.0 * 714.7 * (1.0 + 0.3459)
    )
    assert (tmp_path / "dma_master_curve_18C.csv").exists()
    assert (tmp_path / "reported_measurements.csv").exists()
    assert (tmp_path / "provenance.json").exists()
    assert (tmp_path / "storage_proxy.inp").exists()
    assert (tmp_path / "quality_report.md").exists()
