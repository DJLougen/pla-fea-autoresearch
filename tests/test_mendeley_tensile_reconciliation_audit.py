from src.mendeley_tensile_reconciliation_audit import (
    read_literal_raw_peaks,
    read_summary_specimen_cells,
    reconcile_summary_to_raw_peaks,
    write_reconciliation_audit,
)


def test_reads_literal_and_summary_counts():
    assert len(read_literal_raw_peaks()) == 22
    assert len(read_summary_specimen_cells()) == 21


def test_reconciliation_exposes_cross_label_peak_reuse():
    mappings = reconcile_summary_to_raw_peaks(
        read_literal_raw_peaks(), read_summary_specimen_cells()
    )
    same_label = sum(row["same_label_peak_match"] for row in mappings)
    assert same_label == 12
    y_flat_2 = next(
        row for row in mappings if row["summary_label"] == "Y-F-2"
    )
    assert y_flat_2["matching_raw_labels"] == "X-F-4"
    assert y_flat_2["cross_label_or_missing_match"] is True


def test_audit_blocks_directional_release(tmp_path):
    payload = write_reconciliation_audit(tmp_path)
    assert payload["status"] == "source_reconciliation_failed_non_bijective"
    assert payload["counts"][
        "unique_raw_curves_referenced_by_summary"
    ] == 13
    assert payload["counts"][
        "raw_curves_omitted_from_summary_peak_values"
    ] == 9
    assert payload["gates"]["source_checksums_passed"] is True
    assert payload["gates"][
        "summary_to_raw_mapping_is_one_to_one"
    ] is False
    assert payload["gates"][
        "directional_calibration_release_allowed"
    ] is False
    assert (tmp_path / "summary_to_raw_peak_mapping.csv").exists()
    assert (tmp_path / "provenance.json").exists()
