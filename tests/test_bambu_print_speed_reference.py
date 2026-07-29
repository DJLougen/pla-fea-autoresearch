import csv
from pathlib import Path

import numpy as np

from src.bambu_print_speed_reference import (
    SAMPLE_MAP,
    digitize_representative_curves,
    digitize_summary_figures,
    verify_sources,
    write_reference,
)


def test_print_speed_sources_and_process_matrix_are_complete():
    checksums = verify_sources()
    assert len(checksums) == 10
    rows = digitize_summary_figures()
    assert len(rows) == 8
    assert {int(row["sample_group"]) for row in rows} == set(SAMPLE_MAP)
    assert all(int(row["replicates"]) == 5 for row in rows)
    strengths = [
        float(row["tensile_strength_mean_MPa"]) for row in rows
    ]
    moduli = [float(row["youngs_modulus_mean_MPa"]) for row in rows]
    impacts = [float(row["impact_strength_mean_kJ_m2"]) for row in rows]
    assert min(strengths) > 32.0
    assert max(strengths) < 39.0
    assert min(moduli) > 2950.0
    assert max(moduli) < 3160.0
    assert min(impacts) > 18.0
    assert max(impacts) < 25.0
    assert all(
        float(row["tensile_strength_standard_deviation_MPa"]) > 0.0
        for row in rows
    )


def test_representative_curve_hybrid_has_auditable_branches():
    rows = digitize_summary_figures()
    curves, validation = digitize_representative_curves(rows)
    assert len(curves) == 8
    assert len(validation) == 8
    assert all(row["quality_gate_passed"] for row in validation)
    for sample, curve in curves.items():
        strain = np.asarray(
            [float(point["engineering_strain"]) for point in curve]
        )
        stress = np.asarray(
            [float(point["engineering_stress_MPa"]) for point in curve]
        )
        branches = {str(point["branch"]) for point in curve}
        assert strain[0] == 0.0
        assert stress[0] == 0.0
        assert np.all(np.diff(strain) > 0.0)
        assert stress[-1] == 0.0
        assert {
            "reconstructed_prepeak_cubic",
            "digitized_postpeak_monotone_envelope",
            "assumed_terminal_fracture_drop",
        } <= branches
        measured = float(
            rows[sample - 1]["tensile_strength_mean_MPa"]
        )
        assert abs(np.max(stress) / measured - 1.0) < 1.0e-8
        assert 0.05 < strain[-1] < 0.13


def test_reference_writer_creates_eight_csv_curves(tmp_path: Path):
    payload = write_reference(output_dir=tmp_path)
    assert payload["quality_gate_passed"] is True
    assert len(list((tmp_path / "curves").glob("sample_*.csv"))) == 8
    with (tmp_path / "process_groups.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 8
