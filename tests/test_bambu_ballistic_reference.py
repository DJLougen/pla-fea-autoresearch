import numpy as np

from src.bambu_ballistic_reference import (
    TABLE4_MEASURED,
    damage_curve,
    damage_energy_N_mm,
    digitize_figure6,
    verify_sources,
    write_reference,
)


def test_ballistic_sources_and_specimen_curve_crosschecks():
    assert len(verify_sources()) == 4
    curves, validation = digitize_figure6()
    assert set(curves) == {"tensile", "flexural"}
    assert all(len(response) == 4 for response in curves.values())
    assert len(validation) == 8
    assert all(row["passes_table_crosscheck"] for row in validation)
    assert all(
        "official high-resolution primary-source PNG"
        in row["evidence_class"]
        for row in validation
    )
    assert all(row["digitized_points"] > 350 for row in validation)
    assert all(row["source_columns_observed"] > 350 for row in validation)
    assert all(row["passes_trace_continuity_gate"] for row in validation)
    assert max(
        row["maximum_interpolated_gap_strain"] for row in validation
    ) < 0.005
    tensile = [
        row for row in validation if row["response"] == "tensile"
    ]
    flexural = [
        row for row in validation if row["response"] == "flexural"
    ]
    assert abs(
        np.mean([row["digitized_peak_stress_MPa"] for row in tensile])
        / TABLE4_MEASURED["tensile_strength_MPa"]
        - 1.0
    ) < 0.02
    assert abs(
        np.mean([row["digitized_peak_stress_MPa"] for row in flexural])
        / TABLE4_MEASURED["flexural_strength_MPa"]
        - 1.0
    ) < 0.02


def test_crack_band_curves_preserve_fracture_energy_across_mesh_lengths():
    failure_strains = []
    for length in (0.5, 1.0, 2.0):
        curve = damage_curve(length)
        assert curve[0]["nominal_stress_MPa"] == 0.0
        assert curve[-1]["nominal_stress_MPa"] == 0.0
        assert curve[-1]["damage"] == 1.0
        assert abs(damage_energy_N_mm(curve) / 2.5 - 1.0) < 0.01
        failure_strains.append(curve[-1]["total_strain"])
    assert failure_strains[0] > failure_strains[1] > failure_strains[2]


def test_ballistic_reference_writer_passes_both_gates(tmp_path):
    provenance = write_reference(output_dir=tmp_path)
    assert provenance["curve_quality_gate_passed"] is True
    assert provenance["damage_energy_gate_passed"] is True
    assert provenance["figure6_digitization_asset"]["pixel_dimensions"] == [
        4262,
        1579,
    ]
    assert provenance["separable_colored_curves_per_response"] == 4
    assert len(list((tmp_path / "curves").glob("*.csv"))) == 11
