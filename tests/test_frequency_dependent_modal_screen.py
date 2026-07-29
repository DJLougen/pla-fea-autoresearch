from pathlib import Path

import pytest

from src.frequency_dependent_modal_screen import (
    build_frequency_profile,
    evaluate_frequency_dependent_gate,
    load_relaxation_spectrum,
)


def test_frequency_profile_stiffens_inside_measured_window() -> None:
    spectrum = load_relaxation_spectrum()
    low = build_frequency_profile(1.0, spectrum)
    high = build_frequency_profile(25_000.0, spectrum)
    assert high["shear_storage_modulus_MPa"] > (
        low["shear_storage_modulus_MPa"]
    )
    assert high["screening_engineering_constants"]["E1"] > (
        low["screening_engineering_constants"]["E1"]
    )
    with pytest.raises(ValueError, match="outside"):
        build_frequency_profile(30_000.0, spectrum)


def test_frequency_dependent_gate_requires_convergence_and_ranking() -> None:
    patterns = (
        "triangular",
        "kite_08_08_08_08",
        "kite_08_08_12_08",
    )
    frequencies = {
        "triangular": (1_030.0, 1_020.0),
        "kite_08_08_08_08": (1_210.0, 1_200.0),
        "kite_08_08_12_08": (1_195.0, 1_185.0),
    }
    rows = []
    for pattern in patterns:
        for resolution, frequency in zip((60, 80), frequencies[pattern]):
            rows.append(
                {
                    "pattern": pattern,
                    "resolution_xy": resolution,
                    "iteration": 2,
                    "status": "ok",
                    "out_of_plane_frequency_Hz": frequency,
                    "self_consistency_converged": True,
                    "selected_mode_consistent": True,
                    "terminal_iteration": True,
                }
            )
    gate = evaluate_frequency_dependent_gate(rows)
    assert gate["frequency_dependent_modal_gate_passed"] is True
    assert gate["both_candidates_beat_triangular"] is True


def test_frequency_dependent_gate_rejects_unconverged_terminal() -> None:
    rows = []
    for pattern, frequency in (
        ("triangular", 1_020.0),
        ("kite_08_08_08_08", 1_200.0),
        ("kite_08_08_12_08", 1_185.0),
    ):
        for resolution in (60, 80):
            rows.append(
                {
                    "pattern": pattern,
                    "resolution_xy": resolution,
                    "iteration": 2,
                    "status": "ok",
                    "out_of_plane_frequency_Hz": frequency,
                    "self_consistency_converged": pattern != "triangular",
                    "selected_mode_consistent": True,
                    "terminal_iteration": True,
                }
            )
    gate = evaluate_frequency_dependent_gate(rows)
    assert gate["frequency_dependent_modal_gate_passed"] is False
    assert gate["all_material_frequency_iterations_converged"] is False
