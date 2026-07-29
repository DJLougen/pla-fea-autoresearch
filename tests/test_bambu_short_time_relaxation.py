from pathlib import Path

import numpy as np

from src.bambu_short_time_relaxation import (
    DEFAULT_SOURCE,
    generalized_maxwell_complex,
    generalized_maxwell_relaxation,
    relaxation_times,
    robust_loss_outliers,
    write_short_time_relaxation,
)


def test_positive_maxwell_spectrum_has_physical_limits() -> None:
    times = np.asarray([1.0e-4, 1.0e-2])
    branches = np.asarray([20.0, 10.0])
    storage, loss = generalized_maxwell_complex(
        np.asarray([1.0, 1.0e6]),
        600.0,
        branches,
        times,
    )
    relaxation = generalized_maxwell_relaxation(
        np.asarray([0.0, 1.0, 10.0]),
        600.0,
        branches,
        times,
    )
    assert storage[1] > storage[0] > 600.0
    assert np.all(loss > 0.0)
    assert np.isclose(relaxation[0], 630.0)
    assert np.all(np.diff(relaxation) <= 0.0)
    assert np.isclose(relaxation[-1], 600.0)


def test_relaxation_times_match_measured_frequency_window() -> None:
    times = relaxation_times(5)
    assert np.all(np.diff(times) > 0.0)
    assert np.isclose(times[0], 1.0 / (2.0 * np.pi * 25_000.0))
    assert np.isclose(times[-1], 1.0 / (2.0 * np.pi))


def test_robust_loss_rule_flags_only_extreme_points() -> None:
    values = np.asarray(
        [0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.35, 0.70]
    )
    mask = robust_loss_outliers(values)
    assert mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_exact_brand_fit_exports_bounded_sensitivity(
    tmp_path: Path,
) -> None:
    summary = write_short_time_relaxation(DEFAULT_SOURCE, tmp_path)
    assert summary["overall_status"] == (
        "storage_and_short_time_relaxation_sensitivity_only"
    )
    assert summary["raw_row_count"] == 85
    assert summary["outlier_row_count"] == 2
    assert summary["selected_term_count"] == 5
    assert summary["storage_fit_gate_passed"] is True
    assert summary["loss_fit_gate_passed"] is False
    assert summary["minimum_supported_time_s"] > 0.0
    assert summary["maximum_supported_time_s"] < 1.0
    assert len(list((tmp_path / "frequency_cards").glob("*.inp"))) == 6
    assert (tmp_path / "maxwell_spectrum.csv").exists()
    assert (tmp_path / "short_time_relaxation_sensitivity.csv").exists()
