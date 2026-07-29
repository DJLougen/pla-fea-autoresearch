import numpy as np
import pytest

from src.metrics import aggregate_metrics, curve_metrics, interpolate_common


def test_curve_interpolation_and_metric_are_zero_for_same_line():
    mx = [0.0, 0.5, 1.0]
    my = [0.0, 1.0, 2.0]
    sx = np.linspace(0.0, 2.0, 11)
    sy = sx
    _, measured, simulated = interpolate_common(mx, my, sx, sy)
    np.testing.assert_allclose(measured, simulated)
    metric = curve_metrics(mx, my, sx, sy)
    assert metric.total == pytest.approx(0.0, abs=1e-12)


def test_metric_detects_scale_error():
    metric = curve_metrics([0, 1, 2], [0, 2, 4], [0, 1, 2], [0, 1, 2])
    assert metric.curve_rmse > 0
    assert metric.slope_error == pytest.approx(0.5)
    assert metric.peak_error == pytest.approx(0.5)


def test_weighted_aggregate():
    first = curve_metrics([0, 1], [0, 1], [0, 1], [0, 1])
    second = curve_metrics([0, 1], [0, 1], [0, 1], [0, 2])
    result = aggregate_metrics({"a": first, "b": second}, {"a": 3, "b": 1})
    assert result["total"] == pytest.approx(second.total / 4)


def test_lateral_strain_adds_fixed_poisson_sensitive_term():
    axial = [0.0, 0.01, 0.02]
    stress = [0.0, 20.0, 40.0]
    measured_lateral = [0.0, -0.003, -0.006]
    exact = curve_metrics(
        axial,
        stress,
        axial,
        stress,
        measured_lateral,
        measured_lateral,
    )
    wrong_poisson = curve_metrics(
        axial,
        stress,
        axial,
        stress,
        measured_lateral,
        [0.0, -0.0015, -0.003],
    )
    assert exact.lateral_strain_used
    assert exact.total == pytest.approx(0.0)
    assert wrong_poisson.lateral_strain_error == pytest.approx(0.5)
    assert wrong_poisson.total == pytest.approx(0.1)
