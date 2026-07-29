import numpy as np
import pytest

from src.bambu_curve_calibration import (
    fit_elastic_slope,
    huber_location,
)


def test_fixed_elastic_window_recovers_affine_modulus():
    strain = np.linspace(-0.001, 0.02, 2000)
    stress = 2150.0 * strain + 0.25
    slope, intercept, r_squared, points = fit_elastic_slope(
        strain, stress
    )
    assert slope == pytest.approx(2150.0)
    assert intercept == pytest.approx(0.25)
    assert r_squared == pytest.approx(1.0)
    assert points > 100


def test_huber_location_limits_single_outlier():
    values = [2000.0, 2050.0, 2100.0, 8000.0]
    robust = huber_location(values)
    assert 2000.0 < robust < 2200.0
    assert robust < np.mean(values)
