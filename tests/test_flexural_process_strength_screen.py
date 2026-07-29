import numpy as np
import pytest

from src.flexural_process_strength_screen import (
    screen_flexural_process_strength,
)


def test_flexural_screen_uses_zero_degree_strength():
    screen = screen_flexural_process_strength(
        np.asarray([[30.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        [60.0, 50.0, 40.0],
        [54.0, 45.0, 36.0],
        35.0,
        30.0,
    )
    assert screen is not None
    assert screen.maximum_mean_utilization == pytest.approx(0.5)
    assert screen.governing_in_plane_angle_deg == pytest.approx(0.0)


def test_flexural_screen_uses_45_degree_strength_and_shear():
    # Uniaxial tension of 25 MPa oriented at 45 degrees.
    screen = screen_flexural_process_strength(
        np.asarray([[12.5, 12.5, 0.0, 12.5, 0.0, 0.0]]),
        [60.0, 50.0, 40.0],
        [54.0, 45.0, 36.0],
        35.0,
        30.0,
    )
    assert screen is not None
    assert screen.maximum_mean_utilization == pytest.approx(0.5)
    assert screen.governing_in_plane_angle_deg == pytest.approx(45.0)


def test_flexural_screen_keeps_build_direction_proxy_separate():
    screen = screen_flexural_process_strength(
        np.asarray([[0.0, 0.0, 28.0, 0.0, 0.0, 0.0]]),
        [60.0, 50.0, 40.0],
        [54.0, 45.0, 36.0],
        35.0,
        28.0,
    )
    assert screen is not None
    assert screen.maximum_mean_utilization == pytest.approx(0.8)
    assert screen.maximum_lower_cv_utilization == pytest.approx(1.0)
    assert screen.governing_mode == "build_direction_normal_proxy"
