import numpy as np
import pytest

from src.nonlinear_infill import (
    _yield_from_force_curve,
    _yield_from_peeq_history,
    _yield_from_reference_curve,
)
from src.infill_benchmark import make_staggered_diamond_variant


def test_force_curve_yield_detection():
    displacement = np.array([0.1, 0.2, 0.3, 0.4])
    force = np.array([100.0, 200.0, 297.0, 380.0])
    strain, yield_force = _yield_from_force_curve(
        displacement, force, 1000.0, 40.0
    )
    assert strain == pytest.approx(0.3 / 40.0)
    assert yield_force == pytest.approx(297.0)


def test_matched_reference_curve_yield_detection():
    displacement = np.array([0.1, 0.2, 0.3, 0.4])
    reference = np.array([101.0, 204.0, 309.0, 416.0])
    force = np.array([101.0, 204.0, 305.0, 390.0])
    strain, yield_force, matched = _yield_from_reference_curve(
        displacement,
        force,
        displacement,
        reference,
        40.0,
    )
    assert strain == pytest.approx(0.3 / 40.0)
    assert yield_force == pytest.approx(305.0)
    assert np.array_equal(matched, reference)


def test_peeq_history_yield_detection_interpolates_solver_history():
    time = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    displacement = np.array([0.0, 0.15, 0.3, 0.45, 0.6])
    force = np.array([0.0, 250.0, 490.0, 700.0, 850.0])
    strain, yield_force = _yield_from_peeq_history(
        time,
        displacement,
        force,
        {0.25: 0.0, 0.5: 2.0e-6, 0.75: 1.0e-4},
        40.0,
    )
    assert strain == pytest.approx(0.3 / 40.0)
    assert yield_force == pytest.approx(490.0)


def test_custom_nonlinear_geometry_can_be_constructed_for_runner():
    geometry = make_staggered_diamond_variant(
        "sd_c2_o50", cells=2, offset_fraction=0.5
    )
    assert geometry.name == "sd_c2_o50"
    assert geometry.edges
