import numpy as np
import pytest

from src.strength_screening import (
    directional_tensile_strengths,
    orthotropic_stress_from_tensor_strain,
    screen_bounded_tension_shear_interaction,
    screen_directional_normal_strength,
    screen_interface_initiation,
    stress_rows_to_local,
)


def test_directional_screen_uses_xy_and_z_strengths():
    stress = np.asarray(
        [
            [20.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [-80.0, 5.0, 30.0, 0.0, 0.0, 0.0],
        ]
    )
    screen = screen_directional_normal_strength(
        stress,
        [35.0, 35.0, 31.0],
        [31.0, 31.0, 28.0],
    )
    assert screen is not None
    assert screen.maximum_tensile_sigma1_MPa == pytest.approx(20.0)
    assert screen.maximum_tensile_sigma3_MPa == pytest.approx(30.0)
    assert screen.maximum_mean_strength_utilization == pytest.approx(30.0 / 31.0)
    assert screen.maximum_lower_1sd_strength_utilization == pytest.approx(
        30.0 / 28.0
    )
    assert screen.governing_direction == "local_3"


def test_directional_screen_does_not_treat_compression_as_tensile_failure():
    screen = screen_directional_normal_strength(
        np.asarray([[-100.0, -60.0, -20.0, 0.0, 0.0, 0.0]]),
        [35.0, 35.0, 31.0],
    )
    assert screen is not None
    assert screen.maximum_mean_strength_utilization == 0.0


def test_stress_rotation_follows_local_axes():
    # Local 1 is global Y, local 2 is -global X, and local 3 is global Z.
    axes = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    local = stress_rows_to_local(
        np.asarray([[10.0, 30.0, 5.0, 0.0, 0.0, 0.0]]),
        axes,
    )
    assert local[0, 0] == pytest.approx(30.0)
    assert local[0, 1] == pytest.approx(10.0)
    assert local[0, 2] == pytest.approx(5.0)


def test_orthotropic_stress_recovery_uses_tensor_shear_convention():
    constants = {
        "E1": 2_000.0,
        "E2": 2_000.0,
        "E3": 2_000.0,
        "nu12": 0.25,
        "nu13": 0.25,
        "nu23": 0.25,
        "G12": 800.0,
        "G13": 800.0,
        "G23": 800.0,
    }
    strain = np.asarray(
        [
            [0.01, -0.0025, -0.0025, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.005, 0.0, 0.0],
        ]
    )
    stress = orthotropic_stress_from_tensor_strain(strain, constants)
    np.testing.assert_allclose(
        stress[0],
        [20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        atol=1.0e-12,
    )
    assert stress[1, 3] == pytest.approx(8.0)


def test_profile_strength_extraction_maps_flat_plane_to_local_1_and_2():
    properties = {
        "reported_strength_reference": {
            "overall_tensile_UTS_MPa": 35.0,
            "overall_tensile_UTS_standard_deviation_MPa": 4.0,
            "z_upright_tensile_UTS_MPa": 31.0,
            "z_upright_tensile_UTS_standard_deviation_MPa": 3.0,
        }
    }
    mean, lower = directional_tensile_strengths(properties)
    np.testing.assert_allclose(mean, [35.0, 35.0, 31.0])
    np.testing.assert_allclose(lower, [31.0, 31.0, 28.0])


def test_bounded_interaction_has_explicit_shear_bookends():
    screen = screen_bounded_tension_shear_interaction(
        np.asarray([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]]),
        [30.0, 30.0, 24.0],
    )
    assert screen is not None
    assert screen.maximum_von_mises_shear_bookend_utilization == pytest.approx(
        10.0 / (30.0 / np.sqrt(3.0))
    )
    assert (
        screen.maximum_equal_tensile_shear_bookend_utilization
        == pytest.approx(10.0 / 30.0)
    )
    assert screen.governing_component == "|tau12|"


def test_bounded_interaction_combines_tension_and_shear_quadratically():
    screen = screen_bounded_tension_shear_interaction(
        np.asarray([[15.0, 0.0, 0.0, 15.0 / np.sqrt(3.0), 0.0, 0.0]]),
        [30.0, 30.0, 24.0],
    )
    assert screen is not None
    assert screen.maximum_von_mises_shear_bookend_utilization == pytest.approx(
        np.sqrt(0.5)
    )


def test_bounded_interaction_ignores_pure_normal_compression():
    screen = screen_bounded_tension_shear_interaction(
        np.asarray([[-80.0, -50.0, -20.0, 0.0, 0.0, 0.0]]),
        [35.0, 35.0, 31.0],
    )
    assert screen is not None
    assert screen.maximum_von_mises_shear_bookend_utilization == 0.0
    assert screen.maximum_equal_tensile_shear_bookend_utilization == 0.0


def test_interface_initiation_combines_local3_opening_and_tear():
    screen = screen_interface_initiation(
        np.asarray([[0.0, 0.0, 17.535, 0.0, 6.94, 0.0]]),
        normal_strength_MPa=35.07,
        tear_strength_MPa=13.88,
        lower_1sd_normal_strength_MPa=32.52,
        lower_1sd_tear_strength_MPa=10.90,
    )
    assert screen is not None
    assert screen.maximum_utilization == pytest.approx(np.sqrt(0.5))
    assert screen.maximum_tensile_sigma3_MPa == pytest.approx(17.535)
    assert screen.maximum_resultant_interface_tear_MPa == pytest.approx(6.94)


def test_interface_initiation_ignores_local3_compression_but_not_tear():
    screen = screen_interface_initiation(
        np.asarray([[0.0, 0.0, -100.0, 0.0, 3.0, 4.0]]),
        normal_strength_MPa=35.07,
        tear_strength_MPa=10.0,
        lower_1sd_normal_strength_MPa=32.52,
        lower_1sd_tear_strength_MPa=8.0,
    )
    assert screen is not None
    assert screen.maximum_utilization == pytest.approx(0.5)
    assert screen.governing_component == "resultant_tau23_tau31_tear"
