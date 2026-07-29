import numpy as np
import pandas as pd
import pytest

from src.prepare import (
    force_displacement_to_stress_strain,
    preprocess_frame,
)


def test_force_displacement_conversion_uses_n_mm_mpa():
    stress, strain = force_displacement_to_stress_strain(
        np.array([0.0, 400.0]), np.array([0.0, 0.5]), 10.0, 4.0, 50.0
    )
    np.testing.assert_allclose(stress, [0.0, 10.0])
    np.testing.assert_allclose(strain, [0.0, 0.01])


def test_preprocess_converts_sorts_and_deduplicates():
    frame = pd.DataFrame(
        {
            "Force": [400.0, 0.0, 200.0, 200.0],
            "Displacement": [0.5, 0.0, 0.25, 0.25],
            "width": [10.0] * 4,
            "thickness": [4.0] * 4,
            "gauge_length": [50.0] * 4,
        }
    )
    processed, metadata = preprocess_frame(frame, "coupon.csv")
    assert len(processed) == 3
    assert processed["engineering_strain"].is_monotonic_increasing
    assert processed["engineering_stress_MPa"].iloc[-1] == pytest.approx(10.0)
    assert metadata["partition"] == "calibration"


def test_preprocess_refuses_missing_geometry():
    frame = pd.DataFrame({"force_N": [0, 1, 2], "displacement_mm": [0, 1, 2]})
    with pytest.raises(ValueError, match="requires"):
        preprocess_frame(frame, "bad.csv")


def test_non_tensile_force_displacement_is_preserved_without_geometry():
    frame = pd.DataFrame(
        {
            "test_type": ["flexural"] * 3,
            "force_N": [0.0, 10.0, 20.0],
            "displacement_mm": [0.0, 0.5, 1.0],
        }
    )
    processed, metadata = preprocess_frame(frame, "flexural.csv")
    assert metadata["test_type"] == "flexural"
    assert "engineering_strain" not in processed
    np.testing.assert_allclose(processed["force_N"], [0.0, 10.0, 20.0])


def test_preprocess_detects_obvious_strain_unit_error():
    frame = pd.DataFrame(
        {
            "engineering_strain": [0, 10, 20],
            "engineering_stress_MPa": [0, 10, 20],
        }
    )
    with pytest.raises(ValueError, match="check units"):
        preprocess_frame(frame, "percent-as-fraction.csv")


def test_preprocess_accepts_human_readable_column_names():
    frame = pd.DataFrame(
        {
            "Engineering Strain": [0, 0.01, 0.02],
            "Engineering Stress MPa": [0, 20, 40],
            "Print Direction": ["1", "1", "1"],
        }
    )
    processed, _ = preprocess_frame(frame, "readable.csv")
    assert "engineering_strain" in processed
    assert "engineering_stress_MPa" in processed


def test_transverse_strain_is_numeric_but_not_required_on_every_row():
    frame = pd.DataFrame(
        {
            "engineering_strain": [0.0, 0.01, 0.02],
            "engineering_stress_MPa": [0.0, 20.0, 40.0],
            "lateral_strain": ["", "-0.003", "-0.006"],
        }
    )
    processed, _ = preprocess_frame(frame, "lateral.csv")
    assert processed["transverse_strain"].isna().sum() == 1
    assert processed["transverse_strain"].iloc[-1] == pytest.approx(-0.006)
