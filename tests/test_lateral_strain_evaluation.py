import numpy as np
import pandas as pd
import pytest

from src.evaluate import _effective_global_xy_poisson, _mock_curve
from src.materials import PRIORS


def test_effective_poisson_matches_reciprocal_directional_constants():
    assert _effective_global_xy_poisson(PRIORS, "1") == pytest.approx(
        PRIORS["nu12"]
    )
    assert _effective_global_xy_poisson(PRIORS, "2") == pytest.approx(
        PRIORS["nu12"] * PRIORS["E2"] / PRIORS["E1"]
    )
    assert _effective_global_xy_poisson(PRIORS, "3") == pytest.approx(
        PRIORS["nu13"] * PRIORS["E3"] / PRIORS["E1"]
    )


def test_mock_tensile_curve_returns_matching_lateral_strain():
    frame = pd.DataFrame(
        {
            "engineering_strain": [0.0, 0.01],
            "engineering_stress_MPa": [0.0, 1.0],
        }
    )
    axial, stress, lateral = _mock_curve(
        PRIORS, frame, "tensile", "3"
    )
    assert lateral is not None
    np.testing.assert_allclose(
        lateral,
        -_effective_global_xy_poisson(PRIORS, "3") * axial,
    )
    np.testing.assert_allclose(stress, PRIORS["E3"] * axial)
