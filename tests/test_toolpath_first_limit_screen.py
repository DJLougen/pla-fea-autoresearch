import numpy as np

from src.toolpath_first_limit_screen import (
    evaluate_first_limit_gate,
    first_limit_metrics,
)


PROPERTIES = {
    "reported_strength_reference": {
        "overall_tensile_UTS_MPa": 35.0,
        "overall_tensile_UTS_standard_deviation_MPa": 4.0,
        "z_upright_tensile_UTS_MPa": 31.0,
        "z_upright_tensile_UTS_standard_deviation_MPa": 3.0,
    }
}
INTERFACE = {
    "weak": {
        "normal_tensile_mean_MPa": 35.0,
        "apparent_tear_mean_MPa": 14.0,
        "normal_tensile_lower_1sd_MPa": 32.0,
        "apparent_tear_lower_1sd_MPa": 11.0,
    },
    "strong": {
        "normal_tensile_mean_MPa": 47.0,
        "apparent_tear_mean_MPa": 15.0,
        "normal_tensile_lower_1sd_MPa": 45.0,
        "apparent_tear_lower_1sd_MPa": 12.0,
    },
}


def test_first_limit_band_is_positive_and_ordered() -> None:
    stress = np.asarray(
        [
            [10.0, 0.0, 8.0, 3.0, 5.0, 2.0],
            [20.0, 0.0, 12.0, 5.0, 7.0, 3.0],
        ]
    )
    result = first_limit_metrics(stress, PROPERTIES, INTERFACE)
    assert (
        result["conservative_p99_first_limit_displacement_mm"]
        <= result["optimistic_p99_first_limit_displacement_mm"]
    )
    assert result["recommended_first_ramp_displacement_mm"] <= 0.4
    assert result[
        "p99_lower_1sd_von_mises_shear_bookend_utilization"
    ] > result["p99_equal_tensile_shear_bookend_utilization"]


def test_first_limit_gate_requires_mesh_stability() -> None:
    rows = []
    for pattern, base in (
        ("triangular", 0.50),
        ("kite_08_08_08_08", 0.55),
        ("kite_08_08_12_08", 0.53),
    ):
        for resolution, factor in ((60, 1.03), (80, 1.0)):
            rows.append(
                {
                    "pattern": pattern,
                    "resolution_xy": resolution,
                    "status": "ok",
                    "sampled_strain_points": 100,
                    "conservative_p99_first_limit_displacement_mm": (
                        base * factor
                    ),
                    "optimistic_p99_first_limit_displacement_mm": (
                        1.4 * base * factor
                    ),
                }
            )
    gate = evaluate_first_limit_gate(rows)
    assert gate["bounded_first_limit_screen_admitted"] is True
    rows[0][
        "conservative_p99_first_limit_displacement_mm"
    ] = 0.8
    gate = evaluate_first_limit_gate(rows)
    assert gate["bounded_first_limit_screen_admitted"] is False
