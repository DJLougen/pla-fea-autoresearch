from src.print_speed_infill_sensitivity import process_effect_checks


def _profile(sample, strength, modulus):
    return {
        "sample_group": sample,
        "tensile_strength_mean_MPa": strength,
        "youngs_modulus_mean_MPa": modulus,
    }


def test_matched_process_effect_checks_keep_pair_identity():
    profiles = [
        _profile(1, 35.0, 3100.0),
        _profile(2, 34.0, 3000.0),
        _profile(3, 34.0, 3050.0),
        _profile(4, 33.0, 2950.0),
        _profile(5, 38.0, 3150.0),
        _profile(6, 37.0, 3050.0),
        _profile(7, 37.0, 3100.0),
        _profile(8, 36.0, 3000.0),
    ]
    rows = process_effect_checks(profiles)
    assert len(rows) == 24
    assert all(row["expected_direction_observed"] for row in rows)
    assert all(float(row["ratio"]) > 1.0 for row in rows)
