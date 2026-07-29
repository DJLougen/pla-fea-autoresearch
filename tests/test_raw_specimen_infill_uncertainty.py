import pytest

from src.raw_specimen_infill_uncertainty import (
    read_specimen_moduli,
    scalar_linearity_errors,
    write_scalar_profiles,
)


def test_reads_22_positive_raw_moduli():
    rows = read_specimen_moduli()
    assert len(rows) == 22
    assert len({row["source_specimen_label"] for row in rows}) == 22
    assert min(row["modulus_MPa"] for row in rows) > 1600.0
    assert max(row["modulus_MPa"] for row in rows) < 2700.0


def test_scalar_profiles_use_measured_endpoints(tmp_path):
    paths = write_scalar_profiles(tmp_path)
    assert set(paths) == {"measured_minimum", "measured_maximum"}
    assert all(path.exists() for path in paths.values())


def test_scalar_linearity_error_calculation():
    baseline = [
        {
            "pattern": "triangular",
            "load_case": "bending",
            "stiffness_N_per_mm": 10.0,
        },
        {
            "pattern": "triangular",
            "load_case": "buckling",
            "critical_buckling_load_N": 20.0,
        },
    ]
    endpoints = {
        "low": [
            {
                "pattern": "triangular",
                "load_case": "bending",
                "stiffness_N_per_mm": 5.0,
            },
            {
                "pattern": "triangular",
                "load_case": "buckling",
                "critical_buckling_load_N": 10.0,
            },
        ]
    }
    rows = scalar_linearity_errors(
        baseline, endpoints, 2000.0, {"low": 1000.0}
    )
    assert len(rows) == 2
    assert all(row["relative_error"] == pytest.approx(0.0) for row in rows)
