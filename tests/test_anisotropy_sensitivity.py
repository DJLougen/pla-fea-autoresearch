import pytest

from src.anisotropy_sensitivity import compare_results
from src.solid_multiload import SolidLoadResult


def _row(stiffness: float) -> SolidLoadResult:
    return SolidLoadResult(
        pattern="grid",
        load_case="tension",
        status="ok",
        mesh_size_mm=0.7,
        through_thickness_layers=5,
        nodes=10,
        elements=5,
        strut_width_mm=2.0,
        actual_relative_density=0.35,
        actual_mass_g=2.78,
        applied_displacement_mm=0.4,
        reaction_force_N=0.4 * stiffness,
        stiffness_N_per_mm=stiffness,
        specific_stiffness_N_per_mm_g=stiffness / 2.78,
        apparent_modulus_MPa=100.0,
        maximum_von_mises_MPa=20.0,
        tensile_reference_utilization=0.5,
        critical_buckling_load_N=None,
        critical_buckling_load_N_per_g=None,
        matching_beam_metric=stiffness,
        solid_to_beam_ratio=1.0,
        lower_E_metric=stiffness * 0.9,
        upper_E_metric=stiffness * 1.1,
        run_directory="run",
        mesh_file="mesh",
    )


def test_compare_results_reports_paired_percent_change():
    comparison = compare_results([_row(100.0)], [_row(80.0)])
    assert comparison[0]["transverse_to_isotropic_ratio"] == pytest.approx(
        0.8
    )
    assert comparison[0]["percent_change"] == pytest.approx(-20.0)
