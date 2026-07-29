from dataclasses import replace

from src.nozzle_temperature_sensitivity import (
    summarize_temperature_results,
)
from src.solid_multiload import SolidLoadResult


def _result(pattern, load_case, value):
    row = SolidLoadResult(
        pattern=pattern,
        load_case=load_case,
        status="ok",
        mesh_size_mm=0.7,
        through_thickness_layers=5,
        nodes=10,
        elements=5,
        strut_width_mm=1.0,
        actual_relative_density=0.35,
        actual_mass_g=2.77,
        applied_displacement_mm=0.4,
        reaction_force_N=value,
        stiffness_N_per_mm=value,
        specific_stiffness_N_per_mm_g=value / 2.77,
        apparent_modulus_MPa=value,
        maximum_von_mises_MPa=1.0,
        tensile_reference_utilization=0.1,
        critical_buckling_load_N=value,
        critical_buckling_load_N_per_g=value / 2.77,
        matching_beam_metric=value,
        solid_to_beam_ratio=1.0,
        lower_E_metric=value,
        upper_E_metric=value,
        run_directory="/tmp/test",
        mesh_file="/tmp/test.msh",
    )
    if load_case == "buckling":
        return replace(row, stiffness_N_per_mm=None)
    return row


def test_temperature_summary_preserves_winner_and_builds_220_ratios():
    load_cases = ("tension", "shear", "bending", "compression", "buckling")
    results = {}
    for temperature, scale in ((210, 0.9), (220, 1.0), (230, 1.05), (240, 1.1)):
        rows = []
        for load in load_cases:
            rows.append(_result("triangular", load, 10.0 * scale))
            rows.append(_result("sd_c2_o50", load, 12.0 * scale))
        results[temperature] = rows
    rows, aggregate = summarize_temperature_results(
        results,
        {
            210: "source_disagreement",
            220: "strict_pass",
            230: "source_disagreement",
            240: "strict_pass",
        },
    )
    assert len(rows) == 40
    assert all(
        check["preserved"]
        for check in aggregate["ranking_checks"].values()
    )
    baseline = next(
        row
        for row in rows
        if row["nozzle_temperature_C"] == 220
        and row["pattern"] == "triangular"
        and row["load_case"] == "tension"
    )
    assert baseline["ratio_to_220C"] == 1.0
