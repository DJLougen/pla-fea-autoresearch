from src.ballistic_pattern_validation import (
    PHYSICAL_RANKING,
    analytical_penetration_mm,
    ranking,
    summarize_physical_results,
    velocity_sensitivity,
)


def test_paper_energy_balance_and_physical_ranking():
    assert abs(analytical_penetration_mm() - 155.37) < 0.02
    rows = summarize_physical_results()
    assert len(rows) == 4
    assert ranking(rows) == list(PHYSICAL_RANKING)
    honeycomb = next(row for row in rows if row["pattern"] == "honeycomb")
    grid = next(row for row in rows if row["pattern"] == "grid")
    gyroid = next(row for row in rows if row["pattern"] == "gyroid")
    assert float(honeycomb["mean_penetration_mm"]) < float(
        grid["mean_penetration_mm"]
    )
    assert float(honeycomb["specific_resistance_ratio_to_grid"]) > 1.0
    assert gyroid["censoring"] == "right_censored_all_three"


def test_velocity_sensitivity_is_quadratic_and_keeps_censoring():
    rows = velocity_sensitivity(summarize_physical_results())
    assert len(rows) == 12
    grid_355 = next(
        row for row in rows
        if row["pattern"] == "grid" and row["velocity_m_s"] == 355.0
    )
    grid_368 = next(
        row for row in rows
        if row["pattern"] == "grid" and row["velocity_m_s"] == 368.0
    )
    assert (
        float(grid_368["empirical_mean_penetration_mm"])
        / float(grid_355["empirical_mean_penetration_mm"])
        == (368.0 / 355.0) ** 2
    )
    gyroid = [
        row for row in rows if row["pattern"] == "gyroid"
    ]
    assert all(
        row["empirical_mean_penetration_mm"] is None for row in gyroid
    )
    assert all(
        row["right_censored_penetration_lower_bound_mm"] is not None
        for row in gyroid
    )
