from pathlib import Path

import pytest

from src.infill_benchmark import LOAD_CASES
from src.pattern_robustness_campaign import _sha256
from src.static_poisson_propagation import (
    PATTERNS,
    SCENARIOS,
    aggregate_scenario_rows,
    build_propagation_profiles,
)


def test_profiles_separate_historical_prior_from_measured_nu12(
    tmp_path: Path,
):
    paths = build_propagation_profiles(tmp_path)
    assert set(paths) == set(SCENARIOS)
    import json

    prior = json.loads(paths["prior_nu12_0p33"].read_text())
    measured = json.loads(paths["measured_nu12_0p3459"].read_text())
    assert prior["screening_engineering_constants"]["nu12"] == pytest.approx(
        0.33
    )
    assert measured["screening_engineering_constants"][
        "nu12"
    ] == pytest.approx(0.3459)
    assert prior["screening_engineering_constants"]["G12"] != pytest.approx(
        measured["screening_engineering_constants"]["G12"]
    )
    assert _sha256(paths["prior_nu12_0p33"]) != _sha256(
        paths["measured_nu12_0p3459"]
    )


def test_aggregate_requires_complete_matrix_and_tracks_rank_stability():
    scenarios = {}
    for scenario, scale in (
        ("prior_nu12_0p33", 1.0),
        ("measured_nu12_0p3459", 1.01),
    ):
        rows = []
        for pattern_index, pattern in enumerate(PATTERNS):
            for load_index, load_case in enumerate(LOAD_CASES):
                value = (
                    (100.0 - 5.0 * pattern_index + load_index) * scale
                )
                rows.append(
                    {
                        "pattern": pattern,
                        "load_case": load_case,
                        "status": "ok",
                        (
                            "critical_buckling_load_N"
                            if load_case == "buckling"
                            else "stiffness_N_per_mm"
                        ): value,
                    }
                )
        scenarios[scenario] = rows
    result = aggregate_scenario_rows(scenarios)
    assert result["all_solver_cases_passed"] is True
    assert result["rank_order_preserved"] is True
    assert result["propagation_gate_passed"] is True
    assert result["maximum_absolute_metric_change"] == pytest.approx(0.01)
    assert len(result["metric_rows"]) == len(PATTERNS) * len(LOAD_CASES)
