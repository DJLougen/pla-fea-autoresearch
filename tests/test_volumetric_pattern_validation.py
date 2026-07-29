from pathlib import Path
from types import SimpleNamespace

from src.volumetric_pattern_validation import (
    _curve_energy,
    pairwise_matches,
    run_case,
)


def test_curve_energy_prepends_origin_and_pairwise_score_is_exact():
    assert _curve_energy([0.5, 1.0], [10.0, 20.0]) == 10.0
    physical = ["honeycomb", "cubic", "grid", "gyroid"]
    assert pairwise_matches(physical) == 6
    assert pairwise_matches(list(reversed(physical))) == 0


def test_successful_case_is_cached_before_aggregate_step(tmp_path: Path):
    class FirstRunner:
        def run(self, *_args, **_kwargs):
            return SimpleNamespace(
                ok=True,
                status="ok",
                curve={
                    "displacement": [0.1, 0.2],
                    "reaction_force": [10.0, 20.0],
                },
                run_dir=tmp_path / "solver-run",
                error=None,
            )

    class FailingRunner:
        def run(self, *_args, **_kwargs):
            raise AssertionError("cached case should not launch the solver")

    first, _ = run_case(
        FirstRunner(), tmp_path / "results", "grid", 6, "x", "elastic"
    )
    cached, _ = run_case(
        FailingRunner(), tmp_path / "results", "grid", 6, "x", "elastic"
    )
    assert first.status == "ok"
    assert cached == first
    assert (
        tmp_path
        / "results/case_results/grid-n6-x-elastic.json"
    ).exists()
