from pathlib import Path
from types import SimpleNamespace

import pytest

from src.bambu_common_infill_reference import DEFAULT_OUTPUT
from src.common_infill_compression_validation import (
    _reference_conditions,
    run_case,
)


def test_reference_conditions_join_mass_and_curve_evidence() -> None:
    rows = _reference_conditions(DEFAULT_OUTPUT)
    assert len(rows) == 9
    assert {row["pattern"] for row in rows} == {
        "grid",
        "gyroid",
        "cubic",
    }
    assert {row["nominal_infill_percent"] for row in rows} == {
        10,
        20,
        30,
    }
    assert all(
        0.20 < row["target_total_solid_fraction"] < 0.41
        for row in rows
    )
    assert all(
        0.15 < row["source_initial_effective_modulus_MPa"] < 0.60
        for row in rows
    )


def test_case_matches_target_mass_and_is_resumable(tmp_path: Path) -> None:
    condition = {
        "pattern": "grid",
        "nominal_infill_percent": 10,
        "target_mass_g": 8.1162,
        "target_total_solid_fraction": 0.2424193548387097,
        "source_initial_effective_modulus_MPa": 0.32,
    }

    class FirstRunner:
        def run(self, *_args, **_kwargs):
            return SimpleNamespace(
                ok=True,
                status="ok",
                curve={
                    "displacement": [0.3],
                    "reaction_force": [90.0],
                },
                run_dir=tmp_path / "solver-run",
                error=None,
            )

    class FailingRunner:
        def run(self, *_args, **_kwargs):
            raise AssertionError("cached case should not rerun")

    first = run_case(
        FirstRunner(), tmp_path / "results", condition, resolution=6
    )
    cached = run_case(
        FailingRunner(), tmp_path / "results", condition, resolution=6
    )
    assert first == cached
    assert first.status == "ok"
    assert first.simulated_mass_g == pytest.approx(
        first.target_mass_g, rel=0.02
    )
    assert first.apparent_modulus_MPa == pytest.approx(10.0)
