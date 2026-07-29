import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.bambu_pvh_reference import DEFAULT_OUTPUT as REFERENCE
from src.pvh_material_transfer_sensitivity import (
    PROJECT_CARD,
    material_cards,
    run_case,
)


def test_material_cards_join_independent_and_project_evidence() -> None:
    cards = material_cards(PROJECT_CARD, REFERENCE)
    assert set(cards) == {
        "project_raw_tensile",
        "independent_pvh_author",
    }
    assert cards["independent_pvh_author"][
        "youngs_modulus_MPa"
    ] == pytest.approx(2140.0)
    assert cards["independent_pvh_author"][
        "yield_stress_MPa"
    ] == pytest.approx(36.0)
    assert len(cards["project_raw_tensile"]["plastic_points"]) == 10


def test_transfer_case_is_mass_matched_and_resumable(
    tmp_path: Path,
) -> None:
    card = {
        "youngs_modulus_MPa": 2140.0,
        "poisson_ratio": 0.35,
        "yield_stress_MPa": 36.0,
        "plastic_points": [
            {"stress_MPa": 36.0, "equivalent_plastic_strain": 0.0}
        ],
    }

    class FirstRunner:
        def run(self, *_args, **_kwargs):
            return SimpleNamespace(
                ok=True,
                status="ok",
                curve={
                    "displacement": [-0.05, -0.10, -0.20, -1.0],
                    "reaction_force": [-5.0, -10.0, -18.0, -30.0],
                },
                run_dir=tmp_path / "solver-run",
                error=None,
            )

    class FailingRunner:
        def run(self, *_args, **_kwargs):
            raise AssertionError("cached case should not rerun")

    first = run_case(
        FirstRunner(),
        tmp_path / "results",
        "gyroid",
        "independent_pvh_author",
        card,
        resolution=6,
    )
    cached = run_case(
        FailingRunner(),
        tmp_path / "results",
        "gyroid",
        "independent_pvh_author",
        card,
        resolution=6,
    )
    assert first == cached
    assert first.status == "ok"
    assert first.actual_relative_density == pytest.approx(0.5, rel=0.05)
    assert first.absorbed_energy_N_mm > 0.0
