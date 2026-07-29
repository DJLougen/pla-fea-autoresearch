import json
from pathlib import Path

import pytest

from src.directional_interval_robustness import (
    BOUNDS,
    FIXED_NU12,
    PATTERNS,
    _sample_rows,
    _refinement_change,
    _run_solid_matrix,
    aggregate_interval_rows,
    write_profiles,
)
from src.infill_benchmark import LOAD_CASES
from src.materials import check_admissibility


def test_interval_samples_are_deterministic_admissible_and_cover_bounds():
    first = _sample_rows(lhs_samples=4, seed=7)
    second = _sample_rows(lhs_samples=4, seed=7)
    assert first == second
    assert len(first) == 1 + 2 * len(BOUNDS) + 3 + 4
    assert first[0]["sample_id"] == "canonical"
    for row in first:
        constants = {
            name: float(row[name])
            for name in (
                "E1",
                "E2",
                "E3",
                "nu12",
                "nu13",
                "nu23",
                "G12",
                "G13",
                "G23",
            )
        }
        assert constants["nu12"] == pytest.approx(FIXED_NU12)
        assert check_admissibility(constants).valid
    for name, (lower, upper) in BOUNDS.items():
        observed = [float(row[name]) for row in first]
        assert min(observed) == pytest.approx(lower)
        assert max(observed) == pytest.approx(upper)


def test_written_profiles_keep_sensitivity_and_measurement_labels(
    tmp_path: Path,
):
    paths = write_profiles(tmp_path, lhs_samples=2, seed=11)
    assert "canonical" in paths
    profile = json.loads(paths["axis_G13_low"].read_text())
    assert profile["model_class"] == "orthotropic_interval_sensitivity"
    sample = profile["directional_interval_sample"]
    assert "non-probabilistic" in sample["interpretation"]
    assert sample["fixed_nu12"] == pytest.approx(FIXED_NU12)
    assert profile["screening_engineering_constants"]["G13"] == 390.0
    manifest = json.loads(
        (tmp_path / "profile_manifest.json").read_text()
    )
    assert manifest["sample_count"] == len(paths)
    assert "not a probability" in manifest["interpretation"]


def test_interval_aggregation_selects_adversarial_material_cards():
    scenarios = {}
    for sample_id, custom_factor in (
        ("canonical", 1.20),
        ("adverse", 0.80),
    ):
        rows = []
        for pattern_index, pattern in enumerate(PATTERNS):
            factor = (
                1.0
                if pattern == "triangular"
                else custom_factor - pattern_index * 0.02
            )
            for load_index, load_case in enumerate(LOAD_CASES):
                value = (100.0 + load_index) * factor
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
        scenarios[sample_id] = rows
    result = aggregate_interval_rows(scenarios, fidelity="test")
    assert result["all_solver_cases_passed"] is True
    assert result["solver_case_count"] == (
        len(scenarios) * len(PATTERNS) * len(LOAD_CASES)
    )
    assert "canonical" in result["adversarial_sample_ids"]
    assert "adverse" in result["adversarial_sample_ids"]
    candidate = next(
        row
        for row in result["candidate_summary"]
        if row["pattern"] == PATTERNS[1]
    )
    assert candidate["minimum_balanced_sample"] == "adverse"
    assert candidate["load_envelopes"]["bending"][
        "minimum_sample"
    ] == "adverse"

    refined = {
        **result,
        "detail_rows": [
            {
                **row,
                "pattern_to_triangular_ratio": (
                    float(row["pattern_to_triangular_ratio"]) * 1.02
                ),
            }
            for row in result["detail_rows"]
        ],
    }
    change = _refinement_change(result, refined)
    assert change["maximum_absolute_ratio_change"] == pytest.approx(0.02)
    assert change["ratio_refinement_gate_passed"] is True


def test_solid_matrix_resumes_a_complete_material_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rows = []
    for pattern in PATTERNS:
        for load_case in LOAD_CASES:
            rows.append(
                {
                    "pattern": pattern,
                    "load_case": load_case,
                    "status": "ok",
                }
            )
    summary = tmp_path / "output/coarse/canonical/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(rows), encoding="utf-8")

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("completed material card should be resumed")

    monkeypatch.setattr(
        "src.directional_interval_robustness.run_solid_multiload",
        unexpected_solver,
    )
    result = _run_solid_matrix(
        ["canonical"],
        {"canonical": tmp_path / "unused.json"},
        tmp_path / "output",
        tmp_path / "runs",
        1.2,
        stage="coarse",
    )
    assert result["canonical"] == rows
