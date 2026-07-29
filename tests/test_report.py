import json
from pathlib import Path

from src.report import build_report
from src.results import append_result


CONSTANTS = {
    "E1": 2200.0,
    "E2": 2200.0,
    "E3": 2200.0,
    "nu12": 0.33,
    "nu13": 0.33,
    "nu23": 0.33,
    "G12": 827.0,
    "G13": 827.0,
    "G23": 827.0,
}


def _record(
    experiment_id: str,
    model_class: str,
    objective: float,
    accepted: bool,
) -> dict:
    return {
        "experiment_id": experiment_id,
        "parent_experiment_id": "",
        "timestamp_utc": "2026-07-28T00:00:00+00:00",
        "hypothesis": f"{experiment_id} hypothesis",
        "model_class": model_class,
        "parameters_json": {"E": 2200.0},
        "engineering_constants_json": CONSTANTS,
        "bounds_json": {"E": [500.0, 6000.0]},
        "random_seed": 42,
        "mesh_level": "analytic",
        "objective_score": objective,
        "calibration_score": objective,
        "validation_score": objective,
        "heldout_score": objective,
        "per_test_scores_json": {},
        "runtime_seconds": 1.0,
        "solver_status_json": {},
        "solver_failure_count": 0,
        "objective_evaluations": 10,
        "accepted": accepted,
        "rejection_reason": "" if accepted else "not better",
    }


def test_report_summary_uses_latest_attempt_not_historical_best(
    tmp_path: Path,
):
    results = tmp_path / "results.tsv"
    append_result(
        results, _record("orthotropic", "orthotropic", 0.01, True)
    )
    append_result(
        results, _record("isotropic", "isotropic", 0.10, True)
    )
    append_result(
        results, _record("orthotropic", "orthotropic", 0.05, False)
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "best_experiment_id": "isotropic",
                "attempts": [
                    {"experiment_id": "isotropic"},
                    {"experiment_id": "orthotropic"},
                ],
            }
        )
    )
    report = build_report(results, summary)
    assert "Experiment: `isotropic`" in report
    assert report.count("| orthotropic |") == 1
    assert "did not materially improve prediction" in report
