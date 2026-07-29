from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import yaml

from .config import load_config
from .experiment import validate_research_contract
from .materials import complexity
from .optimize import OptimizationResult, optimize_config, save_best
from .results import append_result, base_record


COMPLEXITY_PENALTY_PER_PARAMETER = 0.002
IMPROVEMENT_TOLERANCE = 1.0e-6


def selection_score(config: Mapping[str, object], result: OptimizationResult) -> float:
    experiment = config["experiment"]
    count = complexity(
        experiment["model_class"],
        bool(experiment.get("fit_poisson", True)),
        experiment.get("parameter_ties"),
    )
    return float(result.validation["score"]) + COMPLEXITY_PENALTY_PER_PARAMETER * count


def _log(
    config: Mapping[str, object],
    result: OptimizationResult,
    score: float,
    accepted: bool,
    reason: str,
) -> None:
    record = base_record(config)
    record.update(
        {
            "parameters_json": result.parameters,
            "engineering_constants_json": result.constants,
            "bounds_json": result.bounds,
            "objective_score": score,
            "calibration_score": result.calibration["score"],
            "validation_score": result.validation["score"],
            "heldout_score": result.heldout["score"],
            "per_test_scores_json": {
                "calibration": result.calibration["per_test"],
                "validation": result.validation["per_test"],
                "heldout": result.heldout["per_test"],
            },
            "runtime_seconds": result.runtime_seconds,
            "solver_status_json": {
                "calibration": result.calibration["statuses"],
                "validation": result.validation["statuses"],
                "heldout": result.heldout["statuses"],
            },
            "solver_failure_count": sum(
                result.to_dict()[partition]["solver_failures"]
                for partition in ("calibration", "validation", "heldout")
            ),
            "objective_evaluations": result.objective_evaluations,
            "accepted": accepted,
            "rejection_reason": reason,
        }
    )
    append_result(Path(config["output"]["results_file"]), record)


def run_research(research_space: Path, max_experiments: int) -> Dict[str, object]:
    payload = yaml.safe_load(Path(research_space).read_text(encoding="utf-8"))
    hypotheses = payload.get("hypotheses", [])
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("research space must contain a non-empty hypotheses list")
    incumbent = None
    incumbent_score = float("inf")
    attempts = []
    reference_config = None
    for proposal in hypotheses[:max_experiments]:
        config = load_config(proposal["config"])
        if reference_config is None:
            reference_config = config
        violations = validate_research_contract(reference_config, config)
        if violations:
            reason = "; ".join(violations)
            record = base_record(config)
            record.update(
                {
                    "accepted": False,
                    "rejection_reason": reason,
                    "objective_evaluations": 0,
                }
            )
            append_result(Path(config["output"]["results_file"]), record)
            attempts.append(
                {
                    "experiment_id": config["experiment"]["id"],
                    "accepted": False,
                    "reason": reason,
                }
            )
            continue
        result = optimize_config(config)
        score = selection_score(config, result)
        accepted = score < incumbent_score - IMPROVEMENT_TOLERANCE
        reason = "" if accepted else (
            f"complexity-adjusted validation score {score:.8g} did not beat "
            f"{incumbent_score:.8g}"
        )
        _log(config, result, score, accepted, reason)
        if accepted:
            incumbent = result
            incumbent_score = score
            save_best(config, result)
        attempts.append(
            {
                "experiment_id": config["experiment"]["id"],
                "model_class": config["experiment"]["model_class"],
                "hypothesis": config["experiment"]["hypothesis"],
                "selection_score": score,
                "accepted": accepted,
                "reason": reason,
            }
        )
    if incumbent is None:
        raise RuntimeError("no research hypothesis completed successfully")
    summary = {
        "best_experiment_id": incumbent.experiment_id,
        "best_model_class": incumbent.model_class,
        "best_selection_score": incumbent_score,
        "attempts": attempts,
    }
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "autoresearch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run constrained deterministic autoresearch")
    parser.add_argument("--program", default="program.md")
    parser.add_argument("--research-space", default="configs/research_space.yaml")
    parser.add_argument("--max-experiments", type=int, default=20)
    args = parser.parse_args()
    if not Path(args.program).exists():
        raise FileNotFoundError(args.program)
    summary = run_research(Path(args.research_space), args.max_experiments)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
