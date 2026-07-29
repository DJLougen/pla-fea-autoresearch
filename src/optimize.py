from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import differential_evolution

from .config import dump_config, load_config
from .evaluate import FixedEvaluator
from .materials import (
    initial_vector,
    material_card,
    parameter_bounds,
    parameter_names,
    vector_to_constants,
)
from .results import append_result, base_record


class BudgetReached(RuntimeError):
    pass


class EarlyStop(RuntimeError):
    pass


@dataclass
class OptimizationResult:
    experiment_id: str
    model_class: str
    parameter_names: List[str]
    parameters: Dict[str, float]
    constants: Dict[str, float]
    bounds: Dict[str, List[float]]
    calibration: Dict[str, object]
    validation: Dict[str, object]
    heldout: Dict[str, object]
    objective_evaluations: int
    cache_hits: int
    runtime_seconds: float
    optimizer_message: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _key(vector: Sequence[float]) -> str:
    return ",".join(f"{float(value):.12g}" for value in vector)


def _data_content_fingerprint(config: Mapping[str, object]) -> str:
    """Hash the fixed manifest and every referenced processed curve."""
    manifest_path = Path(config["data"]["manifest"])
    processed_dir = Path(config["data"]["processed_dir"])
    digest = hashlib.sha256()
    digest.update(manifest_path.read_bytes())
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        filenames = sorted(
            {
                str(row["file"])
                for row in csv.DictReader(handle)
                if row.get("file")
            }
        )
    for filename in filenames:
        path = processed_dir / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class CachedObjective:
    def __init__(
        self,
        evaluator: FixedEvaluator,
        checkpoint: Path,
        budget: int,
        patience: int,
        tolerance: float,
        fingerprint: str = "",
    ):
        self.evaluator = evaluator
        self.checkpoint = Path(checkpoint)
        self.budget = int(budget)
        self.patience = int(patience)
        self.tolerance = float(tolerance)
        self.fingerprint = fingerprint
        self.cache: Dict[str, dict] = {}
        self.cache_hits = 0
        self.best_score = float("inf")
        self.since_improvement = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.inflight = set()
        self._load()

    def _load(self) -> None:
        if not self.checkpoint.exists():
            return
        payload = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        stored_fingerprint = payload.get("fingerprint", "")
        if self.fingerprint and stored_fingerprint != self.fingerprint:
            raise ValueError(
                "checkpoint belongs to a different experiment configuration; "
                "use --no-resume or a new experiment id"
            )
        self.cache = payload.get("cache", {})
        if len(self.cache) > self.budget:
            raise ValueError(
                "checkpoint already contains more evaluations than the requested budget"
            )
        if self.cache:
            self.best_score = min(float(item["score"]) for item in self.cache.values())

    def _save(self) -> None:
        payload = {
            "budget": self.budget,
            "fingerprint": self.fingerprint,
            "cache": self.cache,
            "best_score": self.best_score,
        }
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.checkpoint)

    def __call__(self, vector: Sequence[float]) -> float:
        key = _key(vector)
        with self.condition:
            while key in self.inflight:
                self.condition.wait()
            if key in self.cache:
                self.cache_hits += 1
                return float(self.cache[key]["score"])
            if len(self.cache) + len(self.inflight) >= self.budget:
                raise BudgetReached
            evaluation_number = len(self.cache) + 1
            self.inflight.add(key)
        tag = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        try:
            result = self.evaluator.evaluate_vector(
                vector, "calibration", f"eval-{evaluation_number:05d}-{tag}"
            )
        except Exception:
            with self.condition:
                self.inflight.discard(key)
                self.condition.notify_all()
            raise
        with self.condition:
            self.inflight.discard(key)
            self.cache[key] = {
                "vector": [float(value) for value in vector],
                "score": float(result.score),
                "solver_failures": result.solver_failures,
            }
            if result.score < self.best_score - self.tolerance:
                self.best_score = float(result.score)
                self.since_improvement = 0
            else:
                self.since_improvement += 1
            self._save()
            self.condition.notify_all()
        return float(result.score)

    def best(self) -> tuple:
        if not self.cache:
            raise RuntimeError("optimizer produced no evaluations")
        item = min(self.cache.values(), key=lambda row: float(row["score"]))
        return np.asarray(item["vector"], dtype=float), float(item["score"])


def _initial_population(
    bounds: Sequence[Sequence[float]], initial: np.ndarray, size: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = np.asarray([bound[0] for bound in bounds])
    upper = np.asarray([bound[1] for bound in bounds])
    population = rng.uniform(lower, upper, size=(size, len(bounds)))
    population[0] = np.clip(initial, lower, upper)
    cursor = 1
    for dimension in range(len(bounds)):
        delta = 0.12 * max(abs(float(initial[dimension])), 0.25)
        for sign in (-1.0, 1.0):
            if cursor >= size:
                break
            candidate = np.clip(initial.copy(), lower, upper)
            candidate[dimension] = np.clip(
                initial[dimension] + sign * delta,
                lower[dimension],
                upper[dimension],
            )
            population[cursor] = candidate
            cursor += 1
    return population


def optimize_config(
    config: Mapping[str, object],
    budget_override: Optional[int] = None,
    resume: bool = True,
) -> OptimizationResult:
    start = time.monotonic()
    experiment = config["experiment"]
    optimization = config["optimization"]
    if optimization.get("algorithm", "differential_evolution") != "differential_evolution":
        raise NotImplementedError(
            "this release implements differential_evolution; optimizer adapters "
            "may add CMA-ES or Bayesian optimization without changing evaluation"
        )
    model_class = experiment["model_class"]
    fit_poisson = bool(experiment.get("fit_poisson", True))
    ties = experiment.get("parameter_ties")
    names = parameter_names(model_class, fit_poisson, ties)
    bounds_list = parameter_bounds(
        model_class, fit_poisson, ties, experiment.get("parameter_bounds")
    )
    budget = int(budget_override or optimization["budget"])
    population_size = max(5, int(optimization.get("popsize", 5)) * len(names))
    if budget < population_size:
        population_size = budget
    if population_size < 5:
        raise ValueError("differential evolution needs a budget of at least five")
    checkpoint = (
        Path(config["output"]["runs_dir"]) / str(experiment["id"]) / "checkpoint.json"
    )
    if not resume and checkpoint.exists():
        checkpoint.unlink()
    evaluator = FixedEvaluator(config)
    fingerprint_payload = {
        "experiment": config["experiment"],
        "solver": config["solver"],
        "data": config["data"],
        "algorithm": optimization.get("algorithm", "differential_evolution"),
        "seed": optimization["seed"],
        "data_content_sha256": _data_content_fingerprint(config),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    objective = CachedObjective(
        evaluator,
        checkpoint,
        budget,
        int(optimization.get("early_stopping_patience", 0)),
        float(optimization.get("early_stopping_tolerance", 1.0e-6)),
        fingerprint,
    )
    initial = initial_vector(
        model_class, fit_poisson, ties, experiment.get("initial_parameters")
    )
    population = _initial_population(
        bounds_list, initial, population_size, int(optimization["seed"])
    )
    generations = max(0, int(math.floor(budget / population_size)) - 1)
    message = "completed"
    workers = max(1, int(optimization.get("workers", 1)))
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        scipy_result = differential_evolution(
            objective,
            bounds_list,
            seed=int(optimization["seed"]),
            maxiter=generations,
            popsize=max(1, int(optimization.get("popsize", 5))),
            init=population,
            polish=bool(optimization.get("polish", False)),
            workers=pool.map if pool else 1,
            updating="deferred" if pool else "immediate",
            tol=0.0,
            atol=0.0,
            callback=lambda _vector, _convergence: (
                objective.patience > 0
                and objective.since_improvement >= objective.patience
            ),
        )
        if "callback" in str(scipy_result.message).lower():
            message = "early stopping patience reached"
    except BudgetReached:
        message = "evaluation budget reached"
    finally:
        if pool:
            pool.shutdown(wait=True)
    best_vector, _ = objective.best()
    constants = vector_to_constants(model_class, best_vector, fit_poisson, ties)
    calibration = evaluator.evaluate_constants(
        constants, "calibration", f"{experiment['id']}-best-cal"
    )
    validation = evaluator.evaluate_constants(
        constants, "validation", f"{experiment['id']}-best-val"
    )
    heldout = evaluator.evaluate_constants(
        constants, "heldout", f"{experiment['id']}-best-heldout"
    )
    runtime = time.monotonic() - start
    result = OptimizationResult(
        str(experiment["id"]),
        model_class,
        names,
        dict(zip(names, map(float, best_vector))),
        constants,
        {name: list(map(float, bound)) for name, bound in zip(names, bounds_list)},
        calibration.to_dict(),
        validation.to_dict(),
        heldout.to_dict(),
        len(objective.cache),
        objective.cache_hits,
        runtime,
        message,
    )
    run_dir = Path(config["output"]["runs_dir"]) / str(experiment["id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "optimization.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def save_best(config: Mapping[str, object], result: OptimizationResult) -> None:
    best_dir = Path(config["output"]["best_dir"])
    best_dir.mkdir(parents=True, exist_ok=True)
    (best_dir / "material.inp").write_text(
        material_card(result.constants), encoding="utf-8"
    )
    dump_config(dict(config), best_dir / "config.yaml")
    (best_dir / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def log_standalone(
    config: Mapping[str, object], result: OptimizationResult
) -> None:
    record = base_record(config)
    record.update(
        {
            "parameters_json": result.parameters,
            "engineering_constants_json": result.constants,
            "bounds_json": result.bounds,
            "objective_score": result.validation["score"],
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
            "accepted": True,
            "rejection_reason": "",
        }
    )
    append_result(Path(config["output"]["results_file"]), record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate one elastic material hypothesis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = optimize_config(config, args.budget, resume=not args.no_resume)
    save_best(config, result)
    log_standalone(config, result)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
