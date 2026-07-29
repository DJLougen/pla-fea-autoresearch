from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional


RESULT_FIELDS = [
    "experiment_id",
    "parent_experiment_id",
    "timestamp_utc",
    "git_commit",
    "hypothesis",
    "model_class",
    "parameters_json",
    "engineering_constants_json",
    "bounds_json",
    "random_seed",
    "mesh_level",
    "objective_score",
    "calibration_score",
    "validation_score",
    "heldout_score",
    "per_test_scores_json",
    "runtime_seconds",
    "solver_status_json",
    "solver_failure_count",
    "objective_evaluations",
    "accepted",
    "rejection_reason",
]


def git_commit(cwd: Path = None) -> Optional[str]:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return process.stdout.strip() if process.returncode == 0 else None


def append_result(path: Path, record: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle, delimiter="\t"), [])
        if header != RESULT_FIELDS:
            raise ValueError("existing results file has an incompatible fixed schema")
    normalized = {}
    for field in RESULT_FIELDS:
        value = record.get(field, "")
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if value is None:
            value = ""
        normalized[field] = value
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)
        handle.flush()


def base_record(config: Mapping[str, object]) -> dict:
    experiment = config["experiment"]
    return {
        "experiment_id": experiment["id"],
        "parent_experiment_id": experiment.get("parent_id"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(Path.cwd()),
        "hypothesis": experiment["hypothesis"],
        "model_class": experiment["model_class"],
        "random_seed": config["optimization"]["seed"],
        "mesh_level": experiment.get("mesh_level", "coarse"),
    }

