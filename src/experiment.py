from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping


@dataclass(frozen=True)
class Hypothesis:
    experiment_id: str
    parent_id: str
    statement: str
    model_class: str
    config_path: str


def validate_research_contract(
    reference: Mapping[str, object], candidate: Mapping[str, object]
) -> List[str]:
    """Return fixed-contract violations without mutating either config."""
    violations = []
    if candidate["data"] != reference["data"]:
        violations.append("experimental data paths or fixed manifest changed")
    if candidate["solver"] != reference["solver"]:
        violations.append("solver backend, executable, or timeout changed")
    if (
        candidate["output"]["results_file"]
        != reference["output"]["results_file"]
    ):
        violations.append("append-only logging target changed")
    if candidate["optimization"]["seed"] != reference["optimization"]["seed"]:
        violations.append("fixed random seed changed")
    return violations

