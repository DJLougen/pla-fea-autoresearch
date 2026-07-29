import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.evaluate import FixedEvaluator
from src.optimize import CachedObjective, _data_content_fingerprint
from src.results import RESULT_FIELDS, append_result


def test_append_only_result_schema(tmp_path):
    path = tmp_path / "results.tsv"
    record = {field: "" for field in RESULT_FIELDS}
    record["experiment_id"] = "one"
    record["parameters_json"] = {"E": 2000}
    append_result(path, record)
    record["experiment_id"] = "two"
    append_result(path, record)
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert json.loads(rows[0]["parameters_json"]) == {"E": 2000}


def test_logger_refuses_schema_drift(tmp_path):
    path = tmp_path / "results.tsv"
    path.write_text("wrong\theader\n")
    with pytest.raises(ValueError, match="incompatible"):
        append_result(path, {})


def test_evaluation_cache_survives_reload(tmp_path):
    config = load_config("configs/demo_isotropic.yaml")
    evaluator = FixedEvaluator(config)
    checkpoint = tmp_path / "checkpoint.json"
    first = CachedObjective(evaluator, checkpoint, 10, 0, 1e-8)
    vector = np.array([2300.0, 0.3])
    score = first(vector)
    assert checkpoint.exists()
    assert len(first.cache) == 1
    second = CachedObjective(evaluator, checkpoint, 10, 0, 1e-8)
    assert second(vector) == pytest.approx(score)
    assert len(second.cache) == 1
    assert second.cache_hits == 1


def test_cache_rejects_changed_experiment_fingerprint(tmp_path):
    config = load_config("configs/demo_isotropic.yaml")
    evaluator = FixedEvaluator(config)
    checkpoint = tmp_path / "checkpoint.json"
    first = CachedObjective(evaluator, checkpoint, 10, 0, 1e-8, "config-a")
    first(np.array([2300.0, 0.3]))
    with pytest.raises(ValueError, match="different experiment configuration"):
        CachedObjective(evaluator, checkpoint, 10, 0, 1e-8, "config-b")


def test_data_fingerprint_changes_when_processed_curve_changes(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    curve = processed / "curve.csv"
    curve.write_text("x,y\n0,0\n1,1\n", encoding="utf-8")
    manifest = processed / "manifest.csv"
    manifest.write_text("file\ncurve.csv\n", encoding="utf-8")
    config = {
        "data": {
            "manifest": str(manifest),
            "processed_dir": str(processed),
        }
    }
    before = _data_content_fingerprint(config)
    curve.write_text("x,y\n0,0\n1,2\n", encoding="utf-8")
    after = _data_content_fingerprint(config)
    assert before != after
