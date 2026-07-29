from pathlib import Path

from src.config import load_config
from src.evaluate import FixedEvaluator
from src.materials import initial_vector
from src.prepare import prepare_directory
from src.synthetic import TRUTH, generate
from src.validation import load_manifest


def test_synthetic_prepare_and_fixed_partitions(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    generate(raw)
    manifest = prepare_directory(raw, processed)
    assert manifest.groupby("partition").size().to_dict() == {
        "calibration": 7,
        "heldout": 1,
        "validation": 7,
    }
    loaded = load_manifest(processed / "manifest.csv")
    assert len(loaded) == 15


def test_mock_evaluator_reports_per_test_metrics(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    generate(raw)
    prepare_directory(raw, processed)
    config = load_config("configs/demo_isotropic.yaml")
    config["data"]["processed_dir"] = str(processed)
    config["data"]["manifest"] = str(processed / "manifest.csv")
    evaluator = FixedEvaluator(config)
    vector = initial_vector("isotropic")
    calibration = evaluator.evaluate_vector(vector, "calibration")
    validation = evaluator.evaluate_vector(vector, "validation")
    heldout = evaluator.evaluate_vector(vector, "heldout")
    assert len(calibration.per_test) == 7
    assert len(validation.per_test) == 7
    assert len(heldout.per_test) == 1
    assert calibration.score >= 0


def test_lateral_curves_make_true_poisson_values_score_better(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    generate(raw)
    prepare_directory(raw, processed)
    config = load_config("configs/demo_orthotropic.yaml")
    config["data"]["processed_dir"] = str(processed)
    config["data"]["manifest"] = str(processed / "manifest.csv")
    evaluator = FixedEvaluator(config)
    truth = evaluator.evaluate_constants(TRUTH, "calibration", "truth")
    wrong = dict(TRUTH)
    wrong.update({"nu12": 0.10, "nu13": 0.10, "nu23": 0.10})
    wrong_result = evaluator.evaluate_constants(
        wrong, "calibration", "wrong-poisson"
    )
    assert truth.score < wrong_result.score
    assert (
        sum(
            metric["lateral_strain_used"]
            for metric in truth.per_test.values()
        )
        == 6
    )
