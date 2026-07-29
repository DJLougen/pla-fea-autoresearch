from pathlib import Path

import pytest

from src.bambu_pvh_reference import (
    DEFAULT_SOURCE,
    digitize_compression,
    digitize_tensile_sample_3,
    import_reference,
    validate_digitization,
    verify_source,
)


def test_source_assets_and_digitization_are_quality_gated(
    tmp_path: Path,
) -> None:
    evidence = verify_source(DEFAULT_SOURCE)
    assert set(evidence) == {
        "supplement.pdf",
        "figure-s3-tensile.png",
        "figure-s5-compression.png",
    }
    tensile = digitize_tensile_sample_3(
        DEFAULT_SOURCE / "figure-s3-tensile.png"
    )
    compression = digitize_compression(
        DEFAULT_SOURCE / "figure-s5-compression.png"
    )
    validation = validate_digitization(tensile, compression)
    assert validation["quality_gate_passed"] is True
    assert validation[
        "tensile_sample_3_origin_slope_MPa"
    ] == pytest.approx(2140.0, rel=0.05)
    assert abs(
        validation["simulation_to_experiment_energy_to_0p45_error"]
    ) < 0.10


def test_import_formally_excludes_unreproducible_exact_replay(
    tmp_path: Path,
) -> None:
    result = import_reference(DEFAULT_SOURCE, tmp_path)
    assert result["quality_gate_passed"] is True
    assert result["exact_specimen_replay_allowed"] is False
    assert (tmp_path / "exact_replay_gate.csv").exists()
    assert (tmp_path / "curves/compression_experiment.csv").exists()
    assert (tmp_path / "reported_optimum_validation.csv").exists()
