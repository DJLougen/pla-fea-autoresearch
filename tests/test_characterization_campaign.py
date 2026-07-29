from pathlib import Path
import json

from src.characterization_campaign import campaign_rows, write_campaign


def test_campaign_is_deterministic_and_partitioned_before_measurement():
    first = campaign_rows()
    second = campaign_rows()
    assert first == second
    assert len(first) == 43
    assert sum(row["partition"] == "calibration" for row in first) == 24
    assert sum(row["partition"] == "validation" for row in first) == 16
    assert sum(row["partition"] == "heldout" for row in first) == 3
    assert sorted(row["randomized_print_order"] for row in first) == list(
        range(1, 44)
    )


def test_campaign_writes_matching_cad_and_capture_templates(tmp_path: Path):
    rows = write_campaign(tmp_path)
    assert (tmp_path / "cad/tensile_bar_50x10x4.scad").exists()
    heldout = (tmp_path / "cad/heldout_l_bracket.scad").read_text()
    assert "square([60, 10])" in heldout
    captures = list((tmp_path / "capture_templates").glob("*.csv"))
    assert len(captures) == len(rows)
    header = captures[0].read_text().splitlines()[0]
    assert "transverse_strain" in header
    assert "material_lot" in header
    assert "failure_mode" in header
    manifest = json.loads(
        (tmp_path / "campaign_manifest.json").read_text()
    )
    assert manifest["specimens"] == 43
    assert manifest["partition_counts"] == {
        "calibration": 24,
        "validation": 16,
        "heldout": 3,
    }
