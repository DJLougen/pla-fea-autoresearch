from pathlib import Path

import numpy as np
import pandas as pd

from src.campaign_ingestion import validate_campaign_submission
from src.characterization_campaign import CAPTURE_FIELDS, write_campaign


def _filled_tensile(path: Path, partition: str = "calibration") -> None:
    points = 20
    strain = np.linspace(0.0, 0.004, points)
    shared = {
        "test_id": "bambu-tensile-1-r01",
        "partition": partition,
        "test_type": "tensile",
        "print_direction": "1",
        "specimen_width_mm": 10.02,
        "specimen_thickness_mm": 3.98,
        "gauge_length_mm": 50.0,
        "specimen_mass_g": 2.48,
        "layer_height_mm": 0.08,
        "nozzle_temperature_C": 220.0,
        "print_speed_mm_s": 50.0,
        "infill_percent": 100.0,
        "material_name": "Bambu PLA Basic",
        "material_lot": "lot-a",
        "material_color": "black",
        "machine_id": "x1c-01",
        "slicer_profile": "campaign-v1",
        "drying_protocol": "8h at 45C",
    }
    rows = []
    for index, current in enumerate(strain):
        rows.append(
            {
                **shared,
                "time_s": float(index),
                "engineering_strain": current,
                "engineering_stress_MPa": 2500.0 * current,
                "transverse_strain": -0.32 * current,
            }
        )
    pd.DataFrame(rows, columns=CAPTURE_FIELDS).to_csv(
        path, index=False
    )


def test_metadata_only_template_is_rejected(tmp_path: Path):
    campaign = tmp_path / "campaign"
    write_campaign(campaign)
    payload = validate_campaign_submission(
        campaign / "capture_templates",
        campaign / "test_plan.csv",
        campaign / "campaign_manifest.json",
    )
    assert payload["counts"]["valid"] == 0
    assert payload["counts"]["invalid"] == 43


def test_valid_partial_file_passes_but_campaign_remains_incomplete(
    tmp_path: Path,
):
    campaign = tmp_path / "campaign"
    submission = tmp_path / "submission"
    submission.mkdir()
    write_campaign(campaign)
    _filled_tensile(submission / "bambu-tensile-1-r01.csv")
    payload = validate_campaign_submission(
        submission,
        campaign / "test_plan.csv",
        campaign / "campaign_manifest.json",
    )
    assert payload["counts"] == {
        "valid": 1,
        "invalid": 0,
        "missing": 42,
    }
    assert not payload["ready_for_fixed_preprocessing"]


def test_partition_tampering_is_rejected(tmp_path: Path):
    campaign = tmp_path / "campaign"
    submission = tmp_path / "submission"
    submission.mkdir()
    write_campaign(campaign)
    _filled_tensile(
        submission / "bambu-tensile-1-r01.csv",
        partition="heldout",
    )
    payload = validate_campaign_submission(
        submission,
        campaign / "test_plan.csv",
        campaign / "campaign_manifest.json",
    )
    row = next(
        row
        for row in payload["specimens"]
        if row["specimen_id"] == "bambu-tensile-1-r01"
    )
    assert row["status"] == "invalid"
    assert any("frozen value" in issue for issue in row["issues"])
