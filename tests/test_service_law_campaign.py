import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.service_law_campaign import (
    CAPTURE_FIELDS,
    validate_submission,
    write_campaign,
)


def _fill_common(frame: pd.DataFrame) -> pd.DataFrame:
    frame["material_lot"] = "lot-a"
    frame["material_color"] = "jade-white"
    frame["machine_id"] = "x1c-01"
    frame["print_profile"] = "service-law-v1"
    frame["drying_protocol"] = "8 h at 45 C"
    frame["ambient_temperature_C"] = 23.0
    frame["ambient_relative_humidity_percent"] = 20.0
    return frame


def test_campaign_is_frozen_randomized_and_metadata_only(tmp_path: Path):
    manifest = write_campaign(tmp_path)
    assert manifest["specimen_count"] == 90
    assert manifest["stage_counts"] == {
        "1": 6,
        "2": 18,
        "3": 12,
        "4": 30,
        "5": 24,
    }
    assert len(list((tmp_path / "capture_templates").glob("*.csv"))) == 90
    with (tmp_path / "randomized_run_order.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        order = list(csv.DictReader(handle))
    assert sorted(int(row["global_randomized_order"]) for row in order) == list(
        range(1, 91)
    )
    frozen = tmp_path / "frozen_plan.csv"
    written = json.loads((tmp_path / "campaign_manifest.json").read_text())
    assert written["frozen_plan_sha256"]
    assert written["capture_fields"] == list(CAPTURE_FIELDS)
    assert frozen.exists()


def test_blank_templates_release_no_laws(tmp_path: Path):
    write_campaign(tmp_path)
    payload = validate_submission(
        tmp_path / "capture_templates",
        tmp_path / "frozen_plan.csv",
        tmp_path / "campaign_manifest.json",
    )
    assert payload["status"] == "awaiting_measurements"
    assert payload["counts"] == {
        "valid": 0,
        "invalid": 0,
        "awaiting_measurements": 90,
        "missing": 0,
    }
    assert payload["all_laws_released"] is False
    assert all(
        item["law_released"] is False
        for item in payload["stage_status"].values()
    )


def test_complete_dma_stage_can_release_without_unlocking_other_laws(
    tmp_path: Path,
):
    write_campaign(tmp_path)
    with (tmp_path / "frozen_plan.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        plan = list(csv.DictReader(handle))
    for row in plan:
        if int(row["stage"]) != 1:
            continue
        path = tmp_path / "capture_templates" / f"{row['test_id']}.csv"
        frame = _fill_common(pd.read_csv(path))
        temperature = frame["test_temperature_C"].to_numpy(float)
        frequency = frame["frequency_Hz"].to_numpy(float)
        storage = 2400.0 - 12.0 * (temperature - 20.0) + 4.0 * np.log10(
            frequency
        )
        loss = 0.035 * storage
        frame["storage_modulus_MPa"] = storage
        frame["loss_modulus_MPa"] = loss
        frame["tan_delta"] = loss / storage
        frame.to_csv(path, index=False)
    payload = validate_submission(
        tmp_path / "capture_templates",
        tmp_path / "frozen_plan.csv",
        tmp_path / "campaign_manifest.json",
    )
    assert payload["status"] == "awaiting_measurements"
    assert payload["stage_status"]["1"]["status"] == "ready_for_calibration"
    assert payload["stage_status"]["1"]["law_released"] is True
    assert payload["counts"]["valid"] == 6


def test_fatigue_runout_remains_a_valid_censored_observation(
    tmp_path: Path,
):
    write_campaign(tmp_path)
    with (tmp_path / "frozen_plan.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        plan = list(csv.DictReader(handle))
    expected = next(row for row in plan if int(row["stage"]) == 4)
    path = tmp_path / "capture_templates" / f"{expected['test_id']}.csv"
    frame = _fill_common(pd.read_csv(path))
    fraction = float(
        expected["maximum_stress_fraction_of_directional_UTS"]
    )
    uts = 40.0
    frame["directional_UTS_MPa"] = uts
    frame["maximum_stress_MPa"] = fraction * uts
    frame["minimum_stress_MPa"] = 0.1 * fraction * uts
    frame["secant_stiffness_MPa"] = np.linspace(2400.0, 2200.0, len(frame))
    frame["hysteresis_energy_MJ_m3"] = np.linspace(0.01, 0.04, len(frame))
    frame["outcome"] = "runout"
    frame["failure_mode"] = "none"
    frame.to_csv(path, index=False)
    payload = validate_submission(
        tmp_path / "capture_templates",
        tmp_path / "frozen_plan.csv",
        tmp_path / "campaign_manifest.json",
    )
    result = next(
        row for row in payload["specimens"] if row["test_id"] == expected["test_id"]
    )
    assert result["status"] == "valid"
    assert result["usable_measurement_points"] == 6
    assert payload["stage_status"]["4"]["law_released"] is False


def test_measured_but_tampered_material_is_invalid(tmp_path: Path):
    write_campaign(tmp_path)
    path = next((tmp_path / "capture_templates").glob("DMA-*.csv"))
    frame = _fill_common(pd.read_csv(path))
    frame["material"] = "generic PLA"
    frame["storage_modulus_MPa"] = 2000.0
    frame["loss_modulus_MPa"] = 70.0
    frame["tan_delta"] = 0.035
    frame.to_csv(path, index=False)
    payload = validate_submission(
        tmp_path / "capture_templates",
        tmp_path / "frozen_plan.csv",
        tmp_path / "campaign_manifest.json",
    )
    result = next(
        row for row in payload["specimens"] if row["source_file"] == str(path.resolve())
    )
    assert result["status"] == "invalid"
    assert any("generic PLA" in issue for issue in result["issues"])
