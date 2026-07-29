from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


PARTITIONS = ("calibration", "validation", "heldout")


def load_manifest(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"processed manifest not found: {path}; run python -m src.prepare first"
        )
    manifest = pd.read_csv(path, dtype={"direction": str})
    required = {"test_id", "partition", "test_type", "direction", "file"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    unknown = set(manifest["partition"]) - set(PARTITIONS)
    if unknown:
        raise ValueError(f"unknown partitions: {sorted(unknown)}")
    if manifest["test_id"].duplicated().any():
        raise ValueError("test_id values must be unique")
    for partition in PARTITIONS:
        if not (manifest["partition"] == partition).any():
            raise ValueError(f"manifest has no {partition} specimens")
    return manifest.sort_values(["partition", "test_id"]).reset_index(drop=True)


def partition_manifest(manifest: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        partition: manifest.loc[manifest["partition"] == partition].copy()
        for partition in PARTITIONS
    }


def assert_heldout_not_calibration(manifest: pd.DataFrame) -> None:
    calibration = set(
        manifest.loc[manifest["partition"] == "calibration", "test_id"]
    )
    heldout = set(manifest.loc[manifest["partition"] == "heldout", "test_id"])
    overlap = calibration & heldout
    if overlap:
        raise ValueError(f"held-out tests leak into calibration: {sorted(overlap)}")

