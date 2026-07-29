#!/usr/bin/env python3
"""Download and extract raw Bambu PLA Basic tensile curves from Mendeley Data.

The source workbook contains one five-column block per specimen.  This importer
streams the OOXML sheet so the 97 MB worksheet XML does not need to be loaded
into memory, preserves the literal source labels, and writes one CSV per block.
It deliberately does not infer local material axes or calibration partitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import urllib.request
import zipfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET


DATASET_DOI = "10.17632/8jxw5533py.2"
ARTICLE_DOI = "10.1016/j.dib.2026.113015"
LICENSE = "CC BY 4.0"
DATASET_URL = "https://data.mendeley.com/datasets/8jxw5533py/2"
RAW_URL = (
    "https://data.mendeley.com/public-files/datasets/8jxw5533py/files/"
    "33090214-2017-492c-b464-a30cdbbff2fa/file_downloaded"
)
RAW_SHA256 = "0e9ef3c7ae638cb3544c119f21531a7f963f91ffc490976bd21ae41bb80724cc"
SUMMARY_URL = (
    "https://data.mendeley.com/public-files/datasets/8jxw5533py/files/"
    "8d5a41d1-1c63-454d-b27d-bdf8b5132d04/file_downloaded"
)
SUMMARY_SHA256 = "bc59ed28723893f111c2cf5e3cd8f70c8f51ea627f28a232501cd13557c8fbcd"

ARTICLE_CLAIMED_CURVE_COUNT = 21
EXPECTED_FIELDS = ("Force", "Displacement", "Deformation", "Stress", "Strain")
SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = f"{{{SPREADSHEET_NS}}}"

PRINT_METADATA = {
    "filament": "Bambu PLA Basic, 1.75 mm",
    "printer": "Bambu X1C",
    "slicer": "Bambu Studio",
    "layer_height_mm": 0.08,
    "raster_width_mm": 0.42,
    "infill_percent": 100,
    "infill_pattern": "rectilinear",
    "nozzle_diameter_mm": 0.4,
    "nozzle_temperature_C": 220,
    "infill_speed_mm_s": 350,
    "tensile_loading_speed_mm_min": 20,
    "test_standard": "ASTM D638-22",
}

CSV_FIELDS = [
    "test_id",
    "partition",
    "test_type",
    "print_direction",
    "source_specimen_label",
    "source_platform_axis",
    "source_placement",
    "force_N",
    "crosshead_displacement_mm",
    "extensometer_deformation_mm",
    "engineering_stress_MPa",
    "engineering_strain",
    *PRINT_METADATA,
    "source_dataset_doi",
    "source_article_doi",
    "source_license",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(url: str, expected_sha256: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or sha256(path) != expected_sha256:
        temporary = path.with_suffix(path.suffix + ".download")
        request = urllib.request.Request(url, headers={"User-Agent": "pla-fea-autoresearch"})
        with urllib.request.urlopen(request, timeout=180) as response:
            with temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        temporary.replace(path)
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"source checksum mismatch for {path.name}: {actual}")


def column_index(reference: str) -> int:
    match = re.match(r"[A-Z]+", reference)
    if not match:
        raise ValueError(f"invalid cell reference: {reference}")
    result = 0
    for character in match.group(0):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def shared_strings(archive: zipfile.ZipFile) -> List[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(text or "" for text in item.itertext())
        for item in root.findall(f"{NS}si")
    ]


def cell_value(cell: ET.Element, strings: List[str]) -> object:
    if cell.attrib.get("t") == "inlineStr":
        inline = cell.find(f"{NS}is")
        return "" if inline is None else "".join(inline.itertext())
    value_node = cell.find(f"{NS}v")
    if value_node is None or value_node.text is None:
        return None
    if cell.attrib.get("t") == "s":
        return strings[int(value_node.text)]
    try:
        return float(value_node.text)
    except ValueError:
        return value_node.text


def read_small_sheet(path: Path) -> Dict[int, Dict[int, object]]:
    with zipfile.ZipFile(path) as archive:
        strings = shared_strings(archive)
        rows: Dict[int, Dict[int, object]] = {}
        with archive.open("xl/worksheets/sheet1.xml") as sheet:
            for _, element in ET.iterparse(sheet, events=("end",)):
                if element.tag != f"{NS}row":
                    continue
                row_number = int(element.attrib["r"])
                row = rows.setdefault(row_number, {})
                for cell in element.findall(f"{NS}c"):
                    row[column_index(cell.attrib["r"])] = cell_value(cell, strings)
                element.clear()
    return rows


def normalize_summary_label(label: str) -> str:
    return label.replace("45-", "45Deg-")


def summary_uts(path: Path) -> Dict[str, float]:
    rows = read_small_sheet(path)
    result: Dict[str, float] = {}
    for header_row in (1, 4, 7, 10, 13, 16, 19):
        header = rows.get(header_row, {})
        strengths = rows.get(header_row + 1, {})
        for column in (1, 2, 3):
            label = header.get(column)
            value = strengths.get(column)
            if isinstance(label, str) and isinstance(value, (int, float)):
                result[normalize_summary_label(label)] = float(value)
    return result


def parse_specimen_label(label: str) -> tuple[str, str, int]:
    match = re.fullmatch(r"(45Deg|X|Y|Z)-(F|O|U)-(\d+)", label)
    if not match:
        raise ValueError(f"unexpected source specimen label: {label}")
    axis, placement_code, replicate = match.groups()
    placement = {"F": "flat", "O": "on-edge", "U": "upright"}[placement_code]
    return axis, placement, int(replicate)


def slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def specimen_metadata(label: str) -> Dict[str, object]:
    axis, placement, _ = parse_specimen_label(label)
    return {
        "test_id": f"mendeley8jxw5533py_{slug(label)}",
        "partition": "unassigned",
        "test_type": "tensile",
        # X/Y/45/Z and flat/on-edge/upright are build-placement labels.  They
        # are not a verified mapping to this repository's local axes 1/2/3.
        "print_direction": "",
        "source_specimen_label": label,
        "source_platform_axis": axis,
        "source_placement": placement,
        **PRINT_METADATA,
        "source_dataset_doi": DATASET_DOI,
        "source_article_doi": ARTICLE_DOI,
        "source_license": LICENSE,
    }


def iter_rows(
    archive: zipfile.ZipFile, strings: List[str]
) -> Iterable[tuple[int, Dict[int, object]]]:
    with archive.open("xl/worksheets/sheet1.xml") as sheet:
        for _, element in ET.iterparse(sheet, events=("end",)):
            if element.tag != f"{NS}row":
                continue
            row_number = int(element.attrib["r"])
            values = {
                column_index(cell.attrib["r"]): cell_value(cell, strings)
                for cell in element.findall(f"{NS}c")
            }
            yield row_number, values
            element.clear()


def extract(raw_path: Path, summary_path: Path, output: Path) -> Path:
    curves_dir = output / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    summary_values = summary_uts(summary_path)
    manifest_records: List[Dict[str, object]] = []

    with zipfile.ZipFile(raw_path) as archive:
        strings = shared_strings(archive)
        rows = iter_rows(archive, strings)
        row_number, first_row = next(rows)
        if row_number != 1:
            raise ValueError("raw workbook is missing header row 1")
        labels = {
            column // 5: value
            for column, value in first_row.items()
            if column % 5 == 0 and isinstance(value, str) and value
        }
        row_number, second_row = next(rows)
        if row_number != 2:
            raise ValueError("raw workbook is missing field row 2")
        for block in labels:
            actual = tuple(second_row.get(block * 5 + offset) for offset in range(5))
            if actual != EXPECTED_FIELDS:
                raise ValueError(f"{labels[block]} has unexpected fields: {actual}")

        group_counts = Counter(
            (parse_specimen_label(label)[0], parse_specimen_label(label)[1])
            for label in labels.values()
        )
        if len(labels) != 22:
            raise ValueError(f"expected 22 literal workbook blocks, found {len(labels)}")

        with ExitStack() as stack:
            writers: Dict[int, csv.DictWriter] = {}
            stats: Dict[int, Dict[str, object]] = {}
            for block, label in labels.items():
                metadata = specimen_metadata(label)
                curve_path = curves_dir / f"{metadata['test_id']}.csv"
                handle = stack.enter_context(
                    curve_path.open("w", encoding="utf-8", newline="")
                )
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writers[block] = writer
                stats[block] = {
                    "file": f"curves/{curve_path.name}",
                    "label": label,
                    "metadata": metadata,
                    "points": 0,
                    "peak_stress_MPa": float("-inf"),
                    "max_strain": float("-inf"),
                }

            for row_number, values in rows:
                if row_number <= 2:
                    continue
                for block, label in labels.items():
                    start = block * 5
                    raw_values = [values.get(start + offset) for offset in range(5)]
                    force, displacement, deformation, stress, strain = raw_values
                    if stress is None and strain is None:
                        continue
                    if not isinstance(stress, (int, float)) or not isinstance(
                        strain, (int, float)
                    ):
                        raise ValueError(
                            f"{label} row {row_number}: nonnumeric stress/strain"
                        )
                    record = {
                        **specimen_metadata(label),
                        "force_N": force,
                        "crosshead_displacement_mm": displacement,
                        "extensometer_deformation_mm": deformation,
                        "engineering_stress_MPa": stress,
                        "engineering_strain": strain,
                    }
                    writers[block].writerow(record)
                    stat = stats[block]
                    stat["points"] = int(stat["points"]) + 1
                    stat["peak_stress_MPa"] = max(
                        float(stat["peak_stress_MPa"]), float(stress)
                    )
                    stat["max_strain"] = max(float(stat["max_strain"]), float(strain))

        for block, label in labels.items():
            stat = stats[block]
            axis, placement, replicate = parse_specimen_label(label)
            group_count = group_counts[(axis, placement)]
            reported_uts: Optional[float] = summary_values.get(label)
            if reported_uts is None:
                relative_error: object = ""
            else:
                relative_error = abs(float(stat["peak_stress_MPa"]) - reported_uts) / max(
                    abs(reported_uts), 1.0e-12
                )
            flags = ["workbook_has_22_curves_but_article_claims_21"]
            if group_count != 3:
                flags.append(f"orientation_group_has_{group_count}_curves_not_3")
            if reported_uts is None:
                flags.append("no_same_label_entry_in_summary_workbook")
            elif float(relative_error) > 0.02:
                flags.append("raw_peak_differs_from_same_label_summary_uts_gt_2pct")
            manifest_records.append(
                {
                    "file": stat["file"],
                    "test_id": stat["metadata"]["test_id"],
                    "source_specimen_label": label,
                    "source_platform_axis": axis,
                    "source_placement": placement,
                    "source_replicate_label": replicate,
                    "orientation_group_curve_count": group_count,
                    "points": stat["points"],
                    "extracted_peak_stress_MPa": stat["peak_stress_MPa"],
                    "extracted_max_strain": stat["max_strain"],
                    "same_label_summary_UTS_MPa": (
                        "" if reported_uts is None else reported_uts
                    ),
                    "peak_relative_error_vs_same_label_summary": relative_error,
                    "data_quality_flags": ";".join(flags),
                    "source_dataset_doi": DATASET_DOI,
                    "source_article_doi": ARTICLE_DOI,
                    "source_dataset_url": DATASET_URL,
                    "source_raw_url": RAW_URL,
                    "source_raw_sha256": RAW_SHA256,
                    "source_summary_url": SUMMARY_URL,
                    "source_summary_sha256": SUMMARY_SHA256,
                    "license": LICENSE,
                    "article_claimed_curve_count": ARTICLE_CLAIMED_CURVE_COUNT,
                    "literal_workbook_curve_count": len(labels),
                    "calibration_ready": False,
                    "calibration_blocker": (
                        "Source block labels/count conflict with the article and "
                        "summary workbook; local material-axis mapping, blinded "
                        "partition, flexural curves, and held-out component data "
                        "remain unresolved."
                    ),
                }
            )

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_records[0]))
        writer.writeheader()
        writer.writerows(manifest_records)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/public/mendeley-8jxw5533py-v2",
        help="destination directory",
    )
    args = parser.parse_args()
    output = Path(args.output)
    source_dir = output / "source"
    raw_path = source_dir / "3-2_Original_data_of_material_tests.xlsx"
    summary_path = source_dir / "3-1_Material_property.xlsx"
    download_source(RAW_URL, RAW_SHA256, raw_path)
    download_source(SUMMARY_URL, SUMMARY_SHA256, summary_path)
    manifest = extract(raw_path, summary_path, output)
    with manifest.open(encoding="utf-8") as handle:
        count = sum(1 for _ in handle) - 1
    print(
        f"extracted {count} literal Bambu PLA Basic tensile curve blocks "
        f"to {output.resolve()}"
    )


if __name__ == "__main__":
    main()
