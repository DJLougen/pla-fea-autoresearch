#!/usr/bin/env python3
"""Reproducibly extract public PLA tensile curves embedded in a Zenodo XLSX.

The workbook stores stress-strain measurements as literal scatter-chart caches,
not worksheet cells. This importer reads those OOXML chart caches without
altering the source workbook and writes one CSV per measured specimen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET


SOURCE_URL = (
    "https://zenodo.org/records/16935914/files/"
    "Brick%20Layering%20Experiment.xlsx?download=1"
)
SOURCE_SHA256 = "d6d048d1642c5766abb8e77d8b36d482691c4b06438a528b301fa7581c8bf6e7"
SOURCE_DOI = "10.5281/zenodo.16935914"
LICENSE = "CC BY 4.0"
CONFIGS_WITH_CURVES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14)

NS = {
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or sha256(path) != SOURCE_SHA256:
        temporary = path.with_suffix(".download")
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            temporary.write_bytes(response.read())
        temporary.replace(path)
    actual = sha256(path)
    if actual != SOURCE_SHA256:
        raise ValueError(f"source checksum mismatch: {actual}")


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference).group(0)
    result = 0
    for character in letters:
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _shared_strings(archive: zipfile.ZipFile) -> List[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.findall(".//s:t", NS))
        for item in root.findall("s:si", NS)
    ]


def _sheet_paths(archive: zipfile.ZipFile) -> Dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        relation.attrib["Id"]: relation.attrib["Target"].lstrip("/")
        for relation in relationships.findall("pr:Relationship", NS)
    }
    paths = {}
    for sheet in workbook.findall("s:sheets/s:sheet", NS):
        target = targets[sheet.attrib[f"{{{NS['r']}}}id"]]
        if not target.startswith("xl/"):
            target = "xl/" + target
        paths[sheet.attrib["name"]] = target
    return paths


def _sheet_rows(
    archive: zipfile.ZipFile, path: str, shared: List[str]
) -> List[List[object]]:
    root = ET.fromstring(archive.read(path))
    sparse = {}
    maximum_column = -1
    for cell in root.findall(".//s:sheetData/s:row/s:c", NS):
        reference = cell.attrib["r"]
        row_number = int(re.search(r"\d+", reference).group(0))
        column = _column_index(reference)
        maximum_column = max(maximum_column, column)
        value_node = cell.find("s:v", NS)
        if cell.attrib.get("t") == "inlineStr":
            value = "".join(
                node.text or "" for node in cell.findall(".//s:is/s:t", NS)
            )
        elif value_node is None:
            value = None
        elif cell.attrib.get("t") == "s":
            value = shared[int(value_node.text)]
        else:
            try:
                value = float(value_node.text)
            except (TypeError, ValueError):
                value = value_node.text
        sparse[(row_number, column)] = value
    maximum_row = max((row for row, _ in sparse), default=0)
    return [
        [sparse.get((row, column)) for column in range(maximum_column + 1)]
        for row in range(1, maximum_row + 1)
    ]


def _numeric_points(parent: ET.Element, axis: str) -> List[float]:
    points = []
    for point in parent.findall(f"c:{axis}/c:numLit/c:pt", NS):
        value = point.find("c:v", NS)
        if value is not None:
            points.append((int(point.attrib["idx"]), float(value.text)))
    return [value for _, value in sorted(points)]


def _chart_curves(
    archive: zipfile.ZipFile, chart_number: int
) -> Tuple[int, List[Tuple[str, List[float], List[float]]]]:
    root = ET.fromstring(archive.read(f"xl/charts/chart{chart_number}.xml"))
    title = "".join(node.text or "" for node in root.findall(".//c:title//a:t", NS))
    match = re.search(r"Stress\s+vs\s+Strain\s+(\d+)", title, re.IGNORECASE)
    if not match:
        raise ValueError(f"chart {chart_number} has unexpected title {title!r}")
    config = int(match.group(1))
    curves = []
    for series in root.findall(".//c:scatterChart/c:ser", NS):
        name_node = series.find("c:tx/c:v", NS)
        name = name_node.text if name_node is not None else "unnamed"
        strain = _numeric_points(series, "xVal")
        stress = _numeric_points(series, "yVal")
        if len(strain) != len(stress) or len(strain) < 3:
            raise ValueError(
                f"{name}: mismatched or insufficient chart points "
                f"({len(strain)} strain, {len(stress)} stress)"
            )
        curves.append((name, strain, stress))
    return config, curves


def _configuration_metadata(rows: List[List[object]]) -> Dict[str, dict]:
    result = {}
    for row in rows:
        if row and isinstance(row[0], str) and re.fullmatch(r"C\d+", row[0]):
            result[row[0]] = {
                "layering_type": row[1],
                "material_extrusion_multiplier": row[2],
                "source_printing_orientation": row[3],
                "nozzle_temperature_C": row[4],
                "source_status": row[5],
            }
    return result


def _specimen_metadata(rows: List[List[object]]) -> Dict[Tuple[str, int], dict]:
    result = {}
    for row in rows[1:]:
        if not row or not isinstance(row[0], str) or not re.fullmatch(r"C\d+", row[0]):
            continue
        sample = int(row[1])
        result[(row[0], sample)] = {
            "specimen_height_mm": row[2],
            "specimen_width_mm": row[3],
            "specimen_thickness_mm": row[4],
            "reported_area_mm2": row[5],
            "reported_UTS_MPa": row[9],
            "reported_maximum_strain": row[10],
            # The source labels this MPa, but values around 3-4 indicate GPa.
            # Preserve it as reported and do not use it for calibration.
            "reported_modulus_source_value": row[11],
        }
    return result


def extract(source: Path, output: Path) -> Path:
    curves_dir = output / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    with zipfile.ZipFile(source) as archive:
        shared = _shared_strings(archive)
        paths = _sheet_paths(archive)
        configurations = _configuration_metadata(
            _sheet_rows(archive, paths["1. Overview"], shared)
        )
        specimens = _specimen_metadata(
            _sheet_rows(archive, paths["2. Experiment Results"], shared)
        )
        chart_names = sorted(
            (
                name
                for name in archive.namelist()
                if re.fullmatch(r"xl/charts/chart\d+\.xml", name)
            ),
            key=lambda name: int(re.search(r"\d+", name).group(0)),
        )
        extracted_configs = []
        for chart_name in chart_names:
            number = int(re.search(r"\d+", chart_name).group(0))
            try:
                config_number, curves = _chart_curves(archive, number)
            except ValueError:
                continue
            if config_number not in CONFIGS_WITH_CURVES:
                continue
            extracted_configs.append(config_number)
            config_id = f"C{config_number}"
            config_metadata = configurations[config_id]
            for sample, (curve_name, strain, stress) in enumerate(curves, start=1):
                specimen = specimens[(config_id, sample)]
                test_id = f"zenodo16935914_{config_id}_S{sample}"
                filename = f"{test_id}.csv"
                path = curves_dir / filename
                fields = [
                    "test_id",
                    "partition",
                    "test_type",
                    "print_direction",
                    "source_printing_orientation",
                    "source_layering_type",
                    "material_extrusion_multiplier",
                    "nozzle_temperature_C",
                    "specimen_width_mm",
                    "specimen_thickness_mm",
                    "engineering_strain",
                    "engineering_stress_MPa",
                    "source_curve_name",
                    "source_doi",
                    "source_license",
                ]
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for x_value, y_value in zip(strain, stress):
                        writer.writerow(
                            {
                                "test_id": test_id,
                                # Deliberately invalid for the calibration loader
                                # until a researcher makes a blinded split.
                                "partition": "unassigned",
                                "test_type": "tensile",
                                # Source X/Y/Z is a build placement label, not a
                                # verified local material-axis mapping.
                                "print_direction": "",
                                "source_printing_orientation": config_metadata[
                                    "source_printing_orientation"
                                ],
                                "source_layering_type": config_metadata[
                                    "layering_type"
                                ],
                                "material_extrusion_multiplier": config_metadata[
                                    "material_extrusion_multiplier"
                                ],
                                "nozzle_temperature_C": config_metadata[
                                    "nozzle_temperature_C"
                                ],
                                "specimen_width_mm": specimen["specimen_width_mm"],
                                "specimen_thickness_mm": specimen[
                                    "specimen_thickness_mm"
                                ],
                                "engineering_strain": x_value,
                                "engineering_stress_MPa": y_value,
                                "source_curve_name": curve_name,
                                "source_doi": SOURCE_DOI,
                                "source_license": LICENSE,
                            }
                        )
                peak = max(stress)
                reported_peak = float(specimen["reported_UTS_MPa"])
                manifest_rows.append(
                    {
                        "file": f"curves/{filename}",
                        "test_id": test_id,
                        "configuration": config_id,
                        "sample": sample,
                        "points": len(strain),
                        **config_metadata,
                        **specimen,
                        "extracted_peak_stress_MPa": peak,
                        "peak_relative_error_vs_reported": abs(peak - reported_peak)
                        / max(abs(reported_peak), 1.0e-12),
                        "source_doi": SOURCE_DOI,
                        "source_url": SOURCE_URL,
                        "source_sha256": SOURCE_SHA256,
                        "license": LICENSE,
                        "calibration_ready": False,
                        "calibration_blocker": (
                            "Not Bambu PLA Basic; local material-axis mapping, "
                            "blinded partition, flexural data, and held-out component "
                            "data are not provided by this source."
                        ),
                    }
                )
        if sorted(extracted_configs) != list(CONFIGS_WITH_CURVES):
            raise ValueError(f"unexpected extracted configurations: {extracted_configs}")
    manifest = output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    worst_peak_error = max(
        float(row["peak_relative_error_vs_reported"]) for row in manifest_rows
    )
    if worst_peak_error > 0.02:
        raise ValueError(
            f"chart extraction failed UTS reconciliation: worst error {worst_peak_error:.2%}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/public/zenodo-16935914",
        help="destination directory",
    )
    args = parser.parse_args()
    output = Path(args.output)
    source = output / "source" / "Brick_Layering_Experiment.xlsx"
    download_source(source)
    manifest = extract(source, output)
    with manifest.open(encoding="utf-8") as handle:
        count = sum(1 for _ in handle) - 1
    print(f"extracted {count} measured PLA tensile curves to {output.resolve()}")


if __name__ == "__main__":
    main()

