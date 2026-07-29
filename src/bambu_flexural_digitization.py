from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "data/public/kontaxis-polymers-2025"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_published_flexural_220C"
)
ARTICLE_DOI = "10.3390/polym17070913"
ARTICLE_URL = "https://www.mdpi.com/2073-4360/17/7/913"
EXPECTED_SHA256 = {
    "article.xml": (
        "0bb0387b94ce35c56c5f656e3ee7967470f54da1bf19d2b27c5b4d3b5fb51857"
    ),
    "figure-4-flexural-curves.tif": (
        "b370646dba18476f287ffd3ddfe0cf1143c8b6964fda29ab89834615774a00f6"
    ),
}

# Exact axes of the checksum-pinned 1637x1335 Figure 4 raster.
PLOT_LEFT_PX = 208
PLOT_RIGHT_PX = 1577
PLOT_TOP_PX = 34
PLOT_BOTTOM_PX = 1145
PLOT_MAX_STRAIN = 0.07
PLOT_MAX_STRESS_MPA = 75.0

CURVES = {
    "0": {
        "name": "gray",
        "table_modulus_GPa": 3.03,
        "table_strength_MPa": 66.85,
    },
    "45": {
        "name": "red",
        "table_modulus_GPa": 3.03,
        "table_strength_MPa": 64.67,
    },
    "90": {
        "name": "blue",
        "table_modulus_GPa": 2.98,
        "table_strength_MPa": 57.69,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(source_dir: Path = DEFAULT_SOURCE) -> Dict[str, str]:
    observed = {}
    for filename, expected in EXPECTED_SHA256.items():
        path = Path(source_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"missing published source file: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{filename} checksum changed: expected {expected}, got {actual}"
            )
        observed[filename] = actual
    return observed


def _text(cell: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(cell.itertext())).strip()


def parse_table3(xml_path: Path) -> List[Dict[str, float]]:
    root = ET.parse(xml_path).getroot()
    table_wrap = root.find(
        ".//table-wrap[@id='polymers-17-00913-t003']"
    )
    if table_wrap is None:
        raise ValueError("published XML has no Table 3")
    rows: List[Dict[str, float]] = []
    temperature: float | None = None
    for tr in table_wrap.findall(".//tbody/tr"):
        values = [_text(cell) for cell in tr.findall("./td")]
        if len(values) == 6:
            temperature = float(re.sub(r"[^0-9.+-]", "", values.pop(0)))
        if len(values) != 5 or temperature is None:
            raise ValueError(f"unexpected Table 3 row: {values}")
        orientation = float(
            re.sub(r"[^0-9.+-]", "", values[0])
        )
        rows.append(
            {
                "nozzle_temperature_C": temperature,
                "raster_orientation_deg": orientation,
                "flexural_modulus_GPa": float(values[1]),
                "flexural_modulus_CV_percent": float(values[2]),
                "flexural_strength_MPa": float(values[3]),
                "flexural_strength_CV_percent": float(values[4]),
            }
        )
    if len(rows) != 12:
        raise ValueError(f"expected 12 Table 3 rows, found {len(rows)}")
    return rows


def _color_masks(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    red = rgb[:, :, 0].astype(int)
    green = rgb[:, :, 1].astype(int)
    blue = rgb[:, :, 2].astype(int)
    return {
        "red": (
            (red > 180)
            & (green < 170)
            & (blue < 170)
            & ((red - green) > 50)
        ),
        "blue": (
            (blue > 130)
            & (red < 100)
            & (green < 155)
            & ((blue - red) > 100)
        ),
        "gray": (
            (red >= 40)
            & (red < 140)
            & (np.abs(red - green) < 12)
            & (np.abs(red - blue) < 12)
        ),
    }


def _top_curve_group(y_pixels: np.ndarray) -> float:
    groups = np.split(
        y_pixels, np.where(np.diff(y_pixels) > 2)[0] + 1
    )
    group = min(groups, key=lambda values: float(np.median(values)))
    # The blue fracture line is nearly vertical. Retain the upper loading
    # branch rather than averaging it with the post-fracture drop.
    if int(group[-1] - group[0]) > 15:
        return float(group[0] + 2)
    return float(np.median(group))


def _largest_x_run(
    x_pixels: np.ndarray, y_pixels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cuts = np.where(np.diff(x_pixels) > 15)[0] + 1
    runs = np.split(np.arange(len(x_pixels)), cuts)
    run = max(runs, key=len)
    return x_pixels[run], y_pixels[run]


def digitize_figure4(
    image_path: Path,
) -> Dict[str, List[Dict[str, float]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "figure digitization requires the optional `digitization` dependencies"
        ) from exc
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    if rgb.shape != (1335, 1637, 3):
        raise ValueError(
            f"Figure 4 dimensions changed; got {rgb.shape}, expected (1335,1637,3)"
        )
    masks = _color_masks(rgb)
    output: Dict[str, List[Dict[str, float]]] = {}
    for orientation, definition in CURVES.items():
        mask = masks[str(definition["name"])]
        x_values: List[int] = []
        y_values: List[float] = []
        for x_pixel in range(PLOT_LEFT_PX, PLOT_RIGHT_PX + 1):
            # y>=120 excludes the cyan title box; y<901 excludes legend lines.
            candidates = np.flatnonzero(mask[120:901, x_pixel]) + 120
            if len(candidates) == 0:
                continue
            x_values.append(x_pixel)
            y_values.append(_top_curve_group(candidates))
        x_array, y_array = _largest_x_run(
            np.asarray(x_values, dtype=int),
            np.asarray(y_values, dtype=float),
        )
        strain = (
            (x_array - PLOT_LEFT_PX)
            / (PLOT_RIGHT_PX - PLOT_LEFT_PX)
            * PLOT_MAX_STRAIN
        )
        stress = (
            (PLOT_BOTTOM_PX - y_array)
            / (PLOT_BOTTOM_PX - PLOT_TOP_PX)
            * PLOT_MAX_STRESS_MPA
        )
        output[orientation] = [
            {
                "flexural_strain": float(current_strain),
                "flexural_stress_MPa": float(current_stress),
                "source_pixel_x": int(x_pixel),
                "source_pixel_y": float(y_pixel),
            }
            for current_strain, current_stress, x_pixel, y_pixel in zip(
                strain, stress, x_array, y_array
            )
        ]
    return output


def validate_digitization(
    curves: Mapping[str, Sequence[Mapping[str, float]]],
) -> List[Dict[str, object]]:
    rows = []
    for orientation, points in curves.items():
        strain = np.asarray(
            [point["flexural_strain"] for point in points], dtype=float
        )
        stress = np.asarray(
            [point["flexural_stress_MPa"] for point in points], dtype=float
        )
        slope_window = (strain >= 0.006) & (strain <= 0.012)
        if slope_window.sum() < 20:
            raise ValueError(
                f"{orientation} degree curve lacks a usable visible slope window"
            )
        digitized_modulus = float(
            np.polyfit(strain[slope_window], stress[slope_window], 1)[0]
            / 1000.0
        )
        digitized_strength = float(np.max(stress))
        expected = CURVES[orientation]
        modulus_error = (
            digitized_modulus - float(expected["table_modulus_GPa"])
        ) / float(expected["table_modulus_GPa"])
        strength_error = (
            digitized_strength - float(expected["table_strength_MPa"])
        ) / float(expected["table_strength_MPa"])
        rows.append(
            {
                "raster_orientation_deg": int(orientation),
                "digitized_visible_start_strain": float(np.min(strain)),
                "digitized_visible_end_strain": float(np.max(strain)),
                "digitized_slope_0p006_to_0p012_GPa": digitized_modulus,
                "table_flexural_modulus_GPa": expected[
                    "table_modulus_GPa"
                ],
                "modulus_relative_error": modulus_error,
                "digitized_peak_stress_MPa": digitized_strength,
                "table_flexural_strength_MPa": expected[
                    "table_strength_MPa"
                ],
                "strength_relative_error": strength_error,
                "passes_table_crosscheck": (
                    abs(modulus_error) <= 0.06
                    and abs(strength_error) <= 0.07
                ),
                "digitized_points": len(points),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def import_published_flexural(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "curves").mkdir(exist_ok=True)
    checksums = verify_sources(source_dir)
    table_rows = parse_table3(source_dir / "article.xml")
    _write_csv(output_dir / "table3_flexural_summary.csv", table_rows)
    curves = digitize_figure4(
        source_dir / "figure-4-flexural-curves.tif"
    )
    for orientation, points in curves.items():
        rows = [
            {
                "curve_id": f"kontaxis-flexural-{orientation}deg-220C",
                "material": "Bambu Lab PLA Basic",
                "nozzle_temperature_C": 220.0,
                "raster_orientation_deg": float(orientation),
                "evidence_class": "published_figure_digitization",
                **point,
            }
            for point in points
        ]
        _write_csv(
            output_dir / "curves" / f"flexural_{orientation}deg_220C.csv",
            rows,
        )
    validation = validate_digitization(curves)
    _write_csv(output_dir / "digitization_validation.csv", validation)
    all_pass = all(
        bool(row["passes_table_crosscheck"]) for row in validation
    )
    provenance = {
        "material": "Bambu Lab PLA Basic",
        "article": (
            "3D-Printed PLA Mechanical and Viscoelastic Behavior Dependence "
            "on the Nozzle Temperature and Printing Orientation"
        ),
        "doi": ARTICLE_DOI,
        "url": ARTICLE_URL,
        "license": "CC BY 4.0",
        "source_checksums_sha256": checksums,
        "figure": 4,
        "table": 3,
        "plot_axis_calibration": {
            "left_px": PLOT_LEFT_PX,
            "right_px": PLOT_RIGHT_PX,
            "top_px": PLOT_TOP_PX,
            "bottom_px": PLOT_BOTTOM_PX,
            "strain_range": [0.0, PLOT_MAX_STRAIN],
            "stress_range_MPa": [0.0, PLOT_MAX_STRESS_MPA],
            "one_pixel_strain": (
                PLOT_MAX_STRAIN
                / (PLOT_RIGHT_PX - PLOT_LEFT_PX)
            ),
            "one_pixel_stress_MPa": (
                PLOT_MAX_STRESS_MPA
                / (PLOT_BOTTOM_PX - PLOT_TOP_PX)
            ),
        },
        "quality_gate_passed": all_pass,
        "allowed_use": (
            "qualitative external curve-shape and initial-flexural-response "
            "validation at the published process condition"
        ),
        "prohibited_use": (
            "not raw specimen data; do not place in calibration/model-selection "
            "partitions or infer a fracture law"
        ),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Published Bambu PLA Basic flexural evidence: data-quality audit",
        "",
        "## Dataset and grain",
        "",
        "Table 3 contains 12 orientation/temperature summary rows. Figure 4 "
        "contains three plotted flexural stress–strain traces at 220 °C. "
        "The article reports five or more tested specimens per condition, but "
        "does not state whether each plotted trace is a mean, median, or a "
        "representative specimen.",
        "",
        "## Checks performed",
        "",
        "- SHA-256 pinning of the CC BY article XML and original TIFF.",
        "- Exact Table 3 row count, numeric types, and domain values.",
        "- Fixed pixel-to-axis calibration against the printed axes.",
        "- Color-separated curve extraction with title and legend exclusion.",
        "- Cross-check of visible-region slope and peak against Table 3.",
        "",
        "## Cross-check findings",
        "",
        "| Raster | Visible strain | Digitized slope (GPa) | Table modulus "
        "(GPa) | Digitized peak (MPa) | Table strength (MPa) | Gate |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in validation:
        report.append(
            f"| {row['raster_orientation_deg']}° | "
            f"{float(row['digitized_visible_start_strain']):.4f}–"
            f"{float(row['digitized_visible_end_strain']):.4f} | "
            f"{float(row['digitized_slope_0p006_to_0p012_GPa']):.3f} | "
            f"{float(row['table_flexural_modulus_GPa']):.3f} | "
            f"{float(row['digitized_peak_stress_MPa']):.2f} | "
            f"{float(row['table_flexural_strength_MPa']):.2f} | "
            f"{'pass' if row['passes_table_crosscheck'] else 'fail'} |"
        )
    report.extend(
        [
            "",
            "## Findings and analytical risk",
            "",
            "- **High severity, high confidence:** plotted-trace aggregation is "
            "undefined. These curves cannot be treated as raw replicates or "
            "used in calibration/model selection.",
            "- **Medium severity, high confidence:** the three colored traces "
            "overlap below roughly 0.5% strain, so the extractor retains only "
            "pixels that are visibly attributable to each trace and does not "
            "fabricate the hidden origin segment.",
            "- **Medium severity, high confidence:** 0°, 45°, and 90° are flat "
            "in-plane raster orientations, not build-Z coupons. They constrain "
            "external flexural behavior but not E3 or interlayer fracture.",
            "- Pixel resolution is approximately "
            f"{provenance['plot_axis_calibration']['one_pixel_strain']:.2e} "
            "strain and "
            f"{provenance['plot_axis_calibration']['one_pixel_stress_MPa']:.3f} "
            "MPa before line-thickness uncertainty.",
            "",
            "## Decision",
            "",
            (
                "All three digitized traces pass the summary-table cross-check. "
                "They are admitted only as qualitative external validation "
                "evidence. The missing raw-flexural-data requirement remains "
                "open."
                if all_pass
                else "The digitization failed its table cross-check and is not "
                "admitted for any simulation comparison."
            ),
            "",
        ]
    )
    (output_dir / "data_quality_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return {
        "quality_gate_passed": all_pass,
        "table_rows": len(table_rows),
        "curves": len(curves),
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import and quality-audit published Bambu flexural evidence"
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = import_published_flexural(
        Path(args.source), Path(args.output)
    )
    print(
        f"published flexural import: {result['table_rows']} summary rows, "
        f"{result['curves']} digitized curves, "
        f"quality_gate_passed={result['quality_gate_passed']}; "
        f"report: {(Path(args.output) / 'data_quality_report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
