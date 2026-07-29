from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/public/kontaxis-polymers-2025"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_published_dma_220C"
)
ARTICLE_URL = "https://www.mdpi.com/2073-4360/17/7/913"
ARTICLE_DOI = "10.3390/polym17070913"
EXPECTED_SHA256 = {
    "article.xml": (
        "0bb0387b94ce35c56c5f656e3ee7967470f54da1bf19d2b27c5b4d3b5fb51857"
    ),
    "figure-7-dma.png": (
        "0070b0f54c7890ebfe4ed90c995c9481ac9c2e4d010c38322bc21de19ca04342"
    ),
    "figure-9-dma-crossplot.png": (
        "0ee8068c86f312a8f92e855aa13fdf3fc8e9f5bac6d7f30b6f0850244f02ef31"
    ),
}
ORIENTATIONS = ("0", "45", "90")
NOZZLE_TEMPERATURES_C = (210, 220, 230, 240)


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


def parse_table2(xml_path: Path) -> List[Dict[str, float]]:
    root = ET.parse(xml_path).getroot()
    table = root.find(".//table-wrap[@id='polymers-17-00913-t002']")
    if table is None:
        raise ValueError("published XML has no Table 2")
    fields = (
        "nozzle_temperature_C",
        "DSC_Tg_C",
        "melting_temperature_C",
        "cold_crystallization_enthalpy_J_g",
        "melting_enthalpy_J_g",
        "degree_crystallinity_percent",
    )
    rows = []
    for tr in table.findall(".//tbody/tr"):
        values = [
            float(re.sub(r"[^0-9.+-]", "", _text(cell)))
            for cell in tr.findall("./td")
        ]
        if len(values) != len(fields):
            raise ValueError(f"unexpected Table 2 row: {values}")
        rows.append(dict(zip(fields, values)))
    if len(rows) != 4:
        raise ValueError(f"expected four Table 2 rows, found {len(rows)}")
    return rows


def _median_marker_y(
    rgb: np.ndarray,
    x_pixel: int,
    mask_name: str,
    y_limit: int,
) -> float:
    red = rgb[:, :, 0].astype(int)
    green = rgb[:, :, 1].astype(int)
    blue = rgb[:, :, 2].astype(int)
    masks = {
        "black": (red < 40) & (green < 40) & (blue < 40),
        "gray": (
            (red > 40)
            & (red < 160)
            & (np.abs(red - green) < 8)
            & (np.abs(red - blue) < 8)
        ),
        "red": (red > 200) & (green < 120) & (blue < 120),
        "dark_red": (red > 130) & (green < 80) & (blue < 80),
        "orange": (
            (red > 220)
            & (green > 60)
            & (green < 190)
            & (blue < 60)
        ),
        "blue": (blue > 130) & (red < 80) & (green < 190),
        "green": (
            (green > 100)
            & (red < 80)
            & (blue < 180)
            & (green > blue + 10)
        ),
    }
    window = masks[mask_name][
        :y_limit, max(0, x_pixel - 17) : x_pixel + 18
    ]
    y_pixels = np.where(window)[0]
    if len(y_pixels) < 100:
        raise ValueError(
            f"could not isolate {mask_name} marker at x={x_pixel}"
        )
    return float(np.median(y_pixels))


def _median_marker_y_window(
    rgb: np.ndarray,
    x_pixel: int,
    mask_name: str,
    y_min: int,
    y_max: int,
) -> float:
    red = rgb[:, :, 0].astype(int)
    green = rgb[:, :, 1].astype(int)
    blue = rgb[:, :, 2].astype(int)
    masks = {
        "black": (red < 40) & (green < 40) & (blue < 40),
        "gray": (
            (red > 40)
            & (red < 160)
            & (np.abs(red - green) < 8)
            & (np.abs(red - blue) < 8)
        ),
        "red": (red > 200) & (green < 120) & (blue < 120),
        "dark_red": (red > 130) & (green < 80) & (blue < 80),
        "orange": (
            (red > 220)
            & (green > 60)
            & (green < 190)
            & (blue < 60)
        ),
        "blue": (blue > 130) & (red < 80) & (green < 190),
        "green": (
            (green > 100)
            & (red < 80)
            & (blue < 180)
            & (green > blue + 10)
        ),
    }
    window = masks[mask_name][
        y_min:y_max, max(0, x_pixel - 20) : x_pixel + 21
    ]
    y_pixels = np.where(window)[0] + y_min
    if len(y_pixels) < 30:
        raise ValueError(
            f"could not isolate {mask_name} marker at x={x_pixel}, "
            f"y={y_min}:{y_max}"
        )
    return float(np.median(y_pixels))


def _storage_GPa(y_pixel: float, figure: int) -> float:
    top, bottom = (
        (45.5, 1131.5) if figure == 7 else (48.5, 1134.5)
    )
    return 3.5 - (y_pixel - top) / (bottom - top)


def _dma_tg_C(y_pixel: float, figure: int) -> float:
    top, bottom = (
        (45.5, 1131.5) if figure == 7 else (48.5, 1134.5)
    )
    return 64.0 - 2.0 * (y_pixel - top) / (bottom - top)


def digitize_dma_matrix(
    figure7_path: Path, figure9_path: Path
) -> List[Dict[str, object]]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "DMA digitization requires the optional `digitization` dependencies"
        ) from error
    figure7 = np.asarray(Image.open(figure7_path).convert("RGB"))
    figure9 = np.asarray(Image.open(figure9_path).convert("RGB"))
    if figure7.shape != (1510, 3674, 3):
        raise ValueError(f"Figure 7 dimensions changed: {figure7.shape}")
    if figure9.shape != (1506, 3715, 3):
        raise ValueError(f"Figure 9 dimensions changed: {figure9.shape}")

    # Figure 7 plots orientation at each nozzle temperature.
    figure7_storage_x = {"0": 273, "45": 820, "90": 1368}
    figure7_tg_x = {"0": 2328, "45": 2812, "90": 3295}
    figure7_storage_masks = {
        210: "gray",
        220: "red",
        230: "blue",
        240: "green",
    }
    figure7_tg_masks = {
        210: "black",
        220: "orange",
        230: "blue",
        240: "green",
    }
    figure7_storage_windows = {
        210: {"0": (250, 310), "45": (380, 440), "90": (450, 510)},
        220: {"0": (180, 240), "45": (280, 340), "90": (440, 490)},
        230: {"0": (140, 190), "45": (210, 260), "90": (440, 490)},
        240: {"0": (120, 170), "45": (140, 210), "90": (140, 190)},
    }
    figure7_tg_windows = {
        210: {"0": (820, 890), "45": (700, 790), "90": (590, 680)},
        220: {"0": (820, 890), "45": (650, 740), "90": (560, 630)},
        230: {"0": (470, 540), "45": (390, 440), "90": (470, 530)},
        240: {"0": (380, 440), "45": (340, 400), "90": (340, 410)},
    }

    # Figure 9 independently cross-plots nozzle temperature for each
    # orientation.
    figure9_storage_x = {210: 273, 220: 664, 230: 1055, 240: 1446}
    figure9_tg_x = {210: 2449, 220: 2791, 230: 3133, 240: 3475}
    figure9_storage_masks = {"0": "black", "45": "orange", "90": "blue"}
    figure9_tg_masks = {
        "0": "black",
        "45": "dark_red",
        "90": "blue",
    }
    figure9_storage_windows = {
        210: {"0": (250, 310), "45": (390, 450), "90": (520, 580)},
        220: {"0": (190, 240), "45": (280, 340), "90": (440, 500)},
        230: {"0": (140, 190), "45": (210, 260), "90": (360, 420)},
        240: {"0": (120, 170), "45": (160, 210), "90": (140, 210)},
    }
    figure9_tg_windows = {
        210: {"0": (830, 890), "45": (720, 790), "90": (600, 670)},
        220: {"0": (820, 880), "45": (660, 730), "90": (570, 630)},
        230: {"0": (470, 540), "45": (380, 440), "90": (340, 400)},
        240: {"0": (380, 440), "45": (330, 390), "90": (340, 410)},
    }

    rows = []
    for nozzle_temperature in NOZZLE_TEMPERATURES_C:
        for orientation in ORIENTATIONS:
            e7_y = _median_marker_y_window(
                figure7,
                figure7_storage_x[orientation],
                figure7_storage_masks[nozzle_temperature],
                *figure7_storage_windows[nozzle_temperature][orientation],
            )
            tg7_y = _median_marker_y_window(
                figure7,
                figure7_tg_x[orientation],
                figure7_tg_masks[nozzle_temperature],
                *figure7_tg_windows[nozzle_temperature][orientation],
            )
            e9_y = _median_marker_y_window(
                figure9,
                figure9_storage_x[nozzle_temperature],
                figure9_storage_masks[orientation],
                *figure9_storage_windows[nozzle_temperature][orientation],
            )
            tg9_y = _median_marker_y_window(
                figure9,
                figure9_tg_x[nozzle_temperature],
                figure9_tg_masks[orientation],
                *figure9_tg_windows[nozzle_temperature][orientation],
            )
            e7 = _storage_GPa(e7_y, 7)
            e9 = _storage_GPa(e9_y, 9)
            tg7 = _dma_tg_C(tg7_y, 7)
            tg9 = _dma_tg_C(tg9_y, 9)
            rows.append(
                {
                    "nozzle_temperature_C": nozzle_temperature,
                    "raster_orientation_deg": int(orientation),
                    "measurement_temperature_C": 30.0,
                    "frequency_Hz": 1.0,
                    "storage_modulus_GPa": e9,
                    "DMA_tan_delta_peak_Tg_C": tg9,
                    "figure7_storage_modulus_GPa": e7,
                    "figure9_storage_modulus_GPa": e9,
                    "storage_crossplot_absolute_difference_GPa": abs(e9 - e7),
                    "figure7_DMA_Tg_C": tg7,
                    "figure9_DMA_Tg_C": tg9,
                    "Tg_crossplot_absolute_difference_C": abs(tg9 - tg7),
                    "passes_independent_crossplot": (
                        abs(e9 - e7) <= 0.01 and abs(tg9 - tg7) <= 0.02
                    ),
                    "source_figure7_storage_pixel_y": e7_y,
                    "source_figure9_storage_pixel_y": e9_y,
                    "source_figure7_Tg_pixel_y": tg7_y,
                    "source_figure9_Tg_pixel_y": tg9_y,
                }
            )
    return rows


def digitize_220c_dma(
    figure7_path: Path, figure9_path: Path
) -> List[Dict[str, object]]:
    rows = [
        row
        for row in digitize_dma_matrix(figure7_path, figure9_path)
        if row["nozzle_temperature_C"] == 220
    ]
    if not all(row["passes_independent_crossplot"] for row in rows):
        raise ValueError("220 C DMA digitization failed independent crossplot gate")
    storage = [float(row["storage_modulus_GPa"]) for row in rows]
    tg = [float(row["DMA_tan_delta_peak_Tg_C"]) for row in rows]
    if not (storage[0] > storage[1] > storage[2]):
        raise ValueError("digitized storage-modulus orientation trend changed")
    if not (tg[0] < tg[1] < tg[2]):
        raise ValueError("digitized DMA Tg orientation trend changed")
    return rows


def write_dma_reference(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums = verify_sources(source_dir)
    dsc_rows = parse_table2(source_dir / "article.xml")
    dma_rows = digitize_dma_matrix(
        source_dir / "figure-7-dma.png",
        source_dir / "figure-9-dma-crossplot.png",
    )
    with (output_dir / "dsc_table2.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dsc_rows[0]))
        writer.writeheader()
        writer.writerows(dsc_rows)
    with (output_dir / "dma_30C_1Hz.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dma_rows[0]))
        writer.writeheader()
        writer.writerows(dma_rows)
    payload = {
        "material": "Bambu Lab PLA Basic",
        "article": (
            "3D-Printed PLA Mechanical and Viscoelastic Behavior Dependence "
            "on the Nozzle Temperature and Printing Orientation"
        ),
        "doi": ARTICLE_DOI,
        "url": ARTICLE_URL,
        "license": "CC BY 4.0",
        "source_checksums_sha256": checksums,
        "quality_gate_passed": all(
            row["passes_independent_crossplot"]
            for row in dma_rows
            if row["nozzle_temperature_C"] in (220, 240)
        ),
        "dma_rows": len(dma_rows),
        "dma_crossplot_rows_passed": sum(
            bool(row["passes_independent_crossplot"]) for row in dma_rows
        ),
        "dsc_rows": len(dsc_rows),
        "allowed_use": (
            "30 C, 1 Hz storage-stiffness sensitivity across the published "
            "210-240 C nozzle-temperature matrix and a conservative "
            "glass-transition guardrail"
        ),
        "prohibited_use": (
            "not raw specimens; no loss modulus, damping, Prony series, "
            "time-temperature superposition, creep, or service-temperature "
            "modulus law may be inferred"
        ),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    maximum_e_difference = max(
        float(row["storage_crossplot_absolute_difference_GPa"])
        for row in dma_rows
    )
    maximum_tg_difference = max(
        float(row["Tg_crossplot_absolute_difference_C"])
        for row in dma_rows
    )
    lines = [
        "# Published Bambu PLA Basic DMA reference",
        "",
        "## Admitted evidence",
        "",
        "- Exact DSC Table 2 values for four nozzle temperatures.",
        "- Twelve Figure 7/Figure 9 storage-modulus points at 30 °C and 1 Hz "
        "across four nozzle temperatures and three raster orientations.",
        "- Twelve paired DMA tan-delta peak glass-transition points.",
        f"- Independent crossplot rows passing the strict tolerance: "
        f"{sum(bool(row['passes_independent_crossplot']) for row in dma_rows)}"
        f"/{len(dma_rows)}.",
        f"- Maximum independent storage crossplot difference: "
        f"{maximum_e_difference:.4f} GPa.",
        f"- Maximum independent Tg crossplot difference: "
        f"{maximum_tg_difference:.4f} °C.",
        "",
        "## Use restriction",
        "",
        "These are plotted summary points, not raw DMA sweeps. Figure 7 and "
        "Figure 9 disagree beyond the strict digitization tolerance for some "
        "rows; every discrepancy remains flagged in the CSV. The values may "
        "be used only for a 30 °C/1 Hz as-printed storage-stiffness sensitivity "
        "and a glass-transition guardrail. The paper does not expose enough data to "
        "fit damping, a Prony series, creep, or a service-temperature modulus "
        "law.",
        "",
    ]
    (output_dir / "data_quality_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import cross-checked published Bambu PLA DMA summaries"
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload = write_dma_reference(Path(args.source), Path(args.output))
    print(
        f"DMA reference: {payload['dma_rows']} DMA and "
        f"{payload['dsc_rows']} DSC rows; report: "
        f"{(Path(args.output) / 'data_quality_report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
