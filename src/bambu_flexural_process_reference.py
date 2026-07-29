from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

from .bambu_flexural_digitization import parse_table3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/public/kontaxis-polymers-2025"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_published_flexural_process"
)
ARTICLE_URL = "https://www.mdpi.com/2073-4360/17/7/913"
ARTICLE_DOI = "10.3390/polym17070913"
EXPECTED_SHA256 = {
    "article.xml": (
        "0bb0387b94ce35c56c5f656e3ee7967470f54da1bf19d2b27c5b4d3b5fb51857"
    ),
    "figure-5-flexural-process.png": (
        "3dab27c024242015686756ca3dc1f101fb16ff080353b5b83294f7c882a7d095"
    ),
    "figure-6-flexural-crossplot.png": (
        "fcc4a7b133b16f08179fc2f46c37d89f210f26bb525ba49b4d4c30d5fca06677"
    ),
}
TEMPERATURES = (210, 220, 230, 240)
ORIENTATIONS = (0, 45, 90)

# Fixed marker locations selected from the checksum-pinned images.
FIGURE5_X = {
    "modulus": {0: 283, 45: 832, 90: 1380},
    "strength": {0: 2351, 45: 2899, 90: 3447},
}
FIGURE6_X = {
    "modulus": {210: 283, 220: 674, 230: 1066, 240: 1458},
    "strength": {210: 2311, 220: 2703, 230: 3094, 240: 3485},
}
FIGURE5_Y = {
    "modulus": {
        210: {0: 691, 45: 693, 90: 809},
        220: {0: 577, 45: 582, 90: 636},
        230: {0: 557, 45: 582, 90: 618},
        240: {0: 629, 45: 605, 90: 625},
    },
    "strength": {
        210: {0: 249, 45: 523, 90: 1002},
        220: {0: 232, 45: 352, 90: 738},
        230: {0: 180, 45: 291, 90: 550},
        240: {0: 204, 45: 252, 90: 486},
    },
}
FIGURE6_Y = {
    "modulus": {
        210: {0: 670, 45: 677, 90: 793},
        220: {0: 561, 45: 565, 90: 616},
        230: {0: 541, 45: 562, 90: 607},
        240: {0: 619, 45: 589, 90: 614},
    },
    "strength": {
        210: {0: 227, 45: 507, 90: 988},
        220: {0: 213, 45: 336, 90: 722},
        230: {0: 162, 45: 272, 90: 534},
        240: {0: 189, 45: 239, 90: 473},
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


def _color_masks(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    red = rgb[:, :, 0].astype(int)
    green = rgb[:, :, 1].astype(int)
    blue = rgb[:, :, 2].astype(int)
    return {
        "black": (red < 40) & (green < 40) & (blue < 40),
        "gray": (
            (red > 40)
            & (red < 160)
            & (np.abs(red - green) < 8)
            & (np.abs(red - blue) < 8)
        ),
        "red": (red > 200) & (green < 130) & (blue < 130),
        "orange": (
            (red > 220)
            & (green > 50)
            & (green < 180)
            & (blue < 80)
        ),
        "blue": (blue > 130) & (red < 80) & (green < 190),
        "green": (
            (green > 100)
            & (red < 80)
            & (blue < 180)
            & (green > blue + 10)
        ),
    }


def _marker_y(
    mask: np.ndarray,
    x_pixel: int,
    center_y: int,
    half_height: int,
) -> float:
    y_min = center_y - half_height
    y_max = center_y + half_height + 1
    pixels = np.where(
        mask[y_min:y_max, x_pixel - 20 : x_pixel + 21]
    )[0] + y_min
    if len(pixels) < 30:
        raise ValueError(
            f"could not isolate marker at x={x_pixel}, y≈{center_y}"
        )
    return float(np.median(pixels))


def _modulus_GPa(y_pixel: float, figure: int) -> float:
    top, bottom = ((56.0, 1167.0) if figure == 5 else (40.0, 1151.0))
    return 3.5 - (y_pixel - top) / (bottom - top)


def _strength_MPa(y_pixel: float, figure: int) -> float:
    top, bottom = ((55.0, 1167.0) if figure == 5 else (39.0, 1151.0))
    return 70.0 - 20.0 * (y_pixel - top) / (bottom - top)


def digitize_process_crossplots(
    figure5_path: Path,
    figure6_path: Path,
    table_rows: List[Mapping[str, float]],
) -> List[Dict[str, object]]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "flexural process digitization requires Pillow"
        ) from error
    figure5 = np.asarray(Image.open(figure5_path).convert("RGB"))
    figure6 = np.asarray(Image.open(figure6_path).convert("RGB"))
    if figure5.shape != (1542, 3686, 3):
        raise ValueError(f"Figure 5 dimensions changed: {figure5.shape}")
    if figure6.shape != (1528, 3609, 3):
        raise ValueError(f"Figure 6 dimensions changed: {figure6.shape}")
    masks5 = _color_masks(figure5)
    masks6 = _color_masks(figure6)
    temperature_masks = {
        210: "gray",
        220: "red",
        230: "blue",
        240: "green",
    }
    orientation_masks = {0: "black", 45: "orange", 90: "blue"}
    table = {
        (
            int(row["nozzle_temperature_C"]),
            int(row["raster_orientation_deg"]),
        ): row
        for row in table_rows
    }
    rows: List[Dict[str, object]] = []
    for temperature in TEMPERATURES:
        for orientation in ORIENTATIONS:
            modulus5_y = _marker_y(
                masks5[temperature_masks[temperature]],
                FIGURE5_X["modulus"][orientation],
                FIGURE5_Y["modulus"][temperature][orientation],
                24,
            )
            strength5_y = _marker_y(
                masks5[temperature_masks[temperature]],
                FIGURE5_X["strength"][orientation],
                FIGURE5_Y["strength"][temperature][orientation],
                28,
            )
            modulus6_y = _marker_y(
                masks6[orientation_masks[orientation]],
                FIGURE6_X["modulus"][temperature],
                FIGURE6_Y["modulus"][temperature][orientation],
                24,
            )
            strength6_y = _marker_y(
                masks6[orientation_masks[orientation]],
                FIGURE6_X["strength"][temperature],
                FIGURE6_Y["strength"][temperature][orientation],
                28,
            )
            modulus5 = _modulus_GPa(modulus5_y, 5)
            modulus6 = _modulus_GPa(modulus6_y, 6)
            strength5 = _strength_MPa(strength5_y, 5)
            strength6 = _strength_MPa(strength6_y, 6)
            exact = table[(temperature, orientation)]
            table_modulus = float(exact["flexural_modulus_GPa"])
            table_strength = float(exact["flexural_strength_MPa"])
            row = {
                **dict(exact),
                "figure5_flexural_modulus_GPa": modulus5,
                "figure6_flexural_modulus_GPa": modulus6,
                "figure5_figure6_modulus_difference_GPa": abs(
                    modulus5 - modulus6
                ),
                "figure5_table_modulus_error_GPa": abs(
                    modulus5 - table_modulus
                ),
                "figure6_table_modulus_error_GPa": abs(
                    modulus6 - table_modulus
                ),
                "figure5_flexural_strength_MPa": strength5,
                "figure6_flexural_strength_MPa": strength6,
                "figure5_figure6_strength_difference_MPa": abs(
                    strength5 - strength6
                ),
                "figure5_table_strength_error_MPa": abs(
                    strength5 - table_strength
                ),
                "figure6_table_strength_error_MPa": abs(
                    strength6 - table_strength
                ),
                "source_figure5_modulus_pixel_y": modulus5_y,
                "source_figure6_modulus_pixel_y": modulus6_y,
                "source_figure5_strength_pixel_y": strength5_y,
                "source_figure6_strength_pixel_y": strength6_y,
            }
            row["passes_crossplot_and_table_gate"] = (
                row["figure5_figure6_modulus_difference_GPa"] <= 0.01
                and row["figure5_figure6_strength_difference_MPa"] <= 0.15
                and row["figure5_table_modulus_error_GPa"] <= 0.02
                and row["figure6_table_modulus_error_GPa"] <= 0.02
                and row["figure5_table_strength_error_MPa"] <= 0.2
                and row["figure6_table_strength_error_MPa"] <= 0.2
            )
            rows.append(row)
    return rows


def write_flexural_process_reference(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums = verify_sources(source_dir)
    table_rows = parse_table3(source_dir / "article.xml")
    rows = digitize_process_crossplots(
        source_dir / "figure-5-flexural-process.png",
        source_dir / "figure-6-flexural-crossplot.png",
        table_rows,
    )
    fields = list(rows[0])
    with (output_dir / "flexural_process_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "flexural_process_matrix.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    passed = sum(bool(row["passes_crossplot_and_table_gate"]) for row in rows)
    payload = {
        "material": "Bambu Lab PLA Basic",
        "doi": ARTICLE_DOI,
        "url": ARTICLE_URL,
        "license": "CC BY 4.0",
        "source_checksums_sha256": checksums,
        "rows": len(rows),
        "quality_rows_passed": passed,
        "quality_gate_passed": passed == len(rows),
        "authoritative_values": "exact XML Table 3 means and CV percentages",
        "figure_use": (
            "Figures 5 and 6 are independent layout crosschecks only"
        ),
        "allowed_use": (
            "as-printed nozzle-temperature/raster-orientation flexural "
            "modulus and strength sensitivity for the published process"
        ),
        "prohibited_use": (
            "not raw curves; not tensile, compression, shear, fracture, "
            "service-temperature, fatigue, or creep calibration"
        ),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    max_modulus_crossplot = max(
        float(row["figure5_figure6_modulus_difference_GPa"])
        for row in rows
    )
    max_strength_crossplot = max(
        float(row["figure5_figure6_strength_difference_MPa"])
        for row in rows
    )
    lines = [
        "# Published Bambu PLA Basic flexural process matrix",
        "",
        f"All {passed}/{len(rows)} exact Table 3 rows pass the independent "
        "Figure 5/Figure 6 consistency gate.",
        "",
        f"- Maximum figure-to-figure modulus difference: "
        f"{max_modulus_crossplot:.4f} GPa.",
        f"- Maximum figure-to-figure strength difference: "
        f"{max_strength_crossplot:.3f} MPa.",
        "- Exact table means and coefficients of variation remain the "
        "authoritative values.",
        "",
        "## Scope",
        "",
        "The matrix covers four nozzle temperatures and three in-plane raster "
        "orientations for the paper's three-point-bending process. It supports "
        "a process-specific flexural sensitivity, not a general failure or "
        "service-temperature law. Raw specimen curves are unavailable.",
        "",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Bambu PLA flexural process matrix"
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload = write_flexural_process_reference(
        Path(args.source), Path(args.output)
    )
    print(
        f"flexural process reference: "
        f"{payload['quality_rows_passed']}/{payload['rows']} passed; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
