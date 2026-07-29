from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/public/techrxiv-common-infill-2025"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_common_infill_compression_2025"
)
FIGURE_FILENAME = "figure-4-density-comparison.png"
FIGURE_SHA256 = (
    "16d6946d17660321dfc322b4489eb7c64537f9e9b0881764679201fabb64756f"
)
ARTICLE_TITLE = (
    "Compressive Performance of Common Infill Patterns for Fused "
    "Filament Fabrication"
)
TECHRXIV_DOI = "10.36227/techrxiv.173609864.46510948/v1"
RESEARCHGATE_DOI = "10.13140/RG.2.2.12377.43360"
ARTICLE_URL = f"https://doi.org/{TECHRXIV_DOI}"
PDF_URL = (
    "https://d197for5662m48.cloudfront.net/documents/publicationstatus/"
    "239460/preprint_pdf/4e2859e168e0d386f200523c71a35d0b.pdf"
)

PLOT_BOXES = {
    10: (57, 268),
    20: (335, 546),
    30: (613, 824),
}
PLOT_TOP_PX = 34
PLOT_BOTTOM_PX = 245
PLOT_MAX_STRAIN = 0.75
PLOT_MAX_STRESS_PA = 42_500.0
PATTERNS = ("grid", "gyroid", "cubic")
PLA_DENSITY_G_CM3 = 1.24
SPECIMEN_VOLUME_CM3 = 3.0 ** 3

MASS_ROWS = (
    ("grid", 10, 8.1162, 0.0174),
    ("grid", 20, 10.4049, 0.0045),
    ("grid", 30, 13.2457, 0.0297),
    ("gyroid", 10, 7.7805, 0.0904),
    ("gyroid", 20, 10.3857, 0.0174),
    ("gyroid", 30, 12.8249, 0.0150),
    ("cubic", 10, 7.7483, 0.0091),
    ("cubic", 20, 10.4397, 0.0306),
    ("cubic", 30, 13.0415, 0.0291),
)
REPORTED_YIELD_MPA = {
    (10, "grid"): 11.87,
    (10, "gyroid"): 7.85,
    (10, "cubic"): 7.61,
    (20, "grid"): 18.70,
    (20, "gyroid"): 11.62,
    (20, "cubic"): 11.21,
    (30, "grid"): 26.06,
    (30, "gyroid"): 14.08,
    (30, "cubic"): 14.32,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source_dir: Path = DEFAULT_SOURCE) -> Dict[str, object]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "figure digitization requires Pillow"
        ) from exc
    path = Path(source_dir) / FIGURE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing published figure: {path}")
    observed = _sha256(path)
    if observed != FIGURE_SHA256:
        raise ValueError(
            f"{FIGURE_FILENAME} checksum changed: expected "
            f"{FIGURE_SHA256}, got {observed}"
        )
    shape = np.asarray(Image.open(path).convert("RGB")).shape
    if shape != (292, 850, 3):
        raise ValueError(
            f"figure dimensions changed; expected (292, 850, 3), got {shape}"
        )
    return {"sha256": observed, "shape": list(shape)}


def _color_masks(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    red = rgb[:, :, 0].astype(int)
    green = rgb[:, :, 1].astype(int)
    blue = rgb[:, :, 2].astype(int)
    return {
        "grid": (
            (blue > 140)
            & ((blue - red) > 60)
            & ((blue - green) > 60)
        ),
        "gyroid": (
            (red > 160)
            & ((red - green) > 70)
            & ((red - blue) > 70)
            & (green < 100)
        ),
        "cubic": (
            (red > 160)
            & (green > 50)
            & (green < 190)
            & (blue < 100)
            & ((red - green) > 50)
        ),
    }


def digitize_figure(
    image_path: Path,
) -> Dict[Tuple[int, str], List[Dict[str, float]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "figure digitization requires Pillow"
        ) from exc
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    if rgb.shape != (292, 850, 3):
        raise ValueError(
            f"figure dimensions changed; got {rgb.shape}"
        )
    masks = _color_masks(rgb)
    curves: Dict[Tuple[int, str], List[Dict[str, float]]] = {}
    for nominal_density, (left, right) in PLOT_BOXES.items():
        for pattern in PATTERNS:
            x_pixels: List[int] = []
            y_pixels: List[float] = []
            for x_pixel in range(left, right + 1):
                candidates = np.flatnonzero(
                    masks[pattern][
                        PLOT_TOP_PX:PLOT_BOTTOM_PX + 1,
                        x_pixel,
                    ]
                ) + PLOT_TOP_PX
                # Each panel repeats the same legend. Use panel-local x so
                # legend removal is identical in all three panels.
                if x_pixel - left <= 48:
                    candidates = candidates[candidates > 85]
                if len(candidates) == 0:
                    continue
                x_pixels.append(x_pixel)
                y_pixels.append(float(np.median(candidates)))
            strain = (
                (np.asarray(x_pixels, dtype=float) - left)
                / (right - left)
                * PLOT_MAX_STRAIN
            )
            stress = (
                (PLOT_BOTTOM_PX - np.asarray(y_pixels, dtype=float))
                / (PLOT_BOTTOM_PX - PLOT_TOP_PX)
                * PLOT_MAX_STRESS_PA
            )
            curves[(nominal_density, pattern)] = [
                {
                    "compressive_strain": float(current_strain),
                    "compressive_stress_Pa": float(current_stress),
                    "source_pixel_x": int(x_pixel),
                    "source_pixel_y": float(y_pixel),
                }
                for (
                    current_strain,
                    current_stress,
                    x_pixel,
                    y_pixel,
                ) in zip(strain, stress, x_pixels, y_pixels)
            ]
    return curves


def _origin_constrained_slope(
    strain: np.ndarray, stress: np.ndarray
) -> float:
    window = (strain >= 0.01) & (strain <= 0.04)
    if int(np.sum(window)) < 5:
        raise ValueError("curve lacks five points in the initial-slope window")
    denominator = float(np.dot(strain[window], strain[window]))
    return float(np.dot(strain[window], stress[window]) / denominator)


def _integral_to(
    strain: np.ndarray, stress: np.ndarray, endpoint: float
) -> float:
    x = np.asarray(strain, dtype=float)
    y = np.asarray(stress, dtype=float)
    if x[0] > 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
    if endpoint < x[-1] and not np.any(np.isclose(x, endpoint)):
        y_endpoint = float(np.interp(endpoint, x, y))
        insertion = int(np.searchsorted(x, endpoint))
        x = np.insert(x, insertion, endpoint)
        y = np.insert(y, insertion, y_endpoint)
    selected = x <= endpoint
    return float(np.trapezoid(y[selected], x[selected]))


def validate_curves(
    curves: Mapping[Tuple[int, str], Sequence[Mapping[str, float]]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for nominal_density in sorted(PLOT_BOXES):
        left, right = PLOT_BOXES[nominal_density]
        for pattern in PATTERNS:
            points = curves[(nominal_density, pattern)]
            strain = np.asarray(
                [point["compressive_strain"] for point in points],
                dtype=float,
            )
            stress = np.asarray(
                [point["compressive_stress_Pa"] for point in points],
                dtype=float,
            )
            pixels = np.asarray(
                [point["source_pixel_x"] for point in points], dtype=int
            )
            coverage = len(np.unique(pixels)) / (right - left + 1)
            max_gap = (
                int(np.max(np.diff(pixels))) if len(pixels) > 1 else 999
            )
            peak_window = strain <= 0.12
            plateau_window = (strain >= 0.25) & (strain <= 0.55)
            first_peak = float(np.max(stress[peak_window]))
            reported_yield = REPORTED_YIELD_MPA[
                (nominal_density, pattern)
            ]
            x1000_peak_MPa = first_peak / 1000.0
            yield_error = x1000_peak_MPa / reported_yield - 1.0
            gate = (
                coverage >= 0.90
                and max_gap <= 5
                and len(points) >= 190
                and bool(np.all(np.diff(strain) > 0.0))
                and bool(np.all(stress >= 0.0))
                and int(np.sum(plateau_window)) >= 50
                and abs(yield_error) <= 0.04
            )
            rows.append(
                {
                    "pattern": pattern,
                    "nominal_infill_percent": nominal_density,
                    "digitized_points": len(points),
                    "horizontal_pixel_coverage": coverage,
                    "maximum_horizontal_gap_px": max_gap,
                    "visible_start_strain": float(np.min(strain)),
                    "visible_end_strain": float(np.max(strain)),
                    "initial_effective_modulus_Pa": (
                        _origin_constrained_slope(strain, stress)
                    ),
                    "first_peak_stress_to_0p12_as_labeled_Pa": first_peak,
                    "reported_yield_strength_MPa": reported_yield,
                    "first_peak_x1000_unit_sensitivity_MPa": (
                        x1000_peak_MPa
                    ),
                    "x1000_peak_to_reported_yield_relative_error": (
                        yield_error
                    ),
                    "plateau_mean_stress_0p25_to_0p55_Pa": float(
                        np.mean(stress[plateau_window])
                    ),
                    "energy_density_to_0p70_J_m3": _integral_to(
                        strain, stress, 0.70
                    ),
                    "maximum_visible_stress_Pa": float(np.max(stress)),
                    "quality_gate_passed": gate,
                }
            )
    return rows


def mass_records() -> List[Dict[str, object]]:
    solid_mass = PLA_DENSITY_G_CM3 * SPECIMEN_VOLUME_CM3
    return [
        {
            "material": "Bambu Lab PLA Basic, purple",
            "pattern": pattern,
            "nominal_infill_percent": nominal_density,
            "mean_mass_g": mean_mass,
            "mass_standard_deviation_g": standard_deviation,
            "replicates": 3,
            "inferred_total_solid_fraction": mean_mass / solid_mass,
        }
        for (
            pattern,
            nominal_density,
            mean_mass,
            standard_deviation,
        ) in MASS_ROWS
    ]


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def import_reference(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_dir = output_dir / "curves"
    curve_dir.mkdir(exist_ok=True)
    source_check = verify_source(source_dir)
    curves = digitize_figure(source_dir / FIGURE_FILENAME)
    validation = validate_curves(curves)
    masses = mass_records()
    _write_csv(output_dir / "reported_mass_summary.csv", masses)
    _write_csv(output_dir / "curve_validation.csv", validation)
    for (nominal_density, pattern), points in curves.items():
        rows = [
            {
                "curve_id": (
                    f"woody-common-infill-{pattern}-"
                    f"{nominal_density}pct"
                ),
                "material": "Bambu Lab PLA Basic, purple",
                "pattern": pattern,
                "nominal_infill_percent": nominal_density,
                "evidence_class": "published_figure_digitization",
                **point,
                "compressive_stress_x1000_unit_sensitivity_MPa": (
                    float(point["compressive_stress_Pa"]) / 1000.0
                ),
            }
            for point in points
        ]
        _write_csv(
            curve_dir / f"{pattern}_{nominal_density}pct.csv", rows
        )
    all_pass = all(
        bool(row["quality_gate_passed"]) for row in validation
    )
    provenance = {
        "material": "Bambu Lab PLA Basic, purple",
        "article": ARTICLE_TITLE,
        "authors": [
            "Kyle J. Woody",
            "Ian Zhang",
            "Javid Hamidov",
            "Matthew Hong",
            "Hayden K. Taylor",
        ],
        "publication_status": (
            "TechRxiv preprint; not peer reviewed at source capture"
        ),
        "license": "CC BY 4.0",
        "source_capture_date": "2026-07-28",
        "techrxiv_doi": TECHRXIV_DOI,
        "researchgate_doi": RESEARCHGATE_DOI,
        "article_url": ARTICLE_URL,
        "pdf_url": PDF_URL,
        "source_figure": {
            "filename": FIGURE_FILENAME,
            **source_check,
            "figure_number": 4,
            "acquisition_note": (
                "ResearchGate public figure asset; the publisher PDF host "
                "was unavailable in the local environment"
            ),
        },
        "experimental_grain": {
            "specimen": "30 x 30 x 30 mm compression cube",
            "patterns": list(PATTERNS),
            "nominal_infill_percent": [10, 20, 30],
            "wall_thickness_mm": 0.42,
            "replicates_per_condition": 3,
            "physical_tests": 27,
            "compression_endpoint_strain": 0.75,
            "strain_rate_per_s": 0.005,
            "printer": "Bambu Lab X1 Carbon",
        },
        "plot_axis_calibration": {
            "panel_boxes_px": {
                str(key): list(value)
                for key, value in PLOT_BOXES.items()
            },
            "top_px": PLOT_TOP_PX,
            "bottom_px": PLOT_BOTTOM_PX,
            "strain_range": [0.0, PLOT_MAX_STRAIN],
            "stress_range_Pa": [0.0, PLOT_MAX_STRESS_PA],
            "stress_axis_note": (
                "The published figure explicitly labels Pa and displays "
                "a x10^4 multiplier. Literal values are retained. A separate "
                "x1000 sensitivity is provided because all nine adjusted "
                "curve peaks reproduce the paper's independently stated MPa "
                "yield strengths within 4%."
            ),
        },
        "curve_grain": (
            "nine aggregate plotted responses with visible standard-error "
            "bands; not 27 raw specimen curves"
        ),
        "internal_unit_consistency": {
            "status": "conflict",
            "literal_curve_units": "Pa with x10^4 plot multiplier",
            "independent_text_units": "yield strength in MPa",
            "sensitivity_factor": 1000.0,
            "maximum_adjusted_peak_to_yield_relative_error": max(
                abs(
                    float(
                        row[
                            "x1000_peak_to_reported_yield_relative_error"
                        ]
                    )
                )
                for row in validation
            ),
            "interpretation": (
                "The figure is likely off by a factor of 1000 in its stress "
                "unit label or upstream conversion, but the importer does not "
                "choose between the conflicting publication statements."
            ),
        },
        "quality_gate_passed": all_pass,
        "allowed_use": (
            "external topology-ranking and qualitative curve-shape "
            "validation for matched compression screens"
        ),
        "prohibited_use": (
            "not calibration data, not raw replicates, and not sufficient "
            "to identify failure, creep, fatigue, moisture, or rate laws"
        ),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Exact-brand common-infill compression evidence",
        "",
        "## Result",
        "",
        "Figure 4 yields nine checksum-pinned aggregate stress–strain curves "
        "for Bambu Lab purple PLA Basic: grid, gyroid, and cubic infill at "
        "10%, 20%, and 30% nominal slicer infill. All nine digitizations pass "
        "the pixel-coverage, continuity, monotonic-strain, and nonnegative-"
        "stress gates.",
        "",
        "The accompanying mass table contains three-replicate mean and "
        "standard deviation values for all nine conditions. Because the "
        "30 mm cubes include 0.42 mm walls, total measured solid fraction is "
        "substantially greater than the nominal internal-infill setting.",
        "",
        "## Data-quality boundary",
        "",
        "- The source is a TechRxiv preprint and was not peer reviewed at "
        "capture.",
        "- The nine traces are aggregate plotted responses, not 27 raw "
        "specimen curves.",
        "- The stress axis is retained exactly as printed: Pa with a "
        "×10^4 multiplier. It conflicts with the paper's MPa yield-strength "
        "text by a factor of 1,000.",
        "- A separate ×1,000 sensitivity is exported. Its nine curve peaks "
        "cross-check the independently stated yield strengths within 4%, "
        "which strongly suggests a publication-unit error without proving "
        "which upstream field is wrong.",
        "- The curves can challenge topology ranking and response shape, but "
        "cannot identify a transferable material failure law.",
        "",
        "## Digitized metrics",
        "",
        "| Infill | Pattern | Literal modulus (Pa) | ×1000 peak (MPa) | "
        "Reported yield (MPa) | Peak error | Coverage |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in validation:
        report.append(
            f"| {row['nominal_infill_percent']}% | "
            f"{row['pattern']} | "
            f"{float(row['initial_effective_modulus_Pa']):.0f} | "
            f"{float(row['first_peak_x1000_unit_sensitivity_MPa']):.2f} | "
            f"{float(row['reported_yield_strength_MPa']):.2f} | "
            f"{float(row['x1000_peak_to_reported_yield_relative_error']):.1%} | "
            f"{float(row['horizontal_pixel_coverage']):.1%} |"
        )
    (output_dir / "data_quality_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return {
        "curves": len(curves),
        "mass_conditions": len(masses),
        "quality_gate_passed": all_pass,
        "output_directory": str(output_dir.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import exact-brand common-infill compression evidence"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = import_reference(args.source, args.output)
    print(
        f"imported {result['curves']} curves and "
        f"{result['mass_conditions']} mass conditions; "
        f"quality_gate_passed={result['quality_gate_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
